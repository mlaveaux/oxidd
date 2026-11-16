//! Recursive saturation, after
//!
//! > T. van Dijk, J. Meijer, J. van de Pol, *Multi-core on-the-fly
//! > saturation*, TACAS 2019
//! > ([doi:10.1007/978-3-030-17465-1_4](https://doi.org/10.1007/978-3-030-17465-1_4)).
//!
//! Given a set of vectors and a list of events, saturation computes the vectors
//! that are reachable by firing the events in any order. An event is a
//! transition relation that only touches the positions from its `top` on (see
//! [`SaturationEvent`]). Saturation closes the set under the events of the
//! deeper levels before those of a level are fired, which keeps the
//! intermediate diagrams small.
//!
//! Unlike the node-wise saturation of Ciardo et al., this never updates nodes
//! in place and fires the events with the ordinary relational product, so the
//! only extra state is the memoization of the saturated nodes. At the `top`
//! level of a group of events, the node is closed by firing each event on the
//! whole node, and saturating the children again whenever that grew it. Above
//! it, the children of a node are independent and saturated in parallel.
//!
//! The relations of the events are provided by a closure that is called with
//! the node an event is about to fire on. This allows learning the relations
//! on the fly, for exactly the states that saturation reaches.

use oxidd_core::util::{AllocResult, EdgeDropGuard, Own, Ref};
use oxidd_core::{ApplyCache, LevelNo, Node};

use crate::apply_rec::{apply_relational_successors, apply_union, spine};
use crate::recursor::{Recursor, SequentialRecursor};
use crate::{LDDManager, LDDOp, LDDTerminal, SaturationEvent, make_node, stat};

/// Returns `N*(q)`, the smallest superset of the vectors denoted by `q` that is
/// closed under every event in `events` whose `top` is `>= level`, where `q`
/// resides at `level`.
///
/// Before event `e` fires on a node `set` at its `top` level, saturation calls
/// `relation(e, set)` and fires the relation it returns. The relation may grow
/// between calls, for example by learning the transitions of the vectors in
/// `set`. With a parallel recursor, `relation` is called from any worker, also
/// concurrently.
///
/// The results are memoized in the apply cache, keyed on `epoch`, which stands
/// in for the relations: calls with the same `epoch` share their cached
/// results, so it has to differ whenever the relations differ. In particular, a
/// result that was memoized before a relation grew does not include the new
/// transitions, so a caller that learns relations has to repeat the call with a
/// new epoch until the relations no longer change.
pub(crate) fn saturate<M, R, F>(
    manager: &M,
    rec: R,
    level: LevelNo,
    q: Ref<'_, M::Edge>,
    events: &[SaturationEvent<Ref<'_, M::Edge>>],
    epoch: u32,
    relation: &F,
) -> AllocResult<Own<M::Edge>>
where
    M: LDDManager,
    R: SatRecursor<M, F>,
    F: ?Sized + Fn(usize, Ref<'_, M::Edge>) -> AllocResult<Own<M::Edge>>,
{
    let levels = events.iter().map(|e| e.top as usize + 1).max().unwrap_or(0);
    let mut events_at = vec![Vec::new(); levels];
    for (index, event) in events.iter().enumerate() {
        events_at[event.top as usize].push(index);
    }

    let saturation = Saturation {
        manager,
        events,
        events_at,
        epoch,
        relation,
    };
    saturation.saturate(rec, level, q)
}

/// The information that stays the same during a whole saturation run.
pub(crate) struct Saturation<'a, M: LDDManager, F: ?Sized> {
    manager: &'a M,
    events: &'a [SaturationEvent<Ref<'a, M::Edge>>],
    /// The indices of the events per `top` level.
    events_at: Vec<Vec<usize>>,
    /// The number that the cache entries of this call are keyed on.
    epoch: u32,
    /// Returns the relation of an event, see [`saturate`].
    relation: &'a F,
}

/// The `(value, saturated child)` pairs of a node, in spine order.
type Children<M> = Vec<(
    <M as oxidd_core::Manager>::InnerNodeValue,
    Own<<M as oxidd_core::Manager>::Edge>,
)>;

/// How saturation recurses on the independent sub-problems: one after the
/// other, or forked onto the worker pool.
pub(crate) trait SatRecursor<M: LDDManager, F: ?Sized>: Recursor<M> {
    /// Saturates the first `len` children of the spine `spine` at `level + 1`.
    fn saturate_children(
        self,
        sat: &Saturation<'_, M, F>,
        level: LevelNo,
        spine: Ref<'_, M::Edge>,
        len: usize,
    ) -> AllocResult<Children<M>>;

    /// Fires the events `events` of `level` on `current`, and returns
    /// `current` united with the successors.
    ///
    /// Not every successor of a successor needs to be included: the caller
    /// repeats this until nothing changes.
    fn fire_events(
        self,
        sat: &Saturation<'_, M, F>,
        level: LevelNo,
        current: Ref<'_, M::Edge>,
        events: &[usize],
    ) -> AllocResult<Own<M::Edge>>;
}

impl<M, F> SatRecursor<M, F> for SequentialRecursor
where
    M: LDDManager,
    F: ?Sized + Fn(usize, Ref<'_, M::Edge>) -> AllocResult<Own<M::Edge>>,
{
    fn saturate_children(
        self,
        sat: &Saturation<'_, M, F>,
        level: LevelNo,
        spine: Ref<'_, M::Edge>,
        len: usize,
    ) -> AllocResult<Children<M>> {
        sat.saturate_children_seq(self, level, spine, len)
    }

    fn fire_events(
        self,
        sat: &Saturation<'_, M, F>,
        level: LevelNo,
        current: Ref<'_, M::Edge>,
        events: &[usize],
    ) -> AllocResult<Own<M::Edge>> {
        sat.fire_events_seq(self, level, current, events)
    }
}

impl<M, F> Saturation<'_, M, F>
where
    M: LDDManager,
    F: ?Sized + Fn(usize, Ref<'_, M::Edge>) -> AllocResult<Own<M::Edge>>,
{
    /// Returns `N*(q)` for `q` at `level`, memoized on `q` and the epoch.
    fn saturate<R: SatRecursor<M, F>>(
        &self,
        rec: R,
        level: LevelNo,
        q: Ref<'_, M::Edge>,
    ) -> AllocResult<Own<M::Edge>> {
        let manager = self.manager;
        stat!(call LDDOp::Saturate);

        // Terminals are fixed points, and below the deepest top no event applies.
        if matches!(manager.get_node(q), Node::Terminal(_))
            || level as usize >= self.events_at.len()
        {
            return Ok(manager.clone_edge(q));
        }

        stat!(cache_query LDDOp::Saturate);
        if let Some(([result], [])) =
            manager
                .apply_cache()
                .get_extended(manager, LDDOp::Saturate, (&[q], &[self.epoch]))
        {
            stat!(cache_hit LDDOp::Saturate);
            return Ok(result);
        }

        let mut current = EdgeDropGuard::new(manager, self.saturate_below(rec, level, q)?);
        let events_here = &self.events_at[level as usize];
        if !events_here.is_empty() {
            // Fire the events of this level until nothing changes, saturating
            // the children again after a round that grew the node.
            loop {
                let before = EdgeDropGuard::new(manager, manager.clone_edge(current.borrowed()));
                current = EdgeDropGuard::new(
                    manager,
                    rec.fire_events(self, level, current.borrowed(), events_here)?,
                );

                if current.borrowed() == before.borrowed() {
                    break;
                }
                current = EdgeDropGuard::new(
                    manager,
                    self.saturate_below(rec, level, current.borrowed())?,
                );
            }
        }

        let result = current.into_edge();
        manager.apply_cache().add_extended(
            manager,
            LDDOp::Saturate,
            (&[q], &[self.epoch]),
            (&[result.borrowed()], &[]),
        );
        Ok(result)
    }

    /// Fires event `e` (of `level`) on `current` and returns `current` united
    /// with its successors.
    fn fire<R: Recursor<M>>(
        &self,
        rec: R,
        level: LevelNo,
        current: Ref<'_, M::Edge>,
        e: usize,
    ) -> AllocResult<Own<M::Edge>> {
        let manager = self.manager;
        // Operations of the closure continue this recursion's split budget.
        let relation = EdgeDropGuard::new(manager, rec.scoped(|| (self.relation)(e, current))?);
        let successors = EdgeDropGuard::new(
            manager,
            apply_relational_successors(
                manager,
                rec,
                level,
                current,
                relation.borrowed(),
                self.events[e].meta_at_top,
            )?,
        );
        apply_union(manager, rec, level, current, successors.borrowed())
    }

    /// Returns the node at `level` with the values of `q` and their children
    /// saturated, which is `q` closed under the events below `level`.
    fn saturate_below<R: SatRecursor<M, F>>(
        &self,
        rec: R,
        level: LevelNo,
        q: Ref<'_, M::Edge>,
    ) -> AllocResult<Own<M::Edge>> {
        let manager = self.manager;
        let len = spine(manager, q).count();
        let mut children = rec
            .saturate_children(self, level, q, len)?
            .into_iter()
            .rev();

        let mut result = EdgeDropGuard::new(manager, manager.get_terminal(LDDTerminal::Empty)?);
        while let Some((value, down)) = children.next() {
            match make_node(manager, level, &value, down, result.into_edge()) {
                Ok(node) => result = EdgeDropGuard::new(manager, node),
                Err(err) => {
                    for (_, down) in children {
                        manager.drop_edge(down);
                    }
                    return Err(err);
                }
            }
        }
        Ok(result.into_edge())
    }

    /// Implements [`SatRecursor::fire_events`] without splitting: each event
    /// fires on the result of the previous one.
    fn fire_events_seq<R: Recursor<M>>(
        &self,
        rec: R,
        level: LevelNo,
        current: Ref<'_, M::Edge>,
        events: &[usize],
    ) -> AllocResult<Own<M::Edge>> {
        let manager = self.manager;
        let mut current = EdgeDropGuard::new(manager, manager.clone_edge(current));
        for &e in events {
            current = EdgeDropGuard::new(manager, self.fire(rec, level, current.borrowed(), e)?);
        }
        Ok(current.into_edge())
    }

    /// Implements [`SatRecursor::saturate_children`] without splitting.
    fn saturate_children_seq<R: SatRecursor<M, F>>(
        &self,
        rec: R,
        level: LevelNo,
        q: Ref<'_, M::Edge>,
        len: usize,
    ) -> AllocResult<Children<M>> {
        let mut children = Vec::with_capacity(len);
        for (value, down) in spine(self.manager, q).take(len) {
            match self.saturate(rec, level + 1, down) {
                Ok(down) => children.push((value.clone(), down)),
                Err(err) => {
                    for (_, down) in children {
                        self.manager.drop_edge(down);
                    }
                    return Err(err);
                }
            }
        }
        Ok(children)
    }
}

#[cfg(feature = "multi-threading")]
mod mt {
    use oxidd_core::util::{AllocResult, Own, Ref};
    use oxidd_core::{HasWorkers, LevelNo};

    use oxidd_core::util::EdgeDropGuard;

    use crate::apply_rec::apply_union;
    use crate::recursor::mt::ParallelRecursor;
    use crate::recursor::{Recursor, SequentialRecursor};
    use crate::{LDDManager, collect_children};

    use super::{Children, SatRecursor, Saturation};

    /// Returns the spine of `set` without its first `n` entries.
    fn spine_skip<'a, M: LDDManager>(
        manager: &'a M,
        mut set: Ref<'a, M::Edge>,
        n: usize,
    ) -> Ref<'a, M::Edge> {
        for _ in 0..n {
            let (_, right) = collect_children(manager.get_node(set).unwrap_inner());
            set = right;
        }
        set
    }

    /// Splits the children of a node, and the events of a level, in two halves
    /// that are processed in parallel, until the split depth is reached. The
    /// relation closure is called from every worker, so it must be `Sync`.
    impl<M, F> SatRecursor<M, F> for ParallelRecursor
    where
        M: LDDManager + HasWorkers,
        M::Edge: Send + Sync,
        M::InnerNodeValue: Send + Sync,
        F: ?Sized + Fn(usize, Ref<'_, M::Edge>) -> AllocResult<Own<M::Edge>> + Sync,
    {
        fn saturate_children(
            self,
            sat: &Saturation<'_, M, F>,
            level: LevelNo,
            spine: Ref<'_, M::Edge>,
            len: usize,
        ) -> AllocResult<Children<M>> {
            if Recursor::<M>::should_switch_to_sequential(self) {
                return sat.saturate_children_seq(SequentialRecursor, level, spine, len);
            }
            if len < 2 {
                return sat.saturate_children_seq(self, level, spine, len);
            }

            let manager = sat.manager;
            let mid = len / 2;
            let rest = spine_skip(manager, spine, mid);
            let (first, second) = self.join(
                manager,
                |rec| rec.saturate_children(sat, level, spine, mid),
                |rec| rec.saturate_children(sat, level, rest, len - mid),
            );
            match (first, second) {
                (Ok(mut first), Ok(second)) => {
                    first.extend(second);
                    Ok(first)
                }
                (first, second) => {
                    let error = match (&first, &second) {
                        (Err(error), _) | (_, Err(error)) => *error,
                        _ => unreachable!("one of the halves failed"),
                    };
                    for (_, down) in first.into_iter().chain(second).flatten() {
                        manager.drop_edge(down);
                    }
                    Err(error)
                }
            }
        }

        /// Fires both halves of the events on `current` (Jacobi) and unites
        /// the results; within a half, the events fire one after the other.
        fn fire_events(
            self,
            sat: &Saturation<'_, M, F>,
            level: LevelNo,
            current: Ref<'_, M::Edge>,
            events: &[usize],
        ) -> AllocResult<Own<M::Edge>> {
            if Recursor::<M>::should_switch_to_sequential(self) {
                return sat.fire_events_seq(SequentialRecursor, level, current, events);
            }
            if events.len() < 2 {
                return sat.fire_events_seq(self, level, current, events);
            }

            let manager = sat.manager;
            let (first, second) = events.split_at(events.len() / 2);
            let (first, second) = self.join(
                manager,
                |rec| rec.fire_events(sat, level, current, first),
                |rec| rec.fire_events(sat, level, current, second),
            );
            let first = EdgeDropGuard::new(manager, first?);
            let second = EdgeDropGuard::new(manager, second?);
            apply_union(manager, self, level, first.borrowed(), second.borrowed())
        }
    }
}
