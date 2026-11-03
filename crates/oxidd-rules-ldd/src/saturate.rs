//! Node-wise MDD/LDD saturation as described in
//!
//! > Ciardo, Marmorstein, Siminiceanu. *The saturation algorithm for symbolic
//! > state-space exploration*, STTT 2006.
//!
//! Given a set of vectors `q` and a list of events, saturation computes the
//! vectors that are reachable from `q` by firing the events in any order. An
//! event is a transition relation that only touches the positions `top..=bot`
//! of the vectors (see [`SaturationEvent`]). Instead of firing all events on
//! the whole set until nothing changes, saturation brings one *node* at a time
//! to a fixed point under every event confined to its own level and below,
//! bottom-up, before that node is ever used as anyone's child. So a node is
//! only ever built once, in its final form, instead of accumulating
//! intermediate breadth-first versions.
//!
//! Level `i` holds the nodes for position `i` of the vectors, so levels run
//! from `0` (top) to `K - 1` (bottom), where `K` is the length of the vectors,
//! which is the reverse of the paper. LDD nodes are `(value, down, right)`
//! triples: `down` continues at the next level, and `right` is the next value
//! for the same position. The spine formed by the `right` edges is sorted
//! ascending and ends at the Empty terminal.
//!
//! Like the apply algorithms, saturation is generic over the recursion strategy
//! (see [`SatRecursor`]). The independent sub-problems are the children of a
//! node, which are saturated before the node itself, and the two halves of
//! every step of firing an event. The node-local fixpoint itself is inherently
//! sequential in its accumulator, but the unions and firings it performs are
//! split as well.

// spell-checker:ignore Ciardo,Marmorstein,Siminiceanu,STTT,unioned

use std::cmp::Ordering;
use std::ops::ControlFlow;

use oxidd_core::util::{AllocResult, EdgeDropGuard, Own, Ref};
use oxidd_core::{ApplyCache, InnerNode, LevelNo, Node};

use crate::apply_rec::{
    SuccessorsArgs, SuccessorsRecursion, apply_union, relational_successors_step,
    relational_successors_terminal, spine,
};
use crate::recursor::{Recursor, SequentialRecursor};
use crate::{
    LDDManager, LDDOp, LDDTerminal, LDDValue, SaturationEvent, collect_children, make_node, stat,
};

/// Returns `N*(q)`, the smallest superset of the vectors denoted by `q` that is
/// closed under every event in `events` whose `top` is `>= level`, where `q`
/// resides at `level`.
///
/// The results are memoized in the apply cache, keyed on `epoch`, which
/// identifies `events`: calls with the same `epoch` share their cached results,
/// so it has to differ whenever the events differ (see
/// [`LDDFunction::saturate_edge`][crate::LDDFunction::saturate_edge]).
pub(crate) fn saturate<M: LDDManager, R: SatRecursor<M>>(
    manager: &M,
    rec: R,
    level: LevelNo,
    q: Ref<'_, M::Edge>,
    events: &[SaturationEvent<Ref<'_, M::Edge>>],
    epoch: u32,
) -> AllocResult<Own<M::Edge>> {
    let saturation = Saturation {
        manager,
        events,
        epoch,
    };
    let mut scratch = Scratch::new();

    let result = saturation.saturate_rec(rec, &mut scratch, level, q);
    debug_assert!(scratch.is_empty());
    result
}

/// How saturation recurses on independent sub-problems
///
/// This is the counterpart of [`Recursor`] for saturation: its recursive calls
/// need the events and the [`Scratch`] buffers, so they cannot be expressed by
/// the plain operator functions that a [`Recursor`] takes. A
/// [`SequentialRecursor`] evaluates the sub-problems one after the other in
/// the shared buffers, while a [parallel
/// recursor][crate::recursor::mt::ParallelRecursor] forks them onto the
/// worker pool, giving the forked half buffers of its own.
pub(crate) trait SatRecursor<M: LDDManager>: Recursor<M> {
    /// Saturates the down-branches of the first `len` entries of the spine
    /// `spine`, which resides at `level`, and pushes the `(value, saturated
    /// down-branch)` pairs onto `scratch.arcs` in spine order.
    ///
    /// If it fails, the pairs that were already pushed are left in
    /// `scratch.arcs` for the caller to release.
    fn saturate_children(
        self,
        sat: &Saturation<'_, M>,
        scratch: &mut Scratch<M>,
        level: LevelNo,
        spine: Ref<'_, M::Edge>,
        len: usize,
    ) -> AllocResult<()>;

    /// Runs [`Saturation::fire`] for `first` and `second`, which are the two
    /// recursive calls of a step that builds a node at `node_level`.
    fn fire_pair<'m>(
        self,
        manager: &'m M,
        sat: &Saturation<'_, M>,
        scratch: &mut Scratch<M>,
        node_level: LevelNo,
        first: SuccessorsArgs<'_, M>,
        second: SuccessorsArgs<'_, M>,
    ) -> AllocResult<(EdgeDropGuard<'m, M>, EdgeDropGuard<'m, M>)>;
}

impl<M: LDDManager> SatRecursor<M> for SequentialRecursor {
    #[inline(always)]
    fn saturate_children(
        self,
        sat: &Saturation<'_, M>,
        scratch: &mut Scratch<M>,
        level: LevelNo,
        spine: Ref<'_, M::Edge>,
        len: usize,
    ) -> AllocResult<()> {
        sat.saturate_children_seq(self, scratch, level, spine, len)
    }

    #[inline(always)]
    fn fire_pair<'m>(
        self,
        manager: &'m M,
        sat: &Saturation<'_, M>,
        scratch: &mut Scratch<M>,
        node_level: LevelNo,
        first: SuccessorsArgs<'_, M>,
        second: SuccessorsArgs<'_, M>,
    ) -> AllocResult<(EdgeDropGuard<'m, M>, EdgeDropGuard<'m, M>)> {
        let first = EdgeDropGuard::new(manager, sat.fire(self, scratch, node_level, first)?);
        let second = EdgeDropGuard::new(manager, sat.fire(self, scratch, node_level, second)?);
        Ok((first, second))
    }
}

/// The information that stays the same during a whole saturation run.
///
/// Keeping it in one place, rather than in the parameters of every recursive
/// call, keeps the stack frames small.
pub(crate) struct Saturation<'a, M: LDDManager> {
    manager: &'a M,
    /// Every event that can be fired.
    events: &'a [SaturationEvent<Ref<'a, M::Edge>>],
    /// The number that identifies `events`, and that the cache entries of this
    /// call are keyed on.
    epoch: u32,
}

/// Buffers that are shared by every recursive call, such that saturating a
/// node does not allocate.
///
/// All of them are used as stacks: a call remembers the length on entry, only
/// touches the entries above it, and truncates back to that length before
/// returning. Nested calls therefore never observe or disturb the entries of
/// their callers. All of them are empty again when the outermost call returns.
///
/// A sub-problem that is forked onto another thread gets buffers of its own.
pub(crate) struct Scratch<M: LDDManager> {
    /// The accumulator of the node being saturated: its `(value,
    /// continuation)` pairs, sorted ascending by value. The continuation of a
    /// value is replaced by a bigger one whenever firing an event grows it, and
    /// values that were not in the node can be inserted anywhere.
    arcs: Vec<(M::InnerNodeValue, Own<M::Edge>)>,
    /// The values of the node being saturated whose continuation still has to
    /// be fired, used as a FIFO queue.
    frontier: Vec<M::InnerNodeValue>,
    /// The `(value, subtree)` pairs produced by firing the events out of one
    /// value.
    fired: Vec<(M::InnerNodeValue, Own<M::Edge>)>,
}

impl<M: LDDManager> Scratch<M> {
    fn new() -> Self {
        Scratch {
            arcs: Vec::new(),
            frontier: Vec::new(),
            fired: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.arcs.is_empty() && self.frontier.is_empty() && self.fired.is_empty()
    }
}

impl<M: LDDManager> Saturation<'_, M> {
    /// Implements [`saturate`], using (and restoring) the shared `scratch`
    /// buffers.
    ///
    /// Memoized on node identity and the epoch via [`LDDOp::Saturate`]; see
    /// [`Self::saturate_node`] for the computation itself. If it fails,
    /// everything it left in `scratch` is released.
    fn saturate_rec<R: SatRecursor<M>>(
        &self,
        rec: R,
        scratch: &mut Scratch<M>,
        level: LevelNo,
        q: Ref<'_, M::Edge>,
    ) -> AllocResult<Own<M::Edge>> {
        if rec.should_switch_to_sequential() {
            return self.saturate_rec(SequentialRecursor, scratch, level, q);
        }
        
        let manager = self.manager;
        stat!(call LDDOp::Saturate);

        // Terminals are trivial fixed points of every event.
        if let Node::Terminal(_) = manager.get_node(q) {
            return Ok(manager.clone_edge(q));
        }

        // `q` is an inner node, so it determines `level`.
        stat!(cache_query LDDOp::Saturate);
        if let Some(([res], [])) =
            manager
                .apply_cache()
                .get_extended(manager, LDDOp::Saturate, (&[q], &[self.epoch]))
        {
            stat!(cache_hit LDDOp::Saturate);
            return Ok(res);
        }

        let arcs_base = scratch.arcs.len();
        let frontier_base = scratch.frontier.len();
        let fired_base = scratch.fired.len();
        let result = match self.saturate_node(rec, scratch, level, q, arcs_base, frontier_base) {
            Ok(result) => result,
            Err(err) => {
                for (_, edge) in scratch.arcs.drain(arcs_base..) {
                    manager.drop_edge(edge);
                }
                scratch.frontier.truncate(frontier_base);
                for (_, edge) in scratch.fired.drain(fired_base..) {
                    manager.drop_edge(edge);
                }
                return Err(err);
            }
        };

        manager.apply_cache().add_extended(
            manager,
            LDDOp::Saturate,
            (&[q], &[self.epoch]),
            (&[result.borrowed()], &[]),
        );

        // `result` is already saturated (it is the fixed point we just
        // computed), so this is a free cache hit for any later node of this
        // call that happens to reach it directly.
        if result.borrowed() != q {
            manager.apply_cache().add_extended(
                manager,
                LDDOp::Saturate,
                (&[result.borrowed()], &[self.epoch]),
                (&[result.borrowed()], &[]),
            );
        }

        Ok(result)
    }

    /// Computes the saturation of the inner node `q` at `level` on a cache
    /// miss.
    ///
    /// The node is saturated in three steps: (1) saturate the children, which
    /// gives the continuations of a node that is closed under all events
    /// *below* `level`, (2) close it under the events *at* `level`, and (3)
    /// build the resulting node once.
    fn saturate_node<R: SatRecursor<M>>(
        &self,
        rec: R,
        scratch: &mut Scratch<M>,
        level: LevelNo,
        q: Ref<'_, M::Edge>,
        arcs_base: usize,
        frontier_base: usize,
    ) -> AllocResult<Own<M::Edge>> {
        let manager = self.manager;

        // (1) Saturate every child. These are independent sub-problems, so the
        // recursor may split them. The spine is ascending, so the accumulator
        // starts out sorted.
        let len = spine(manager, q).count();
        rec.saturate_children(self, scratch, level, q, len)?;

        // (2) Fixpoint over the events confined to this level (`top(e) ==
        // level`): fire every such event out of every value and unite the
        // results into the accumulator, queueing the values whose continuation
        // actually grew (the paper's pipelining).
        scratch.frontier.extend(
            scratch.arcs[arcs_base..]
                .iter()
                .map(|(value, _)| value.clone()),
        );

        let mut next = frontier_base;
        while next < scratch.frontier.len() {
            let i = scratch.frontier[next].clone();
            next += 1;

            let arc_i = {
                let pos = find_arc(&scratch.arcs[arcs_base..], &i)
                    .expect("a queued value is in the accumulator");
                EdgeDropGuard::new(
                    manager,
                    manager.clone_edge(scratch.arcs[arcs_base + pos].1.borrowed()),
                )
            };

            let fired_base = scratch.fired.len();
            for event in self.events.iter().filter(|e| e.top == level) {
                self.sat_fire_top(rec, scratch, level, event, &i, arc_i.borrowed())?;
            }
            drop(arc_i);

            // Popping (instead of draining) keeps the remaining entries in
            // `scratch.fired` if the union fails, so that `saturate_rec`
            // releases them.
            while scratch.fired.len() > fired_base {
                let (j, f) = scratch.fired.pop().unwrap();
                let f = EdgeDropGuard::new(manager, f);

                match scratch.arcs[arcs_base..].binary_search_by(|(value, _)| value.cmp(&j)) {
                    Ok(pos) => {
                        let arc_j = &mut scratch.arcs[arcs_base + pos].1;

                        // Nothing to add if `j` already continues with exactly
                        // `f`.
                        if *arc_j == *f {
                            continue;
                        }

                        // The union of two saturated sets is saturated (the
                        // relational image distributes over union), so no
                        // re-saturation is needed here.
                        let unioned =
                            apply_union(manager, rec, level + 1, arc_j.borrowed(), f.borrowed())?;
                        if unioned == *arc_j {
                            manager.drop_edge(unioned);
                            continue;
                        }

                        manager.drop_edge(std::mem::replace(arc_j, unioned));
                        if !scratch.frontier[next..].contains(&j) {
                            scratch.frontier.push(j);
                        }
                    }
                    Err(pos) => {
                        scratch
                            .arcs
                            .insert(arcs_base + pos, (j.clone(), f.into_edge()));
                        scratch.frontier.push(j);
                    }
                }
            }
        }
        scratch.frontier.truncate(frontier_base);

        // (3) Build the node once, right to left, so the spine ends up
        // ascending.
        let mut result = manager.get_terminal(LDDTerminal::Empty)?;
        while scratch.arcs.len() > arcs_base {
            let (value, down) = scratch.arcs.pop().unwrap();
            result = make_node(manager, level, &value, down, result)?;
        }
        Ok(result)
    }

    /// Implements [`SatRecursor::saturate_children`] without splitting, but
    /// still using `rec` for the children themselves.
    fn saturate_children_seq<R: SatRecursor<M>>(
        &self,
        rec: R,
        scratch: &mut Scratch<M>,
        level: LevelNo,
        q: Ref<'_, M::Edge>,
        len: usize,
    ) -> AllocResult<()> {
        for (value, down) in spine(self.manager, q).take(len) {
            let down_sat = self.saturate_rec(rec, scratch, level + 1, down)?;
            scratch.arcs.push((value.clone(), down_sat));
        }
        Ok(())
    }

    /// Fires `event`, whose `top` is `level`, the level of the node that is
    /// being saturated, out of the single value `i` of that node. The
    /// continuation of `i` is `arc_i`, and it is already saturated. Pushes the
    /// resulting `(value, subtree)` pairs onto `scratch.fired`, where every
    /// subtree is already saturated by [`Self::sat_rec_fire`].
    ///
    /// This is the case analysis of [`relational_successors_step`] for the
    /// first meta level, restricted to the single source value `i` instead of a
    /// whole spine: the source is `(i, arc_i)`, so nothing has to be built for
    /// it, and the entries of the relation are looked up directly.
    ///
    /// It is deliberately *not* cached (it is only reached from
    /// [`Self::saturate_rec`], which is), and it does *not* saturate the node
    /// it contributes to: only [`Self::sat_rec_fire`], which works strictly
    /// below `top(e)`, saturates. Saturating here would call `saturate` for a
    /// node at this same level, while the saturation of a node at this level
    /// may still be on the call stack, which would not terminate.
    fn sat_fire_top<R: SatRecursor<M>>(
        &self,
        rec: R,
        scratch: &mut Scratch<M>,
        level: LevelNo,
        event: &SaturationEvent<Ref<'_, M::Edge>>,
        i: &M::InnerNodeValue,
        arc_i: Ref<'_, M::Edge>,
    ) -> AllocResult<()> {
        let manager = self.manager;

        // Everything is borrowed from the event or from `arc_i`, both of which
        // outlive this call.
        let Node::Inner(meta_node) = manager.get_node(event.meta_at_top) else {
            unreachable!("an event always reads or writes at least one position")
        };
        let meta_value = meta_node.value();
        let (meta_down, _meta_right) = collect_children(meta_node);

        if *meta_value == M::InnerNodeValue::READ_ONLY {
            // Read-only: the value is unaffected; only fire if the relation has
            // an entry matching `i`.
            if let Some(rel_down) = spine_lookup(manager, event.relation, i) {
                let f = self.sat_rec_fire(rec, scratch, level + 1, arc_i, rel_down, meta_down)?;
                self.push_fired(scratch, i, f);
            }
        } else if *meta_value == M::InnerNodeValue::WRITE_ONLY {
            // Write-only: every relation value is a possible successor, all
            // sharing `arc_i` as their (unconstrained by `i`) source.
            for (value, down) in spine(manager, event.relation) {
                let f = self.sat_rec_fire(rec, scratch, level + 1, arc_i, down, meta_down)?;
                self.push_fired(scratch, value, f);
            }
        } else if *meta_value == M::InnerNodeValue::READ_OF_PAIR {
            // Read half of a pair: match the relation entry equal to `i`.
            // `meta_down` is the write half, whose own down-branch enumerates
            // the possible written values.
            if let Some(rel_down) = spine_lookup(manager, event.relation, i) {
                let write_meta_node = manager.get_node(meta_down).unwrap_inner();
                debug_assert!(*write_meta_node.value() == M::InnerNodeValue::WRITE_OF_PAIR);
                let (write_meta_down, _) = collect_children(write_meta_node);

                for (value, down) in spine(manager, rel_down) {
                    let f =
                        self.sat_rec_fire(rec, scratch, level + 1, arc_i, down, write_meta_down)?;
                    self.push_fired(scratch, value, f);
                }
            }
        } else {
            panic!("meta_at_top has an unexpected value");
        }

        Ok(())
    }

    /// Pushes `(value, f)` onto `scratch.fired`, unless `f` is empty.
    fn push_fired(&self, scratch: &mut Scratch<M>, value: &M::InnerNodeValue, f: Own<M::Edge>) {
        if self
            .manager
            .get_node(f.borrowed())
            .is_terminal(&LDDTerminal::Empty)
        {
            self.manager.drop_edge(f);
        } else {
            scratch.fired.push((value.clone(), f));
        }
    }

    /// Fires the part `rel`/`meta` of an event on `q`, where the result resides
    /// at `level`, which lies below the event's `top`. The result is saturated
    /// at `level` before it is returned, so it is safe to use as anyone's
    /// child.
    ///
    /// Memoized on node identity, the epoch and `level` via
    /// [`LDDOp::SatRecFire`]. As for the relational successors, the operands
    /// do not always determine the result level, so it is part of the key.
    /// Only calls that end up at a new level are memoized, and those are
    /// exactly the ones that saturate.
    fn sat_rec_fire<R: SatRecursor<M>>(
        &self,
        rec: R,
        scratch: &mut Scratch<M>,
        level: LevelNo,
        q: Ref<'_, M::Edge>,
        rel: Ref<'_, M::Edge>,
        meta: Ref<'_, M::Edge>,
    ) -> AllocResult<Own<M::Edge>> {
        if rec.should_switch_to_sequential() {
            return self.sat_rec_fire(SequentialRecursor, scratch, level, q, rel, meta);
        }
        let manager = self.manager;
        stat!(call LDDOp::SatRecFire);

        // `meta == True` means that every meta level of the event is consumed,
        // i.e. `level > bot(e)`. Below `bot(e)` the event is the identity, and
        // `q` is already saturated (its creator did that before it was ever
        // used as a child), so it is returned unchanged.
        let nodes = match relational_successors_terminal(manager, q, rel, meta)? {
            ControlFlow::Break(result) => return Ok(result),
            ControlFlow::Continue(nodes) => nodes,
        };

        stat!(cache_query LDDOp::SatRecFire);
        if let Some(([res], [])) = manager.apply_cache().get_extended(
            manager,
            LDDOp::SatRecFire,
            (&[q, rel, meta], &[self.epoch, level]),
        ) {
            stat!(cache_hit LDDOp::SatRecFire);
            return Ok(res);
        }

        let raw = relational_successors_step(
            manager,
            &mut FireRecursion {
                saturation: self,
                scratch: &mut *scratch,
                rec,
                level,
            },
            level,
            q,
            rel,
            meta,
            nodes,
        )?;
        let raw = EdgeDropGuard::new(manager, raw);
        let result = self.saturate_rec(rec, scratch, level, raw.borrowed())?;

        manager.apply_cache().add_extended(
            manager,
            LDDOp::SatRecFire,
            (&[q, rel, meta], &[self.epoch, level]),
            (&[result.borrowed()], &[]),
        );

        Ok(result)
    }

    /// The uncached counterpart of [`Self::sat_rec_fire`] for continuations
    /// that stay at `level`: the remaining entries of a spine, or the write
    /// half of a read+write pair. Nothing is saturated here, since the node
    /// that is being built is not finished yet; its creator saturates it.
    ///
    /// This is a linear spine walk, so it needs no cache of its own (mirroring
    /// the paper's split between `SatFire` at the top and `SatRecFire` below
    /// it).
    fn rec_fire<R: SatRecursor<M>>(
        &self,
        rec: R,
        scratch: &mut Scratch<M>,
        level: LevelNo,
        q: Ref<'_, M::Edge>,
        rel: Ref<'_, M::Edge>,
        meta: Ref<'_, M::Edge>,
    ) -> AllocResult<Own<M::Edge>> {
        if rec.should_switch_to_sequential() {
            return self.rec_fire(SequentialRecursor, scratch, level, q, rel, meta);
        }
        let manager = self.manager;

        // A continuation to the right may run off the end of a spine, or a
        // relation branch may be empty: both simply mean there is nothing left
        // to fire.
        let nodes = match relational_successors_terminal(manager, q, rel, meta)? {
            ControlFlow::Break(result) => return Ok(result),
            ControlFlow::Continue(nodes) => nodes,
        };

        relational_successors_step(
            manager,
            &mut FireRecursion {
                saturation: self,
                scratch,
                rec,
                level,
            },
            level,
            q,
            rel,
            meta,
            nodes,
        )
    }

    /// Computes the relational successors of `args` while firing an event, in
    /// a step that builds a node at `node_level`: continuations that stay at
    /// `node_level` are fired by [`Self::rec_fire`], and those that descend to
    /// the next level by the saturating [`Self::sat_rec_fire`].
    fn fire<R: SatRecursor<M>>(
        &self,
        rec: R,
        scratch: &mut Scratch<M>,
        node_level: LevelNo,
        SuccessorsArgs {
            level,
            set,
            rel,
            meta,
        }: SuccessorsArgs<'_, M>,
    ) -> AllocResult<Own<M::Edge>> {
        if level == node_level {
            self.rec_fire(rec, scratch, level, set, rel, meta)
        } else {
            debug_assert_eq!(level, node_level + 1);
            self.sat_rec_fire(rec, scratch, level, set, rel, meta)
        }
    }
}

/// The recursion of the relational successors that is used to fire events: it
/// is the plain relational successors, except that every node that is built at
/// a new level is saturated.
struct FireRecursion<'s, 'a, M: LDDManager, R> {
    saturation: &'s Saturation<'a, M>,
    scratch: &'s mut Scratch<M>,
    rec: R,
    /// The level of the node built by the current step.
    level: LevelNo,
}

impl<M: LDDManager, R: SatRecursor<M>> SuccessorsRecursion<M> for FireRecursion<'_, '_, M, R> {
    type Rec = R;

    fn recursor(&self) -> R {
        self.rec
    }

    fn recurse(
        &mut self,
        level: LevelNo,
        set: Ref<'_, M::Edge>,
        rel: Ref<'_, M::Edge>,
        meta: Ref<'_, M::Edge>,
    ) -> AllocResult<Own<M::Edge>> {
        self.saturation.fire(
            self.rec,
            self.scratch,
            self.level,
            SuccessorsArgs {
                level,
                set,
                rel,
                meta,
            },
        )
    }

    fn recurse_pair<'m>(
        &mut self,
        manager: &'m M,
        first: SuccessorsArgs<'_, M>,
        second: SuccessorsArgs<'_, M>,
    ) -> AllocResult<(EdgeDropGuard<'m, M>, EdgeDropGuard<'m, M>)> {
        self.rec.fire_pair(
            manager,
            self.saturation,
            self.scratch,
            self.level,
            first,
            second,
        )
    }
}

/// Walks the spine of `rel` looking for the entry equal to `i`, returning its
/// down-branch. `rel`'s spine is sorted ascending, so the search stops as soon
/// as a strictly greater value is seen. Returns `None` if `rel` is the Empty
/// terminal or has no matching entry.
fn spine_lookup<'a, M: LDDManager>(
    manager: &'a M,
    rel: Ref<'a, M::Edge>,
    i: &M::InnerNodeValue,
) -> Option<Ref<'a, M::Edge>> {
    for (value, down) in spine(manager, rel) {
        match value.cmp(i) {
            Ordering::Less => {}
            Ordering::Equal => return Some(down),
            Ordering::Greater => return None,
        }
    }
    None
}

/// Returns the position of `value` in the accumulator `arcs`, which is sorted
/// ascending by value.
fn find_arc<V: Ord, E>(arcs: &[(V, E)], value: &V) -> Option<usize> {
    arcs.binary_search_by(|(v, _)| v.cmp(value)).ok()
}

/// Returns the rest of the spine of `set` after its first `n` entries.
#[cfg(feature = "multi-threading")]
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

#[cfg(feature = "multi-threading")]
mod mt {
    use oxidd_core::HasWorkers;
    use oxidd_core::LevelNo;
    use oxidd_core::util::{AllocResult, EdgeDropGuard, Ref};

    use crate::LDDManager;
    use crate::apply_rec::SuccessorsArgs;
    use crate::recursor::Recursor;
    use crate::recursor::mt::ParallelRecursor;

    use super::{SatRecursor, Saturation, Scratch, spine_skip};

    impl<M> SatRecursor<M> for ParallelRecursor
    where
        M: LDDManager + HasWorkers,
        M::Edge: Send + Sync,
        M::InnerNodeValue: Send,
    {
        fn saturate_children(
            self,
            sat: &Saturation<'_, M>,
            scratch: &mut Scratch<M>,
            level: LevelNo,
            spine: Ref<'_, M::Edge>,
            len: usize,
        ) -> AllocResult<()> {
            if len < 2 || Recursor::<M>::should_switch_to_sequential(self) {
                return sat.saturate_children_seq(self, scratch, level, spine, len);
            }

            // Split the spine in two halves. The first one pushes its pairs
            // onto `scratch.arcs` directly, the second one onto buffers of its
            // own, which are appended afterwards to keep the spine order.
            let manager = sat.manager;
            let mid = len / 2;
            let rest = spine_skip(manager, spine, mid);
            let (first, (second, mut second_arcs)) = self.join(
                manager,
                |rec| rec.saturate_children(sat, scratch, level, spine, mid),
                |rec| {
                    let mut scratch = Scratch::new();
                    let result = rec.saturate_children(sat, &mut scratch, level, rest, len - mid);
                    debug_assert!(scratch.frontier.is_empty() && scratch.fired.is_empty());
                    (result, scratch.arcs)
                },
            );

            if let Err(err) = first.and(second) {
                // The caller releases the pairs of the first half.
                for (_, edge) in second_arcs {
                    manager.drop_edge(edge);
                }
                return Err(err);
            }
            scratch.arcs.append(&mut second_arcs);
            Ok(())
        }

        fn fire_pair<'m>(
            self,
            manager: &'m M,
            sat: &Saturation<'_, M>,
            scratch: &mut Scratch<M>,
            node_level: LevelNo,
            first: SuccessorsArgs<'_, M>,
            second: SuccessorsArgs<'_, M>,
        ) -> AllocResult<(EdgeDropGuard<'m, M>, EdgeDropGuard<'m, M>)> {
            if Recursor::<M>::should_switch_to_sequential(self) {
                let first =
                    EdgeDropGuard::new(manager, sat.fire(self, scratch, node_level, first)?);
                let second =
                    EdgeDropGuard::new(manager, sat.fire(self, scratch, node_level, second)?);
                return Ok((first, second));
            }

            let (first, second) = self.join(
                manager,
                |rec| {
                    let edge = sat.fire(rec, scratch, node_level, first)?;
                    Ok(EdgeDropGuard::new(manager, edge))
                },
                |rec| {
                    let mut scratch = Scratch::new();
                    let edge = sat.fire(rec, &mut scratch, node_level, second);
                    debug_assert!(scratch.is_empty());
                    Ok(EdgeDropGuard::new(manager, edge?))
                },
            );
            Ok((first?, second?))
        }
    }
}
