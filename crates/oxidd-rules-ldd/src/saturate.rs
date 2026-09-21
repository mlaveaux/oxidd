//! Node-wise MDD/LDD saturation as described in
//!
//! > Ciardo, Marmorstein, Siminiceanu. *The saturation algorithm for symbolic
//! > state-space exploration*, STTT 2006.
//!
//! Given a set of vectors `q` and a list of events, saturation computes the
//! vectors that are reachable from `q` by firing the events in any order. An
//! event is a transition relation that only touches the positions
//! `top..=bot` of the vector (see [`SaturationEvent`]). Instead of firing all
//! events on the whole set until nothing changes, saturation brings one *node*
//! at a time to a fixed point under every event confined to its own position
//! and below, bottom-up, before that node is ever used as anyone's child. So a
//! node is only ever built once, in its final form, instead of accumulating
//! intermediate breadth-first versions.
//!
//! Positions in the state vector run `0` (top) to `K - 1` (bottom), where `K` is the length of the
//! vectors, which is the reverse of the paper. LDD nodes are `(value, down, right)`
//! triples: `down` continues with the next position, and `right` is the next
//! value for the same position. The spine formed by the `right` edges is sorted
//! ascending and ends at the Empty terminal.

use std::borrow::Borrow;
use std::cmp::Ordering;

use oxidd_core::util::{AllocResult, Borrowed, EdgeDropGuard};
use oxidd_core::{ApplyCache, Edge, InnerNode, Node};

use crate::apply::{
    apply_union, collect_children, make_node, relational_product_step, relational_product_terminal,
    spine, Position, RelationalProductRecursion,
};
use crate::recursor::SequentialRecursor;
use crate::stat;
use crate::{LDDManager, LDDOp, LDDTerminal, LDDValue, SaturationEvent};

/// Returns `N*(q)`, the smallest superset of the vectors denoted by `q` that is
/// closed under every event in `events` whose `top` is `>= p`, where `q` is
/// known to sit at state-vector position `p`.
///
/// The results are memoised in the apply cache, keyed on `epoch`, which identifies `events`: calls
/// with the same `epoch` share their cached results, so it has to differ whenever the events differ
/// (see [`LDDFunction::saturate_edge`][crate::LDDFunction::saturate_edge]).
pub(crate) fn saturate<M: LDDManager>(
    manager: &M,
    q: Borrowed<M::Edge>,
    p: u32,
    events: &[SaturationEvent<M::Edge>],
    epoch: u32,
) -> AllocResult<M::Edge> {
    let saturation = Saturation {
        manager,
        events,
        epoch,
    };
    let mut scratch = Scratch {
        arcs: Vec::new(),
        frontier: Vec::new(),
        fired: Vec::new(),
    };

    let result = saturation.saturate_rec(&mut scratch, q, p);
    debug_assert!(
        scratch.arcs.is_empty() && scratch.frontier.is_empty() && scratch.fired.is_empty()
    );
    result
}

/// The information that stays the same during a whole saturation run.
///
/// Keeping it in one place, rather than in the parameters of every recursive call, keeps the stack
/// frames small.
struct Saturation<'a, M: LDDManager> {
    manager: &'a M,
    /// Every event that can be fired.
    events: &'a [SaturationEvent<M::Edge>],
    /// The number that identifies `events`, and that the cache entries of this call are keyed on.
    epoch: u32,
}

/// Buffers that are shared by every recursive call, such that saturating a node does not allocate.
///
/// All of them are used as stacks: a call remembers the length on entry, only touches the entries
/// above it, and truncates back to that length before returning. Nested calls therefore never
/// observe or disturb the entries of their callers. All of them are empty again when the outermost
/// call returns.
struct Scratch<M: LDDManager> {
    /// The accumulator of the node being saturated: its `(value, continuation)` pairs, sorted
    /// ascending by value. The continuation of a value is replaced by a bigger one whenever firing
    /// an event grows it, and values that were not in the node can be inserted anywhere.
    arcs: Vec<(M::InnerNodeValue, M::Edge)>,
    /// The values of the node being saturated whose continuation still has to be fired, used as a
    /// FIFO queue.
    frontier: Vec<M::InnerNodeValue>,
    /// The `(value, subtree)` pairs produced by firing the events out of one value.
    fired: Vec<(M::InnerNodeValue, M::Edge)>,
}

impl<M: LDDManager> Saturation<'_, M> {
    /// Implements [`saturate`], using (and restoring) the shared `scratch` buffers.
    ///
    /// Memoised on node identity and the epoch via `LDDOp::Saturate`; see [`Self::saturate_node`] for the
    /// computation itself. If it fails, everything it left in `scratch` is released.
    fn saturate_rec(
        &self,
        scratch: &mut Scratch<M>,
        q: Borrowed<M::Edge>,
        p: u32,
    ) -> AllocResult<M::Edge> {
        let manager = self.manager;

        stat!(call LDDOp::Saturate);

        // Terminals are trivial fixed points of every event.
        if let Node::Terminal(_) = manager.get_node(&q) {
            return Ok(manager.clone_edge(&q));
        }

        stat!(cache_query LDDOp::Saturate);
        if let Some(([res], [])) = manager.apply_cache().get_extended::<1, 0>(
            manager,
            LDDOp::Saturate,
            (&[q.borrowed()], &[self.epoch]),
        ) {
            stat!(cache_hit LDDOp::Saturate);
            return Ok(res);
        }

        let arcs_base = scratch.arcs.len();
        let frontier_base = scratch.frontier.len();
        let fired_base = scratch.fired.len();
        let result = match self.saturate_node(scratch, q.borrowed(), p, arcs_base, frontier_base) {
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
            (&[q.borrowed()], &[self.epoch]),
            (&[result.borrowed()], &[]),
        );

        // `result` is already saturated (it is the fixed point we just computed), so this is a
        // free cache hit for any later node of this call that happens to reach it directly.
        if result != *q {
            manager.apply_cache().add_extended(
                manager,
                LDDOp::Saturate,
                (&[result.borrowed()], &[self.epoch]),
                (&[result.borrowed()], &[]),
            );
        }

        Ok(result)
    }

    /// Computes the saturation of `q`, which sits at position `p`, on a cache miss.
    ///
    /// The node is saturated in three steps: (1) saturate the children, which gives the
    /// continuations of a node that is closed under all events *below* position `p`, (2) close it
    /// under the events *at* `p`, and (3) build the resulting node once.
    fn saturate_node(
        &self,
        scratch: &mut Scratch<M>,
        q: Borrowed<M::Edge>,
        p: u32,
        arcs_base: usize,
        frontier_base: usize,
    ) -> AllocResult<M::Edge> {
        let manager = self.manager;

        // (1) Saturate every child. The spine is ascending, so the accumulator starts out sorted.
        for (value, down) in spine(manager, q) {
            let down_sat = self.saturate_rec(scratch, down, p + 1)?;
            scratch.arcs.push((value.clone(), down_sat));
        }

        // (2) Fixpoint over the events confined to this level (`top(e) == p`): fire every such
        // event out of every value and unite the results into the accumulator, queueing the values
        // whose continuation actually grew (the paper's pipelining).
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
                    manager.clone_edge(&scratch.arcs[arcs_base + pos].1),
                )
            };

            let fired_base = scratch.fired.len();
            for event in self.events.iter().filter(|e| e.top == p) {
                self.sat_fire_top(scratch, event, &i, arc_i.borrowed())?;
            }
            drop(arc_i);

            for (j, f) in scratch.fired.drain(fired_base..) {
                let f = EdgeDropGuard::new(manager, f);

                match scratch.arcs[arcs_base..].binary_search_by(|(value, _)| value.cmp(&j)) {
                    Ok(pos) => {
                        let arc_j = &mut scratch.arcs[arcs_base + pos].1;

                        // Nothing to add if `j` already continues with exactly `f`.
                        if *arc_j == *f {
                            continue;
                        }

                        // Union of two already-saturated sets is saturated (relational image
                        // distributes over union), so no re-saturation is needed here.
                        let unioned = apply_union(
                            manager,
                            SequentialRecursor,
                            arc_j.borrowed(),
                            f.borrowed(),
                        )?;
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

        // (3) Build the node once, right to left, so the spine ends up ascending, matching every
        // other LDD builder in this crate (see `apply_union`'s `Ordering::Less` case).
        let mut result = manager.get_terminal(LDDTerminal::Empty)?;
        while scratch.arcs.len() > arcs_base {
            let (value, down) = scratch.arcs.pop().expect("the length was checked above");
            let tail = EdgeDropGuard::new(manager, result);
            result = make_node(manager, &value, down, tail.into_edge())?;
        }
        Ok(result)
    }

    /// Fires `event`, whose `top` is the position of the node that is being saturated, out of the
    /// single value `i` of that node. The continuation of `i` is `arc_i`, and it is already
    /// saturated. Pushes the resulting `(value, subtree)` pairs onto `scratch.fired`, where every
    /// subtree is already saturated by [`Self::sat_rec_fire`].
    ///
    /// This is the case analysis of [`relational_product_step`] for the first meta level, restricted
    /// to the single source value `i` instead of a whole spine: the source is `(i, arc_i)`, so
    /// nothing has to be built for it, and the entries of the relation are looked up directly.
    ///
    /// It is deliberately *not* cached (it is only reached from [`Self::saturate_rec`], which is),
    /// and it does *not* saturate the node it contributes to: only [`Self::sat_rec_fire`], which
    /// works strictly below `top(e)`, saturates. Saturating here would call `saturate` for a node
    /// at this same position, while the saturation of a node at this position may still be on the
    /// call stack, which would not terminate.
    fn sat_fire_top(
        &self,
        scratch: &mut Scratch<M>,
        event: &SaturationEvent<M::Edge>,
        i: &M::InnerNodeValue,
        arc_i: Borrowed<M::Edge>,
    ) -> AllocResult<()> {
        let manager = self.manager;
        let l = event.top + 1;

        // Everything is borrowed from the event or from `arc_i`, both of which outlive this call.
        let meta_node = match manager.get_node(&event.meta_at_top) {
            Node::Inner(n) => n.borrow(),
            Node::Terminal(_) => {
                unreachable!("an event always reads or writes at least one position")
            }
        };
        let meta_value = meta_node.get_value();
        let (meta_down, _meta_right) = collect_children(meta_node);

        if *meta_value == M::InnerNodeValue::read_only_value() {
            // 1: read only — the local value is unaffected; only fire if the relation has an entry
            // matching `i`.
            if let Some(rel_down) = spine_lookup(manager, event.relation.borrowed(), i) {
                let f = self.sat_rec_fire(scratch, arc_i, rel_down, meta_down, l)?;
                if manager.get_node(&f).is_terminal(&LDDTerminal::Empty) {
                    manager.drop_edge(f);
                } else {
                    scratch.fired.push((i.clone(), f));
                }
            }
        } else if *meta_value == M::InnerNodeValue::write_only_value() {
            // 2: write only — every relation value is a possible successor, all sharing `arc_i` as
            // their (unconstrained by `i`) source.
            for (value, down) in spine(manager, event.relation.borrowed()) {
                let f =
                    self.sat_rec_fire(scratch, arc_i.borrowed(), down, meta_down.borrowed(), l)?;
                if manager.get_node(&f).is_terminal(&LDDTerminal::Empty) {
                    manager.drop_edge(f);
                } else {
                    scratch.fired.push((value.clone(), f));
                }
            }
        } else if *meta_value == M::InnerNodeValue::read_of_pair_value() {
            // 3: read half of a read+write pair — match the relation entry equal to `i`; `meta_down`
            // is the paired write-half, whose own down-branch enumerates the possible written values.
            if let Some(rel_down) = spine_lookup(manager, event.relation.borrowed(), i) {
                let write_meta_node = match manager.get_node(&meta_down) {
                    Node::Inner(n) => n.borrow(),
                    Node::Terminal(_) => {
                        unreachable!("read-of-pair meta_down must be write-of-pair")
                    }
                };
                debug_assert!(
                    *write_meta_node.get_value() == M::InnerNodeValue::write_of_pair_value()
                );
                let (write_meta_down, _) = collect_children(write_meta_node);

                for (value, down) in spine(manager, rel_down) {
                    let f = self.sat_rec_fire(
                        scratch,
                        arc_i.borrowed(),
                        down,
                        write_meta_down.borrowed(),
                        l,
                    )?;
                    if manager.get_node(&f).is_terminal(&LDDTerminal::Empty) {
                        manager.drop_edge(f);
                    } else {
                        scratch.fired.push((value.clone(), f));
                    }
                }
            }
        } else {
            panic!("meta_at_top has an unexpected value");
        }

        Ok(())
    }

    /// Fires the part `rel`/`meta_l` of an event on `q`, where the result belongs to position `l`, which
    /// lies below the event's `top`. The result is saturated at `l` before it is returned, so it is safe to
    /// use as anyone's child.
    ///
    /// Memoised on node identity and the epoch via `LDDOp::SatRecFire`. Only calls that end up at a new
    /// position are memoised, and those are exactly the ones that saturate.
    fn sat_rec_fire(
        &self,
        scratch: &mut Scratch<M>,
        q: Borrowed<M::Edge>,
        rel: Borrowed<M::Edge>,
        meta_l: Borrowed<M::Edge>,
        l: u32,
    ) -> AllocResult<M::Edge> {
        let manager = self.manager;
        stat!(call LDDOp::SatRecFire);

        // `meta_l == True` means that every meta level of the event is consumed, i.e. `l > bot(e)`.
        // Below `bot(e)` the event is the identity, and `q` is already saturated (its creator did
        // that before it was ever used as a child), so it is returned unchanged.
        if let Some(result) =
            relational_product_terminal(manager, q.borrowed(), rel.borrowed(), meta_l.borrowed())?
        {
            return Ok(result);
        }

        stat!(cache_query LDDOp::SatRecFire);
        if let Some(([res], [])) = manager.apply_cache().get_extended::<1, 0>(
            manager,
            LDDOp::SatRecFire,
            (
                &[q.borrowed(), rel.borrowed(), meta_l.borrowed()],
                &[self.epoch],
            ),
        ) {
            stat!(cache_hit LDDOp::SatRecFire);
            return Ok(res);
        }

        let raw = relational_product_step(
            manager,
            &mut FireRecursion {
                saturation: self,
                scratch: &mut *scratch,
                l,
            },
            q.borrowed(),
            rel.borrowed(),
            meta_l.borrowed(),
        )?;
        let raw_guard = EdgeDropGuard::new(manager, raw);
        let result = self.saturate_rec(scratch, raw_guard.borrowed(), l)?;

        manager.apply_cache().add_extended(
            manager,
            LDDOp::SatRecFire,
            (&[q, rel, meta_l], &[self.epoch]),
            (&[result.borrowed()], &[]),
        );

        Ok(result)
    }

    /// The uncached counterpart of [`Self::sat_rec_fire`] for continuations that stay at position `l`:
    /// the remaining entries of a spine, or the write half of a read+write pair. Nothing is saturated
    /// here, since the node that is being built is not finished yet; its creator saturates it.
    ///
    /// This is a linear spine walk, so it needs no cache of its own (mirroring the paper's split
    /// between `SatFire` at the top and `SatRecFire` below it).
    fn rec_fire(
        &self,
        scratch: &mut Scratch<M>,
        q: Borrowed<M::Edge>,
        rel: Borrowed<M::Edge>,
        meta_l: Borrowed<M::Edge>,
        l: u32,
    ) -> AllocResult<M::Edge> {
        let manager = self.manager;

        // A continuation to the right may run off the end of a spine, or a relation branch may be
        // empty: both simply mean there is nothing left to fire.
        if let Some(result) =
            relational_product_terminal(manager, q.borrowed(), rel.borrowed(), meta_l.borrowed())?
        {
            return Ok(result);
        }

        relational_product_step(
            manager,
            &mut FireRecursion {
                saturation: self,
                scratch,
                l,
            },
            q,
            rel,
            meta_l,
        )
    }
}

/// The recursion of the relational product that is used to fire events: it is the plain relational
/// product, except that every node that is built for a new position is saturated.
struct FireRecursion<'s, 'a, M: LDDManager> {
    saturation: &'s Saturation<'a, M>,
    scratch: &'s mut Scratch<M>,
    /// The state-vector position of the node built by the current step.
    l: u32,
}

impl<M: LDDManager> RelationalProductRecursion<M> for FireRecursion<'_, '_, M> {
    type Rec = SequentialRecursor;

    fn recursor(&self) -> SequentialRecursor {
        SequentialRecursor
    }

    fn recurse(
        &mut self,
        position: Position,
        set: Borrowed<M::Edge>,
        rel: Borrowed<M::Edge>,
        meta: Borrowed<M::Edge>,
    ) -> AllocResult<M::Edge> {
        match position {
            Position::Next => {
                self.saturation
                    .sat_rec_fire(self.scratch, set, rel, meta, self.l + 1)
            }
            Position::Same => self
                .saturation
                .rec_fire(self.scratch, set, rel, meta, self.l),
        }
    }
}

/// Walks the spine of `rel` looking for the entry equal to `i`, returning its down-branch.
/// `rel`'s spine is sorted ascending, so the search stops as soon as a strictly greater value is
/// seen. Returns `None` if `rel` is the Empty terminal or has no matching entry.
fn spine_lookup<'a, M: LDDManager>(
    manager: &'a M,
    rel: Borrowed<'a, M::Edge>,
    i: &M::InnerNodeValue,
) -> Option<Borrowed<'a, M::Edge>> {
    for (value, down) in spine(manager, rel) {
        match value.cmp(i) {
            Ordering::Less => {}
            Ordering::Equal => return Some(down),
            Ordering::Greater => return None,
        }
    }
    None
}

/// Returns the position of `value` in the accumulator `arcs`, which is sorted ascending by value.
fn find_arc<V: Ord, E>(arcs: &[(V, E)], value: &V) -> Option<usize> {
    arcs.binary_search_by(|(v, _)| v.cmp(value)).ok()
}
