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
/// The saturation cache entries do not depend on the events, so the caller must clear them
/// (see [`LDDFunction::clear_saturation_cache`][crate::LDDFunction::clear_saturation_cache])
/// between calls with different `events`.
pub(crate) fn saturate<M: LDDManager>(
    manager: &M,
    q: Borrowed<M::Edge>,
    p: u32,
    events: &[SaturationEvent<M::Edge>],
) -> AllocResult<M::Edge> {
    let saturation = Saturation { manager, events };
    let mut scratch = Scratch {
        frontier: Vec::new(),
        fired: Vec::new(),
    };

    let result = saturation.saturate_rec(&mut scratch, q, p);
    debug_assert!(scratch.frontier.is_empty() && scratch.fired.is_empty());
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
}

/// Buffers that are shared by every recursive call, such that saturating a node does not allocate.
///
/// Both are used as stacks: a call remembers the length on entry, only touches the entries above
/// it, and truncates back to that length before returning. Nested calls therefore never observe or
/// disturb the entries of their callers. All of them are empty again when the outermost call
/// returns.
struct Scratch<M: LDDManager> {
    /// The values of the node being saturated whose continuation still has to be fired.
    frontier: Vec<M::InnerNodeValue>,
    /// The `(value, subtree)` pairs produced by firing the events out of the frontier.
    fired: Vec<(M::InnerNodeValue, M::Edge)>,
}

impl<M: LDDManager> Saturation<'_, M> {
    /// Implements [`saturate`], using (and restoring) the shared `scratch` buffers.
    ///
    /// The node is saturated in two steps: (1) saturate the children, which gives a node that is
    /// closed under all events *below* position `p`, and (2) close it under the events *at* `p`.
    ///
    /// Memoised on node identity via `LDDOp::Saturate`.
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
        if let Some(res) = manager
            .apply_cache()
            .get(manager, LDDOp::Saturate, &[q.borrowed()])
        {
            stat!(cache_hit LDDOp::Saturate);
            return Ok(res);
        }

        // (1) Right to left: saturate the tail of the spine first, so it is already closed under
        // the events at this level, and then put the head value in front of it. The head can only
        // ever add to the tail (events may also write values smaller than the head, so this must
        // be a union rather than a `make_node`), which means no intermediate accumulator is
        // needed: every intermediate result is an ordinary, immutable node.
        //
        // The children are only borrowed: `q` keeps them alive for the whole call.
        let q_node = match manager.get_node(&q) {
            Node::Inner(n) => n.borrow(),
            Node::Terminal(_) => unreachable!("terminals are handled above"),
        };
        let value = q_node.get_value();
        let (down, right) = collect_children(q_node);

        let right_sat = EdgeDropGuard::new(manager, self.saturate_rec(scratch, right, p)?);
        let down_sat = EdgeDropGuard::new(manager, self.saturate_rec(scratch, down, p + 1)?);

        // If everything in the saturated tail is larger than the head value, then the head can be
        // put in front of it directly. Otherwise events wrote values smaller than the head into the
        // tail, and the two have to be merged.
        let mut node = if spine_starts_after(manager, right_sat.borrowed(), value) {
            EdgeDropGuard::new(
                manager,
                make_node(manager, value, down_sat.into_edge(), right_sat.into_edge())?,
            )
        } else {
            let head = EdgeDropGuard::new(
                manager,
                make_node(
                    manager,
                    value,
                    down_sat.into_edge(),
                    manager.get_terminal(LDDTerminal::Empty)?,
                )?,
            );
            EdgeDropGuard::new(
                manager,
                apply_union(
                    manager,
                    SequentialRecursor,
                    head.borrowed(),
                    right_sat.borrowed(),
                )?,
            )
        };

        // (2) Fixpoint over the events confined to this level (`top(e) == p`): fire every such
        // event out of the values on the frontier and union the result into the node. Only the
        // head can have anything new to fire initially, since the tail is closed already.
        // Afterwards only the values whose continuation actually changed need to fire again (the
        // paper's pipelining).
        let frontier_base = scratch.frontier.len();
        scratch.frontier.push(value.clone());
        while scratch.frontier.len() > frontier_base {
            let fired_base = scratch.fired.len();

            // The nested calls made while firing push above and truncate back to the current
            // length, so the frontier can be walked by index.
            for index in frontier_base..scratch.frontier.len() {
                let i = scratch.frontier[index].clone();
                // `node` is not replaced while firing, so it keeps `arc_i` alive.
                let Some(arc_i) = spine_lookup(manager, node.borrowed(), &i) else {
                    continue;
                };

                for event in self.events.iter().filter(|e| e.top == p) {
                    self.sat_fire_top(scratch, event, &i, arc_i.borrowed())?;
                }
            }
            scratch.frontier.truncate(frontier_base);

            for (j, f) in scratch.fired.drain(fired_base..) {
                let f = EdgeDropGuard::new(manager, f);

                // Nothing to add if `j` already continues with exactly `f`.
                if spine_lookup(manager, node.borrowed(), &j)
                    .is_some_and(|existing| *existing == *f)
                {
                    continue;
                }

                let single = EdgeDropGuard::new(
                    manager,
                    make_node(
                        manager,
                        &j,
                        f.into_edge(),
                        manager.get_terminal(LDDTerminal::Empty)?,
                    )?,
                );

                // Union of two already-saturated sets is saturated (relational image distributes
                // over union), so no re-saturation is needed here.
                let unioned = EdgeDropGuard::new(
                    manager,
                    apply_union(
                        manager,
                        SequentialRecursor,
                        node.borrowed(),
                        single.borrowed(),
                    )?,
                );

                // Only `j`'s continuation can differ, so if the union changed anything, `j` is
                // what changed.
                if *unioned != *node {
                    node = unioned;
                    if !scratch.frontier[frontier_base..].contains(&j) {
                        scratch.frontier.push(j);
                    }
                }
            }
        }

        let result = node.into_edge();

        manager
            .apply_cache()
            .add(manager, LDDOp::Saturate, &[q.borrowed()], result.borrowed());

        // `result` is already saturated (it is the fixed point we just computed), so this is a
        // free cache hit for any later `saturate` call that happens to reach the same node
        // directly.
        if result != *q {
            manager.apply_cache().add(
                manager,
                LDDOp::Saturate,
                &[result.borrowed()],
                result.borrowed(),
            );
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
    /// Memoised on node identity via `LDDOp::SatRecFire`. Only calls that end up at a new
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
        if let Some(res) = manager.apply_cache().get(
            manager,
            LDDOp::SatRecFire,
            &[q.borrowed(), rel.borrowed(), meta_l.borrowed()],
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

        manager.apply_cache().add(
            manager,
            LDDOp::SatRecFire,
            &[q, rel, meta_l],
            result.borrowed(),
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

/// Returns whether every value on the spine of `rel` is strictly greater than `value`, which is
/// trivially the case if `rel` is the Empty terminal.
fn spine_starts_after<M: LDDManager>(
    manager: &M,
    rel: Borrowed<M::Edge>,
    value: &M::InnerNodeValue,
) -> bool {
    spine(manager, rel)
        .next()
        .is_none_or(|(first, _)| first > value)
}
