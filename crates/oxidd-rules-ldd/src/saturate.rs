//! Node-wise MDD/LDD saturation as described in 
//! 
//! > Ciardo, Marmorstein, Siminiceanu. *The saturation algorithm for symbolic
//! > state-space exploration*, STTT 2006.
//!
//! The main idea of saturation is bringing one node at a time to a fixed point
//! under every event confined to its own level and below, bottom-up, before
//! that node is ever used as anyone's child. So a node is only ever built once,
//! in its final form, instead of accumulating intermediate breadth-first
//! versions.
//!
//! Positions in the state vector run `0` (top) to `num_levels - 1` (bottom),
//! which is the reverse of the paper.

use std::borrow::Borrow;
use std::cmp::Ordering;

use oxidd_core::util::{AllocResult, Borrowed, EdgeDropGuard};
use oxidd_core::{ApplyCache, Edge, InnerNode, Node};

use crate::apply::{apply_union, collect_children, make_node};
use crate::recursor::SequentialRecursor;
use crate::stat;
use crate::{LDDManager, LDDOp, LDDTerminal, LDDValue, SaturationEvent};

/// Returns `N*(q)`, the smallest superset of the vectors denoted by `q` that is
/// closed under every event in `events` whose `top` is `>= p`, where `q` is
/// known to sit at state-vector position `p`.
pub(crate) fn saturate<M: LDDManager>(
    manager: &M,
    q: Borrowed<M::Edge>,
    p: u32,
    events: &[SaturationEvent<M::Edge>],
    num_levels: u32,
    epoch: u32,
) -> AllocResult<M::Edge> {
    stat!(call LDDOp::Saturate);
    debug_assert!(p <= num_levels);

    // Terminals are trivial fixed points of every event.
    if let Node::Terminal(_) = manager.get_node(&q) {
        return Ok(manager.clone_edge(&q));
    }

    stat!(cache_query LDDOp::Saturate);
    if let Some(([res], [])) =
        manager
            .apply_cache()
            .get_extended(manager, LDDOp::Saturate, (&[q.borrowed()], &[epoch]))
    {
        stat!(cache_hit LDDOp::Saturate);
        return Ok(res);
    }

    // (1) Right to left: saturate the tail of the spine first, so it is already closed under the
    // events at this level, and then put the head value in front of it. The head can only ever add
    // to the tail (events may also write values smaller than the head, so this must be a union
    // rather than a `make_node`), which means no intermediate accumulator is needed: every
    // intermediate result is an ordinary, immutable node.
    let (value, down, right) = {
        let node = match manager.get_node(&q) {
            Node::Inner(n) => n.borrow(),
            Node::Terminal(_) => unreachable!("terminals are handled above"),
        };
        let value = node.get_value().clone();
        let (down, right) = collect_children(node);
        (value, manager.clone_edge(&down), manager.clone_edge(&right))
    };
    let down_guard = EdgeDropGuard::new(manager, down);
    let right_guard = EdgeDropGuard::new(manager, right);

    let right_sat = EdgeDropGuard::new(
        manager,
        saturate(manager, right_guard.borrowed(), p, events, num_levels, epoch)?,
    );
    let down_sat = saturate(manager, down_guard.borrowed(), p + 1, events, num_levels, epoch)?;
    let head = EdgeDropGuard::new(
        manager,
        make_node(manager, &value, down_sat, manager.get_terminal(LDDTerminal::Empty)?)?,
    );
    let mut node = EdgeDropGuard::new(
        manager,
        apply_union(manager, SequentialRecursor, head.borrowed(), right_sat.borrowed())?,
    );

    // (2) Fixpoint over the events confined to this level (`top(e) == p`): fire every such event
    // out of the values on the frontier and union the result into the node. Only the head can
    // have anything new to fire initially, since the tail is closed already. Afterwards only the
    // values whose continuation actually changed need to fire again (the paper's pipelining).
    let mut frontier = vec![value];
    while !frontier.is_empty() {
        let mut fired = Vec::new();
        for i in frontier.drain(..) {
            let Some(arc_i) = spine_lookup(manager, node.borrowed(), &i) else {
                continue;
            };
            let arc_i = EdgeDropGuard::new(manager, arc_i);

            for event in events.iter().filter(|e| e.top == p) {
                fired.extend(sat_fire_top(manager, event, &i, arc_i.borrowed(), events, num_levels, epoch)?);
            }
        }

        for (j, f) in fired {
            let single = EdgeDropGuard::new(
                manager,
                make_node(manager, &j, f, manager.get_terminal(LDDTerminal::Empty)?)?,
            );

            // Union of two already-saturated sets is saturated (relational image distributes over
            // union), so no re-saturation is needed here.
            let unioned = EdgeDropGuard::new(
                manager,
                apply_union(manager, SequentialRecursor, node.borrowed(), single.borrowed())?,
            );

            // Only `j`'s continuation can differ, so if the union changed anything, `j` is what
            // changed.
            if *unioned != *node {
                node = unioned;
                if !frontier.contains(&j) {
                    frontier.push(j);
                }
            }
        }
    }

    let result = node.into_edge();

    manager.apply_cache().add_extended(
        manager,
        LDDOp::Saturate,
        (&[q.borrowed()], &[epoch]),
        (&[result.borrowed()], &[]),
    );

    // `result` is already saturated (it is the fixed point we just computed), so this is a free
    // cache hit for any later `saturate` call that happens to reach the same node directly.
    if result != *q {
        manager.apply_cache().add_extended(
            manager,
            LDDOp::Saturate,
            (&[result.borrowed()], &[epoch]),
            (&[result.borrowed()], &[]),
        );
    }

    Ok(result)
}

/// Fires event `event` (whose `top` equals the level of `arc_i`'s parent) out
/// of the single local value `i`, whose current (`Sat_{top(e)+1}`-closed)
/// continuation is `arc_i`. Returns the resulting `(value, subtree)` pairs,
/// each subtree already saturated by [`sat_rec_fire`].
///
/// This is the same case analysis as [`crate::apply::apply_relational_product`],
/// restricted to a single source value instead of a whole spine. It is
/// deliberately *not* cached (it is only ever reached from [`saturate`],
/// which is) and deliberately does *not* saturate its own node — only
/// [`sat_rec_fire`], which is strictly below `top(e)`, does that. Saturating
/// here would call `saturate` on a node at this same level while `saturate`
/// for it may still be on the call stack, which would not terminate.
#[allow(clippy::too_many_arguments)]
fn sat_fire_top<M: LDDManager>(
    manager: &M,
    event: &SaturationEvent<M::Edge>,
    i: &M::InnerNodeValue,
    arc_i: Borrowed<M::Edge>,
    events: &[SaturationEvent<M::Edge>],
    num_levels: u32,
    epoch: u32,
) -> AllocResult<Vec<(M::InnerNodeValue, M::Edge)>> {
    let l = event.top + 1;
    let mut result = Vec::new();

    let (meta_value, meta_down) = {
        let meta_node = match manager.get_node(&event.meta_at_top) {
            Node::Inner(n) => n.borrow(),
            Node::Terminal(_) => {
                unreachable!("an event always reads or writes at least one position")
            }
        };
        let meta_value = meta_node.get_value().clone();
        let (meta_down, _meta_right) = collect_children(meta_node);
        (meta_value, manager.clone_edge(&meta_down))
    };
    let meta_down_guard = EdgeDropGuard::new(manager, meta_down);

    if meta_value == M::InnerNodeValue::read_only_value() {
        // 1: read only — the local value is unaffected; only fire if the relation has an entry
        // matching `i`.
        if let Some(rel_down) = spine_lookup(manager, event.relation.borrowed(), i) {
            let rel_down_guard = EdgeDropGuard::new(manager, rel_down);
            let f = sat_rec_fire(
                manager,
                arc_i.borrowed(),
                rel_down_guard.borrowed(),
                meta_down_guard.borrowed(),
                l,
                events,
                num_levels,
                epoch,
            )?;
            if manager.get_node(&f).is_terminal(&LDDTerminal::Empty) {
                manager.drop_edge(f);
            } else {
                result.push((i.clone(), f));
            }
        }
    } else if meta_value == M::InnerNodeValue::write_only_value() {
        // 2: write only — every relation value is a possible successor, all sharing `arc_i` as
        // their (unconstrained by `i`) source.
        for (value, down) in spine_entries(manager, event.relation.borrowed()) {
            let down_guard = EdgeDropGuard::new(manager, down);
            let f = sat_rec_fire(
                manager,
                arc_i.borrowed(),
                down_guard.borrowed(),
                meta_down_guard.borrowed(),
                l,
                events,
                num_levels,
                epoch,
            )?;
            if manager.get_node(&f).is_terminal(&LDDTerminal::Empty) {
                manager.drop_edge(f);
            } else {
                result.push((value, f));
            }
        }
    } else if meta_value == M::InnerNodeValue::read_of_pair_value() {
        // 3: read half of a read+write pair — match the relation entry equal to `i`; `meta_down`
        // is the paired write-half, whose own down-branch enumerates the possible written values.
        if let Some(rel_down) = spine_lookup(manager, event.relation.borrowed(), i) {
            let rel_down_guard = EdgeDropGuard::new(manager, rel_down);

            let write_meta_down = {
                let write_meta_node = match manager.get_node(&meta_down_guard) {
                    Node::Inner(n) => n.borrow(),
                    Node::Terminal(_) => {
                        unreachable!("read-of-pair meta_down must be write-of-pair")
                    }
                };
                debug_assert!(*write_meta_node.get_value() == M::InnerNodeValue::write_of_pair_value());
                let (write_meta_down, _) = collect_children(write_meta_node);
                manager.clone_edge(&write_meta_down)
            };
            let write_meta_guard = EdgeDropGuard::new(manager, write_meta_down);

            for (value, down) in spine_entries(manager, rel_down_guard.borrowed()) {
                let down_guard = EdgeDropGuard::new(manager, down);
                let f = sat_rec_fire(
                    manager,
                    arc_i.borrowed(),
                    down_guard.borrowed(),
                    write_meta_guard.borrowed(),
                    l,
                    events,
                    num_levels,
                    epoch,
                )?;
                if manager.get_node(&f).is_terminal(&LDDTerminal::Empty) {
                    manager.drop_edge(f);
                } else {
                    result.push((value, f));
                }
            }
        }
    } else {
        panic!("meta_at_top has an unexpected value");
    }

    Ok(result)
}

/// Computes `N*_{rec}(q)` for a single event `rel`/`meta_l` below its `top`: mirrors
/// [`crate::apply::apply_relational_product`]'s case dispatch via the uncached [`rec_fire`], then
/// saturates the freshly built node at its own level `l` before returning it, so it is safe to use
/// as anyone's child. Memoised on node identity and `epoch` via `LDDOp::SatRecFire`.
#[allow(clippy::too_many_arguments)]
fn sat_rec_fire<M: LDDManager>(
    manager: &M,
    q: Borrowed<M::Edge>,
    rel: Borrowed<M::Edge>,
    meta_l: Borrowed<M::Edge>,
    l: u32,
    events: &[SaturationEvent<M::Edge>],
    num_levels: u32,
    epoch: u32,
) -> AllocResult<M::Edge> {
    stat!(call LDDOp::SatRecFire);

    // meta == True: every meta level for this event is consumed, i.e. l > bot(e). Below Bot(e) an
    // event is the identity, and `q` is already saturated (it was saturated by the caller before
    // this event ever fired), so it is returned unchanged.
    if let Node::Terminal(t) = manager.get_node(&meta_l) {
        debug_assert_eq!(*t.borrow(), LDDTerminal::True, "meta should never reach the Empty terminal");
        return Ok(manager.clone_edge(&q));
    }
    if manager.get_node(&q).is_terminal(&LDDTerminal::Empty) {
        return manager.get_terminal(LDDTerminal::Empty);
    }
    if manager.get_node(&rel).is_terminal(&LDDTerminal::Empty) {
        return manager.get_terminal(LDDTerminal::Empty);
    }

    stat!(cache_query LDDOp::SatRecFire);
    if let Some(([res], [])) = manager.apply_cache().get_extended(
        manager,
        LDDOp::SatRecFire,
        (&[q.borrowed(), rel.borrowed(), meta_l.borrowed()], &[epoch]),
    ) {
        stat!(cache_hit LDDOp::SatRecFire);
        return Ok(res);
    }

    let raw = rec_fire(manager, q.borrowed(), rel.borrowed(), meta_l.borrowed(), l, events, num_levels, epoch)?;
    let raw_guard = EdgeDropGuard::new(manager, raw);
    let result = saturate(manager, raw_guard.borrowed(), l, events, num_levels, epoch)?;

    manager.apply_cache().add_extended(
        manager,
        LDDOp::SatRecFire,
        (&[q, rel, meta_l], &[epoch]),
        (&[result.borrowed()], &[]),
    );

    Ok(result)
}

/// Uncached helper for [`sat_rec_fire`]. Mirrors
/// [`crate::apply::apply_relational_product`]'s case dispatch exactly, except every "down" branch
/// that finishes resolving the *current* logical position and descends into the *next* one calls
/// [`sat_rec_fire`] instead of recursing into itself, so every newly built node is saturated before
/// it is used as anyone's child. Same-level continuations — further spine entries, or resolving the
/// write half of a read+write pair, which does not itself advance the state position — recurse into
/// `rec_fire` directly and are therefore *not* memoised: this is a linear spine walk, so it needs no
/// cache of its own (mirroring the paper's split between `SatFire`, at the top, and `SatRecFire`
/// below it).
#[allow(clippy::too_many_arguments)]
fn rec_fire<M: LDDManager>(
    manager: &M,
    q: Borrowed<M::Edge>,
    rel: Borrowed<M::Edge>,
    meta_l: Borrowed<M::Edge>,
    l: u32,
    events: &[SaturationEvent<M::Edge>],
    num_levels: u32,
    epoch: u32,
) -> AllocResult<M::Edge> {
    // A "right" continuation may run off the end of a spine, or a relation branch may be empty:
    // both simply mean there is nothing left to fire.
    if manager.get_node(&q).is_terminal(&LDDTerminal::Empty) {
        return manager.get_terminal(LDDTerminal::Empty);
    }
    if manager.get_node(&rel).is_terminal(&LDDTerminal::Empty) {
        return manager.get_terminal(LDDTerminal::Empty);
    }

    let meta_node = match manager.get_node(&meta_l) {
        Node::Inner(n) => n.borrow(),
        Node::Terminal(_) => unreachable!("rec_fire is never called with meta already at True"),
    };
    let meta_value = meta_node.get_value();
    let (meta_down, _meta_right) = collect_children(meta_node);

    let result = if *meta_value == M::InnerNodeValue::false_value() {
        // 0: position not in the relation — keep q's value, advance meta (and hence the state
        // position) for the down-branch only.
        let q_node = match manager.get_node(&q) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("q must have as many levels as the event's meta"),
        };
        let q_value = q_node.get_value();
        let (q_down, q_right) = collect_children(q_node);

        let down_result = EdgeDropGuard::new(
            manager,
            sat_rec_fire(manager, q_down, rel.borrowed(), meta_down, l + 1, events, num_levels, epoch)?,
        );
        let right_result = EdgeDropGuard::new(
            manager,
            rec_fire(manager, q_right, rel.borrowed(), meta_l.borrowed(), l, events, num_levels, epoch)?,
        );

        if manager.get_node(&down_result).is_terminal(&LDDTerminal::Empty) {
            right_result.into_edge()
        } else {
            make_node(manager, q_value, down_result.into_edge(), right_result.into_edge())?
        }
    } else if *meta_value == M::InnerNodeValue::read_only_value() {
        // 1: read only — match q and rel values; keep the matched value in the output.
        let q_node = match manager.get_node(&q) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("q must have as many levels as the event's meta"),
        };
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as the event's meta"),
        };
        let q_value = q_node.get_value();
        let (q_down, q_right) = collect_children(q_node);
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match q_value.cmp(rel_value) {
            Ordering::Less => rec_fire(manager, q_right, rel.borrowed(), meta_l.borrowed(), l, events, num_levels, epoch)?,
            Ordering::Greater => {
                rec_fire(manager, q.borrowed(), rel_right, meta_l.borrowed(), l, events, num_levels, epoch)?
            }
            Ordering::Equal => {
                let down_result = EdgeDropGuard::new(
                    manager,
                    sat_rec_fire(manager, q_down, rel_down, meta_down, l + 1, events, num_levels, epoch)?,
                );
                let right_result = EdgeDropGuard::new(
                    manager,
                    rec_fire(manager, q_right, rel_right, meta_l.borrowed(), l, events, num_levels, epoch)?,
                );
                if manager.get_node(&down_result).is_terminal(&LDDTerminal::Empty) {
                    right_result.into_edge()
                } else {
                    make_node(manager, q_value, down_result.into_edge(), right_result.into_edge())?
                }
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_only_value() {
        // 2: write only — union all of q's down-branches (the written value is unconstrained by
        // what was read), then write each rel value with that combined continuation.
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as the event's meta"),
        };
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        let combined = combined_down(manager, q.borrowed())?;
        let combined_guard = EdgeDropGuard::new(manager, combined);

        let down_result = EdgeDropGuard::new(
            manager,
            sat_rec_fire(manager, combined_guard.borrowed(), rel_down, meta_down, l + 1, events, num_levels, epoch)?,
        );
        let right_result = EdgeDropGuard::new(
            manager,
            rec_fire(manager, q.borrowed(), rel_right, meta_l.borrowed(), l, events, num_levels, epoch)?,
        );

        if manager.get_node(&down_result).is_terminal(&LDDTerminal::Empty) {
            right_result.into_edge()
        } else {
            make_node(manager, rel_value, down_result.into_edge(), right_result.into_edge())?
        }
    } else if *meta_value == M::InnerNodeValue::read_of_pair_value() {
        // 3: read half of a read+write pair — match values (nothing is emitted for this level
        // itself: the write half, one meta level down but at the *same* state position `l`,
        // resolves it). `q_down` is already the position-`l+1` node, reused unchanged by the write
        // half below.
        let q_node = match manager.get_node(&q) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("q must have as many levels as the event's meta"),
        };
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as the event's meta"),
        };
        let q_value = q_node.get_value();
        let (q_down, q_right) = collect_children(q_node);
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match q_value.cmp(rel_value) {
            Ordering::Less => rec_fire(manager, q_right, rel.borrowed(), meta_l.borrowed(), l, events, num_levels, epoch)?,
            Ordering::Greater => {
                rec_fire(manager, q.borrowed(), rel_right, meta_l.borrowed(), l, events, num_levels, epoch)?
            }
            Ordering::Equal => {
                let down_result = EdgeDropGuard::new(
                    manager,
                    rec_fire(manager, q_down, rel_down, meta_down, l, events, num_levels, epoch)?,
                );
                let right_result = EdgeDropGuard::new(
                    manager,
                    rec_fire(manager, q_right, rel_right, meta_l.borrowed(), l, events, num_levels, epoch)?,
                );
                apply_union(manager, SequentialRecursor, down_result.borrowed(), right_result.borrowed())?
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_of_pair_value() {
        // 4: write half of a read+write pair — this is the genuine transition into state position
        // `l + 1`; `q` here is already that position's node (obtained by the paired read half).
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as the event's meta"),
        };
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        let down_result = EdgeDropGuard::new(
            manager,
            sat_rec_fire(manager, q.borrowed(), rel_down, meta_down, l + 1, events, num_levels, epoch)?,
        );
        let right_result = EdgeDropGuard::new(
            manager,
            rec_fire(manager, q.borrowed(), rel_right, meta_l.borrowed(), l, events, num_levels, epoch)?,
        );

        if manager.get_node(&down_result).is_terminal(&LDDTerminal::Empty) {
            right_result.into_edge()
        } else {
            make_node(manager, rel_value, down_result.into_edge(), right_result.into_edge())?
        }
    } else {
        panic!("meta has an unexpected value");
    };

    Ok(result)
}

/// Returns the union of all down-branches of `q`'s spine, ignoring the specific values — the
/// continuation shared by every value once a `write_only` position makes it unconstrained.
fn combined_down<M: LDDManager>(manager: &M, q: Borrowed<M::Edge>) -> AllocResult<M::Edge> {
    let mut acc = EdgeDropGuard::new(manager, manager.get_terminal(LDDTerminal::Empty)?);
    let mut cur = EdgeDropGuard::new(manager, manager.clone_edge(&q));
    loop {
        let (down, right, right_is_empty) = {
            let node = match manager.get_node(&cur) {
                Node::Inner(n) => n.borrow(),
                Node::Terminal(_) => unreachable!("q's right spine ends at the Empty terminal"),
            };
            let (down, right) = collect_children(node);
            let right_is_empty = manager.get_node(&right).is_terminal(&LDDTerminal::Empty);
            (
                manager.clone_edge(&down),
                manager.clone_edge(&right),
                right_is_empty,
            )
        };

        let down_guard = EdgeDropGuard::new(manager, down);
        let new_acc = EdgeDropGuard::new(
            manager,
            apply_union(manager, SequentialRecursor, acc.borrowed(), down_guard.borrowed())?,
        );
        acc = new_acc;

        if right_is_empty {
            manager.drop_edge(right);
            break;
        }
        cur = EdgeDropGuard::new(manager, right);
    }
    Ok(acc.into_edge())
}

/// Walks the spine of `rel` looking for the entry equal to `i`, returning its (owned) down-branch.
/// `rel`'s spine is sorted ascending, so the search stops as soon as a strictly greater value is
/// seen. Returns `None` if `rel` is the Empty terminal or has no matching entry.
fn spine_lookup<M: LDDManager>(manager: &M, rel: Borrowed<M::Edge>, i: &M::InnerNodeValue) -> Option<M::Edge> {
    if manager.get_node(&rel).is_terminal(&LDDTerminal::Empty) {
        return None;
    }

    let mut cur = EdgeDropGuard::new(manager, manager.clone_edge(&rel));
    loop {
        let (cmp, down, right, right_is_empty) = {
            let node = match manager.get_node(&cur) {
                Node::Inner(n) => n.borrow(),
                Node::Terminal(_) => unreachable!("rel's right spine ends at the Empty terminal"),
            };
            let cmp = node.get_value().cmp(i);
            let (down, right) = collect_children(node);
            let right_is_empty = manager.get_node(&right).is_terminal(&LDDTerminal::Empty);
            (cmp, manager.clone_edge(&down), manager.clone_edge(&right), right_is_empty)
        };

        match cmp {
            Ordering::Equal => {
                manager.drop_edge(right);
                return Some(down);
            }
            Ordering::Greater => {
                manager.drop_edge(down);
                manager.drop_edge(right);
                return None;
            }
            Ordering::Less => {
                manager.drop_edge(down);
                if right_is_empty {
                    manager.drop_edge(right);
                    return None;
                }
                cur = EdgeDropGuard::new(manager, right);
            }
        }
    }
}

/// Walks the entire spine of `rel`, returning owned `(value, down)` pairs. Returns an empty `Vec`
/// if `rel` is the Empty terminal.
fn spine_entries<M: LDDManager>(manager: &M, rel: Borrowed<M::Edge>) -> Vec<(M::InnerNodeValue, M::Edge)> {
    if manager.get_node(&rel).is_terminal(&LDDTerminal::Empty) {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut cur = EdgeDropGuard::new(manager, manager.clone_edge(&rel));
    loop {
        let (value, down, right, right_is_empty) = {
            let node = match manager.get_node(&cur) {
                Node::Inner(n) => n.borrow(),
                Node::Terminal(_) => unreachable!("rel's right spine ends at the Empty terminal"),
            };
            let value = node.get_value().clone();
            let (down, right) = collect_children(node);
            let right_is_empty = manager.get_node(&right).is_terminal(&LDDTerminal::Empty);
            (value, manager.clone_edge(&down), manager.clone_edge(&right), right_is_empty)
        };

        result.push((value, down));
        if right_is_empty {
            manager.drop_edge(right);
            break;
        }
        cur = EdgeDropGuard::new(manager, right);
    }
    result
}
