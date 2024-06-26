//! Recursive, multi-threaded apply algorithms

use oxidd_core::function::BooleanFunction;
use oxidd_core::function::BooleanFunctionQuant;
use oxidd_core::function::EdgeOfFunc;
use oxidd_core::function::Function;
use oxidd_core::function::FunctionSubst;
use oxidd_core::util::AllocResult;
use oxidd_core::util::Borrowed;
use oxidd_core::util::EdgeDropGuard;
use oxidd_core::util::OptBool;
use oxidd_core::util::SatCountCache;
use oxidd_core::util::SatCountNumber;
use oxidd_core::ApplyCache;
use oxidd_core::Edge;
use oxidd_core::HasApplyCache;
use oxidd_core::HasLevel;
use oxidd_core::InnerNode;
use oxidd_core::LevelNo;
use oxidd_core::Manager;
use oxidd_core::Node;
use oxidd_core::Tag;
use oxidd_core::WorkerManager;
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

use crate::stat;

use super::apply_rec_st;
use super::collect_cofactors;
use super::get_terminal;
use super::not;
use super::not_owned;
use super::reduce;
use super::BCDDOp;
use super::BCDDTerminal;
use super::EdgeTag;
use super::NodesOrDone;
#[cfg(feature = "statistics")]
use super::STAT_COUNTERS;

// spell-checker:ignore fnode,gnode,hnode,vnode,flevel,glevel,hlevel,vlevel

/// Recursively apply the binary operator `OP` to `f` and `g`
///
/// `depth` is decremented for each recursive call. If it reaches 0, this
/// function simply calls [`apply_rec_st::apply_bin()`].
///
/// We use a `const` parameter `OP` to have specialized version of this function
/// for each operator.
fn apply_bin<M, const OP: u8>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal>
        + HasApplyCache<M, BCDDOp>
        + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    if depth == 0 {
        return apply_rec_st::apply_bin::<M, OP>(manager, f, g);
    }
    stat!(call OP);
    let (op, f, fnode, g, gnode) = if OP == BCDDOp::And as u8 {
        match super::terminal_and(manager, &f, &g) {
            NodesOrDone::Nodes(fnode, gnode) if f < g => {
                (BCDDOp::And, f.borrowed(), fnode, g.borrowed(), gnode)
            }
            // `And` is commutative, hence we swap `f` and `g` in the apply
            // cache key if `f > g` to have a unique representation of the set
            // `{f, g}`.
            NodesOrDone::Nodes(fnode, gnode) => {
                (BCDDOp::And, g.borrowed(), gnode, f.borrowed(), fnode)
            }
            NodesOrDone::Done(h) => return Ok(h),
        }
    } else {
        assert_eq!(OP, BCDDOp::Xor as u8);
        match super::terminal_xor(manager, &f, &g) {
            NodesOrDone::Nodes(fnode, gnode) if f < g => {
                (BCDDOp::Xor, f.borrowed(), fnode, g.borrowed(), gnode)
            }
            NodesOrDone::Nodes(fnode, gnode) => {
                (BCDDOp::Xor, g.borrowed(), gnode, f.borrowed(), fnode)
            }
            NodesOrDone::Done(h) => {
                return Ok(h);
            }
        }
    };

    // Query apply cache
    stat!(cache_query OP);
    if let Some(h) = manager
        .apply_cache()
        .get(manager, op, &[f.borrowed(), g.borrowed()])
    {
        stat!(cache_hit OP);
        return Ok(h);
    }

    let flevel = fnode.level();
    let glevel = gnode.level();
    let level = std::cmp::min(flevel, glevel);

    // Collect cofactors of all top-most nodes
    let (ft, fe) = if flevel == level {
        collect_cofactors(f.tag(), fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (gt, ge) = if glevel == level {
        collect_cofactors(g.tag(), gnode)
    } else {
        (g.borrowed(), g.borrowed())
    };

    let d = depth - 1;
    let (t, e) = manager.join(
        || {
            let t = apply_bin::<M, OP>(manager, d, ft, gt)?;
            Ok(EdgeDropGuard::new(manager, t))
        },
        || {
            let e = apply_bin::<M, OP>(manager, d, fe, ge)?;
            Ok(EdgeDropGuard::new(manager, e))
        },
    );
    let (t, e) = (t?, e?);
    let h = reduce(manager, level, t.into_edge(), e.into_edge(), op)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, op, &[f, g], h.borrowed());

    Ok(h)
}

/// Shorthand for `apply_bin_rec_mt::<M, { BCDDOp::And as u8 }>(manager, depth,
/// f, g)`
#[inline(always)]
fn apply_and<M>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal>
        + HasApplyCache<M, BCDDOp>
        + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    apply_bin::<M, { BCDDOp::And as u8 }>(manager, depth, f, g)
}

/// Recursively apply the if-then-else operator (`if f { g } else { h }`),
/// multi-threaded version
///
/// `depth` is decremented for each recursive call. If it reaches 0, this
/// function simply calls [`apply_rec_st::apply_ite()`].
fn apply_ite<M>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    h: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal>
        + HasApplyCache<M, BCDDOp>
        + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    if depth == 0 {
        return apply_rec_st::apply_ite(manager, f, g, h);
    }
    stat!(call BCDDOp::Ite);

    // Terminal cases
    let gu = g.with_tag(EdgeTag::None); // untagged
    let hu = h.with_tag(EdgeTag::None);
    if gu == hu {
        return Ok(if g.tag() == h.tag() {
            manager.clone_edge(&g)
        } else {
            not_owned(apply_bin::<M, { BCDDOp::Xor as u8 }>(manager, depth, f, g)?)
            // f ↔ g
        });
    }
    let fu = f.with_tag(EdgeTag::None);
    if fu == gu {
        return if f.tag() == g.tag() {
            Ok(not_owned(apply_and(manager, depth, not(&f), not(&h))?)) // f ∨ h
        } else {
            apply_and(manager, depth, not(&f), h) // f < h
        };
    }
    if fu == hu {
        return if f.tag() == h.tag() {
            apply_and(manager, depth, f, g)
        } else {
            // f → g = ¬f ∨ g = ¬(f ∧ ¬g)
            Ok(not_owned(apply_and(manager, depth, f, not(&g))?))
        };
    }
    let fnode = match manager.get_node(&f) {
        Node::Inner(n) => n,
        Node::Terminal(_) => {
            return Ok(manager.clone_edge(&*if f.tag() == EdgeTag::None { g } else { h }))
        }
    };
    let (gnode, hnode) = match (manager.get_node(&g), manager.get_node(&h)) {
        (Node::Inner(gn), Node::Inner(hn)) => (gn, hn),
        (Node::Terminal(_), Node::Inner(_)) => {
            return if g.tag() == EdgeTag::None {
                // f ∨ h
                Ok(not_owned(apply_and(manager, depth, not(&f), not(&h))?))
            } else {
                apply_and(manager, depth, not(&f), h) // f < h
            };
        }
        (_gnode, Node::Terminal(_)) => {
            debug_assert!(_gnode.is_inner());
            return if h.tag() == EdgeTag::None {
                Ok(not_owned(apply_and(manager, depth, f, not(&g))?)) // f → g
            } else {
                apply_and(manager, depth, f, g)
            };
        }
    };

    // Query apply cache
    stat!(cache_query BCDDOp::Ite);
    if let Some(res) = manager.apply_cache().get(
        manager,
        BCDDOp::Ite,
        &[f.borrowed(), g.borrowed(), h.borrowed()],
    ) {
        stat!(cache_hit BCDDOp::Ite);
        return Ok(res);
    }

    // Get the top-most level of the three
    let flevel = fnode.level();
    let glevel = gnode.level();
    let hlevel = hnode.level();
    let level = std::cmp::min(std::cmp::min(flevel, glevel), hlevel);

    // Collect cofactors of all top-most nodes
    let (ft, fe) = if flevel == level {
        collect_cofactors(f.tag(), fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (gt, ge) = if glevel == level {
        collect_cofactors(g.tag(), gnode)
    } else {
        (g.borrowed(), g.borrowed())
    };
    let (ht, he) = if hlevel == level {
        collect_cofactors(h.tag(), hnode)
    } else {
        (h.borrowed(), h.borrowed())
    };

    let d = depth - 1;
    let (t, e) = manager.join(
        || {
            let t = apply_ite(manager, d, ft, gt, ht)?;
            Ok(EdgeDropGuard::new(manager, t))
        },
        || {
            let e = apply_ite(manager, d, fe, ge, he)?;
            Ok(EdgeDropGuard::new(manager, e))
        },
    );
    let (t, e) = (t?, e?);
    let res = reduce(manager, level, t.into_edge(), e.into_edge(), BCDDOp::Ite)?;

    manager
        .apply_cache()
        .add(manager, BCDDOp::Ite, &[f, g, h], res.borrowed());

    Ok(res)
}

fn substitute<M>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    subst: &[M::Edge],
    cache_id: u32,
) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal>
        + HasApplyCache<M, BCDDOp>
        + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    if depth == 0 {
        return apply_rec_st::substitute(manager, f, subst, cache_id);
    }
    stat!(call BCDDOp::Substitute);

    let Node::Inner(node) = manager.get_node(&f) else {
        return Ok(manager.clone_edge(&f));
    };
    let level = node.level();
    if level as usize >= subst.len() {
        return Ok(manager.clone_edge(&f));
    }

    // Query apply cache
    stat!(cache_query BCDDOp::Substitute);
    if let Some(h) = manager.apply_cache().get_with_numeric(
        manager,
        BCDDOp::Substitute,
        &[f.borrowed()],
        &[cache_id],
    ) {
        stat!(cache_hit BCDDOp::Substitute);
        return Ok(h);
    }

    let (t, e) = collect_cofactors(f.tag(), node);
    let d = depth - 1;
    let (t, e) = manager.join(
        || {
            let t = substitute(manager, d, t, subst, cache_id)?;
            Ok(EdgeDropGuard::new(manager, t))
        },
        || {
            let e = substitute(manager, d, e, subst, cache_id)?;
            Ok(EdgeDropGuard::new(manager, e))
        },
    );
    let (t, e) = (t?, e?);
    let res = apply_ite(
        manager,
        d,
        subst[level as usize].borrowed(),
        t.borrowed(),
        e.borrowed(),
    )?;

    // Insert into apply cache
    manager.apply_cache().add_with_numeric(
        manager,
        BCDDOp::Substitute,
        &[f.borrowed()],
        &[cache_id],
        res.borrowed(),
    );

    Ok(res)
}

fn restrict<M>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag>
        + HasApplyCache<M, BCDDOp>
        + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    if depth == 0 {
        return apply_rec_st::restrict(manager, f, vars);
    }
    stat!(call BCDDOp::Restrict);

    let (Node::Inner(fnode), Node::Inner(vnode)) = (manager.get_node(&f), manager.get_node(&vars))
    else {
        return Ok(manager.clone_edge(&f));
    };

    let inner_res = {
        let f_neg = f.tag() == EdgeTag::Complemented;
        let flevel = fnode.level();
        let vars_neg = vars.tag() == EdgeTag::Complemented;
        apply_rec_st::restrict_inner(manager, f, f_neg, fnode, flevel, vars, vars_neg, vnode)
    };
    match inner_res {
        apply_rec_st::RestrictInnerResult::Done(result) => Ok(result),
        apply_rec_st::RestrictInnerResult::Rec {
            vars,
            f,
            f_neg,
            fnode,
        } => {
            // f above top-most restrict variable
            let f_untagged = f.with_tag(EdgeTag::None);
            let f_tag = if f_neg {
                EdgeTag::Complemented
            } else {
                EdgeTag::None
            };

            // Query apply cache
            stat!(cache_query BCDDOp::Restrict);
            if let Some(result) = manager.apply_cache().get(
                manager,
                BCDDOp::Restrict,
                &[f_untagged.borrowed(), vars.borrowed()],
            ) {
                stat!(cache_hit BCDDOp::Restrict);
                let result_tag = result.tag();
                return Ok(result.with_tag_owned(result_tag ^ f_tag));
            }

            let d = depth - 1;
            let (ft, fe) = (fnode.child(0), fnode.child(1));
            let (t, e) = manager.join(
                || {
                    let t = restrict(manager, d, ft, vars.borrowed())?;
                    Ok(EdgeDropGuard::new(manager, t))
                },
                || {
                    let e = restrict(manager, d, fe, vars.borrowed())?;
                    Ok(EdgeDropGuard::new(manager, e))
                },
            );
            let (t, e) = (t?, e?);

            let result = reduce(
                manager,
                fnode.level(),
                t.into_edge(),
                e.into_edge(),
                BCDDOp::Restrict,
            )?;

            manager.apply_cache().add(
                manager,
                BCDDOp::Restrict,
                &[f_untagged, vars],
                result.borrowed(),
            );

            let result_tag = result.tag();
            Ok(result.with_tag_owned(result_tag ^ f_tag))
        }
    }
}

/// Compute the quantification `Q` over `vars`
///
/// `Q` is one of `BCDDOp::Forall`, `BCDDOp::Exist`, or `BCDDOp::Forall` as
/// `u8`.
///
/// `depth` is decremented for each recursive call. If it reaches 0, this
/// function simply calls [`apply_rec_st::quant()`].
fn quant<M, const Q: u8>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag>
        + HasApplyCache<M, BCDDOp>
        + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    if depth == 0 {
        return apply_rec_st::quant::<M, Q>(manager, f, vars);
    }
    let operator = match () {
        _ if Q == BCDDOp::Forall as u8 => BCDDOp::Forall,
        _ if Q == BCDDOp::Exist as u8 => BCDDOp::Exist,
        _ if Q == BCDDOp::Unique as u8 => BCDDOp::Unique,
        _ => unreachable!("invalid quantifier"),
    };

    stat!(call operator);
    // Terminal cases
    let fnode = match manager.get_node(&f) {
        Node::Inner(n) => n,
        Node::Terminal(_) => {
            return Ok(
                if operator != BCDDOp::Unique || manager.get_node(&vars).is_any_terminal() {
                    manager.clone_edge(&f)
                } else {
                    // ∃! x. ⊤ ≡ ⊤ ⊕ ⊤ ≡ ⊥
                    get_terminal(manager, false)
                },
            );
        }
    };
    let flevel = fnode.level();

    let vars = if operator != BCDDOp::Unique {
        // We can ignore all variables above the top-most variable. Removing
        // them before querying the apply cache should increase the hit ratio by
        // a lot.
        crate::set_pop(manager, vars, flevel)
    } else {
        // No need to pop variables here, if the variable is above `fnode`,
        // i.e., does not occur in `f`, then the result is `f ⊕ f ≡ ⊥`. We
        // handle this below.
        vars
    };
    let vnode = match manager.get_node(&vars) {
        Node::Inner(n) => n,
        Node::Terminal(_) => return Ok(manager.clone_edge(&f)),
    };
    let vlevel = vnode.level();
    if operator == BCDDOp::Unique && vlevel < flevel {
        // `vnode` above `fnode`, i.e., the variable does not occur in `f` (see above)
        return Ok(get_terminal(manager, false));
    }
    debug_assert!(flevel <= vlevel);
    let vars = vars.borrowed();

    // Query apply cache
    stat!(cache_query operator);
    if let Some(res) =
        manager
            .apply_cache()
            .get(manager, operator, &[f.borrowed(), vars.borrowed()])
    {
        stat!(cache_hit operator);
        return Ok(res);
    }

    let d = depth - 1;
    let (ft, fe) = collect_cofactors(f.tag(), fnode);
    let vt = if vlevel == flevel {
        vnode.child(0)
    } else {
        vars.borrowed()
    };
    let (t, e) = manager.join(
        || {
            let t = quant::<M, Q>(manager, d, ft, vt.borrowed())?;
            Ok(EdgeDropGuard::new(manager, t))
        },
        || {
            let e = quant::<M, Q>(manager, d, fe, vt.borrowed())?;
            Ok(EdgeDropGuard::new(manager, e))
        },
    );
    let (t, e) = (t?, e?);

    let res = if flevel == vlevel {
        match operator {
            BCDDOp::Forall => apply_and(manager, d, t.borrowed(), e.borrowed())?,
            BCDDOp::Exist => not_owned(apply_and(manager, d, not(&t), not(&e))?),
            BCDDOp::Unique => {
                apply_bin::<M, { BCDDOp::Xor as u8 }>(manager, d, t.borrowed(), e.borrowed())?
            }
            _ => unreachable!(),
        }
    } else {
        reduce(manager, flevel, t.into_edge(), e.into_edge(), operator)?
    };

    manager
        .apply_cache()
        .add(manager, operator, &[f, vars], res.borrowed());

    Ok(res)
}

// --- Function Interface ------------------------------------------------------

/// Boolean function backed by a complement edge binary decision diagram
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
#[repr(transparent)]
pub struct BCDDFunctionMT<F: Function>(F);

impl<F: Function> From<F> for BCDDFunctionMT<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        BCDDFunctionMT(value)
    }
}

impl<F: Function> BCDDFunctionMT<F>
where
    for<'id> F::Manager<'id>: WorkerManager,
{
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }

    fn init_depth(manager: &F::Manager<'_>) -> u32 {
        let n = manager.current_num_threads();
        if n > 1 {
            (4096 * n).ilog2()
        } else {
            0
        }
    }
}

impl<F: Function> FunctionSubst for BCDDFunctionMT<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag>
        + super::HasBCDDOpApplyCache<F::Manager<'id>>
        + WorkerManager,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
{
    fn substitute_edge<'id, 'a>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        substitution: impl oxidd_core::util::Substitution<
            Var = Borrowed<'a, EdgeOfFunc<'id, Self>>,
            Replacement = Borrowed<'a, EdgeOfFunc<'id, Self>>,
        >,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let subst = apply_rec_st::substitute_prepare(manager, substitution.pairs())?;
        let depth = Self::init_depth(manager);
        substitute(manager, depth, edge.borrowed(), &subst, substitution.id())
    }
}

impl<F: Function> BooleanFunction for BCDDFunctionMT<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag>
        + super::HasBCDDOpApplyCache<F::Manager<'id>>
        + WorkerManager,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
{
    #[inline]
    fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self> {
        let t = get_terminal(manager, true);
        let e = get_terminal(manager, false);
        let edge = manager.add_level(|level| InnerNode::new(level, [t, e]))?;
        Ok(Self::from_edge(manager, edge))
    }

    #[inline]
    fn f_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        get_terminal(manager, false)
    }
    #[inline]
    fn t_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        get_terminal(manager, true)
    }

    #[inline]
    fn not_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Ok(not_owned(manager.clone_edge(edge)))
    }
    #[inline]
    fn not_edge_owned<'id>(
        _manager: &Self::Manager<'id>,
        edge: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Ok(not_owned(edge))
    }

    #[inline]
    fn and_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let d = Self::init_depth(manager);
        apply_and(manager, d, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn or_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let d = Self::init_depth(manager);
        Ok(not_owned(apply_and(manager, d, not(lhs), not(rhs))?))
    }
    #[inline]
    fn nand_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Ok(not_owned(Self::and_edge(manager, lhs, rhs)?))
    }
    #[inline]
    fn nor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_and(manager, Self::init_depth(manager), not(lhs), not(rhs))
    }
    #[inline]
    fn xor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let d = Self::init_depth(manager);
        apply_bin::<_, { BCDDOp::Xor as u8 }>(manager, d, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn equiv_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Ok(not_owned(Self::xor_edge(manager, lhs, rhs)?))
    }
    #[inline]
    fn imp_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let d = Self::init_depth(manager);
        Ok(not_owned(apply_and(manager, d, lhs.borrowed(), not(rhs))?))
    }
    #[inline]
    fn imp_strict_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_and(manager, Self::init_depth(manager), not(lhs), rhs.borrowed())
    }

    #[inline]
    fn ite_edge<'id>(
        manager: &Self::Manager<'id>,
        if_edge: &EdgeOfFunc<'id, Self>,
        then_edge: &EdgeOfFunc<'id, Self>,
        else_edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_ite(
            manager,
            Self::init_depth(manager),
            if_edge.borrowed(),
            then_edge.borrowed(),
            else_edge.borrowed(),
        )
    }

    #[inline]
    fn sat_count_edge<'id, N: SatCountNumber, S: std::hash::BuildHasher>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        vars: LevelNo,
        cache: &mut SatCountCache<N, S>,
    ) -> N {
        apply_rec_st::BCDDFunction::<F>::sat_count_edge(manager, edge, vars, cache)
    }

    #[inline]
    fn pick_cube_edge<'id, 'a, I>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        order: impl IntoIterator<IntoIter = I>,
        choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>) -> bool,
    ) -> Option<Vec<OptBool>>
    where
        I: ExactSizeIterator<Item = &'a EdgeOfFunc<'id, Self>>,
    {
        apply_rec_st::BCDDFunction::<F>::pick_cube_edge(manager, edge, order, choice)
    }

    #[inline]
    fn eval_edge<'id, 'a>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        args: impl IntoIterator<Item = (Borrowed<'a, EdgeOfFunc<'id, Self>>, bool)>,
    ) -> bool {
        apply_rec_st::BCDDFunction::<F>::eval_edge(manager, edge, args)
    }
}

impl<F: Function> BooleanFunctionQuant for BCDDFunctionMT<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag>
        + super::HasBCDDOpApplyCache<F::Manager<'id>>
        + WorkerManager,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
{
    #[inline]
    fn restrict_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let d = Self::init_depth(manager);
        restrict(manager, d, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn forall_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let d = Self::init_depth(manager);
        quant::<_, { BCDDOp::Forall as u8 }>(manager, d, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn exist_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let d = Self::init_depth(manager);
        quant::<_, { BCDDOp::Exist as u8 }>(manager, d, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn unique_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let d = Self::init_depth(manager);
        quant::<_, { BCDDOp::Unique as u8 }>(manager, d, root.borrowed(), vars.borrowed())
    }
}

impl<F: Function, T: Tag> DotStyle<T> for BCDDFunctionMT<F> {}
