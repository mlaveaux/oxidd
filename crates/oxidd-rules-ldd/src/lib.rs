//! List decision diagrams (LDDs) for OxiDD.

use std::hash::{BuildHasher, Hash};

use oxidd_core::{
    function::{EdgeOfFunc, Function},
    util::{AllocResult, Borrowed, EdgeDropGuard},
    ApplyCache, DiagramRules, Edge, HasApplyCache, HasLevel, InnerNode, LevelNo, Manager,
    ManagerRef, ReducedOrNew,
};
use oxidd_derive::{Countable, Function};

use oxidd_core::util::{SatCountCache, SatCountNumber};

use crate::apply::*;
use crate::recursor::SequentialRecursor;

mod apply;
mod recursor;
mod saturate;

#[cfg(feature = "statistics")]
pub use apply::print_stats;

/// Terminal nodes in simple binary decision diagrams
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Countable, Debug)]
#[repr(u8)]
pub enum LDDTerminal {
    /// This represents the empty set, also denoted by `false`.
    Empty,
    /// This represents the set containing only the empty list, also denoted by
    /// `true`.
    True,
}

/// Native operators of this LDD implementation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash, Ord, Countable, Debug)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum LDDOp {
    Union,

    Project,

    RelationalProduct,

    RelationalPredecessor,

    Intersect,

    Minus,

    /// Node-wise saturation (`Sat_p`, see [`crate::saturate`]).
    Saturate,

    /// The recursive, node-saturating half of one event's firing below its `top` level (see
    /// [`crate::saturate`]).
    SatRecFire,
}

/// For LDDs it is essential that values are ordered and can be cloned.
trait LDDValue: Clone + Ord + Eq + Hash {
    /// The value used to indicate "included" in a projection meta-LDD.
    fn true_value() -> Self;

    /// The value used to indicate "excluded" in a projection meta-LDD, and
    /// "skip" (neither read nor write) in a relation-product meta-LDD.
    fn false_value() -> Self;

    /// Value encoding a read-only position in a relation-product meta-LDD.
    fn read_only_value() -> Self;

    /// Value encoding a write-only position in a relation-product meta-LDD.
    fn write_only_value() -> Self;

    /// Value encoding the *read* half of a read+write position pair in a
    /// relation-product meta-LDD.
    fn read_of_pair_value() -> Self;

    /// Value encoding the *write* half of a read+write position pair in a
    /// relation-product meta-LDD.
    fn write_of_pair_value() -> Self;
}

/// The default LDDValue to be used.
impl LDDValue for u32 {
    #[inline(always)]
    fn true_value() -> Self {
        1
    }

    #[inline(always)]
    fn false_value() -> Self {
        0
    }

    #[inline(always)]
    fn read_only_value() -> Self {
        1
    }

    #[inline(always)]
    fn write_only_value() -> Self {
        2
    }

    #[inline(always)]
    fn read_of_pair_value() -> Self {
        3
    }

    #[inline(always)]
    fn write_of_pair_value() -> Self {
        4
    }
}

trait LDDManager:
    Manager<Terminal = LDDTerminal, InnerNodeValue: LDDValue, InnerNode: HasLevel>
    + HasApplyCache<Self, LDDOp>
{
}
impl<M> LDDManager for M
where
    M: Manager<Terminal = LDDTerminal> + HasApplyCache<M, LDDOp>,
    M::InnerNodeValue: LDDValue,
    M::InnerNode: HasLevel,
{
}

// --- Function Interface ------------------------------------------------------

/// Result of [`LDDFunction::relation_product_meta`].
///
/// Bundles the meta-LDD encoding the read/write projection together with the
/// positions of the read and write variables in that encoding.
pub struct RelationProductMeta<E> {
    /// The meta-LDD encoding the read/write projection.
    pub meta: E,
    /// The positions of the read variables in the meta encoding.
    pub read_positions: Vec<usize>,
    /// The positions of the write variables in the meta encoding.
    pub write_positions: Vec<usize>,
}

/// One event (transition group) used by [`LDDFunction::saturate_edge`].
///
/// `relation` and `meta_at_top` are as produced by [`LDDFunction::relation_product_meta`], except
/// `meta_at_top` must already be descended `top` times, so that its own root describes state
/// position `top` — the meta-LDD is otherwise padded with `false_value` entries for every position
/// below the actual bottom, which [`crate::saturate`] never needs to see.
pub struct SaturationEvent<E> {
    /// The event's transition relation (short-vector encoding).
    pub relation: E,
    /// The event's relation-product meta, descended `top` times.
    pub meta_at_top: E,
    /// First (topmost) state-vector position this event reads or writes.
    pub top: u32,
    /// Last (bottommost) state-vector position this event reads or writes.
    pub bot: u32,
}

/// Boolean function backed by a list decision diagram
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
#[repr_id = "LDD"]
#[repr(transparent)]
pub struct LDDFunction<F: Function>(F);

impl<F: Function> From<F> for LDDFunction<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        LDDFunction(value)
    }
}

impl<F: Function> LDDFunction<F> {
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }
}

#[allow(private_bounds)]
impl<F: Function> LDDFunction<F>
where
    for<'id> F::Manager<'id>: LDDManager,
{
    pub fn relation_product_meta<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        read_proj: &[u32],
        write_proj: &[u32],
    ) -> AllocResult<
        RelationProductMeta<<<LDDFunction<F> as Function>::Manager<'id> as Manager>::Edge>,
    > {
        relation_product_meta(manager, read_proj, write_proj)
    }

    #[inline]
    pub fn empty_set<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
    ) -> AllocResult<Self> {
        Ok(Self::from_edge(
            manager,
            manager.get_terminal(LDDTerminal::Empty)?,
        ))
    }

    #[inline]
    pub fn empty_vector<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
    ) -> AllocResult<Self> {
        Ok(Self::from_edge(
            manager,
            manager.get_terminal(LDDTerminal::True)?,
        ))
    }

    /// Returns an LDD containing only the given vector, i.e., { vector }.
    #[inline]
    pub fn singleton<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        vector: &[<<LDDFunction<F> as Function>::Manager<'id> as Manager>::InnerNodeValue],
    ) -> AllocResult<LDDFunction<F>> {
        Ok(Self::from_edge(manager, singleton(manager, vector)?))
    }

    #[inline]
    pub fn singleton_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        vector: &[<<LDDFunction<F> as Function>::Manager<'id> as Manager>::InnerNodeValue],
    ) -> AllocResult<<<LDDFunction<F> as Function>::Manager<'id> as Manager>::Edge> {
        singleton(manager, vector)
    }

    #[inline]
    pub fn union_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        f: EdgeOfFunc<'id, Self>,
        g: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let f = EdgeDropGuard::new(manager, f);
        let g = EdgeDropGuard::new(manager, g);
        apply_union(manager, SequentialRecursor, f.borrowed(), g.borrowed())
    }

    /// Computes the intersection `a ∩ b` of the two sets of vectors.
    #[inline]
    pub fn intersect_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        a: EdgeOfFunc<'id, Self>,
        b: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let a = EdgeDropGuard::new(manager, a);
        let b = EdgeDropGuard::new(manager, b);
        apply_intersect(manager, SequentialRecursor, a.borrowed(), b.borrowed())
    }

    /// Returns the largest subset of `a` that does not contain any element of
    /// `b` (set difference `a \ b`).
    #[inline]
    pub fn minus_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        a: EdgeOfFunc<'id, Self>,
        b: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let a = EdgeDropGuard::new(manager, a);
        let b = EdgeDropGuard::new(manager, b);
        apply_minus(manager, SequentialRecursor, a.borrowed(), b.borrowed())
    }

    /// Computes the set of vectors reachable in one step from `set` via the
    /// sparse relation `rel`.  `meta` must be produced by
    /// [`relation_product_meta`][Self::relation_product_meta].
    #[inline]
    pub fn relational_product_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        set: EdgeOfFunc<'id, Self>,
        rel: EdgeOfFunc<'id, Self>,
        meta: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let set = EdgeDropGuard::new(manager, set);
        let rel = EdgeDropGuard::new(manager, rel);
        let meta = EdgeDropGuard::new(manager, meta);
        apply_relational_product(
            manager,
            SequentialRecursor,
            set.borrowed(),
            rel.borrowed(),
            meta.borrowed(),
        )
    }

    /// Computes the set of source vectors in `universe` that can reach a vector
    /// in `set` in one step via the sparse relation `rel`.  `meta` must be
    /// produced by [`relation_product_meta`][Self::relation_product_meta].
    ///
    /// This is the inverse of
    /// [`relational_product_edge`][Self::relational_product_edge].
    #[inline]
    pub fn relational_predecessor_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        set: EdgeOfFunc<'id, Self>,
        rel: EdgeOfFunc<'id, Self>,
        meta: EdgeOfFunc<'id, Self>,
        universe: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let set = EdgeDropGuard::new(manager, set);
        let rel = EdgeDropGuard::new(manager, rel);
        let meta = EdgeDropGuard::new(manager, meta);
        let universe = EdgeDropGuard::new(manager, universe);
        apply_relational_predecessor(
            manager,
            SequentialRecursor,
            set.borrowed(),
            rel.borrowed(),
            meta.borrowed(),
            universe.borrowed(),
        )
    }

    /// Returns the number of vectors (lists) contained in the set rooted at
    /// `set`.
    ///
    /// Like [`sat_count_edge`][oxidd_core::function::BooleanFunction::sat_count_edge], the count
    /// is computed in the number type `N` (for example `u128`, or `F64` for an approximation)
    /// and the results per node are kept in `cache`, which can be reused for other sets.
    #[inline]
    pub fn len_edge<'id, N: SatCountNumber, S: BuildHasher>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        set: EdgeOfFunc<'id, Self>,
        cache: &mut SatCountCache<N, S>,
    ) -> N {
        // The LDD has no variables to count over, only its number of vectors matters.
        cache.clear_if_invalid(manager, 0);
        let set = EdgeDropGuard::new(manager, set);
        len(manager, set.borrowed(), &mut cache.map)
    }

    /// Computes a meta-LDD encoding the projection onto the indices in `proj`,
    /// suitable as the second argument of [`project_edge`][Self::project_edge].
    #[inline]
    pub fn projection_meta<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        proj: &[u32],
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        compute_proj(manager, proj)
    }

    /// Projects the vectors in `set` onto the indices encoded by `proj` (built
    /// via [`projection_meta`][Self::projection_meta]).
    #[inline]
    pub fn project_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        set: EdgeOfFunc<'id, Self>,
        proj: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let set = EdgeDropGuard::new(manager, set);
        let proj = EdgeDropGuard::new(manager, proj);
        project(manager, SequentialRecursor, set.borrowed(), proj.borrowed())
    }

    /// Computes the union `self ∪ other`.
    pub fn union(&self, other: &Self) -> AllocResult<Self> {
        self.manager_ref().with_manager_shared(|manager| {
            let edge = Self::union_edge(
                manager,
                manager.clone_edge(self.as_edge(manager)),
                manager.clone_edge(other.as_edge(manager)),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Computes the intersection `self ∩ other`.
    pub fn intersect(&self, other: &Self) -> AllocResult<Self> {
        self.manager_ref().with_manager_shared(|manager| {
            let edge = Self::intersect_edge(
                manager,
                manager.clone_edge(self.as_edge(manager)),
                manager.clone_edge(other.as_edge(manager)),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Computes the set difference `self \ other`.
    pub fn minus(&self, other: &Self) -> AllocResult<Self> {
        self.manager_ref().with_manager_shared(|manager| {
            let edge = Self::minus_edge(
                manager,
                manager.clone_edge(self.as_edge(manager)),
                manager.clone_edge(other.as_edge(manager)),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Computes the set of vectors reachable in one step from `self` via the
    /// sparse relation `rel`, guided by `meta` (produced by
    /// [`relation_product_meta`][Self::relation_product_meta]).
    pub fn relational_product(&self, rel: &Self, meta: &Self) -> AllocResult<Self> {
        self.manager_ref().with_manager_shared(|manager| {
            let edge = Self::relational_product_edge(
                manager,
                manager.clone_edge(self.as_edge(manager)),
                manager.clone_edge(rel.as_edge(manager)),
                manager.clone_edge(meta.as_edge(manager)),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Computes the set of source vectors in `universe` from which a vector in
    /// `self` is reachable in one step via the sparse relation `rel`, guided by
    /// `meta` (produced by
    /// [`relation_product_meta`][Self::relation_product_meta]).
    ///
    /// This is the inverse of
    /// [`relational_product`][Self::relational_product].
    pub fn relational_predecessor(
        &self,
        rel: &Self,
        meta: &Self,
        universe: &Self,
    ) -> AllocResult<Self> {
        self.manager_ref().with_manager_shared(|manager| {
            let edge = Self::relational_predecessor_edge(
                manager,
                manager.clone_edge(self.as_edge(manager)),
                manager.clone_edge(rel.as_edge(manager)),
                manager.clone_edge(meta.as_edge(manager)),
                manager.clone_edge(universe.as_edge(manager)),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Projects the vectors in `self` onto the indices encoded by `proj`
    /// (produced by [`projection_meta`][Self::projection_meta]).
    pub fn project(&self, proj: &Self) -> AllocResult<Self> {
        self.manager_ref().with_manager_shared(|manager| {
            let edge = Self::project_edge(
                manager,
                manager.clone_edge(self.as_edge(manager)),
                manager.clone_edge(proj.as_edge(manager)),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Returns the number of vectors (lists) contained in `self`, see
    /// [`len_edge`][Self::len_edge].
    pub fn len<N: SatCountNumber, S: BuildHasher>(&self, cache: &mut SatCountCache<N, S>) -> N {
        self.manager_ref().with_manager_shared(|manager| {
            Self::len_edge(manager, manager.clone_edge(self.as_edge(manager)), cache)
        })
    }

    /// Returns `true` if `self` is the empty set `∅`, i.e. contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.manager_ref().with_manager_shared(|manager| {
            manager
                .get_node(self.as_edge(manager))
                .is_terminal(&LDDTerminal::Empty)
        })
    }

    /// Returns `N*(set)`, the smallest superset of `set` closed under every event in `events`,
    /// computed by node-wise saturation (see [`crate::saturate`]) rather than whole-set fixpoint
    /// iteration.
    ///
    /// The vectors in `set` may have any length, as long as it is the same for all of them and the
    /// events fit within it.
    ///
    /// The results are cached, and the cache does not know the events. Call
    /// [`clear_saturation_cache`][Self::clear_saturation_cache] before every call whose events differ
    /// from those of an earlier call on this manager, otherwise the result can be silently wrong.
    #[inline]
    pub fn saturate_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        set: EdgeOfFunc<'id, Self>,
        events: &[SaturationEvent<EdgeOfFunc<'id, Self>>],
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let set = EdgeDropGuard::new(manager, set);
        crate::saturate::saturate(manager, set.borrowed(), 0, events)
    }

    /// Removes the cached results of [`saturate_edge`][Self::saturate_edge] (`LDDOp::Saturate` and
    /// `LDDOp::SatRecFire`) from the apply cache, and nothing else. Results of every other operation
    /// do not depend on the events and are kept.
    #[inline]
    pub fn clear_saturation_cache<'id>(manager: &<LDDFunction<F> as Function>::Manager<'id>) {
        manager.apply_cache().clear_operators(manager, |op| {
            matches!(op, LDDOp::Saturate | LDDOp::SatRecFire)
        });
    }
}

/// Multi-threaded list decision diagrams
#[cfg(feature = "multi-threading")]
pub mod mt {
    use oxidd_core::HasWorkers;

    use crate::recursor::mt::ParallelRecursor;

    use super::*;

    /// Set of vectors backed by a list decision diagram, multi-threaded version
    ///
    /// This is the parallel counterpart of [`LDDFunction`]: the apply
    /// algorithms split independent sub-problems onto the manager's worker
    /// pool. All other operations behave exactly like the sequential
    /// [`LDDFunction`].
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
    #[repr_id = "LDD"]
    #[repr(transparent)]
    pub struct LDDFunctionMT<F: Function>(F);

    impl<F: Function> From<F> for LDDFunctionMT<F> {
        #[inline(always)]
        fn from(value: F) -> Self {
            LDDFunctionMT(value)
        }
    }

    impl<F: Function> LDDFunctionMT<F> {
        /// Convert `self` into the underlying [`Function`]
        #[inline(always)]
        pub fn into_inner(self) -> F {
            self.0
        }
    }

    #[allow(private_bounds)]
    impl<F: Function> LDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>: LDDManager + HasWorkers,
        for<'id> EdgeOfFunc<'id, Self>: Send + Sync,
    {
        /// See [`LDDFunction::relation_product_meta`].
        pub fn relation_product_meta<'id>(
            manager: &<Self as Function>::Manager<'id>,
            read_proj: &[u32],
            write_proj: &[u32],
        ) -> AllocResult<RelationProductMeta<EdgeOfFunc<'id, Self>>> {
            relation_product_meta(manager, read_proj, write_proj)
        }

        /// See [`LDDFunction::empty_set`].
        #[inline]
        pub fn empty_set<'id>(manager: &<Self as Function>::Manager<'id>) -> AllocResult<Self> {
            Ok(Self::from_edge(
                manager,
                manager.get_terminal(LDDTerminal::Empty)?,
            ))
        }

        /// See [`LDDFunction::empty_vector`].
        #[inline]
        pub fn empty_vector<'id>(manager: &<Self as Function>::Manager<'id>) -> AllocResult<Self> {
            Ok(Self::from_edge(
                manager,
                manager.get_terminal(LDDTerminal::True)?,
            ))
        }

        /// See [`LDDFunction::singleton`].
        #[inline]
        pub fn singleton<'id>(
            manager: &<Self as Function>::Manager<'id>,
            vector: &[<<Self as Function>::Manager<'id> as Manager>::InnerNodeValue],
        ) -> AllocResult<Self> {
            Ok(Self::from_edge(manager, singleton(manager, vector)?))
        }

        /// See [`LDDFunction::singleton_edge`].
        #[inline]
        pub fn singleton_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            vector: &[<<Self as Function>::Manager<'id> as Manager>::InnerNodeValue],
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            singleton(manager, vector)
        }

        /// See [`LDDFunction::union_edge`].
        #[inline]
        pub fn union_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            f: EdgeOfFunc<'id, Self>,
            g: EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let f = EdgeDropGuard::new(manager, f);
            let g = EdgeDropGuard::new(manager, g);
            apply_union(
                manager,
                ParallelRecursor::new(manager),
                f.borrowed(),
                g.borrowed(),
            )
        }

        /// See [`LDDFunction::intersect_edge`].
        #[inline]
        pub fn intersect_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            a: EdgeOfFunc<'id, Self>,
            b: EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let a = EdgeDropGuard::new(manager, a);
            let b = EdgeDropGuard::new(manager, b);
            apply_intersect(
                manager,
                ParallelRecursor::new(manager),
                a.borrowed(),
                b.borrowed(),
            )
        }

        /// See [`LDDFunction::minus_edge`].
        #[inline]
        pub fn minus_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            a: EdgeOfFunc<'id, Self>,
            b: EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let a = EdgeDropGuard::new(manager, a);
            let b = EdgeDropGuard::new(manager, b);
            apply_minus(
                manager,
                ParallelRecursor::new(manager),
                a.borrowed(),
                b.borrowed(),
            )
        }

        /// See [`LDDFunction::relational_product_edge`].
        #[inline]
        pub fn relational_product_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            set: EdgeOfFunc<'id, Self>,
            rel: EdgeOfFunc<'id, Self>,
            meta: EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let set = EdgeDropGuard::new(manager, set);
            let rel = EdgeDropGuard::new(manager, rel);
            let meta = EdgeDropGuard::new(manager, meta);
            apply_relational_product(
                manager,
                ParallelRecursor::new(manager),
                set.borrowed(),
                rel.borrowed(),
                meta.borrowed(),
            )
        }

        /// See [`LDDFunction::relational_predecessor_edge`].
        #[inline]
        pub fn relational_predecessor_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            set: EdgeOfFunc<'id, Self>,
            rel: EdgeOfFunc<'id, Self>,
            meta: EdgeOfFunc<'id, Self>,
            universe: EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let set = EdgeDropGuard::new(manager, set);
            let rel = EdgeDropGuard::new(manager, rel);
            let meta = EdgeDropGuard::new(manager, meta);
            let universe = EdgeDropGuard::new(manager, universe);
            apply_relational_predecessor(
                manager,
                ParallelRecursor::new(manager),
                set.borrowed(),
                rel.borrowed(),
                meta.borrowed(),
                universe.borrowed(),
            )
        }

        /// See [`LDDFunction::len_edge`].
        #[inline]
        pub fn len_edge<'id, N: SatCountNumber, S: BuildHasher>(
            manager: &<Self as Function>::Manager<'id>,
            set: EdgeOfFunc<'id, Self>,
            cache: &mut SatCountCache<N, S>,
        ) -> N {
            cache.clear_if_invalid(manager, 0);
            let set = EdgeDropGuard::new(manager, set);
            len(manager, set.borrowed(), &mut cache.map)
        }

        /// See [`LDDFunction::projection_meta`].
        #[inline]
        pub fn projection_meta<'id>(
            manager: &<Self as Function>::Manager<'id>,
            proj: &[u32],
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            compute_proj(manager, proj)
        }

        /// See [`LDDFunction::project_edge`].
        #[inline]
        pub fn project_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            set: EdgeOfFunc<'id, Self>,
            proj: EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let set = EdgeDropGuard::new(manager, set);
            let proj = EdgeDropGuard::new(manager, proj);
            project(
                manager,
                ParallelRecursor::new(manager),
                set.borrowed(),
                proj.borrowed(),
            )
        }

        /// See [`LDDFunction::union`].
        pub fn union(&self, other: &Self) -> AllocResult<Self> {
            self.manager_ref().with_manager_shared(|manager| {
                let edge = Self::union_edge(
                    manager,
                    manager.clone_edge(self.as_edge(manager)),
                    manager.clone_edge(other.as_edge(manager)),
                )?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::intersect`].
        pub fn intersect(&self, other: &Self) -> AllocResult<Self> {
            self.manager_ref().with_manager_shared(|manager| {
                let edge = Self::intersect_edge(
                    manager,
                    manager.clone_edge(self.as_edge(manager)),
                    manager.clone_edge(other.as_edge(manager)),
                )?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::minus`].
        pub fn minus(&self, other: &Self) -> AllocResult<Self> {
            self.manager_ref().with_manager_shared(|manager| {
                let edge = Self::minus_edge(
                    manager,
                    manager.clone_edge(self.as_edge(manager)),
                    manager.clone_edge(other.as_edge(manager)),
                )?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::relational_product`].
        pub fn relational_product(&self, rel: &Self, meta: &Self) -> AllocResult<Self> {
            self.manager_ref().with_manager_shared(|manager| {
                let edge = Self::relational_product_edge(
                    manager,
                    manager.clone_edge(self.as_edge(manager)),
                    manager.clone_edge(rel.as_edge(manager)),
                    manager.clone_edge(meta.as_edge(manager)),
                )?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::relational_predecessor`].
        pub fn relational_predecessor(
            &self,
            rel: &Self,
            meta: &Self,
            universe: &Self,
        ) -> AllocResult<Self> {
            self.manager_ref().with_manager_shared(|manager| {
                let edge = Self::relational_predecessor_edge(
                    manager,
                    manager.clone_edge(self.as_edge(manager)),
                    manager.clone_edge(rel.as_edge(manager)),
                    manager.clone_edge(meta.as_edge(manager)),
                    manager.clone_edge(universe.as_edge(manager)),
                )?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::project`].
        pub fn project(&self, proj: &Self) -> AllocResult<Self> {
            self.manager_ref().with_manager_shared(|manager| {
                let edge = Self::project_edge(
                    manager,
                    manager.clone_edge(self.as_edge(manager)),
                    manager.clone_edge(proj.as_edge(manager)),
                )?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::len`].
        pub fn len<N: SatCountNumber, S: BuildHasher>(&self, cache: &mut SatCountCache<N, S>) -> N {
            self.manager_ref().with_manager_shared(|manager| {
                Self::len_edge(manager, manager.clone_edge(self.as_edge(manager)), cache)
            })
        }

        /// See [`LDDFunction::is_empty`].
        pub fn is_empty(&self) -> bool {
            self.manager_ref().with_manager_shared(|manager| {
                manager
                    .get_node(self.as_edge(manager))
                    .is_terminal(&LDDTerminal::Empty)
            })
        }

        /// See [`LDDFunction::saturate_edge`].
        ///
        /// Saturation is not yet parallelised (its node-local fixpoint is inherently sequential in
        /// its accumulator): this dispatches to the same sequential implementation regardless of
        /// the manager's worker pool.
        #[inline]
        pub fn saturate_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            set: EdgeOfFunc<'id, Self>,
            events: &[SaturationEvent<EdgeOfFunc<'id, Self>>],
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let set = EdgeDropGuard::new(manager, set);
            crate::saturate::saturate(manager, set.borrowed(), 0, events)
        }

        /// See [`LDDFunction::clear_saturation_cache`].
        #[inline]
        pub fn clear_saturation_cache<'id>(manager: &<Self as Function>::Manager<'id>) {
            manager.apply_cache().clear_operators(manager, |op| {
                matches!(op, LDDOp::Saturate | LDDOp::SatRecFire)
            });
        }
    }
}

/// [`DiagramRules`] for list decision diagrams
pub struct LDDRules;

impl<E: Edge, N: InnerNode<E, Value: LDDValue>> DiagramRules<E, N, LDDTerminal> for LDDRules {
    type Cofactors<'a>
        = N::ChildrenIter<'a>
    where
        N: 'a,
        E: 'a;

    #[inline(always)]
    fn reduce<M: Manager<Edge = E, InnerNode = N>>(
        _manager: &M,
        _level: LevelNo,
        _children: impl IntoIterator<Item = E>,
    ) -> ReducedOrNew<E, N> {
        unimplemented!("Missing the value to construct a new node");
    }

    #[inline(always)]
    fn cofactors(_tag: E::Tag, node: &N) -> Self::Cofactors<'_> {
        node.children()
    }

    #[inline(always)]
    fn cofactor(_tag: E::Tag, node: &N, n: usize) -> Borrowed<'_, E> {
        node.child(n)
    }
}

macro_rules! stat {
    (call $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        crate::apply::STAT_COUNTERS[$op as usize]
            .calls
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
    (cache_query $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        crate::apply::STAT_COUNTERS[$op as usize]
            .cache_queries
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
    (cache_hit $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        crate::apply::STAT_COUNTERS[$op as usize]
            .cache_hits
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
    (reduced $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        crate::apply::STAT_COUNTERS[$op as usize]
            .reduced
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
}

pub(crate) use stat;
