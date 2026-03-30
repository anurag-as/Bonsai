//! Analytical cost model for the four spatial index backends.
//!
//! [`CostModel`] estimates the expected query cost for each backend given a
//! [`DataShape`] and a [`QueryKind`]. All estimates are dimensionless relative
//! values — lower is cheaper.

use crate::profiler::QueryKind;
use crate::types::{BackendKind, DataShape};

/// Estimated query cost for a single backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostEstimate {
    pub backend: BackendKind,
    pub cost: f64,
}

/// Analytical cost model parameterised by dimensionality `D`.
///
/// | Backend  | Range cost formula |
/// |----------|--------------------|
/// | R-tree   | `ln(n) × sel × (1 + overlap × 2)` |
/// | KD-tree  | `n^(1 − 1/D) × clustering × sel` |
/// | Grid     | `sel^(1/D) × n^(1/D) × clustering^D` |
/// | Quadtree | `ln(n)^(D/2) × sel`  (or `f64::MAX` when D > 4) |
pub struct CostModel<const D: usize>;

impl<const D: usize> CostModel<D> {
    /// R-tree range cost: `ln(n) × selectivity × (1 + overlap_ratio × 2)`.
    pub fn rtree_cost(shape: &DataShape<D>) -> f64 {
        let n = shape.point_count as f64;
        if n < 1.0 {
            return 0.0;
        }
        let sel = shape.query_mix.mean_selectivity.clamp(0.0, 1.0);
        let overlap = shape.overlap_ratio.max(0.0);
        n.ln() * sel * (1.0 + overlap * 2.0)
    }

    /// KD-tree range cost: `n^(1 − 1/D) × clustering_coef × selectivity`.
    ///
    /// The `n^(1 − 1/D)` exponent captures the curse of dimensionality: as D
    /// grows the exponent approaches 1 and cost trends toward O(n).
    pub fn kdtree_cost(shape: &DataShape<D>) -> f64 {
        let n = shape.point_count as f64;
        if n < 1.0 {
            return 0.0;
        }
        let sel = shape.query_mix.mean_selectivity.clamp(0.0, 1.0);
        let clustering = shape.clustering_coef.max(0.0);
        n.powf(1.0 - 1.0 / D as f64) * clustering * sel
    }

    /// Grid range cost: `(selectivity^(1/D) × n^(1/D)) × clustering_coef^D`.
    pub fn grid_cost(shape: &DataShape<D>) -> f64 {
        let n = shape.point_count as f64;
        if n < 1.0 {
            return 0.0;
        }
        let d = D as f64;
        let sel = shape.query_mix.mean_selectivity.clamp(0.0, 1.0);
        let clustering = shape.clustering_coef.max(0.0);
        sel.powf(1.0 / d) * n.powf(1.0 / d) * clustering.powf(d)
    }

    /// Quadtree range cost: `ln(n)^(D/2) × selectivity`.
    ///
    /// Returns `f64::MAX` for D > 4 — the Quadtree is excluded from the
    /// candidate set at high dimensions (2^D children per node is prohibitive).
    pub fn quadtree_cost(shape: &DataShape<D>) -> f64 {
        if D > 4 {
            return f64::MAX;
        }
        let n = shape.point_count as f64;
        if n < 1.0 {
            return 0.0;
        }
        let sel = shape.query_mix.mean_selectivity.clamp(0.0, 1.0);
        n.ln().max(0.0).powf(D as f64 / 2.0) * sel
    }

    /// Estimate cost for all four backends. Results are ordered: R-tree,
    /// KD-tree, Grid, Quadtree.
    pub fn estimate_all(shape: &DataShape<D>, _kind: QueryKind) -> [CostEstimate; 4] {
        [
            CostEstimate {
                backend: BackendKind::RTree,
                cost: Self::rtree_cost(shape),
            },
            CostEstimate {
                backend: BackendKind::KDTree,
                cost: Self::kdtree_cost(shape),
            },
            CostEstimate {
                backend: BackendKind::Grid,
                cost: Self::grid_cost(shape),
            },
            CostEstimate {
                backend: BackendKind::Quadtree,
                cost: Self::quadtree_cost(shape),
            },
        ]
    }

    /// Return the [`BackendKind`] with the lowest estimated cost.
    ///
    /// Ties are broken by the order: R-tree, KD-tree, Grid, Quadtree.
    pub fn cheapest(shape: &DataShape<D>, kind: QueryKind) -> BackendKind {
        Self::estimate_all(shape, kind)
            .iter()
            .min_by(|a, b| {
                a.cost
                    .partial_cmp(&b.cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| e.backend)
            .unwrap_or(BackendKind::RTree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BBox, DataShape, Point, QueryMix};
    use proptest::prelude::*;

    fn make_shape<const D: usize>(
        n: usize,
        clustering_coef: f64,
        overlap_ratio: f64,
        selectivity: f64,
    ) -> DataShape<D> {
        DataShape {
            point_count: n,
            bbox: BBox::new(Point::new([0.0; D]), Point::new([1.0; D])),
            skewness: [0.0; D],
            clustering_coef,
            overlap_ratio,
            effective_dim: D as f64,
            query_mix: QueryMix {
                range_frac: 1.0,
                knn_frac: 0.0,
                join_frac: 0.0,
                mean_selectivity: selectivity,
            },
        }
    }

    #[test]
    fn zero_points_returns_zero_for_all_backends() {
        let shape = make_shape::<2>(0, 1.0, 0.0, 0.01);
        assert_eq!(CostModel::<2>::rtree_cost(&shape), 0.0);
        assert_eq!(CostModel::<2>::kdtree_cost(&shape), 0.0);
        assert_eq!(CostModel::<2>::grid_cost(&shape), 0.0);
        assert_eq!(CostModel::<2>::quadtree_cost(&shape), 0.0);
    }

    #[test]
    fn quadtree_returns_max_above_d4() {
        assert_eq!(
            CostModel::<5>::quadtree_cost(&make_shape::<5>(1000, 1.0, 0.0, 0.01)),
            f64::MAX
        );
        assert_eq!(
            CostModel::<8>::quadtree_cost(&make_shape::<8>(1000, 1.0, 0.0, 0.01)),
            f64::MAX
        );
    }

    #[test]
    fn quadtree_finite_at_d4() {
        let cost = CostModel::<4>::quadtree_cost(&make_shape::<4>(1000, 1.0, 0.0, 0.01));
        assert!(cost.is_finite());
    }

    #[test]
    fn rtree_cost_increases_with_overlap() {
        let low = CostModel::<2>::rtree_cost(&make_shape::<2>(10_000, 1.0, 0.0, 0.01));
        let high = CostModel::<2>::rtree_cost(&make_shape::<2>(10_000, 1.0, 1.0, 0.01));
        assert!(high > low);
    }

    #[test]
    fn kdtree_cost_increases_with_clustering() {
        let uniform = CostModel::<2>::kdtree_cost(&make_shape::<2>(10_000, 1.0, 0.0, 0.01));
        let clustered = CostModel::<2>::kdtree_cost(&make_shape::<2>(10_000, 3.0, 0.0, 0.01));
        assert!(clustered > uniform);
    }

    #[test]
    fn estimate_all_covers_all_backends() {
        let shape = make_shape::<2>(100_000, 1.0, 0.0, 0.01);
        let estimates = CostModel::<2>::estimate_all(&shape, QueryKind::Range);
        let backends: Vec<_> = estimates.iter().map(|e| e.backend).collect();
        assert!(backends.contains(&BackendKind::RTree));
        assert!(backends.contains(&BackendKind::KDTree));
        assert!(backends.contains(&BackendKind::Grid));
        assert!(backends.contains(&BackendKind::Quadtree));
    }

    #[test]
    // For any fixed dataset size n and fixed clustering coefficient, the KD-tree
    // range cost estimate must increase monotonically with D, following the
    // `n^(1 - 1/D)` formula (curse of dimensionality).
    fn cheapest_uniform_2d_prefers_kdtree_or_rtree() {
        // n=100k, uniform (clustering=1.0, overlap=0.0), selectivity=0.01
        // KD-tree: 100000^0.5 * 1.0 * 0.01 ≈ 3.16
        // R-tree:  ln(100000) * 0.01 * 1.0 ≈ 1.15
        // R-tree should win here.
        let shape = make_shape::<2>(100_000, 1.0, 0.0, 0.01);
        let best = CostModel::<2>::cheapest(&shape, QueryKind::Range);
        assert_eq!(best, BackendKind::RTree);
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_kdtree_cost_increases_with_dimension(
            n in 100usize..100_000,
            clustering_coef in 0.5f64..5.0,
            selectivity in 0.001f64..0.1,
        ) {
            let cost_d2 = CostModel::<2>::kdtree_cost(&make_shape::<2>(n, clustering_coef, 0.0, selectivity));
            let cost_d3 = CostModel::<3>::kdtree_cost(&make_shape::<3>(n, clustering_coef, 0.0, selectivity));
            let cost_d4 = CostModel::<4>::kdtree_cost(&make_shape::<4>(n, clustering_coef, 0.0, selectivity));

            prop_assert!(cost_d2 <= cost_d3,
                "D=2 cost {cost_d2:.6} should be <= D=3 cost {cost_d3:.6}");
            prop_assert!(cost_d3 <= cost_d4,
                "D=3 cost {cost_d3:.6} should be <= D=4 cost {cost_d4:.6}");
        }
    }

    // For any D > 4, the CostModel must return `f64::MAX` for any Quadtree
    // cost estimate.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_quadtree_excluded_for_high_dimensions(
            n in 1usize..1_000_000,
            clustering_coef in 0.1f64..10.0,
            selectivity in 1e-6f64..1.0,
        ) {
            prop_assert_eq!(
                CostModel::<5>::quadtree_cost(&make_shape::<5>(n, clustering_coef, 0.0, selectivity)),
                f64::MAX
            );
            prop_assert_eq!(
                CostModel::<8>::quadtree_cost(&make_shape::<8>(n, clustering_coef, 0.0, selectivity)),
                f64::MAX
            );
            let cost_d4 = CostModel::<4>::quadtree_cost(&make_shape::<4>(n, clustering_coef, 0.0, selectivity));
            prop_assert!(cost_d4 < f64::MAX, "D=4 quadtree cost must be finite, got {cost_d4}");
        }
    }
}
