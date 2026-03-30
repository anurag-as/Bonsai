//! Policy engine for adaptive backend migration decisions.
//!
//! [`PolicyEngine`] evaluates migration candidates on every tick, applying an
//! improvement threshold and a hysteresis guard to avoid thrashing.

use crate::profiler::CostModel;
use crate::types::{BackendKind, DataShape};

/// A decision to migrate to a new backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MigrationDecision {
    /// The backend to migrate to.
    pub target: BackendKind,
    /// The estimated cost ratio: `target_cost / current_cost`.
    pub cost_ratio: f64,
}

/// Decides when and where to migrate based on cost model estimates.
///
/// Migration is triggered only when all three conditions hold:
/// - No migration is currently in progress.
/// - The hysteresis window has elapsed since the last migration.
/// - The best alternative backend's cost is less than
///   `current_cost × improvement_threshold`.
///
/// `BackendKind::Quadtree` is excluded from the candidate set when `D > 4`.
#[derive(Debug, Clone)]
pub struct PolicyEngine<const D: usize> {
    improvement_threshold: f64,
    hysteresis_window: usize,
    observations_since_migration: usize,
    migrating: bool,
    current_backend: BackendKind,
}

impl<const D: usize> PolicyEngine<D> {
    /// Create a new `PolicyEngine` with default thresholds.
    ///
    /// Defaults: `improvement_threshold = 0.77`, `hysteresis_window = 1000`.
    pub fn new(initial_backend: BackendKind) -> Self {
        Self {
            improvement_threshold: 0.77,
            hysteresis_window: 1000,
            observations_since_migration: 0,
            migrating: false,
            current_backend: initial_backend,
        }
    }

    /// Create a `PolicyEngine` with custom thresholds.
    pub fn with_config(
        initial_backend: BackendKind,
        improvement_threshold: f64,
        hysteresis_window: usize,
    ) -> Self {
        Self {
            improvement_threshold,
            hysteresis_window,
            observations_since_migration: 0,
            migrating: false,
            current_backend: initial_backend,
        }
    }

    /// Return the currently active backend.
    pub fn current_backend(&self) -> BackendKind {
        self.current_backend
    }

    /// Return the configured improvement threshold.
    pub fn improvement_threshold(&self) -> f64 {
        self.improvement_threshold
    }

    /// Return the configured hysteresis window.
    pub fn hysteresis_window(&self) -> usize {
        self.hysteresis_window
    }

    /// Return the number of observations since the last migration.
    pub fn observations_since_migration(&self) -> usize {
        self.observations_since_migration
    }

    /// Return whether a migration is currently in progress.
    pub fn is_migrating(&self) -> bool {
        self.migrating
    }

    /// Notify the engine that a migration has started.
    ///
    /// Sets the `migrating` flag and resets the hysteresis counter.
    /// `current_backend` is updated only when the migration completes via
    /// [`Self::on_migration_complete`].
    pub fn on_migration_started(&mut self) {
        self.migrating = true;
        self.observations_since_migration = 0;
    }

    /// Notify the engine that the in-progress migration has completed.
    ///
    /// Updates `current_backend` to `target` and clears the `migrating` flag.
    pub fn on_migration_complete(&mut self, target: BackendKind) {
        self.current_backend = target;
        self.migrating = false;
    }

    /// Evaluate the current data shape and decide whether to migrate.
    ///
    /// Returns `Some(MigrationDecision)` when a migration should be triggered,
    /// or `None` when no action is needed. Increments the observation counter
    /// on every call regardless of outcome.
    pub fn tick(&mut self, shape: &DataShape<D>) -> Option<MigrationDecision> {
        self.observations_since_migration += 1;

        if self.migrating {
            return None;
        }

        if self.observations_since_migration <= self.hysteresis_window {
            return None;
        }

        let current_cost = self.cost_for(self.current_backend, shape);
        if current_cost <= 0.0 {
            return None;
        }

        let (target, best_cost) = [
            BackendKind::RTree,
            BackendKind::KDTree,
            BackendKind::Grid,
            BackendKind::Quadtree,
        ]
        .iter()
        .copied()
        .filter(|&c| c != self.current_backend && !(c == BackendKind::Quadtree && D > 4))
        .map(|c| (c, self.cost_for(c, shape)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        let cost_ratio = best_cost / current_cost;
        if cost_ratio >= self.improvement_threshold {
            return None;
        }

        Some(MigrationDecision { target, cost_ratio })
    }

    fn cost_for(&self, backend: BackendKind, shape: &DataShape<D>) -> f64 {
        match backend {
            BackendKind::RTree => CostModel::<D>::rtree_cost(shape),
            BackendKind::KDTree => CostModel::<D>::kdtree_cost(shape),
            BackendKind::Grid => CostModel::<D>::grid_cost(shape),
            BackendKind::Quadtree => CostModel::<D>::quadtree_cost(shape),
        }
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

    // For any migration event, the PolicyEngine must not trigger another migration for the next
    // `hysteresis_window` observations, regardless of cost model estimates
    // during that window.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_policy_hysteresis_guard(
            hysteresis_window in 1usize..200,
            ticks_in_window in 1usize..200,
        ) {
            // Strongly clustered shape: without the hysteresis guard this shape
            // would trigger migration from KD-tree to R-tree.
            let shape = make_shape::<2>(100_000, 5.0, 0.0, 0.01);

            let mut engine = PolicyEngine::<2>::with_config(
                BackendKind::KDTree,
                0.77,
                hysteresis_window,
            );

            // Simulate a migration having just completed so the counter resets.
            engine.on_migration_started();
            engine.on_migration_complete(BackendKind::KDTree);

            // All ticks strictly inside the window must return None.
            let ticks = ticks_in_window.min(hysteresis_window);
            for _ in 0..ticks {
                prop_assert!(engine.tick(&shape).is_none());
            }
        }
    }

    // For any data shape where no alternative backend's estimated cost is less than
    // `current_cost × improvement_threshold`, the PolicyEngine must not trigger
    // a migration.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_policy_improvement_threshold(
            n in 100usize..100_000,
            selectivity in 0.001f64..0.1,
        ) {
            // threshold=0.0 means the alternative must be 100% cheaper (impossible).
            // No migration should ever fire regardless of data shape.
            let shape = make_shape::<2>(n, 1.0, 0.0, selectivity);
            let mut engine = PolicyEngine::<2>::with_config(BackendKind::KDTree, 0.0, 0);

            for _ in 0..5 {
                prop_assert!(engine.tick(&shape).is_none());
            }
        }
    }

    // For any scenario where a migration is already in progress, a second call to trigger migration must
    // be rejected (PolicyEngine returns None and migrating flag remains true).
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_no_concurrent_migrations(
            n in 100usize..100_000,
            clustering_coef in 1.0f64..10.0,
            selectivity in 0.001f64..0.1,
        ) {
            let shape = make_shape::<2>(n, clustering_coef, 0.0, selectivity);
            let mut engine = PolicyEngine::<2>::with_config(BackendKind::KDTree, 0.77, 0);

            engine.on_migration_started();
            prop_assert!(engine.is_migrating());

            for _ in 0..10 {
                prop_assert!(engine.tick(&shape).is_none());
                prop_assert!(engine.is_migrating());
            }
        }
    }

    #[test]
    fn hysteresis_prevents_immediate_remigration() {
        let shape = make_shape::<2>(100_000, 10.0, 0.0, 0.01);
        let mut engine = PolicyEngine::<2>::with_config(BackendKind::KDTree, 0.77, 5);

        for _ in 0..6 {
            engine.tick(&shape);
        }
        let decision = engine.tick(&shape);
        assert!(
            decision.is_some(),
            "expected migration decision after window"
        );

        let target = decision.unwrap().target;
        engine.on_migration_started();
        engine.on_migration_complete(target);

        assert!(
            engine.tick(&shape).is_none(),
            "expected None immediately after migration"
        );
    }

    #[test]
    fn no_migration_when_already_migrating() {
        let shape = make_shape::<2>(100_000, 10.0, 0.0, 0.01);
        let mut engine = PolicyEngine::<2>::with_config(BackendKind::KDTree, 0.77, 0);

        engine.on_migration_started();
        assert!(engine.is_migrating());
        assert!(engine.tick(&shape).is_none());
        assert!(engine.is_migrating());
    }

    #[test]
    fn quadtree_excluded_at_d5() {
        let shape = make_shape::<5>(10_000, 1.0, 0.0, 0.01);
        let mut engine = PolicyEngine::<5>::with_config(BackendKind::KDTree, 0.77, 0);

        for _ in 0..2 {
            engine.tick(&shape);
        }
        if let Some(d) = engine.tick(&shape) {
            assert_ne!(d.target, BackendKind::Quadtree);
        }
    }

    #[test]
    fn migration_fires_after_hysteresis_window() {
        let shape = make_shape::<2>(100_000, 10.0, 0.0, 0.01);
        let mut engine = PolicyEngine::<2>::with_config(BackendKind::KDTree, 0.77, 3);

        for i in 1..=3 {
            assert!(
                engine.tick(&shape).is_none(),
                "tick {i}: expected None within window"
            );
        }
        assert!(
            engine.tick(&shape).is_some(),
            "tick 4: expected migration decision after window"
        );
    }

    #[test]
    fn no_migration_when_threshold_not_met() {
        let shape = make_shape::<2>(10_000, 1.0, 0.0, 0.01);
        let mut engine = PolicyEngine::<2>::with_config(BackendKind::RTree, 0.0, 0);

        for _ in 0..5 {
            assert!(engine.tick(&shape).is_none());
        }
    }

    #[test]
    fn observations_counter_increments_each_tick() {
        let shape = make_shape::<2>(1000, 1.0, 0.0, 0.01);
        let mut engine = PolicyEngine::<2>::new(BackendKind::RTree);

        assert_eq!(engine.observations_since_migration(), 0);
        engine.tick(&shape);
        assert_eq!(engine.observations_since_migration(), 1);
        engine.tick(&shape);
        assert_eq!(engine.observations_since_migration(), 2);
    }

    #[test]
    fn on_migration_started_resets_counter() {
        let shape = make_shape::<2>(1000, 1.0, 0.0, 0.01);
        let mut engine = PolicyEngine::<2>::new(BackendKind::RTree);

        for _ in 0..10 {
            engine.tick(&shape);
        }
        assert_eq!(engine.observations_since_migration(), 10);

        engine.on_migration_started();
        assert_eq!(engine.observations_since_migration(), 0);
        assert!(engine.is_migrating());
    }

    #[test]
    fn current_backend_updates_on_complete_not_start() {
        let mut engine = PolicyEngine::<2>::new(BackendKind::KDTree);
        assert_eq!(engine.current_backend(), BackendKind::KDTree);

        engine.on_migration_started();
        assert_eq!(engine.current_backend(), BackendKind::KDTree);

        engine.on_migration_complete(BackendKind::RTree);
        assert_eq!(engine.current_backend(), BackendKind::RTree);
    }
}
