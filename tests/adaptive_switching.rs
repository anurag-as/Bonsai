use bonsai::backends::{KDTree, RTree, SpatialBackend};
use bonsai::migration::SimpleIndex;
use bonsai::profiler::{CostModel, Observation, PolicyEngine, Profiler, QueryKind};
use bonsai::router::IndexRouter;
use bonsai::types::{BBox, BackendKind, EntryId, Point};
use std::sync::Arc;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn uniform_points(n: usize, seed: u64) -> Vec<Point<f64, 2>> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| Point::new([rng.next_f64() * 1000.0, rng.next_f64() * 1000.0]))
        .collect()
}

fn clustered_points(n: usize, clusters: usize, seed: u64) -> Vec<Point<f64, 2>> {
    let mut rng = Lcg::new(seed);
    let centres: Vec<[f64; 2]> = (0..clusters)
        .map(|_| {
            [
                rng.next_f64() * 800.0 + 100.0,
                rng.next_f64() * 800.0 + 100.0,
            ]
        })
        .collect();
    (0..n)
        .map(|i| {
            let c = &centres[i % clusters];
            let x = (c[0] + rng.next_normal() * 20.0).clamp(0.0, 1000.0);
            let y = (c[1] + rng.next_normal() * 20.0).clamp(0.0, 1000.0);
            Point::new([x, y])
        })
        .collect()
}

#[test]
fn policy_triggers_migration_on_workload_change() {
    let uniform = uniform_points(2000, 1);
    let clustered = clustered_points(2000, 8, 2);

    // Build DataShape for uniform data.
    let mut profiler_u = Profiler::<f64, 2>::new(512);
    for p in &uniform {
        profiler_u.observe(Observation::Insert(*p));
    }
    profiler_u.observe(Observation::Query {
        kind: QueryKind::Range,
        selectivity: 0.01,
        hit: true,
    });
    profiler_u.flush();
    let uniform_shape = profiler_u.data_shape().cloned().unwrap();

    // Build DataShape for clustered data.
    let mut profiler_c = Profiler::<f64, 2>::new(512);
    for p in &clustered {
        profiler_c.observe(Observation::Insert(*p));
    }
    profiler_c.observe(Observation::Query {
        kind: QueryKind::Range,
        selectivity: 0.01,
        hit: true,
    });
    profiler_c.flush();
    let clustered_shape = profiler_c.data_shape().cloned().unwrap();

    // Clustered data must have a higher clustering coefficient than uniform.
    assert!(
        clustered_shape.clustering_coef > uniform_shape.clustering_coef,
        "clustered data must have higher clustering_coef: clustered={:.4} uniform={:.4}",
        clustered_shape.clustering_coef,
        uniform_shape.clustering_coef,
    );

    // The cost model must prefer R-tree over KD-tree for strongly clustered data.
    let rtree_clustered = CostModel::<2>::rtree_cost(&clustered_shape);
    let kdtree_clustered = CostModel::<2>::kdtree_cost(&clustered_shape);
    assert!(
        rtree_clustered < kdtree_clustered,
        "R-tree must be cheaper than KD-tree for clustered data: rtree={rtree_clustered:.4} kdtree={kdtree_clustered:.4}"
    );

    // Start the engine on KD-tree; tick on clustered data — must fire a migration.
    let mut engine2 = PolicyEngine::<2>::with_config(BackendKind::KDTree, 0.99, 5);

    // Tick on clustered data — must fire a migration away from KD-tree.
    let mut migration_fired = false;
    for _ in 0..20 {
        if let Some(d) = engine2.tick(&clustered_shape) {
            // The target must be cheaper than KD-tree for clustered data.
            let target_cost = match d.target {
                BackendKind::RTree => CostModel::<2>::rtree_cost(&clustered_shape),
                BackendKind::KDTree => CostModel::<2>::kdtree_cost(&clustered_shape),
                BackendKind::Grid => CostModel::<2>::grid_cost(&clustered_shape),
                BackendKind::Quadtree => CostModel::<2>::quadtree_cost(&clustered_shape),
            };
            assert!(
                target_cost < kdtree_clustered,
                "migration target must be cheaper than KD-tree: target={target_cost:.4} kdtree={kdtree_clustered:.4}"
            );
            engine2.on_migration_started();
            engine2.on_migration_complete(d.target);
            migration_fired = true;
            break;
        }
    }

    assert!(
        migration_fired,
        "policy engine must trigger migration from KD-tree on clustered data (clustering_coef={:.4})",
        clustered_shape.clustering_coef,
    );

    // After migration, the engine must be on a backend that is cheaper than KD-tree.
    let final_cost = match engine2.current_backend() {
        BackendKind::RTree => CostModel::<2>::rtree_cost(&clustered_shape),
        BackendKind::KDTree => CostModel::<2>::kdtree_cost(&clustered_shape),
        BackendKind::Grid => CostModel::<2>::grid_cost(&clustered_shape),
        BackendKind::Quadtree => CostModel::<2>::quadtree_cost(&clustered_shape),
    };
    assert!(
        final_cost < kdtree_clustered,
        "final backend must be cheaper than KD-tree: final={final_cost:.4} kdtree={kdtree_clustered:.4}"
    );
}

#[test]
fn migration_correctness_range_query_identical() {
    let pts = uniform_points(1000, 42);
    let mut index: SimpleIndex<usize, f64, 2> =
        SimpleIndex::new(Box::new(KDTree::<(EntryId, usize), f64, 2>::new()));

    let mut pt_ids: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
    for (i, &p) in pts.iter().enumerate() {
        let id = index.insert(p, i);
        pt_ids.push((p, id));
    }

    let bbox = BBox::new(Point::new([200.0, 200.0]), Point::new([800.0, 800.0]));

    let mut before: Vec<EntryId> = index.range_query(&bbox);
    before.sort_by_key(|id| id.0);

    // Brute-force oracle.
    let mut oracle: Vec<EntryId> = pt_ids
        .iter()
        .filter(|(p, _)| bbox.contains_point(p))
        .map(|(_, id)| *id)
        .collect();
    oracle.sort_by_key(|id| id.0);

    assert_eq!(before, oracle, "pre-migration query must match oracle");

    let result = index
        .migrate(RTree::<(EntryId, usize), f64, 2>::bulk_load)
        .expect("migration must succeed");
    assert_eq!(result.new_backend, BackendKind::RTree);

    let mut after: Vec<EntryId> = index.range_query(&bbox);
    after.sort_by_key(|id| id.0);

    assert_eq!(
        before, after,
        "range query results must be identical before and after migration"
    );

    // Full scan must still return all entries.
    let full = BBox::new(Point::new([0.0, 0.0]), Point::new([1000.0, 1000.0]));
    assert_eq!(
        index.range_query(&full).len(),
        pts.len(),
        "all entries must survive migration"
    );
}

#[test]
fn concurrent_stress_rayon_insert_and_query() {
    use rayon::prelude::*;

    const TOTAL_INSERTS: usize = 100_000;
    const INSERT_THREADS: usize = 4;
    const QUERY_THREADS: usize = 4;
    const QUERIES_PER_THREAD: usize = 50;

    let backend: Box<dyn bonsai::backends::SpatialBackend<usize, f64, 2>> =
        Box::new(KDTree::<usize, f64, 2>::new());
    let router = Arc::new(IndexRouter::new(backend));

    let chunk_size = TOTAL_INSERTS / INSERT_THREADS;

    (0..INSERT_THREADS).into_par_iter().for_each(|t| {
        let mut rng = Lcg::new(t as u64 + 1);
        let r = Arc::clone(&router);
        for i in 0..chunk_size {
            let x = rng.next_f64() * 1000.0;
            let y = rng.next_f64() * 1000.0;
            r.insert(Point::new([x, y]), t * chunk_size + i);
        }
    });

    assert_eq!(
        router.len(),
        TOTAL_INSERTS,
        "all inserts must be present after concurrent inserts"
    );

    let full_bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([1000.0, 1000.0]));
    let partial_bbox = BBox::new(Point::new([250.0, 250.0]), Point::new([750.0, 750.0]));

    let query_results: Vec<(usize, usize)> = (0..QUERY_THREADS)
        .into_par_iter()
        .map(|_| {
            let r = Arc::clone(&router);
            let mut full_count = 0usize;
            let mut partial_count = 0usize;
            for _ in 0..QUERIES_PER_THREAD {
                full_count = r.range_query(&full_bbox).len();
                partial_count = r.range_query(&partial_bbox).len();
            }
            (full_count, partial_count)
        })
        .collect();

    for (full, partial) in &query_results {
        assert_eq!(
            *full, TOTAL_INSERTS,
            "full bbox query must return all {TOTAL_INSERTS} points"
        );
        assert!(
            *partial <= TOTAL_INSERTS,
            "partial bbox query must not exceed total inserts"
        );
    }
}

#[test]
fn concurrent_insert_query_interleaved() {
    use rayon::prelude::*;

    const INSERTS_PER_THREAD: usize = 5_000;
    const INSERT_THREADS: usize = 4;
    const QUERY_THREADS: usize = 4;

    let backend: Box<dyn bonsai::backends::SpatialBackend<usize, f64, 2>> =
        Box::new(KDTree::<usize, f64, 2>::new());
    let router = Arc::new(IndexRouter::new(backend));

    let full_bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([1000.0, 1000.0]));

    let insert_handle = {
        let r = Arc::clone(&router);
        std::thread::spawn(move || {
            (0..INSERT_THREADS).into_par_iter().for_each(|t| {
                let mut rng = Lcg::new(t as u64 + 100);
                let r2 = Arc::clone(&r);
                for i in 0..INSERTS_PER_THREAD {
                    let x = rng.next_f64() * 1000.0;
                    let y = rng.next_f64() * 1000.0;
                    r2.insert(Point::new([x, y]), t * INSERTS_PER_THREAD + i);
                }
            });
        })
    };

    let query_handle = {
        let r = Arc::clone(&router);
        std::thread::spawn(move || {
            (0..QUERY_THREADS).into_par_iter().for_each(|_| {
                let r2 = Arc::clone(&r);
                for _ in 0..20 {
                    let count = r2.range_query(&full_bbox).len();
                    assert!(
                        count <= INSERT_THREADS * INSERTS_PER_THREAD,
                        "query returned more results than total inserts: {count}"
                    );
                }
            });
        })
    };

    insert_handle.join().expect("insert thread panicked");
    query_handle.join().expect("query thread panicked");

    let final_count = router.len();
    assert_eq!(
        final_count,
        INSERT_THREADS * INSERTS_PER_THREAD,
        "final count must equal total inserts"
    );
}
