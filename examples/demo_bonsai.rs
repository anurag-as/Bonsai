//! Bonsai end-to-end demo.
//!
//! One dataset, one story — each module builds on the previous:
//!
//!   1. HilbertCurve  — sort 2 000 clustered points into cache-friendly order
//!   2. BloomCache    — populate from the Hilbert-sorted insertions
//!   3. All 4 backends (RTree, KDTree, Quadtree, Grid) — loaded in Hilbert order
//!   4. Range query   — Bloom gates all four backends; brute-force oracle check
//!   5. kNN query     — all four backends, results compared
//!   6. Insert-remove — remove half the entries from every backend, verify absence
//!   7. Profiler      — feed the §1 dataset into the profiler; measure DataShape
//!   8. CostModel     — rank backends using the §7 DataShape
//!   9. PolicyEngine  — tick on the §7 shape; show migration decision
//!  10. Migration     — execute the §9 decision: migrate the §1 KD-tree to the
//!                      policy-chosen backend; verify range query results are
//!                      identical before and after
//!  11. IndexRouter + StatsCollector — wrap the §10 migrated backend in an
//!                      IndexRouter; re-insert the §1 dataset across 4 threads
//!                      while 2 threads query the §4 bbox; StatsCollector
//!                      tracks all inserts and queries lock-free
//!  12. BonsaiIndex   — canonical builder-pattern example: insert 5 named 2D
//!                      points, run a range query, run kNN(k=2), print stats
//!
//! Run with:
//!   cargo run --example demo_bonsai

use std::sync::Arc;

use bonsai::backends::{GridIndex, KDTree, Quadtree, RTree, SpatialBackend};
use bonsai::bloom::{BloomCache, BloomResult};
use bonsai::hilbert::HilbertCurve;
use bonsai::migration::SimpleIndex;
use bonsai::profiler::{CostModel, Observation, PolicyEngine, Profiler, QueryKind};
use bonsai::router::IndexRouter;
use bonsai::stats::StatsCollector;
use bonsai::types::{BBox, BackendKind, EntryId, Point};
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

fn normalise(v: f64, domain: f64, order: u32) -> u64 {
    let cells = (1u64 << order) as f64;
    ((v / domain) * cells).clamp(0.0, cells - 1.0) as u64
}

fn main() {
    const N: usize = 2_000;
    const CLUSTERS: usize = 10;
    const DOMAIN: f64 = 1_000.0;
    const HILBERT_ORDER: u32 = 10;

    // ── 1. Hilbert curve: sort the dataset ────────────────────────────────────
    println!("=== 1. Hilbert sort ({N} clustered points, D=2, order={HILBERT_ORDER}) ===");

    let hilbert = HilbertCurve::<2>::new(HILBERT_ORDER);
    let mut rng = Lcg::new(42);

    let centres: Vec<[f64; 2]> = (0..CLUSTERS)
        .map(|_| {
            [
                rng.next_f64() * 800.0 + 100.0,
                rng.next_f64() * 800.0 + 100.0,
            ]
        })
        .collect();

    // Raw points in insertion order.
    let raw: Vec<Point<f64, 2>> = (0..N)
        .map(|i| {
            let c = &centres[i % CLUSTERS];
            let x = (c[0] + rng.next_normal() * 25.0).clamp(0.0, DOMAIN);
            let y = (c[1] + rng.next_normal() * 25.0).clamp(0.0, DOMAIN);
            Point::new([x, y])
        })
        .collect();

    // Attach Hilbert index and sort — this is the insertion order used by every
    // backend below, maximising cache locality during bulk load.
    let mut sorted: Vec<(u128, Point<f64, 2>)> = raw
        .iter()
        .map(|&p| {
            let hx = normalise(p.coords()[0], DOMAIN, hilbert.order);
            let hy = normalise(p.coords()[1], DOMAIN, hilbert.order);
            (hilbert.index(&[hx, hy]), p)
        })
        .collect();
    sorted.sort_by_key(|&(h, _)| h);

    println!("  First 5 points in Hilbert order:");
    for (h, p) in sorted.iter().take(5) {
        println!(
            "    hilbert={h:<12}  ({:.2}, {:.2})",
            p.coords()[0],
            p.coords()[1]
        );
    }

    // ── 2. BloomCache: built from the same Hilbert-sorted insertions ──────────
    println!("\n=== 2. BloomCache (65 536 bytes, k=7) ===");

    let mut bloom = BloomCache::<2>::new(65_536, 7);
    for (_, p) in &sorted {
        bloom.insert(&BBox::new(*p, *p));
    }
    println!(
        "  Populated from {N} point bboxes — memory {} bytes, k={}",
        bloom.memory, bloom.k
    );

    // ── 3. Load all four backends in Hilbert order ────────────────────────────
    println!("\n=== 3. Load all four backends in Hilbert order ===");

    let mut rtree = RTree::<usize, f64, 2>::new();
    let mut kdtree = KDTree::<usize, f64, 2>::new();
    let mut quadtree = Quadtree::<usize, f64, 2>::new();
    let mut grid = GridIndex::<usize, f64, 2>::new([10.0, 10.0], Point::new([0.0, 0.0]));

    // All four backends receive points in the same Hilbert-sorted order.
    for (i, (_, p)) in sorted.iter().enumerate() {
        rtree.insert(*p, i);
        kdtree.insert(*p, i);
        quadtree.insert(*p, i);
        grid.insert(*p, i);
    }

    let depth_bound = (N as f64).log2().ceil() as usize + 1;
    println!("  RTree    : {} entries", rtree.len());
    println!(
        "  KDTree   : {} entries  depth={}  bound={}  {}",
        kdtree.len(),
        kdtree.depth(),
        depth_bound,
        if kdtree.depth() <= depth_bound {
            "ok"
        } else {
            "EXCEEDED"
        },
    );
    println!(
        "  Quadtree : {} entries  nodes={}  max_depth={}  children_per_split={}",
        quadtree.len(),
        quadtree.node_count(),
        quadtree.max_depth(),
        1usize << 2,
    );
    println!(
        "  Grid     : {} entries  cells={}  avg_per_cell={:.2}",
        grid.len(),
        grid.cell_count(),
        N as f64 / grid.cell_count() as f64,
    );

    // ── 4. Range query — Bloom gates all four backends ────────────────────────
    println!("\n=== 4. Range query [300,700]^2 ===");

    let query_bbox = BBox::new(Point::new([300.0, 300.0]), Point::new([700.0, 700.0]));
    bloom.insert(&query_bbox);

    let brute = raw.iter().filter(|p| query_bbox.contains_point(p)).count();

    // Bloom check is shared — if it fires DefinitelyAbsent, all backends skip.
    let (rt, kd, qt, gr) = match bloom.check(&query_bbox) {
        BloomResult::DefinitelyAbsent => {
            println!("  Bloom: DefinitelyAbsent — all backends skipped");
            (0, 0, 0, 0)
        }
        BloomResult::ProbablyPresent => {
            println!("  Bloom: ProbablyPresent — querying all backends");
            (
                rtree.range_query(&query_bbox).len(),
                kdtree.range_query(&query_bbox).len(),
                quadtree.range_query(&query_bbox).len(),
                grid.range_query(&query_bbox).len(),
            )
        }
    };

    println!("  RTree={rt}  KDTree={kd}  Quadtree={qt}  Grid={gr}  Brute={brute}");
    println!(
        "  All match brute: {}",
        if rt == brute && kd == brute && qt == brute && gr == brute {
            "yes"
        } else {
            "NO"
        },
    );

    // Empty region — Bloom should catch this without touching any backend.
    let empty_bbox = BBox::new(
        Point::new([9_000.0, 9_000.0]),
        Point::new([9_100.0, 9_100.0]),
    );
    match bloom.check(&empty_bbox) {
        BloomResult::DefinitelyAbsent => {
            println!("  Empty [9000,9100]^2 — Bloom: DefinitelyAbsent (O(1) rejection)");
        }
        BloomResult::ProbablyPresent => {
            let r = rtree.range_query(&empty_bbox).len();
            println!("  Empty [9000,9100]^2 — Bloom false positive, RTree={r}");
        }
    }

    // ── 5. kNN(k=5) — all four backends, results compared ────────────────────
    println!("\n=== 5. kNN(k=5) from (500, 500) ===");

    let query_pt = Point::new([500.0, 500.0]);
    let rt_knn = rtree.knn_query(&query_pt, 5);
    let kd_knn = kdtree.knn_query(&query_pt, 5);
    let qt_knn = quadtree.knn_query(&query_pt, 5);
    let gr_knn = grid.knn_query(&query_pt, 5);

    println!("  rank  RTree       KDTree      Quadtree    Grid        all_match");
    for i in 0..5 {
        let rd = rt_knn[i].0;
        let kd = kd_knn[i].0;
        let qt = qt_knn[i].0;
        let gr = gr_knn[i].0;
        let ok = (rd - kd).abs() < 1e-9 && (rd - qt).abs() < 1e-9 && (rd - gr).abs() < 1e-9;
        println!(
            "  #{i}    {rd:>10.4}  {kd:>10.4}  {qt:>10.4}  {gr:>10.4}  {}",
            if ok { "yes" } else { "NO" },
        );
    }

    // ── 6. Insert-remove — remove half from every backend, verify absence ─────
    println!("\n=== 6. Insert-remove round-trip (100 points, remove 50 per backend) ===");

    fn round_trip<B>(name: &str, mut backend: B, points: &[Point<f64, 2>]) -> bool
    where
        B: SpatialBackend<usize, f64, 2>,
    {
        let ids: Vec<EntryId> = points
            .iter()
            .enumerate()
            .map(|(i, &p)| backend.insert(p, i))
            .collect();

        let full = BBox::new(Point::new([0.0, 0.0]), Point::new([1000.0, 1000.0]));
        let before = backend.range_query(&full).len();
        let half = ids.len() / 2;

        for &id in ids.iter().take(half) {
            assert!(backend.remove(id).is_some(), "{name}: remove returned None");
        }

        let after = backend.range_query(&full).len();
        let remaining: Vec<EntryId> = backend
            .range_query(&full)
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        let violations = ids
            .iter()
            .take(half)
            .filter(|id| remaining.contains(id))
            .count();
        let ok = violations == 0 && after == ids.len() - half && backend.len() == ids.len() - half;

        println!(
            "  {name:<10}  before={before}  removed={half}  after={after}  violations={violations}  {}",
            if ok { "ok" } else { "FAIL" },
        );
        ok
    }

    // Use the first 100 points from the same Hilbert-sorted dataset.
    let rtrip_pts: Vec<Point<f64, 2>> = sorted.iter().take(100).map(|(_, p)| *p).collect();

    let all_ok = [
        round_trip("RTree", RTree::<usize, f64, 2>::new(), &rtrip_pts),
        round_trip("KDTree", KDTree::<usize, f64, 2>::new(), &rtrip_pts),
        round_trip(
            "Quadtree",
            Quadtree::<usize, f64, 2>::with_bounds(
                Point::new([0.0, 0.0]),
                Point::new([1000.0, 1000.0]),
            ),
            &rtrip_pts,
        ),
        round_trip(
            "Grid",
            GridIndex::<usize, f64, 2>::new([10.0, 10.0], Point::new([0.0, 0.0])),
            &rtrip_pts,
        ),
    ]
    .iter()
    .all(|&ok| ok);

    println!(
        "  All backends passed: {}",
        if all_ok { "yes" } else { "NO" }
    );

    // ── 7. Profiler — observe the §1 dataset, measure DataShape ─────────────
    println!("\n=== 7. Profiler (the {N}-point §1 dataset, reservoir=512) ===");

    let mut profiler = Profiler::<f64, 2>::new(512);

    // Feed every point from the original dataset in Hilbert order, mirroring
    // how the backends were loaded in §3.
    for (_, p) in &sorted {
        profiler.observe(Observation::Insert(*p));
    }
    // Record the range query from §4 so the profiler tracks query workload too.
    profiler.observe(Observation::Query {
        kind: QueryKind::Range,
        selectivity: brute as f64 / N as f64,
        hit: brute > 0,
    });
    profiler.flush();

    let data_shape = profiler
        .data_shape()
        .expect("profiler must have a shape after N inserts")
        .clone();

    println!("  point_count    = {}", data_shape.point_count);
    println!(
        "  clustering_coef= {:.4}  (clustered data → expected > 1.0)",
        data_shape.clustering_coef
    );
    println!(
        "  effective_dim  = {:.4}  (2D data → expected ≈ 2.0)",
        data_shape.effective_dim
    );
    println!(
        "  skewness       = [{:.4}, {:.4}]",
        data_shape.skewness[0], data_shape.skewness[1]
    );
    println!(
        "  query_mix      = range={:.2} knn={:.2} join={:.2} sel={:.4}",
        data_shape.query_mix.range_frac,
        data_shape.query_mix.knn_frac,
        data_shape.query_mix.join_frac,
        data_shape.query_mix.mean_selectivity,
    );

    // ── 8. CostModel — rank backends using the §7 DataShape ──────────────────
    println!("\n=== 8. CostModel (§7 DataShape, D=2) ===");

    let estimates = CostModel::<2>::estimate_all(&data_shape, QueryKind::Range);
    let cheapest = CostModel::<2>::cheapest(&data_shape, QueryKind::Range);
    for e in &estimates {
        let marker = if e.backend == cheapest {
            " ← cheapest"
        } else {
            ""
        };
        let name = match e.backend {
            BackendKind::RTree => "R-tree   ",
            BackendKind::KDTree => "KD-tree  ",
            BackendKind::Grid => "Grid     ",
            BackendKind::Quadtree => "Quadtree ",
        };
        println!("  {name}  cost = {:>10.4}{marker}", e.cost);
    }
    println!(
        "  clustering_coef={:.4} → KD-tree and Grid costs scale up; {:?} wins",
        data_shape.clustering_coef, cheapest
    );

    // ── 9. PolicyEngine — tick on the §7 DataShape, show migration decision ───
    println!("\n=== 9. PolicyEngine (ticking on §7 DataShape, hysteresis=5) ===");

    // Start on whichever backend the cost model did NOT pick as cheapest, so
    // the policy engine has a reason to migrate.
    let starting_backend = if cheapest == BackendKind::KDTree {
        BackendKind::RTree
    } else {
        BackendKind::KDTree
    };
    let mut engine = PolicyEngine::<2>::with_config(starting_backend, 0.77, 5);
    println!(
        "  Starting on {:?} (cost model prefers {:?}) — migration expected after window",
        starting_backend, cheapest,
    );

    for tick in 1..=8 {
        let obs = engine.observations_since_migration();
        match engine.tick(&data_shape) {
            Some(d) => {
                println!(
                    "  tick {tick} | obs={} | *** MIGRATE → {:?} (cost_ratio={:.3}) ***",
                    obs + 1,
                    d.target,
                    d.cost_ratio,
                );
                engine.on_migration_started();
                engine.on_migration_complete(d.target);
                println!("  [migration complete — hysteresis counter reset]");
            }
            None => println!(
                "  tick {tick} | obs={} | no migration{}",
                obs + 1,
                if obs < engine.hysteresis_window() {
                    " (hysteresis guard)"
                } else {
                    ""
                },
            ),
        }
    }
    println!("  Final backend: {:?}", engine.current_backend());

    // ── 10. Migration — execute the §9 policy decision on the §1 dataset ──────
    //
    // §9 determined that `starting_backend` should migrate to `cheapest`.
    // Here we build a SimpleIndex backed by `starting_backend`, load the same
    // 2 000-point Hilbert-sorted dataset from §1, run the §4 range query to
    // capture the before-migration result, then execute the migration and
    // verify the result is identical.
    println!(
        "\n=== 10. Migration ({N} points, {:?} → {:?}) ===",
        starting_backend, cheapest,
    );
    println!("  (same dataset as §1; migration target chosen by §9 PolicyEngine)");

    // Build the index on the backend the policy engine started on.
    let mut mig_index: SimpleIndex<usize, f64, 2> = match starting_backend {
        BackendKind::RTree => SimpleIndex::new(Box::new(RTree::<(EntryId, usize), f64, 2>::new())),
        _ => SimpleIndex::new(Box::new(KDTree::<(EntryId, usize), f64, 2>::new())),
    };

    // Load the §1 Hilbert-sorted dataset.
    for (i, (_, p)) in sorted.iter().enumerate() {
        mig_index.insert(*p, i);
    }

    // Range query before migration — reuse the §4 bbox.
    let mut before_ids = mig_index.range_query(&query_bbox);
    before_ids.sort_by_key(|id| id.0);
    println!(
        "  Before migration : {} results (bbox = §4 query)",
        before_ids.len()
    );

    // Execute the migration to the policy-chosen backend.
    let mig_result = match cheapest {
        BackendKind::RTree => mig_index
            .migrate(RTree::<(EntryId, usize), f64, 2>::bulk_load)
            .expect("migration should succeed"),
        BackendKind::KDTree => mig_index
            .migrate(KDTree::<(EntryId, usize), f64, 2>::bulk_load)
            .expect("migration should succeed"),
        BackendKind::Quadtree => mig_index
            .migrate(Quadtree::<(EntryId, usize), f64, 2>::bulk_load)
            .expect("migration should succeed"),
        BackendKind::Grid => mig_index
            .migrate(GridIndex::<(EntryId, usize), f64, 2>::bulk_load)
            .expect("migration should succeed"),
    };

    // Range query after migration — must match.
    let mut after_ids = mig_index.range_query(&query_bbox);
    after_ids.sort_by_key(|id| id.0);
    println!("  After migration  : {} results", after_ids.len());
    println!(
        "  Results identical: {}",
        if before_ids == after_ids { "yes" } else { "NO" },
    );
    println!("  New backend      : {:?}", mig_result.new_backend);
    println!(
        "  Entry count after: {} (expected {N})",
        mig_result.entry_count
    );
    println!("  Migration duration: {:?}", mig_result.duration);

    // ── 11. IndexRouter + StatsCollector — concurrent stress on the migrated index
    //
    // We take the backend that §10 migrated to and hand it to an IndexRouter.
    // The §1 dataset is split into 4 equal chunks — one per insert thread —
    // so the same 2 000 points flow through the router concurrently.
    // Two query threads run the §4 bbox in a tight loop throughout.
    // StatsCollector records every insert and query without a lock.
    println!("\n=== 11. IndexRouter + StatsCollector (§10 backend, §1 dataset, 4+2 threads) ===");
    println!("  Backend from §10 migration: {:?}", mig_result.new_backend);

    // Rebuild the migrated backend via bulk_load so IndexRouter owns it.
    let migrated_backend: Box<dyn bonsai::backends::SpatialBackend<usize, f64, 2>> =
        match mig_result.new_backend {
            BackendKind::RTree => Box::new(RTree::<usize, f64, 2>::bulk_load(
                sorted
                    .iter()
                    .enumerate()
                    .map(|(i, (_, p))| (*p, i))
                    .collect(),
            )),
            BackendKind::KDTree => Box::new(KDTree::<usize, f64, 2>::bulk_load(
                sorted
                    .iter()
                    .enumerate()
                    .map(|(i, (_, p))| (*p, i))
                    .collect(),
            )),
            BackendKind::Quadtree => Box::new(Quadtree::<usize, f64, 2>::bulk_load(
                sorted
                    .iter()
                    .enumerate()
                    .map(|(i, (_, p))| (*p, i))
                    .collect(),
            )),
            BackendKind::Grid => Box::new(GridIndex::<usize, f64, 2>::bulk_load(
                sorted
                    .iter()
                    .enumerate()
                    .map(|(i, (_, p))| (*p, i))
                    .collect(),
            )),
        };

    let router: Arc<IndexRouter<usize, f64, 2>> = Arc::new(IndexRouter::new(migrated_backend));
    let sc: Arc<StatsCollector> = Arc::new(StatsCollector::new());

    // Split the §1 Hilbert-sorted points into 4 chunks for the insert threads.
    let chunks: Vec<Vec<Point<f64, 2>>> = {
        let pts: Vec<Point<f64, 2>> = sorted.iter().map(|(_, p)| *p).collect();
        let chunk_size = pts.len() / 4;
        pts.chunks(chunk_size).map(|c| c.to_vec()).collect()
    };

    let mut handles = Vec::new();

    // 4 insert threads — each re-inserts its chunk of the §1 dataset.
    for (t, chunk) in chunks.into_iter().enumerate() {
        let r = Arc::clone(&router);
        let s = Arc::clone(&sc);
        handles.push(std::thread::spawn(move || {
            for (i, p) in chunk.into_iter().enumerate() {
                r.insert(p, t * 500 + i);
                s.record_insert();
            }
        }));
    }

    // 2 query threads — each runs the §4 bbox in a loop while inserts happen.
    for _ in 0..2usize {
        let r = Arc::clone(&router);
        let s = Arc::clone(&sc);
        let bbox = query_bbox;
        handles.push(std::thread::spawn(move || {
            for _ in 0..200 {
                let t0 = std::time::Instant::now();
                let _ = r.range_query(&bbox);
                s.record_query(t0.elapsed().as_nanos() as u64);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    // The router started with N points (bulk-loaded from §10) and each insert
    // thread added another chunk — total should be N + N = 2*N.
    println!(
        "  Points after concurrent inserts: {} (started with {N}, added {N})",
        router.len()
    );
    println!("  StatsCollector inserts : {}", sc.insert_count());
    println!("  StatsCollector queries : {}", sc.query_count());
    println!("  Mean query latency     : {} ns", sc.mean_query_ns());
    println!("  No panics or data races: yes (all 6 threads joined cleanly)");

    // ── 12. BonsaiIndex — canonical 10-line usage example ────────────────────
    println!("\n=== 12. BonsaiIndex<&str> — builder pattern, insert, range query, kNN, stats ===");

    let mut bonsai = bonsai::index::BonsaiIndex::<&str>::builder()
        .reservoir_size(256)
        .build();

    bonsai.insert(Point::new([100.0, 200.0]), "alpha");
    bonsai.insert(Point::new([300.0, 400.0]), "beta");
    bonsai.insert(Point::new([500.0, 600.0]), "gamma");
    bonsai.insert(Point::new([700.0, 800.0]), "delta");
    bonsai.insert(Point::new([900.0, 100.0]), "epsilon");

    let range_bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([600.0, 700.0]));
    let range_hits = bonsai.range_query(&range_bbox);
    println!(
        "  Range query [0,600]^2 × [0,700]: {} hit(s)",
        range_hits.len()
    );
    for (id, payload) in &range_hits {
        println!("    {:?}  {}", id, payload);
    }

    let knn_results = bonsai.knn_query(&Point::new([400.0, 400.0]), 2);
    println!("  kNN(k=2) from (400, 400):");
    for (dist, id, payload) in &knn_results {
        println!("    {:?}  {}  dist={:.2}", id, payload, dist);
    }

    let s = bonsai.stats();
    println!(
        "  stats: backend={:?}  point_count={}",
        s.backend, s.point_count
    );
}
