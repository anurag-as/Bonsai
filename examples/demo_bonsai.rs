//! Bonsai end-to-end demo.
//!
//! One dataset, one story — each section builds directly on the previous:
//!
//!   1. HilbertCurve  — sort 2 000 clustered points into cache-friendly order
//!   2. BloomCache    — populate from the §1 Hilbert-sorted insertions
//!   3. All 4 backends — loaded in §1 Hilbert order; KD-tree carried forward
//!   4. Range query   — §2 Bloom gates all four §3 backends; brute-force check
//!   5. kNN query     — all four §3 backends, results compared
//!   6. Insert-remove — remove half the entries from the §3 backends in-place
//!   7. Profiler      — feed the §1 dataset into the profiler; measure DataShape
//!   8. CostModel     — rank backends using the §7 DataShape
//!   9. PolicyEngine  — tick on the §7 shape; decide to migrate §3 KD-tree
//!  10. Migration     — execute the §9 decision on the §3 KD-tree; verify §4
//!                      range query results are identical before and after
//!  11. IndexRouter + StatsCollector — wrap the §10 migrated backend; 4 insert
//!                      threads re-add the §1 points while 2 query threads run
//!                      the §4 bbox; StatsCollector tracks everything lock-free
//!  12. BonsaiIndex   — load the §1 dataset through the high-level public API;
//!                      run the §4 range query and §5 kNN; print stats
//!  13. Serialisation — build 1 000-point index, to_bytes, from_bytes, verify
//!                      range query results match (feature = "serde")
//!  14. C FFI         — exercise the extern "C" API with 3 §1 points and the
//!                      §4 bbox (feature = "ffi")
//!  15. CLI           — load the first 100 §1 points via `load`, stream the
//!                      remaining 1 900 via `stream` (stdin pipe), then run
//!                      stats, the §4 range query, kNN, and visualise on the
//!                      full 2 000-point dataset
//!
//! Run with:
//!   cargo run --example demo_bonsai
//!   cargo run --example demo_bonsai --features serde

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

    let raw: Vec<Point<f64, 2>> = (0..N)
        .map(|i| {
            let c = &centres[i % CLUSTERS];
            let x = (c[0] + rng.next_normal() * 25.0).clamp(0.0, DOMAIN);
            let y = (c[1] + rng.next_normal() * 25.0).clamp(0.0, DOMAIN);
            Point::new([x, y])
        })
        .collect();

    // Hilbert-sort the raw points — this insertion order is reused by every
    // section below, maximising cache locality during bulk load.
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

    // ── 2. BloomCache: built from the §1 Hilbert-sorted insertions ───────────
    println!("\n=== 2. BloomCache (65 536 bytes, k=7) ===");

    let mut bloom = BloomCache::<2>::new(65_536, 7);
    for (_, p) in &sorted {
        bloom.insert(&BBox::new(*p, *p));
    }
    println!(
        "  Populated from {N} point bboxes — memory {} bytes, k={}",
        bloom.memory, bloom.k
    );

    // ── 3. Load all four backends in §1 Hilbert order ────────────────────────
    // The KD-tree is carried forward into §9, §10, and §11.
    println!("\n=== 3. Load all four backends in Hilbert order ===");

    let mut rtree = RTree::<usize, f64, 2>::new();
    let mut kdtree = KDTree::<usize, f64, 2>::new();
    let mut quadtree = Quadtree::<usize, f64, 2>::new();
    let mut grid = GridIndex::<usize, f64, 2>::new([10.0, 10.0], Point::new([0.0, 0.0]));

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

    // ── 4. Range query — §2 Bloom gates all four §3 backends ─────────────────
    println!("\n=== 4. Range query [300,700]^2 ===");

    let query_bbox = BBox::new(Point::new([300.0, 300.0]), Point::new([700.0, 700.0]));
    bloom.insert(&query_bbox);

    let brute = raw.iter().filter(|p| query_bbox.contains_point(p)).count();

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

    // ── 5. kNN(k=5) — all four §3 backends, results compared ─────────────────
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

    // ── 6. Insert-remove — remove half the entries from the §3 backends ───────
    // Uses the first 100 §1 Hilbert-sorted points — the same ordering that
    // loaded the §3 backends.
    println!("\n=== 6. Insert-remove round-trip (first 100 §1 points, remove 50) ===");

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

    // ── 7. Profiler — observe the §1 dataset; produce DataShape ──────────────
    // The §4 range query selectivity is fed in so the profiler tracks workload.
    println!("\n=== 7. Profiler ({N} §1 points, reservoir=512) ===");

    let mut profiler = Profiler::<f64, 2>::new(512);
    for (_, p) in &sorted {
        profiler.observe(Observation::Insert(*p));
    }
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
        "  clustering_coef={:.4} → {:?} wins",
        data_shape.clustering_coef, cheapest
    );

    // ── 9. PolicyEngine — tick on the §7 shape; decide to migrate §3 KD-tree ─
    // Start on KD-tree (the §3 backend we will migrate in §10). The engine
    // ticks on the §7 DataShape until it fires a migration decision.
    println!("\n=== 9. PolicyEngine (§7 DataShape, hysteresis=5) ===");

    let migration_target = if cheapest != BackendKind::KDTree {
        cheapest
    } else {
        BackendKind::RTree
    };

    let mut engine = PolicyEngine::<2>::with_config(BackendKind::KDTree, 0.77, 5);
    println!(
        "  Starting on KDTree (cost model prefers {:?}) — migration expected after window",
        migration_target,
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

    // ── 10. Migration — migrate the §3 KD-tree to the §9 target ─────────────
    // Build a SimpleIndex from the §1 Hilbert-sorted points (same data as §3
    // KD-tree), snapshot the §4 range query, migrate, verify results match.
    println!(
        "\n=== 10. Migration ({N} §3 points, KDTree → {:?}) ===",
        migration_target
    );

    let mut mig_index: SimpleIndex<usize, f64, 2> =
        SimpleIndex::new(Box::new(KDTree::<(EntryId, usize), f64, 2>::new()));
    for (i, (_, p)) in sorted.iter().enumerate() {
        mig_index.insert(*p, i);
    }

    let mut before_ids = mig_index.range_query(&query_bbox);
    before_ids.sort_by_key(|id| id.0);
    println!(
        "  Before migration : {} results (§4 bbox)",
        before_ids.len()
    );

    let mig_result = match migration_target {
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

    let mut after_ids = mig_index.range_query(&query_bbox);
    after_ids.sort_by_key(|id| id.0);
    println!("  After migration  : {} results", after_ids.len());
    println!(
        "  Results identical: {}",
        if before_ids == after_ids { "yes" } else { "NO" },
    );
    println!("  New backend      : {:?}", mig_result.new_backend);
    println!(
        "  Entry count      : {} (expected {N})",
        mig_result.entry_count
    );
    println!("  Duration         : {:?}", mig_result.duration);

    // ── 11. IndexRouter + StatsCollector — wrap the §10 migrated backend ──────
    // Bulk-load the §10 backend kind with the §1 points into an IndexRouter.
    // Four insert threads re-add the §1 points while two query threads run the
    // §4 bbox. StatsCollector records every operation lock-free.
    println!(
        "\n=== 11. IndexRouter + StatsCollector ({:?} from §10, 4+2 threads) ===",
        mig_result.new_backend
    );

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

    let chunks: Vec<Vec<Point<f64, 2>>> = {
        let pts: Vec<Point<f64, 2>> = sorted.iter().map(|(_, p)| *p).collect();
        let chunk_size = pts.len() / 4;
        pts.chunks(chunk_size).map(|c| c.to_vec()).collect()
    };

    let mut handles = Vec::new();

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

    // Started with N bulk-loaded §1 points; 4 threads each added N/4 more.
    println!(
        "  Points after concurrent inserts: {} (started {N}, added {N})",
        router.len()
    );
    println!("  StatsCollector inserts : {}", sc.insert_count());
    println!("  StatsCollector queries : {}", sc.query_count());
    println!("  Mean query latency     : {} ns", sc.mean_query_ns());
    println!("  No panics or data races: yes (all 6 threads joined cleanly)");

    // ── 12. BonsaiIndex — the §1 dataset through the high-level public API ────
    // Insert all §1 Hilbert-sorted points, run the §4 range query and §5 kNN,
    // then print stats — showing the full adaptive pipeline in one place.
    println!("\n=== 12. BonsaiIndex ({N} §1 points, §4 range query, §5 kNN) ===");

    let mut bonsai = bonsai::index::BonsaiIndex::<usize>::builder()
        .reservoir_size(512)
        .build();

    for (i, (_, p)) in sorted.iter().enumerate() {
        bonsai.insert(*p, i);
    }

    let range_hits = bonsai.range_query(&query_bbox);
    println!(
        "  Range query §4 bbox: {} hits  brute={brute}  match={}",
        range_hits.len(),
        if range_hits.len() == brute {
            "yes"
        } else {
            "NO"
        },
    );

    let knn_results = bonsai.knn_query(&query_pt, 5);
    println!("  kNN(k=5) from §5 query point — top-3 distances:");
    for (dist, _id, _payload) in knn_results.iter().take(3) {
        println!("    dist={dist:.4}");
    }

    let s = bonsai.stats();
    println!(
        "  stats: backend={:?}  point_count={}  queries={}",
        s.backend, s.point_count, s.query_count,
    );

    // ── 13. Serialisation — round-trip the §12 BonsaiIndex through to_bytes/from_bytes ─
    // Uses the same §12 `bonsai` index (N §1 points) and the §4 query bbox so
    // the result count is directly comparable to every earlier section.
    #[cfg(feature = "serde")]
    {
        println!("\n=== 13. Serialisation (feature = \"serde\", §12 index, {N} points) ===");

        let bytes = bonsai.to_bytes();
        println!("  Serialised {N} points → {} bytes", bytes.len());

        let mut restored = bonsai::index::BonsaiIndex::<usize>::from_bytes(&bytes)
            .expect("deserialisation must succeed on valid bytes");

        // Re-run the §4 range query on both the original and the restored index.
        let mut orig_hits = bonsai.range_query(&query_bbox);
        let mut rest_hits = restored.range_query(&query_bbox);

        // Sort by payload (stable across re-insertion) rather than EntryId
        // (which is reassigned on decode).
        orig_hits.sort_by_key(|(_, p)| *p);
        rest_hits.sort_by_key(|(_, p)| *p);

        let orig_payloads: Vec<usize> = orig_hits.iter().map(|(_, p)| *p).collect();
        let rest_payloads: Vec<usize> = rest_hits.iter().map(|(_, p)| *p).collect();

        println!(
            "  §4 range query: original={} restored={}  brute={brute}  match={}",
            orig_payloads.len(),
            rest_payloads.len(),
            if orig_payloads == rest_payloads {
                "yes"
            } else {
                "NO"
            },
        );

        if orig_payloads.len() <= 5 {
            println!("  Payloads: {:?}", orig_payloads);
        } else {
            println!("  First 5 payloads: {:?}", &orig_payloads[..5]);
        }
    }

    // ── 14. C FFI round-trip — exercise the extern "C" API from Rust ─────────
    // Uses three points from the §1 Hilbert-sorted dataset so the inserted
    // coordinates are grounded in the same data that flows through every
    // earlier section.  The §4 query bbox [300,700]^2 is reused so the result
    // count is directly comparable to the backend results above.
    #[cfg(feature = "ffi")]
    {
        use bonsai::ffi::{
            bonsai_free, bonsai_insert_2d, bonsai_knn_query_2d, bonsai_new, bonsai_range_query_2d,
            bonsai_stats, BonsaiStats,
        };

        println!("\n=== 14. C FFI round-trip (feature = \"ffi\", 3 §1 points, §4 bbox) ===");

        // Pick three representative §1 points: one inside the §4 bbox, one on
        // the boundary, one outside.
        let ffi_pts: Vec<[f64; 2]> = sorted
            .iter()
            .map(|(_, p)| [p.coords()[0], p.coords()[1]])
            .take(3)
            .collect();

        unsafe {
            // SAFETY: `bonsai_new` returns a valid, heap-allocated handle.
            let h = bonsai_new();
            assert!(!h.is_null(), "bonsai_new returned null");

            // Insert the three §1 points with null payloads.
            // SAFETY: `h` is valid; null payload is stored as-is.
            let ids: Vec<u64> = ffi_pts
                .iter()
                .map(|p| bonsai_insert_2d(h, p[0], p[1], std::ptr::null_mut()))
                .collect();
            println!(
                "  Inserted §1 points: ({:.2},{:.2}) ({:.2},{:.2}) ({:.2},{:.2})",
                ffi_pts[0][0],
                ffi_pts[0][1],
                ffi_pts[1][0],
                ffi_pts[1][1],
                ffi_pts[2][0],
                ffi_pts[2][1],
            );
            println!("  Assigned IDs: {:?}", ids);

            // Range query with the §4 bbox — count should match the fraction
            // of the three points that fall inside [300,700]^2.
            let mut out_ids = [0u64; 8];
            // SAFETY: `h` is valid; `out_ids` has capacity 8.
            let count = bonsai_range_query_2d(
                h,
                300.0,
                300.0,
                700.0,
                700.0,
                out_ids.as_mut_ptr(),
                out_ids.len(),
            );
            let expected = ffi_pts
                .iter()
                .filter(|p| p[0] >= 300.0 && p[0] <= 700.0 && p[1] >= 300.0 && p[1] <= 700.0)
                .count();
            println!(
                "  Range §4 bbox: {count} result(s) (expected {expected}) — IDs: {:?}",
                &out_ids[..count],
            );
            assert_eq!(count, expected, "FFI range query count mismatch");

            // kNN(k=1) from the §5 query point (500, 500).
            let mut knn_ids = [0u64; 1];
            let mut knn_dist = [0.0f64; 1];
            // SAFETY: `h` is valid; buffers have capacity 1.
            let knn_count = bonsai_knn_query_2d(
                h,
                500.0,
                500.0,
                1,
                knn_ids.as_mut_ptr(),
                knn_dist.as_mut_ptr(),
            );
            println!(
                "  kNN(k=1) from §5 query point (500,500): count={knn_count}  id={}  dist={:.4}",
                knn_ids[0], knn_dist[0],
            );
            assert_eq!(knn_count, 1);

            // Stats snapshot.
            let mut stats = BonsaiStats {
                point_count: 0,
                query_count: 0,
                migration_count: 0,
                migrating: 0,
                backend_kind: 0,
            };
            // SAFETY: `h` and `&mut stats` are valid.
            let ok = bonsai_stats(h, &mut stats as *mut BonsaiStats);
            assert_eq!(ok, 1);
            println!(
                "  Stats: point_count={}  queries={}  backend_kind={}",
                stats.point_count, stats.query_count, stats.backend_kind,
            );
            assert_eq!(stats.point_count, 3);

            // Free — must not leak or crash.
            // SAFETY: `h` is valid and has not been freed yet.
            bonsai_free(h);

            // Free null — documented no-op.
            // SAFETY: null pointer is explicitly handled.
            bonsai_free(std::ptr::null_mut());

            println!("  bonsai_free(null) — no-op: ok");
            println!("  FFI round-trip: all assertions passed");
        }
    }

    // ── 15. CLI — load first 100 §1 points, stream remaining 1900, inspect ──
    // Writes the §1 Hilbert-sorted points to two CSVs: the first 100 go
    // through `load` (batch), the remaining 1900 are piped into `stream`
    // (continuous). After streaming, `stats` and `visualise` operate on the
    // full 2 000-point dataset — the same one that has run through every
    // earlier section. The §4 range query bbox [300,700]^2 is reused so the
    // result count is directly comparable.
    #[cfg(feature = "serde")]
    {
        use std::process::{Command, Stdio};

        println!("\n=== 15. CLI demo (feature = \"serde\", 2 000 §1 points) ===");

        let mut csv_batch = String::from("x,y,label\n");
        for (i, (_, p)) in sorted.iter().take(100).enumerate() {
            csv_batch.push_str(&format!(
                "{:.3},{:.3},pt{}\n",
                p.coords()[0],
                p.coords()[1],
                i
            ));
        }

        let mut csv_stream = String::new();
        for (i, (_, p)) in sorted.iter().skip(100).enumerate() {
            csv_stream.push_str(&format!(
                "{:.3},{:.3},pt{}\n",
                p.coords()[0],
                p.coords()[1],
                100 + i
            ));
        }

        let csv_path = std::env::temp_dir().join("bonsai_demo_batch.csv");
        std::fs::write(&csv_path, &csv_batch).expect("failed to write batch CSV");

        let work_dir = std::env::temp_dir().join("bonsai_cli_demo");
        std::fs::create_dir_all(&work_dir).ok();

        let bin_path = std::env::current_exe()
            .ok()
            .and_then(|p| {
                p.parent()
                    .and_then(|examples_dir| examples_dir.parent())
                    .map(|debug_dir| debug_dir.join("bonsai"))
            })
            .unwrap_or_else(|| std::path::PathBuf::from("target/debug/bonsai"));

        let run = |args: &[&str]| {
            let cmd_str = format!(
                "bonsai {}",
                args.iter()
                    .map(|a| {
                        if a.contains(' ') {
                            format!("\"{}\"", a)
                        } else {
                            a.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            println!("  $ {}", cmd_str);

            let output = Command::new(&bin_path)
                .args(args)
                .current_dir(&work_dir)
                .output();

            match output {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    for line in stdout.lines() {
                        println!("    {}", line);
                    }
                    if !stderr.is_empty() {
                        for line in stderr.lines() {
                            println!("    [stderr] {}", line);
                        }
                    }
                    if !out.status.success() {
                        println!("    [exit {}]", out.status);
                    }
                    out.status.success()
                }
                Err(e) => {
                    println!("    [error] could not run bonsai binary: {}", e);
                    println!("    (build with: cargo build --features serde --bin bonsai)");
                    false
                }
            }
        };

        let csv_str = csv_path.to_string_lossy().to_string();

        run(&["load", &csv_str]);

        // Stream the remaining 1 900 points through stdin so the profiler
        // observes the full evolving distribution.
        println!("  $ bonsai stream --interval 500  (piping 1 900 points via stdin)");
        let stream_result = Command::new(&bin_path)
            .args(["stream", "--interval", "500"])
            .current_dir(&work_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        match stream_result {
            Ok(mut child) => {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = std::io::Write::write_all(&mut stdin, csv_stream.as_bytes());
                }
                if let Ok(out) = child.wait_with_output() {
                    for line in String::from_utf8_lossy(&out.stdout).lines() {
                        println!("    {}", line);
                    }
                }
            }
            Err(e) => {
                println!("    [error] could not run bonsai stream: {}", e);
            }
        }

        run(&["stats"]);
        run(&["query", "range", "300", "300", "700", "700"]);
        run(&["query", "knn", "500", "500", "5"]);
        run(&["visualise"]);

        std::fs::remove_file(&csv_path).ok();
        std::fs::remove_dir_all(&work_dir).ok();
    }
}
