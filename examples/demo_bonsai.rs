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
//!
//! Run with:
//!   cargo run --example demo_bonsai

use bonsai::backends::{GridIndex, KDTree, Quadtree, RTree, SpatialBackend};
use bonsai::bloom::{BloomCache, BloomResult};
use bonsai::hilbert::HilbertCurve;
use bonsai::profiler::{Observation, Profiler, QueryKind};
use bonsai::types::{BBox, EntryId, Point};
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

    // ── 7. Profiler demo — uniform vs clustered data ──────────────────────────
    println!("\n=== 7. Profiler demo (50k uniform vs 50k clustered points, D=2) ===");
    println!("  (reservoir capped at 4096; stats computed on reservoir sample)");

    const PROF_N: usize = 50_000;
    const PROF_DOMAIN: f64 = 1_000.0;
    const PROF_CLUSTERS: usize = 10;
    const PROF_SIGMA: f64 = 5.0; // σ << domain → tight clusters

    // ── 7a. Uniform data ──────────────────────────────────────────────────────
    {
        let mut profiler = Profiler::<f64, 2>::new(4096);
        let mut rng = Lcg::new(0xdeadbeef);

        for _ in 0..PROF_N {
            let x = rng.next_f64() * PROF_DOMAIN;
            let y = rng.next_f64() * PROF_DOMAIN;
            profiler.observe(Observation::Insert(Point::new([x, y])));
        }
        // Also record some range queries.
        for _ in 0..200 {
            profiler.observe(Observation::Query {
                kind: QueryKind::Range,
                selectivity: 0.01,
                hit: true,
            });
        }
        profiler.flush();

        if let Some(shape) = profiler.data_shape() {
            println!("  Uniform data ({PROF_N} points):");
            println!("    point_count    = {}", shape.point_count);
            println!(
                "    clustering_coef= {:.4}  (expected ≈ 1.0)",
                shape.clustering_coef
            );
            println!(
                "    effective_dim  = {:.4}  (expected ≈ 2.0)",
                shape.effective_dim
            );
            println!(
                "    skewness       = [{:.4}, {:.4}]",
                shape.skewness[0], shape.skewness[1]
            );
            println!(
                "    query_mix      = range={:.2} knn={:.2} join={:.2} sel={:.4}",
                shape.query_mix.range_frac,
                shape.query_mix.knn_frac,
                shape.query_mix.join_frac,
                shape.query_mix.mean_selectivity,
            );
        }
    }

    // ── 7b. Clustered data ────────────────────────────────────────────────────
    {
        let mut profiler = Profiler::<f64, 2>::new(4096);
        let mut rng = Lcg::new(0xcafebabe);

        let centres: Vec<[f64; 2]> = (0..PROF_CLUSTERS)
            .map(|_| {
                [
                    rng.next_f64() * 800.0 + 100.0,
                    rng.next_f64() * 800.0 + 100.0,
                ]
            })
            .collect();

        for i in 0..PROF_N {
            let c = &centres[i % PROF_CLUSTERS];
            let x = (c[0] + rng.next_normal() * PROF_SIGMA).clamp(0.0, PROF_DOMAIN);
            let y = (c[1] + rng.next_normal() * PROF_SIGMA).clamp(0.0, PROF_DOMAIN);
            profiler.observe(Observation::Insert(Point::new([x, y])));
        }
        profiler.flush();

        if let Some(shape) = profiler.data_shape() {
            println!(
                "  Clustered data ({PROF_N} points, {PROF_CLUSTERS} clusters, σ={PROF_SIGMA}):"
            );
            println!("    point_count    = {}", shape.point_count);
            println!(
                "    clustering_coef= {:.4}  (expected > 1.0)",
                shape.clustering_coef
            );
            println!(
                "    effective_dim  = {:.4}  (expected ≈ 2.0)",
                shape.effective_dim
            );
            println!(
                "    skewness       = [{:.4}, {:.4}]",
                shape.skewness[0], shape.skewness[1]
            );
        }
    }

    // ── 7c. Linear data (1D manifold in 2D space) ─────────────────────────────
    {
        // Use a smaller reservoir for the linear demo to keep it fast.
        let mut profiler = Profiler::<f64, 2>::new(1024);
        let mut rng = Lcg::new(0x12345678);

        for i in 0..PROF_N {
            let t = (i as f64 / PROF_N as f64) * PROF_DOMAIN;
            let noise = rng.next_f64() * 0.1;
            profiler.observe(Observation::Insert(Point::new([
                t + noise,
                t * 0.5 + noise,
            ])));
        }
        profiler.flush();

        if let Some(shape) = profiler.data_shape() {
            println!("  Linear data ({PROF_N} points on a line in 2D, reservoir=1024):");
            println!("    point_count    = {}", shape.point_count);
            println!("    clustering_coef= {:.4}", shape.clustering_coef);
            println!(
                "    effective_dim  = {:.4}  (expected ≈ 1.0)",
                shape.effective_dim
            );
        }
    }
}
