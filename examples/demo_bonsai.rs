//! Bonsai end-to-end demo.
//!
//! Exercises every module that has been built so far:
//!   - HilbertCurve  — sort points for cache-friendly insertion
//!   - BloomCache    — O(1) negative-result fast path
//!   - RTree         — R*-tree range and kNN queries
//!   - KDTree        — KD-tree range and kNN queries, depth bound
//!   - Quadtree      — 2D quadtree and 3D octree
//!
//! Run with:
//!   cargo run --example demo_bonsai

use bonsai::backends::kdtree::KDTree;
use bonsai::backends::quadtree::Quadtree;
use bonsai::backends::rtree::RTree;
use bonsai::backends::SpatialBackend;
use bonsai::bloom::{BloomCache, BloomResult};
use bonsai::hilbert::HilbertCurve;
use bonsai::types::{BBox, Point};

// ── Minimal deterministic RNG ─────────────────────────────────────────────────

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

    // ── Hilbert curve ─────────────────────────────────────────────────────────
    println!("=== Hilbert curve (D=2, order=10) ===");
    let hilbert = HilbertCurve::<2>::new(10);
    let mut rng = Lcg::new(42);

    let centres: Vec<[f64; 2]> = (0..CLUSTERS)
        .map(|_| [rng.next_f64() * 800.0 + 100.0, rng.next_f64() * 800.0 + 100.0])
        .collect();

    let raw: Vec<Point<f64, 2>> = (0..N)
        .map(|i| {
            let c = &centres[i % CLUSTERS];
            let x = (c[0] + rng.next_normal() * 25.0).clamp(0.0, DOMAIN);
            let y = (c[1] + rng.next_normal() * 25.0).clamp(0.0, DOMAIN);
            Point::new([x, y])
        })
        .collect();

    let mut indexed: Vec<(u128, Point<f64, 2>)> = raw
        .iter()
        .map(|&p| {
            let hx = normalise(p.coords()[0], DOMAIN, hilbert.order);
            let hy = normalise(p.coords()[1], DOMAIN, hilbert.order);
            (hilbert.index(&[hx, hy]), p)
        })
        .collect();
    indexed.sort_by_key(|&(h, _)| h);

    println!("First 5 points in Hilbert order:");
    for (h, p) in indexed.iter().take(5) {
        println!("  hilbert={:<12}  ({:.2}, {:.2})", h, p.coords()[0], p.coords()[1]);
    }

    // ── R-tree and KD-tree ────────────────────────────────────────────────────
    println!("\n=== R-tree and KD-tree ({N} points) ===");
    let mut rtree = RTree::<usize, f64, 2>::new();
    let mut kdtree = KDTree::<usize, f64, 2>::new();
    let mut bloom = BloomCache::<2>::new(65_536, 7);

    for (i, (_, p)) in indexed.iter().enumerate() {
        rtree.insert(*p, i);
        kdtree.insert(*p, i);
        bloom.insert(&BBox::new(*p, *p));
    }

    let depth_bound = (N as f64).log2().ceil() as usize + 1;
    println!("R-tree  entries : {}", rtree.len());
    println!(
        "KD-tree entries : {}  depth={}  bound={}  {}",
        kdtree.len(),
        kdtree.depth(),
        depth_bound,
        if kdtree.depth() <= depth_bound { "ok" } else { "EXCEEDED" }
    );
    println!("Bloom   memory  : {} bytes, k={}", bloom.memory, bloom.k);

    // ── Range query with Bloom fast-path ──────────────────────────────────────
    println!("\n=== Range query [300,700]^2 ===");
    let query_bbox = BBox::new(Point::new([300.0, 300.0]), Point::new([700.0, 700.0]));
    bloom.insert(&query_bbox);

    let brute_count = raw.iter().filter(|p| query_bbox.contains_point(p)).count();

    let (rt_count, kd_count) = match bloom.check(&query_bbox) {
        BloomResult::DefinitelyAbsent => {
            println!("Bloom: DefinitelyAbsent — trees skipped");
            (0, 0)
        }
        BloomResult::ProbablyPresent => {
            let r = rtree.range_query(&query_bbox).len();
            let k = kdtree.range_query(&query_bbox).len();
            (r, k)
        }
    };

    println!("R-tree  : {rt_count}");
    println!("KD-tree : {kd_count}");
    println!("Brute   : {brute_count}");
    println!(
        "Match   : {}",
        if rt_count == brute_count && kd_count == brute_count { "yes" } else { "NO" }
    );

    // ── Bloom catches an empty-region query ───────────────────────────────────
    println!("\n=== Empty-region query [9000,9100]^2 ===");
    let empty_bbox = BBox::new(Point::new([9_000.0, 9_000.0]), Point::new([9_100.0, 9_100.0]));
    match bloom.check(&empty_bbox) {
        BloomResult::DefinitelyAbsent => {
            println!("Bloom: DefinitelyAbsent — trees skipped (O(1) rejection)");
        }
        BloomResult::ProbablyPresent => {
            let r = rtree.range_query(&empty_bbox).len();
            let k = kdtree.range_query(&empty_bbox).len();
            println!("Bloom: ProbablyPresent (false positive) — R-tree={r}, KD-tree={k}");
        }
    }

    // ── kNN query ─────────────────────────────────────────────────────────────
    println!("\n=== kNN(k=5) from (500, 500) ===");
    let query_pt = Point::new([500.0, 500.0]);
    let rtree_knn = rtree.knn_query(&query_pt, 5);
    let kdtree_knn = kdtree.knn_query(&query_pt, 5);

    println!("  rank  R-tree dist    KD-tree dist   match");
    for i in 0..5 {
        let rd = rtree_knn[i].0;
        let kd = kdtree_knn[i].0;
        println!(
            "  #{i}    {rd:>10.4}    {kd:>10.4}     {}",
            if (rd - kd).abs() < 1e-9 { "yes" } else { "NO" }
        );
    }

    // ── Quadtree D=2 ─────────────────────────────────────────────────────────
    println!("\n=== Quadtree D=2 (200 points) ===");
    let mut qt2 = Quadtree::<usize, f64, 2>::new();
    let mut rng2 = Lcg::new(77);
    for i in 0..200usize {
        qt2.insert(Point::new([rng2.next_f64() * 1000.0, rng2.next_f64() * 1000.0]), i);
    }
    let qt2_range = qt2
        .range_query(&BBox::new(Point::new([200.0, 200.0]), Point::new([800.0, 800.0])))
        .len();
    println!("  entries    : {}", qt2.len());
    println!("  node count : {}", qt2.node_count());
    println!("  max depth  : {}", qt2.max_depth());
    println!("  children per split node : {} (2^2)", 1usize << 2);
    println!("  range query [200,800]^2 : {qt2_range} results");

    // ── Octree D=3 ────────────────────────────────────────────────────────────
    println!("\n=== Octree D=3 (200 points) ===");
    let mut qt3 = Quadtree::<usize, f64, 3>::new();
    let mut rng3 = Lcg::new(88);
    for i in 0..200usize {
        qt3.insert(
            Point::new([
                rng3.next_f64() * 1000.0,
                rng3.next_f64() * 1000.0,
                rng3.next_f64() * 1000.0,
            ]),
            i,
        );
    }
    let qt3_range = qt3
        .range_query(&BBox::new(
            Point::new([200.0, 200.0, 200.0]),
            Point::new([800.0, 800.0, 800.0]),
        ))
        .len();
    println!("  entries    : {}", qt3.len());
    println!("  node count : {}", qt3.node_count());
    println!("  max depth  : {}", qt3.max_depth());
    println!("  children per split node : {} (2^3)", 1usize << 3);
    println!("  range query [200,800]^3 : {qt3_range} results");

    // ── Cross-backend range query comparison ──────────────────────────────────
    println!("\n=== Cross-backend range query comparison [300,700]^2 ===");
    let mut qt2_full = Quadtree::<usize, f64, 2>::new();
    for (i, &p) in raw.iter().enumerate() {
        qt2_full.insert(p, i);
    }
    let qt_count = qt2_full.range_query(&query_bbox).len();
    println!("  R-tree   : {rt_count}");
    println!("  KD-tree  : {kd_count}");
    println!("  Quadtree : {qt_count}");
    println!("  Brute    : {brute_count}");
    println!(
        "  All match: {}",
        if rt_count == brute_count && kd_count == brute_count && qt_count == brute_count {
            "yes"
        } else {
            "NO"
        }
    );
}
