//! End-to-end demo showing all modules working together as a system:
//!
//!  1. Generate clustered 2D points
//!  2. Sort them by Hilbert index for cache-friendly insertion order
//!  3. Insert into both an R-tree and a KD-tree
//!  4. Warm a shared BloomCache from the inserted bounding boxes
//!  5. Run range queries through the Bloom fast-path on both backends
//!  6. Run kNN queries on both backends and compare results
//!  7. Show the Bloom filter catching an empty-region query before either tree is touched
//!
//! Run with:
//!   cargo run --example demo_bonsai

use bonsai::backends::kdtree::KDTree;
use bonsai::backends::rtree::RTree;
use bonsai::backends::SpatialBackend;
use bonsai::bloom::{BloomCache, BloomResult};
use bonsai::hilbert::HilbertCurve;
use bonsai::types::{BBox, Point};

// ── Minimal deterministic RNG (no external deps) ──────────────────────────────

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

    // ── 1. Generate clustered points ──────────────────────────────────────────
    println!("=== Generating {} points in {} clusters ===", N, CLUSTERS);
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

    // ── 2. Sort by Hilbert index ───────────────────────────────────────────────
    println!("\n=== Sorting by Hilbert index (order=10, D=2) ===");
    let hilbert = HilbertCurve::<2>::new(10);
    let mut indexed: Vec<(u128, Point<f64, 2>)> = raw
        .iter()
        .map(|&p| {
            let hx = normalise(p.coords()[0], DOMAIN, hilbert.order);
            let hy = normalise(p.coords()[1], DOMAIN, hilbert.order);
            (hilbert.index(&[hx, hy]), p)
        })
        .collect();
    indexed.sort_by_key(|&(h, _)| h);

    println!("First 5 points in Hilbert insertion order:");
    for (h, p) in indexed.iter().take(5) {
        println!(
            "  hilbert={:<10}  ({:.2}, {:.2})",
            h,
            p.coords()[0],
            p.coords()[1]
        );
    }

    // ── 3. Insert into R-tree and KD-tree in Hilbert order ────────────────────
    println!("\n=== Inserting into R-tree and KD-tree ===");
    let mut rtree = RTree::<usize, f64, 2>::new();
    let mut kdtree = KDTree::<usize, f64, 2>::new();
    let mut bloom = BloomCache::<2>::new(65_536, 7);

    for (i, (_, p)) in indexed.iter().enumerate() {
        rtree.insert(*p, i);
        kdtree.insert(*p, i);
        // Warm the Bloom filter with a unit bbox around each point.
        let pt_bbox = BBox::new(*p, *p);
        bloom.insert(&pt_bbox);
    }

    println!("R-tree entries : {}", rtree.len());
    println!(
        "KD-tree entries: {}  depth={}",
        kdtree.len(),
        kdtree.depth()
    );
    let depth_bound = (N as f64).log2().ceil() as usize + 1;
    println!(
        "Depth bound    : ceil(log2({})) + 1 = {}  {}",
        N,
        depth_bound,
        if kdtree.depth() <= depth_bound {
            "ok"
        } else {
            "EXCEEDED"
        }
    );
    println!("Bloom memory   : {} bytes, k={}", bloom.memory, bloom.k);

    // ── 4. Range query through Bloom fast-path on both backends ───────────────
    println!("\n=== Range query [300,700]^2 through Bloom fast-path ===");
    let query_bbox = BBox::new(Point::new([300.0, 300.0]), Point::new([700.0, 700.0]));

    // Insert the query region into Bloom so it's known.
    bloom.insert(&query_bbox);

    let bloom_result = bloom.check(&query_bbox);
    println!("Bloom check: {:?}", bloom_result);

    let (rtree_results, kdtree_results) = match bloom_result {
        BloomResult::DefinitelyAbsent => {
            println!("Bloom says empty — both trees skipped.");
            (vec![], vec![])
        }
        BloomResult::ProbablyPresent => {
            let r = rtree.range_query(&query_bbox);
            let k = kdtree.range_query(&query_bbox);
            (r, k)
        }
    };

    let brute_count = raw.iter().filter(|p| query_bbox.contains_point(p)).count();
    println!("R-tree results : {}", rtree_results.len());
    println!("KD-tree results: {}", kdtree_results.len());
    println!("Brute-force    : {}", brute_count);
    println!(
        "Both match brute-force: {}",
        if rtree_results.len() == brute_count && kdtree_results.len() == brute_count {
            "yes"
        } else {
            "NO"
        }
    );

    // ── 5. Empty-region query — Bloom catches it before either tree is touched ─
    println!("\n=== Empty-region query [9000,9100]^2 (outside all data) ===");
    let empty_bbox = BBox::new(
        Point::new([9_000.0, 9_000.0]),
        Point::new([9_100.0, 9_100.0]),
    );
    match bloom.check(&empty_bbox) {
        BloomResult::DefinitelyAbsent => {
            println!("Bloom: DefinitelyAbsent — both trees skipped (O(1) rejection).");
        }
        BloomResult::ProbablyPresent => {
            let r = rtree.range_query(&empty_bbox);
            let k = kdtree.range_query(&empty_bbox);
            println!(
                "Bloom: ProbablyPresent (false positive) — R-tree={}, KD-tree={}",
                r.len(),
                k.len()
            );
        }
    }

    // ── 6. kNN from centre — compare R-tree and KD-tree ──────────────────────
    println!("\n=== kNN(k=5) from centre (500, 500) — R-tree vs KD-tree ===");
    let query_pt = Point::new([500.0, 500.0]);
    let k = 5;
    let rtree_knn = rtree.knn_query(&query_pt, k);
    let kdtree_knn = kdtree.knn_query(&query_pt, k);

    println!("  rank  R-tree dist    KD-tree dist   match");
    for i in 0..k {
        let rd = rtree_knn[i].0;
        let kd = kdtree_knn[i].0;
        let matches = (rd - kd).abs() < 1e-9;
        println!(
            "  #{i}    {rd:>10.4}    {kd:>10.4}     {}",
            if matches { "yes" } else { "NO" }
        );
    }
}
