//! End-to-end demo covering all core components:
//!   - Point / BBox types (2D and 3D): construction, contains_point, intersects
//!   - HilbertCurve: 4x4 grid view, D=8 128-bit budget, spatial locality
//!   - BloomCache: insert/check, false negative / false positive rate summary
//!   - R-tree: Hilbert-ordered insertion, range query with Bloom fast-path, kNN
//!
//! Run with: cargo run --example demo_bonsai

use bonsai::backends::rtree::RTree;
use bonsai::backends::SpatialBackend;
use bonsai::bloom::{BloomCache, BloomResult};
use bonsai::hilbert::HilbertCurve;
use bonsai::types::{BBox, EntryId, Point};

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn normalise(v: f64, order: u32) -> u64 {
    let cells = (1u64 << order) as f64;
    ((v / 1000.0) * cells).min(cells - 1.0) as u64
}

fn hash_bbox(i: u64, offset: f64) -> BBox<f64, 2> {
    let mut z = i.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    let x = offset + (z as f64 / u64::MAX as f64) * 10_000.0;
    let mut z2 = z.wrapping_add(1);
    z2 = (z2 ^ (z2 >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z2 ^= z2 >> 31;
    let y = offset + (z2 as f64 / u64::MAX as f64) * 10_000.0;
    BBox::new(Point::new([x, y]), Point::new([x + 1.0, y + 1.0]))
}

fn main() {
    // ── Types ─────────────────────────────────────────────────────────────────
    println!("=== Types ===");
    let p3 = Point::<f64, 3>::new([1.5, 2.5, 3.5]);
    let bbox3 = BBox::<f64, 3>::new(
        Point::new([0.0, 0.0, 0.0]),
        Point::new([5.0, 5.0, 5.0]),
    );
    println!("3D point: {:?}", p3);
    println!("3D bbox:  {:?}", bbox3);
    println!("contains (1.5,2.5,3.5): {}", bbox3.contains_point(&p3));
    println!(
        "contains (6.0,2.5,3.5): {}",
        bbox3.contains_point(&Point::new([6.0, 2.5, 3.5]))
    );
    let overlapping = BBox::<f64, 3>::new(
        Point::new([3.0, 3.0, 3.0]),
        Point::new([8.0, 8.0, 8.0]),
    );
    let disjoint = BBox::<f64, 3>::new(
        Point::new([6.0, 6.0, 6.0]),
        Point::new([9.0, 9.0, 9.0]),
    );
    println!("intersects overlapping: {}", bbox3.intersects(&overlapping));
    println!("intersects disjoint:    {}", bbox3.intersects(&disjoint));

    // ── Hilbert curve ─────────────────────────────────────────────────────────
    println!("\n=== Hilbert curve ===");
    let curve2 = HilbertCurve::<2>::new(2);
    let mut cells: Vec<(u128, u64, u64)> = (0u64..4)
        .flat_map(|y| (0u64..4).map(move |x| (curve2.index(&[x, y]), x, y)))
        .collect();
    cells.sort_by_key(|&(idx, _, _)| idx);
    println!("4x4 grid in Hilbert order (index -> x,y):");
    for (idx, x, y) in &cells {
        println!("  {:2} -> ({},{})", idx, x, y);
    }
    println!("\nGrid view (Hilbert index at each cell):");
    println!("  y\\x  0   1   2   3");
    for y in (0u64..4).rev() {
        print!("   {}  ", y);
        for x in 0u64..4 {
            print!("{:3} ", curve2.index(&[x, y]));
        }
        println!();
    }
    let c8 = HilbertCurve::<8>::new(16);
    let max16 = (1u64 << 16) - 1;
    println!("\nD=8, order=16 (128-bit budget):");
    println!("  index([0;8])   = {}", c8.index(&[0u64; 8]));
    println!("  index([max;8]) = {}", c8.index(&[max16; 8]));
    println!("  u128::MAX      = {}", u128::MAX);
    let c16 = HilbertCurve::<2>::new(4);
    let i_base = c16.index(&[7, 7]);
    let i_near = c16.index(&[8, 7]);
    let i_far = c16.index(&[0, 15]);
    println!("\nSpatial locality (16x16 grid):");
    println!("  base (7,7)  -> {:4}", i_base);
    println!("  near (8,7)  -> {:4}  diff={}", i_near, i_base.abs_diff(i_near));
    println!("  far  (0,15) -> {:4}  diff={}", i_far, i_base.abs_diff(i_far));

    // ── Bloom filter ──────────────────────────────────────────────────────────
    println!("\n=== Bloom filter ===");
    const N_INSERT: usize = 1000;
    const N_CHECK: usize = 200;
    let mut bloom_test = BloomCache::<2>::default_settings();
    let inserted: Vec<BBox<f64, 2>> = (0..N_INSERT as u64).map(|i| hash_bbox(i, 0.0)).collect();
    for b in &inserted {
        bloom_test.insert(b);
    }
    let false_negatives = inserted
        .iter()
        .take(N_CHECK)
        .filter(|b| bloom_test.check(b) == BloomResult::DefinitelyAbsent)
        .count();
    let false_positives = (0..N_CHECK as u64)
        .filter(|&i| bloom_test.check(&hash_bbox(i, 100_000.0)) == BloomResult::ProbablyPresent)
        .count();
    let fp_rate = false_positives as f64 / N_CHECK as f64 * 100.0;
    println!("Memory: {} bytes, k={}", bloom_test.memory, bloom_test.k);
    println!("Inserted {N_INSERT} bboxes, checked {N_CHECK} inserted + {N_CHECK} absent");
    println!(
        "  False negatives: {false_negatives}  {}",
        if false_negatives == 0 { "ok" } else { "BUG" }
    );
    println!(
        "  False positives: {false_positives}/{N_CHECK}  rate={fp_rate:.2}%  {}",
        if fp_rate <= 1.0 { "ok" } else { "BUG" }
    );

    // ── R-tree with Hilbert ordering + Bloom fast-path ────────────────────────
    println!("\n=== R-tree (Hilbert-ordered insertion + Bloom fast-path) ===");
    let mut rng = Lcg::new(12345);
    let n = 500usize;
    let raw: Vec<Point<f64, 2>> = (0..n)
        .map(|_| Point::new([rng.next_f64() * 1000.0, rng.next_f64() * 1000.0]))
        .collect();

    let curve = HilbertCurve::<2>::new(10);
    let order = curve.order;
    let mut indexed: Vec<(u128, Point<f64, 2>)> = raw
        .iter()
        .map(|&p| {
            let hx = normalise(p.coords()[0], order);
            let hy = normalise(p.coords()[1], order);
            (curve.index(&[hx, hy]), p)
        })
        .collect();
    indexed.sort_by_key(|&(h, _)| h);

    println!("First 5 points in Hilbert insertion order:");
    for (h, p) in indexed.iter().take(5) {
        println!(
            "  hilbert={:<8}  point=({:.2}, {:.2})",
            h,
            p.coords()[0],
            p.coords()[1]
        );
    }

    let mut tree = RTree::<usize, f64, 2>::new();
    let mut bloom = BloomCache::<2>::new(65_536, 7);
    let mut points: Vec<(Point<f64, 2>, EntryId)> = Vec::with_capacity(n);
    for (i, (_, p)) in indexed.iter().enumerate() {
        let id = tree.insert(*p, i);
        points.push((*p, id));
    }
    println!(
        "\nInserted {}  backend={:?}  bloom: {} bytes k={}",
        tree.len(),
        tree.kind(),
        bloom.memory,
        bloom.k
    );

    let query_bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([500.0, 500.0]));
    println!("\nRange query [0,500]^2 (~25% of space):");
    println!("  Bloom (first):  {:?}", bloom.check(&query_bbox));
    let range_results = tree.range_query(&query_bbox);
    println!("  R-tree results: {}", range_results.len());
    if !range_results.is_empty() {
        bloom.insert(&query_bbox);
    }
    println!("  Bloom (repeat): {:?}", bloom.check(&query_bbox));
    println!("  First 5:");
    for (id, payload) in range_results.iter().take(5) {
        let pt = points.iter().find(|(_, eid)| eid == id).map(|(p, _)| p);
        if let Some(p) = pt {
            println!(
                "    id={:?}  payload={:3}  point=({:.2}, {:.2})",
                id,
                payload,
                p.coords()[0],
                p.coords()[1]
            );
        }
    }

    let empty_bbox = BBox::new(
        Point::new([50_000.0, 50_000.0]),
        Point::new([51_000.0, 51_000.0]),
    );
    println!("\nEmpty region [50000,51000]^2:");
    match bloom.check(&empty_bbox) {
        BloomResult::DefinitelyAbsent => {
            println!("  Bloom: DefinitelyAbsent — R-tree skipped.");
        }
        BloomResult::ProbablyPresent => {
            let r = tree.range_query(&empty_bbox);
            println!(
                "  Bloom: ProbablyPresent (false positive) — R-tree returned {}",
                r.len()
            );
        }
    }

    let centre = Point::new([500.0, 500.0]);
    let knn = tree.knn_query(&centre, 3);
    println!("\nkNN(k=3) from centre (500, 500):");
    for (rank, (dist, id, payload)) in knn.iter().enumerate() {
        let pt = points.iter().find(|(_, eid)| eid == id).map(|(p, _)| p);
        if let Some(p) = pt {
            println!(
                "  #{rank}  id={:?}  payload={:3}  dist={:.4}  point=({:.2}, {:.2})",
                id,
                payload,
                dist,
                p.coords()[0],
                p.coords()[1]
            );
        }
    }
}
