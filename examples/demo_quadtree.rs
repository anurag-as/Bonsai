//! Quadtree / Octree demo.
//!
//! Builds a D=2 quadtree and a D=3 octree, each with 200 random points, and
//! prints structural statistics.
//!
//! Run with:
//!   cargo run --example demo_quadtree

use bonsai::types::{BBox, Point};
use bonsai::SpatialBackend;
use bonsai::Quadtree;

/// Minimal deterministic LCG RNG (same as used in the unit tests).
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
}

fn main() {
    const N: usize = 200;

    // ── D=2 Quadtree ──────────────────────────────────────────────────────────
    println!("=== D=2 Quadtree ({N} random points) ===");
    let mut qt2 = Quadtree::<usize, f64, 2>::new();
    let mut rng = Lcg::new(42);
    for i in 0..N {
        qt2.insert(
            Point::new([rng.next_f64() * 1000.0, rng.next_f64() * 1000.0]),
            i,
        );
    }
    println!("  entries              : {}", qt2.len());
    println!("  node count           : {}", qt2.node_count());
    println!("  max depth            : {}", qt2.max_depth());
    println!("  children per split   : {} (2^D = 2^2)", 1usize << 2);

    let bbox2 = BBox::new(Point::new([200.0, 200.0]), Point::new([800.0, 800.0]));
    let hits2 = qt2.range_query(&bbox2).len();
    println!("  range [200,800]^2    : {hits2} results");

    // ── D=3 Octree ────────────────────────────────────────────────────────────
    println!("\n=== D=3 Octree ({N} random points) ===");
    let mut qt3 = Quadtree::<usize, f64, 3>::new();
    let mut rng3 = Lcg::new(99);
    for i in 0..N {
        qt3.insert(
            Point::new([
                rng3.next_f64() * 1000.0,
                rng3.next_f64() * 1000.0,
                rng3.next_f64() * 1000.0,
            ]),
            i,
        );
    }
    println!("  entries              : {}", qt3.len());
    println!("  node count           : {}", qt3.node_count());
    println!("  max depth            : {}", qt3.max_depth());
    // Each split node in a D=3 tree has exactly 2^3 = 8 children (octree).
    println!("  children per split   : {} (2^D = 2^3)", 1usize << 3);

    let bbox3 = BBox::new(
        Point::new([200.0, 200.0, 200.0]),
        Point::new([800.0, 800.0, 800.0]),
    );
    let hits3 = qt3.range_query(&bbox3).len();
    println!("  range [200,800]^3    : {hits3} results");
}
