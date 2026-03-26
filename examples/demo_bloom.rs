//! Demo: BloomCache<2> — insert 1000 random bboxes, check 200 inserted and
//! 200 never-inserted, print a summary of the false positive rate.

use bonsai::bloom::{BloomCache, BloomResult};
use bonsai::types::{BBox, Point};

fn make_bbox(x: f64, y: f64) -> BBox<f64, 2> {
    BBox::new(Point::new([x, y]), Point::new([x + 1.0, y + 1.0]))
}

fn pseudo_rand(i: u64, scale: f64) -> f64 {
    let mut z = i.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);
    (z as f64 / u64::MAX as f64) * scale
}

fn main() {
    const N_INSERT: usize = 1000;
    const N_CHECK_INSERTED: usize = 200;
    const N_CHECK_ABSENT: usize = 200;

    let mut cache = BloomCache::<2>::default_settings();

    let inserted: Vec<BBox<f64, 2>> = (0..N_INSERT as u64)
        .map(|i| {
            make_bbox(
                pseudo_rand(i * 2, 10_000.0),
                pseudo_rand(i * 2 + 1, 10_000.0),
            )
        })
        .collect();

    for b in &inserted {
        cache.insert(b);
    }

    let mut false_negatives = 0usize;
    for b in inserted.iter().take(N_CHECK_INSERTED) {
        if cache.check(b) == BloomResult::DefinitelyAbsent {
            false_negatives += 1;
        }
    }

    let mut false_positives = 0usize;
    for i in 0..N_CHECK_ABSENT as u64 {
        let b = make_bbox(
            100_000.0 + pseudo_rand(i * 2, 10_000.0),
            100_000.0 + pseudo_rand(i * 2 + 1, 10_000.0),
        );
        if cache.check(&b) == BloomResult::ProbablyPresent {
            false_positives += 1;
        }
    }

    let fp_rate = false_positives as f64 / N_CHECK_ABSENT as f64 * 100.0;

    println!("=== BloomCache<2> Demo ===");
    println!(
        "Memory budget : {} bytes ({} bits)",
        cache.memory,
        cache.memory * 8
    );
    println!("Hash functions: k = {}", cache.k);
    println!("Inserted      : {N_INSERT} bboxes");
    println!();
    println!("Checked {N_CHECK_INSERTED} inserted bboxes:");
    println!(
        "  ProbablyPresent : {}  (expected {N_CHECK_INSERTED})",
        N_CHECK_INSERTED - false_negatives
    );
    println!("  DefinitelyAbsent: {false_negatives}  (false negatives — must be 0)");
    println!();
    println!("Checked {N_CHECK_ABSENT} never-inserted bboxes:");
    println!("  DefinitelyAbsent: {}", N_CHECK_ABSENT - false_positives);
    println!("  ProbablyPresent : {false_positives}  (false positives)");
    println!("  False positive rate: {fp_rate:.2}%  (spec limit: 1.00%)");

    if false_negatives == 0 {
        println!("\n✓ Zero false negatives");
    } else {
        println!("\n✗ {false_negatives} false negatives — BUG");
    }
    if fp_rate <= 1.0 {
        println!("✓ FP rate {fp_rate:.2}% ≤ 1%");
    } else {
        println!("✗ FP rate {fp_rate:.2}% exceeds 1% — BUG");
    }
}
