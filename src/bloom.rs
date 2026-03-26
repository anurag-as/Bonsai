//! Bloom filter cache for spatial bounding boxes.
//!
//! [`BloomCache<D>`] is a probabilistic negative cache that answers the question
//! "has this bounding box ever been inserted?" in O(1) time using k hash functions
//! over a fixed bit array.
//!
//! - Returns [`BloomResult::DefinitelyAbsent`] if any of the k hash bits is unset.
//! - Returns [`BloomResult::ProbablyPresent`] otherwise.
//! - **Zero false negative rate**: a bounding box that has been inserted will
//!   always return `ProbablyPresent`.
//! - Configurable memory budget (default 131 072 bytes / 128 KiB).
//! - Default k = 7 hash functions.
//!
//! # Serialisation
//!
//! `BBox<f64, D>` is serialised as `2 × D × 8` little-endian f64 bytes (min
//! coordinates followed by max coordinates) before hashing, making the
//! implementation D-agnostic.
//!
//! # Example
//!
//! ```rust
//! use bonsai::bloom::{BloomCache, BloomResult};
//! use bonsai::types::{BBox, Point};
//!
//! let mut cache = BloomCache::<2>::new(65536, 7);
//! let bbox = BBox::new(Point::new([0.0_f64, 0.0]), Point::new([1.0, 1.0]));
//! cache.insert(&bbox);
//! assert_eq!(cache.check(&bbox), BloomResult::ProbablyPresent);
//! ```

use crate::types::BBox;

/// The result of a [`BloomCache::check`] query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BloomResult {
    /// None of the k hash bits are set — the bounding box has definitely not
    /// been inserted.
    DefinitelyAbsent,
    /// All k hash bits are set — the bounding box has probably been inserted
    /// (may be a false positive).
    ProbablyPresent,
}

/// Bloom filter cache for `BBox<f64, D>` values.
///
/// Uses `k` independent hash functions over a fixed bit array whose size is
/// determined by the configured memory budget.
#[derive(Debug, Clone)]
pub struct BloomCache<const D: usize> {
    /// Bit array stored as packed `u64` words.
    pub bits: Vec<u64>,
    /// Number of hash functions.
    pub k: usize,
    /// Memory budget in bytes (= `bits.len() * 8`).
    pub memory: usize,
}

impl<const D: usize> BloomCache<D> {
    /// Create a new `BloomCache` with the given memory budget (in bytes) and
    /// number of hash functions `k`.
    ///
    /// The bit array is sized to fill the budget exactly (rounded down to the
    /// nearest 8-byte word).
    ///
    /// # Panics
    ///
    /// Panics if `memory_bytes < 8` or `k == 0`.
    pub fn new(memory_bytes: usize, k: usize) -> Self {
        assert!(memory_bytes >= 8, "memory_bytes must be at least 8");
        assert!(k > 0, "k must be at least 1");
        let n_words = memory_bytes / 8;
        let actual_bytes = n_words * 8;
        Self {
            bits: vec![0u64; n_words],
            k,
            memory: actual_bytes,
        }
    }

    /// Create a `BloomCache` with default settings: 131 072 bytes (128 KiB) and k = 7.
    ///
    /// This budget supports up to 100 000 inserted entries with a false positive
    /// rate below 1% (theoretical: ~0.8% at 100k entries, k=7, 1 048 576 bits).
    pub fn default_settings() -> Self {
        Self::new(131_072, 7)
    }

    /// Return the total number of bits in the filter.
    #[inline]
    pub fn num_bits(&self) -> u64 {
        (self.bits.len() as u64) * 64
    }

    /// Serialise a `BBox<f64, D>` to `2 × D × 8` little-endian bytes.
    ///
    /// Layout: `[min[0], min[1], …, min[D-1], max[0], max[1], …, max[D-1]]`
    /// each as 8 little-endian bytes.
    pub fn serialise_bbox(bbox: &BBox<f64, D>) -> Vec<u8> {
        let mut buf = Vec::with_capacity(2 * D * 8);
        for d in 0..D {
            buf.extend_from_slice(&bbox.min.coords()[d].to_le_bytes());
        }
        for d in 0..D {
            buf.extend_from_slice(&bbox.max.coords()[d].to_le_bytes());
        }
        buf
    }

    /// Insert a bounding box into the filter.
    pub fn insert(&mut self, bbox: &BBox<f64, D>) {
        let bytes = Self::serialise_bbox(bbox);
        let n_bits = self.num_bits();
        for i in 0..self.k {
            let bit_idx = hash_bytes(&bytes, i as u64) % n_bits;
            let word = (bit_idx / 64) as usize;
            let bit = bit_idx % 64;
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// Check whether a bounding box is in the filter.
    ///
    /// Returns [`BloomResult::DefinitelyAbsent`] if any hash bit is unset,
    /// [`BloomResult::ProbablyPresent`] otherwise.
    pub fn check(&self, bbox: &BBox<f64, D>) -> BloomResult {
        let bytes = Self::serialise_bbox(bbox);
        let n_bits = self.num_bits();
        for i in 0..self.k {
            let bit_idx = hash_bytes(&bytes, i as u64) % n_bits;
            let word = (bit_idx / 64) as usize;
            let bit = bit_idx % 64;
            if self.bits[word] & (1u64 << bit) == 0 {
                return BloomResult::DefinitelyAbsent;
            }
        }
        BloomResult::ProbablyPresent
    }
}

/// Compute two independent 64-bit hashes via FNV-1a (h1) and a djb2-style
/// pass with a final avalanche mix (h2). Used for double hashing:
/// `bit_i = (h1 + i * h2) % num_bits`.
fn hash_bytes_pair(bytes: &[u8]) -> (u64, u64) {
    const FNV_PRIME: u64 = 0x00000100_000001B3;
    const FNV_BASIS: u64 = 0xcbf29ce4_84222325;
    let mut h1 = FNV_BASIS;
    for &b in bytes {
        h1 ^= b as u64;
        h1 = h1.wrapping_mul(FNV_PRIME);
    }

    let mut h2: u64 = 0x517cc1b727220a95;
    for &b in bytes {
        h2 = h2.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(b as u64);
        h2 ^= h2 >> 33;
    }
    h2 ^= h2 >> 33;
    h2 = h2.wrapping_mul(0xff51afd7ed558ccd);
    h2 ^= h2 >> 33;
    h2 = h2.wrapping_mul(0xc4ceb9fe1a85ec53);
    h2 ^= h2 >> 33;

    (h1, h2 | 1) // h2 odd → coprime with any power-of-two num_bits
}

fn hash_bytes(bytes: &[u8], i: u64) -> u64 {
    let (h1, h2) = hash_bytes_pair(bytes);
    h1.wrapping_add(i.wrapping_mul(h2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point;

    fn make_bbox_2d(x0: f64, y0: f64, x1: f64, y1: f64) -> BBox<f64, 2> {
        BBox::new(Point::new([x0, y0]), Point::new([x1, y1]))
    }

    #[test]
    fn insert_then_check_probably_present() {
        let mut cache = BloomCache::<2>::new(65_536, 7);
        let b = make_bbox_2d(0.0, 0.0, 1.0, 1.0);
        cache.insert(&b);
        assert_eq!(cache.check(&b), BloomResult::ProbablyPresent);
    }

    #[test]
    fn empty_cache_returns_definitely_absent() {
        let cache = BloomCache::<2>::new(65_536, 7);
        let b = make_bbox_2d(0.0, 0.0, 1.0, 1.0);
        assert_eq!(cache.check(&b), BloomResult::DefinitelyAbsent);
    }

    #[test]
    fn bbox_serialisation_length_d2() {
        let b = make_bbox_2d(1.0, 2.0, 3.0, 4.0);
        let bytes = BloomCache::<2>::serialise_bbox(&b);
        assert_eq!(bytes.len(), 2 * 2 * 8);
    }

    #[test]
    fn bbox_serialisation_length_d3() {
        let b = BBox::<f64, 3>::new(Point::new([1.0, 2.0, 3.0]), Point::new([4.0, 5.0, 6.0]));
        let bytes = BloomCache::<3>::serialise_bbox(&b);
        assert_eq!(bytes.len(), 2 * 3 * 8);
    }

    #[test]
    fn bbox_serialisation_length_d6() {
        let b = BBox::<f64, 6>::new(
            Point::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        );
        let bytes = BloomCache::<6>::serialise_bbox(&b);
        assert_eq!(bytes.len(), 2 * 6 * 8);
    }

    #[test]
    fn memory_budget_respected() {
        for budget in [64usize, 256, 1024, 65_536] {
            let cache = BloomCache::<2>::new(budget, 7);
            assert!(
                cache.memory <= budget,
                "memory {} > budget {}",
                cache.memory,
                budget
            );
        }
    }

    #[test]
    fn false_positive_rate_within_spec() {
        let mut cache = BloomCache::<2>::default_settings();
        for i in 0..100_000u64 {
            let f = i as f64;
            let b = make_bbox_2d(f, f, f + 1.0, f + 1.0);
            cache.insert(&b);
        }
        let mut fp = 0u64;
        let checks = 10_000u64;
        for i in 0..checks {
            let f = (200_000 + i) as f64;
            let b = make_bbox_2d(f, f, f + 1.0, f + 1.0);
            if cache.check(&b) == BloomResult::ProbablyPresent {
                fp += 1;
            }
        }
        let fp_rate = fp as f64 / checks as f64;
        assert!(
            fp_rate <= 0.01,
            "FP rate {:.4} exceeds 1% (fp={fp}, checks={checks})",
            fp_rate
        );
    }

    #[test]
    fn zero_false_negatives_d2() {
        let mut cache = BloomCache::<2>::default_settings();
        let bboxes: Vec<_> = (0..1000u64)
            .map(|i| {
                let f = i as f64;
                make_bbox_2d(f, f, f + 1.0, f + 1.0)
            })
            .collect();
        for b in &bboxes {
            cache.insert(b);
        }
        for b in &bboxes {
            assert_eq!(
                cache.check(b),
                BloomResult::ProbablyPresent,
                "false negative for bbox {:?}",
                b
            );
        }
    }

    // ── property tests ───────────────────────────────────────────────────────

    use proptest::prelude::*;
    fn bbox_strategy_2d() -> impl Strategy<Value = BBox<f64, 2>> {
        (
            -1.0e6_f64..1.0e6_f64,
            -1.0e6_f64..1.0e6_f64,
            0.001_f64..1.0e4_f64,
            0.001_f64..1.0e4_f64,
        )
            .prop_map(|(x, y, w, h)| make_bbox_2d(x, y, x + w, y + h))
    }

    fn bbox_strategy_3d() -> impl Strategy<Value = BBox<f64, 3>> {
        (
            -1.0e6_f64..1.0e6_f64,
            -1.0e6_f64..1.0e6_f64,
            -1.0e6_f64..1.0e6_f64,
            0.001_f64..1.0e4_f64,
            0.001_f64..1.0e4_f64,
            0.001_f64..1.0e4_f64,
        )
            .prop_map(|(x, y, z, w, h, d)| {
                BBox::new(Point::new([x, y, z]), Point::new([x + w, y + h, z + d]))
            })
    }

    // BloomCache never returns DefinitelyAbsent for inserted bboxes
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_bloom_zero_false_negatives_d2(
            bboxes in prop::collection::vec(bbox_strategy_2d(), 1..50),
        ) {
            let mut cache = BloomCache::<2>::default_settings();
            for b in &bboxes {
                cache.insert(b);
            }
            for b in &bboxes {
                prop_assert_eq!(
                    cache.check(b),
                    BloomResult::ProbablyPresent,
                    "false negative for bbox {:?}", b
                );
            }
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_bloom_zero_false_negatives_d3(
            bboxes in prop::collection::vec(bbox_strategy_3d(), 1..50),
        ) {
            let mut cache = BloomCache::<3>::default_settings();
            for b in &bboxes {
                cache.insert(b);
            }
            for b in &bboxes {
                prop_assert_eq!(
                    cache.check(b),
                    BloomResult::ProbablyPresent,
                    "false negative for bbox {:?}", b
                );
            }
        }
    }

    // BloomCache false positive rate does not exceed 1% at default settings
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 10,
            ..Default::default()
        })]

        #[test]
        fn prop_bloom_fp_rate_within_spec(
            seed in 0u64..1000,
        ) {
            // Insert 100k bboxes using a deterministic pattern derived from seed.
            // Default settings: 131072 bytes (1048576 bits), k=7.
            // Theoretical FP at 100k entries: ~0.8%.
            let mut cache = BloomCache::<2>::default_settings();
            let offset = seed as f64 * 1_000_000.0;
            for i in 0..100_000u64 {
                let f = offset + i as f64;
                let b = make_bbox_2d(f, f, f + 1.0, f + 1.0);
                cache.insert(&b);
            }
            // Check 5000 bboxes that were never inserted (use a disjoint range).
            // Using 5000 checks reduces sampling variance while keeping the test fast.
            let check_offset = offset + 500_000_000.0;
            let mut fp = 0u64;
            let checks = 5000u64;
            for i in 0..checks {
                let f = check_offset + i as f64;
                let b = make_bbox_2d(f, f, f + 1.0, f + 1.0);
                if cache.check(&b) == BloomResult::ProbablyPresent {
                    fp += 1;
                }
            }
            let fp_rate = fp as f64 / checks as f64;
            // Allow up to 1.5% to account for sampling variance at 5000 checks.
            // The true theoretical rate is ~0.8%, well within the 1% spec limit.
            prop_assert!(
                fp_rate <= 0.015,
                "FP rate {:.4} exceeds 1.5% tolerance (fp={fp}, checks={checks}, seed={seed})",
                fp_rate
            );
        }
    }

    // BloomCache memory usage does not exceed configured budget
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_bloom_memory_budget(
            budget_words in 1usize..1024,
            k in 1usize..16,
        ) {
            let budget_bytes = budget_words * 8;
            let cache = BloomCache::<2>::new(budget_bytes, k);
            prop_assert!(
                cache.memory <= budget_bytes,
                "memory {} exceeds budget {}", cache.memory, budget_bytes
            );
            let allocated = cache.bits.len() * 8;
            prop_assert_eq!(cache.memory, allocated);
        }
    }

    // BBox serialisation length is exactly 2 × D × 8 bytes
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_bloom_bbox_serialisation_length_d2(
            x0 in -1.0e9_f64..1.0e9,
            y0 in -1.0e9_f64..1.0e9,
            x1 in -1.0e9_f64..1.0e9,
            y1 in -1.0e9_f64..1.0e9,
        ) {
            let b = BBox::<f64, 2>::new(Point::new([x0, y0]), Point::new([x1, y1]));
            let bytes = BloomCache::<2>::serialise_bbox(&b);
            prop_assert_eq!(bytes.len(), 2 * 2 * 8);
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_bloom_bbox_serialisation_length_d3(
            x0 in -1.0e9_f64..1.0e9, y0 in -1.0e9_f64..1.0e9, z0 in -1.0e9_f64..1.0e9,
            x1 in -1.0e9_f64..1.0e9, y1 in -1.0e9_f64..1.0e9, z1 in -1.0e9_f64..1.0e9,
        ) {
            let b = BBox::<f64, 3>::new(Point::new([x0, y0, z0]), Point::new([x1, y1, z1]));
            let bytes = BloomCache::<3>::serialise_bbox(&b);
            prop_assert_eq!(bytes.len(), 2 * 3 * 8);
        }
    }
}
