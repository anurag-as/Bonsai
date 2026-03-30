//! Reservoir sampler using Vitter's Algorithm R.
//!
//! Maintains a fixed-size random sample of inserted points such that each point
//! in the stream has equal probability `capacity / n` of being represented.

use crate::types::{CoordType, Point};

/// A fixed-size reservoir sampler using Vitter's Algorithm R.
///
/// Each point in the stream has equal probability `capacity / n` of being
/// retained, where `n` is the total number of points observed.
pub struct ReservoirSampler<C: CoordType, const D: usize> {
    reservoir: Vec<Point<C, D>>,
    capacity: usize,
    count: usize,
    rng: u64,
}

impl<C: CoordType, const D: usize> ReservoirSampler<C, D> {
    /// Create a new `ReservoirSampler` with the given capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "reservoir capacity must be at least 1");
        Self {
            reservoir: Vec::with_capacity(capacity),
            capacity,
            count: 0,
            rng: 0x517cc1b727220a95,
        }
    }

    /// Return the configured capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the current number of points in the reservoir (≤ capacity).
    #[inline]
    pub fn len(&self) -> usize {
        self.reservoir.len()
    }

    /// Return `true` if the reservoir is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.reservoir.is_empty()
    }

    /// Return the total number of points observed (including those not sampled).
    #[inline]
    pub fn total_count(&self) -> usize {
        self.count
    }

    /// Return a reference to the current reservoir contents.
    #[inline]
    pub fn samples(&self) -> &[Point<C, D>] {
        &self.reservoir
    }

    /// Update the reservoir with a new point using Vitter's Algorithm R.
    pub fn update(&mut self, point: Point<C, D>) {
        self.count += 1;
        if self.reservoir.len() < self.capacity {
            self.reservoir.push(point);
        } else {
            let j = self.next_rand_usize(self.count);
            if j < self.capacity {
                self.reservoir[j] = point;
            }
        }
    }

    /// Reset the sampler, clearing all samples and resetting the count.
    pub fn reset(&mut self) {
        self.reservoir.clear();
        self.count = 0;
    }

    /// Generate a pseudo-random `usize` in `[0, n)` using an LCG.
    #[inline]
    fn next_rand_usize(&mut self, n: usize) -> usize {
        self.rng = self
            .rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.rng >> 11) as usize) % n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point;
    use proptest::prelude::*;

    // For any sequence of n insertions into a ReservoirSampler with capacity k,
    // the reservoir size must never exceed k, and each point must have equal
    // probability k/n of being represented.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_reservoir_bounded_size(
            capacity in 1usize..64,
            n in 0usize..512,
        ) {
            let mut sampler = ReservoirSampler::<f64, 2>::new(capacity);
            for i in 0..n {
                let f = i as f64;
                sampler.update(Point::new([f, f]));
                prop_assert!(
                    sampler.len() <= capacity,
                    "reservoir size {} exceeded capacity {} after {} insertions",
                    sampler.len(), capacity, i + 1
                );
            }
            prop_assert_eq!(sampler.len(), n.min(capacity));
            prop_assert_eq!(sampler.total_count(), n);
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 20,
            ..Default::default()
        })]

        #[test]
        fn prop_reservoir_uniformity(
            capacity in 4usize..16,
            seed in 0u64..100,
        ) {
            let n = capacity * 50;
            let trials = 2000usize;
            let mut counts = vec![0usize; n];

            for t in 0..trials {
                let mut sampler = ReservoirSampler::<f64, 2>::new(capacity);
                sampler.rng = seed.wrapping_mul(6364136223846793005)
                    .wrapping_add((t as u64).wrapping_mul(1442695040888963407));
                for i in 0..n {
                    sampler.update(Point::new([i as f64, 0.0]));
                }
                for p in sampler.samples() {
                    let idx = p.coords()[0] as usize;
                    if idx < n {
                        counts[idx] += 1;
                    }
                }
            }

            let expected = (trials * capacity) as f64 / n as f64;
            let chi2: f64 = counts.iter()
                .map(|&c| {
                    let diff = c as f64 - expected;
                    diff * diff / expected
                })
                .sum();

            // Threshold is 3× degrees of freedom — catches gross non-uniformity
            // without being flaky.
            let threshold = 3.0 * (n - 1) as f64;
            prop_assert!(
                chi2 < threshold,
                "chi2={:.2} exceeds threshold={:.2} (capacity={}, n={}, trials={})",
                chi2, threshold, capacity, n, trials
            );
        }
    }

    #[test]
    fn reservoir_fills_to_capacity() {
        let mut s = ReservoirSampler::<f64, 2>::new(10);
        for i in 0..10 {
            s.update(Point::new([i as f64, 0.0]));
        }
        assert_eq!(s.len(), 10);
        assert_eq!(s.total_count(), 10);
    }

    #[test]
    fn reservoir_does_not_grow_beyond_capacity() {
        let mut s = ReservoirSampler::<f64, 2>::new(10);
        for i in 0..1000 {
            s.update(Point::new([i as f64, 0.0]));
            assert!(s.len() <= 10);
        }
        assert_eq!(s.total_count(), 1000);
    }

    #[test]
    fn reservoir_reset_clears_state() {
        let mut s = ReservoirSampler::<f64, 2>::new(10);
        for i in 0..20 {
            s.update(Point::new([i as f64, 0.0]));
        }
        s.reset();
        assert_eq!(s.len(), 0);
        assert_eq!(s.total_count(), 0);
    }
}
