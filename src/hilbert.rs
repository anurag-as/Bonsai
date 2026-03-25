//! D-dimensional Hilbert curve index computation via the Butz (1971) algorithm.
//!
//! [`HilbertCurve<D>`] maps a D-dimensional integer coordinate (each axis in
//! `[0, 2^order - 1]`) to a single `u128` Hilbert index. The bit budget is
//! `D * order` bits, which must not exceed 128.
//!
//! | D | order | total bits |
//! |---|-------|-----------|
//! | 2 | 32    | 64        |
//! | 4 | 16    | 64        |
//! | 6 | 16    | 96        |
//! | 8 | 16    | 128       |
//!
//! # Example
//!
//! ```rust
//! use bonsai::hilbert::HilbertCurve;
//!
//! let curve = HilbertCurve::<2>::new(4); // 4-bit order → 16×16 grid
//! let idx = curve.index(&[3, 5]);
//! println!("Hilbert index for (3,5): {}", idx);
//! ```

/// A D-dimensional Hilbert curve with configurable bit precision per axis.
///
/// `order` is the number of bits per dimension; the total index width is
/// `D * order` bits, which must be ≤ 128.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HilbertCurve<const D: usize> {
    /// Bits of precision per dimension.
    pub order: u32,
}

impl<const D: usize> HilbertCurve<D> {
    /// Create a new `HilbertCurve` with the given `order` (bits per dimension).
    ///
    /// # Panics
    ///
    /// Panics if `D == 0`, `order == 0`, or `D * order > 128`.
    #[inline]
    pub fn new(order: u32) -> Self {
        assert!(D > 0, "D must be at least 1");
        assert!(order > 0, "order must be at least 1");
        assert!(
            D as u32 * order <= 128,
            "D * order = {} exceeds 128-bit budget",
            D as u32 * order
        );
        Self { order }
    }

    /// Return the default order for this `D` such that `D * order <= 128`.
    ///
    /// Prefers `order = 32` for D ≤ 4 and `order = 16` for D ≤ 8, matching
    /// the bit-budget table in the design document.
    #[inline]
    pub fn default_order() -> u32 {
        let max_order = 128u32 / D as u32;
        if max_order >= 32 { 32 } else { max_order.min(16) }
    }

    /// Compute the Hilbert index for the given D-dimensional integer coordinate.
    ///
    /// Each coordinate value must be in `[0, 2^order - 1]`. Values outside
    /// this range are silently masked to `order` bits.
    ///
    /// Returns a `u128` Hilbert index in `[0, 2^(D*order) - 1]`.
    pub fn index(&self, coords: &[u64; D]) -> u128 {
        let order = self.order as usize;
        let mask = if order >= 64 { u64::MAX } else { (1u64 << order) - 1 };

        let mut x = [0u64; D];
        for d in 0..D {
            x[d] = coords[d] & mask;
        }

        let mut hilbert: u128 = 0;
        let mut state: u64 = 0;

        for i in (0..order).rev() {
            let mut entry: u64 = 0;
            for (d, &xd) in x.iter().enumerate() {
                if (xd >> i) & 1 == 1 {
                    entry |= 1 << d;
                }
            }
            let child = Self::rotate_right(entry ^ state, state, D);
            hilbert = (hilbert << D) | child as u128;
            state = Self::next_state(state, child, D);
        }

        hilbert
    }

    #[inline]
    fn rotate_right(bits: u64, amount: u64, width: usize) -> u64 {
        let w = width as u64;
        let amt = amount % w;
        let mask = (1u64 << width) - 1;
        ((bits >> amt) | (bits << (w - amt))) & mask
    }

    #[inline]
    fn next_state(state: u64, child: u64, width: usize) -> u64 {
        let w = width as u64;
        let mask = (1u64 << width) - 1;
        let inv_gray = Self::inverse_gray(child);
        let t = (state ^ inv_gray) & mask;
        let rot = (child + 1).trailing_zeros() as u64 % w;
        Self::rotate_right(t, rot, width)
    }

    #[inline]
    fn inverse_gray(mut n: u64) -> u64 {
        let mut mask = n >> 1;
        while mask != 0 {
            n ^= mask;
            mask >>= 1;
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn d2_order1_bijection() {
        let c = HilbertCurve::<2>::new(1);
        let v00 = c.index(&[0, 0]);
        let v10 = c.index(&[1, 0]);
        let v11 = c.index(&[1, 1]);
        let v01 = c.index(&[0, 1]);
        let mut vals = [v00, v10, v11, v01];
        vals.sort();
        assert_eq!(vals, [0, 1, 2, 3]);
        assert_eq!(v00, 0);
    }

    #[test]
    fn d2_order2_reference_values() {
        let c = HilbertCurve::<2>::new(2);
        assert_eq!(c.index(&[0, 0]), 0);
        assert_eq!(c.index(&[3, 3]), 13);
        assert_eq!(c.index(&[0, 3]), 10);
        assert_eq!(c.index(&[3, 2]), 15);
    }

    #[test]
    fn d2_order2_all_distinct() {
        let c = HilbertCurve::<2>::new(2);
        let mut seen = std::collections::HashSet::new();
        for x in 0u64..4 {
            for y in 0u64..4 {
                let idx = c.index(&[x, y]);
                assert!(idx < 16, "index {idx} out of range for 4×4 grid");
                assert!(seen.insert(idx), "duplicate index {idx} at ({x},{y})");
            }
        }
        assert_eq!(seen.len(), 16);
    }

    #[test]
    fn d3_order1_all_distinct() {
        let c = HilbertCurve::<3>::new(1);
        let mut seen = std::collections::HashSet::new();
        for x in 0u64..2 {
            for y in 0u64..2 {
                for z in 0u64..2 {
                    let idx = c.index(&[x, y, z]);
                    assert!(idx < 8, "index {idx} out of range for 2×2×2 grid");
                    assert!(seen.insert(idx), "duplicate index {idx} at ({x},{y},{z})");
                }
            }
        }
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn d3_order2_all_distinct() {
        let c = HilbertCurve::<3>::new(2);
        let mut seen = std::collections::HashSet::new();
        for x in 0u64..4 {
            for y in 0u64..4 {
                for z in 0u64..4 {
                    let idx = c.index(&[x, y, z]);
                    assert!(idx < 64, "index {idx} out of range");
                    assert!(seen.insert(idx), "duplicate index {idx} at ({x},{y},{z})");
                }
            }
        }
        assert_eq!(seen.len(), 64);
    }

    #[test]
    fn d8_order16_no_overflow() {
        let c = HilbertCurve::<8>::new(16);
        let max_coord = (1u64 << 16) - 1;
        c.index(&[max_coord; 8]); // must not panic or overflow
    }

    #[test]
    fn d8_order16_zero_is_zero() {
        let c = HilbertCurve::<8>::new(16);
        assert_eq!(c.index(&[0u64; 8]), 0);
    }

    #[test]
    fn default_order_within_budget() {
        fn check<const D: usize>() {
            let order = HilbertCurve::<D>::default_order();
            assert!(D as u32 * order <= 128, "D={D} order={order} exceeds 128-bit budget");
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<5>();
        check::<6>();
        check::<7>();
        check::<8>();
    }

    // Feature: bonsai-spatial-index, Property 23: Hilbert spatial locality
    // For any two points that are spatially close in D-dimensional space,
    // their Hilbert indices must be numerically closer on average than the
    // Hilbert indices of two randomly chosen points from the same dataset.
    // Validates: Requirements 11.5
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_hilbert_spatial_locality_d2(
            bx in 2u64..62,
            by in 2u64..62,
            dx in 0u64..3,
            dy in 0u64..3,
            far_xs in prop::collection::vec(0u64..64, 20),
            far_ys in prop::collection::vec(0u64..64, 20),
        ) {
            let curve = HilbertCurve::<2>::new(6);
            let idx_base = curve.index(&[bx, by]);
            let idx_near = curve.index(&[bx + dx, by + dy]);
            let close_diff = idx_base.abs_diff(idx_near);

            let n = far_xs.len().min(far_ys.len());
            let total_far_diff: u128 = (0..n)
                .map(|i| idx_base.abs_diff(curve.index(&[far_xs[i], far_ys[i]])))
                .fold(0u128, |acc, d| acc.saturating_add(d));
            let avg_far_diff = total_far_diff / n as u128;

            prop_assert!(
                close_diff <= avg_far_diff * 4 + 1,
                "close_diff={close_diff} avg_far_diff={avg_far_diff} base=({bx},{by}) near=({},{})",
                bx + dx, by + dy
            );
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_hilbert_spatial_locality_d3(
            bx in 2u64..30,
            by in 2u64..30,
            bz in 2u64..30,
            dx in 0u64..2,
            dy in 0u64..2,
            dz in 0u64..2,
            far_xs in prop::collection::vec(0u64..32, 20),
            far_ys in prop::collection::vec(0u64..32, 20),
            far_zs in prop::collection::vec(0u64..32, 20),
        ) {
            let curve = HilbertCurve::<3>::new(5);
            let idx_base = curve.index(&[bx, by, bz]);
            let idx_near = curve.index(&[bx + dx, by + dy, bz + dz]);
            let close_diff = idx_base.abs_diff(idx_near);

            let n = far_xs.len().min(far_ys.len()).min(far_zs.len());
            let total_far_diff: u128 = (0..n)
                .map(|i| idx_base.abs_diff(curve.index(&[far_xs[i], far_ys[i], far_zs[i]])))
                .fold(0u128, |acc, d| acc.saturating_add(d));
            let avg_far_diff = total_far_diff / n as u128;

            prop_assert!(
                close_diff <= avg_far_diff * 4 + 1,
                "close_diff={close_diff} avg_far_diff={avg_far_diff} base=({bx},{by},{bz}) near=({},{},{})",
                bx + dx, by + dy, bz + dz
            );
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 50,
            ..Default::default()
        })]

        #[test]
        fn prop_hilbert_injective_d2(
            x1 in 0u64..4, y1 in 0u64..4,
            x2 in 0u64..4, y2 in 0u64..4,
        ) {
            let c = HilbertCurve::<2>::new(2);
            let i1 = c.index(&[x1, y1]);
            let i2 = c.index(&[x2, y2]);
            if x1 == x2 && y1 == y2 {
                prop_assert_eq!(i1, i2);
            } else {
                prop_assert_ne!(i1, i2, "({},{}) and ({},{}) map to same index {}", x1, y1, x2, y2, i1);
            }
        }
    }
}
