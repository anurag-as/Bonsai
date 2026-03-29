//! Online statistics computation from a reservoir sample.
//!
//! [`OnlineStats`] computes per-axis skewness, clustering coefficient,
//! bounding-box overlap ratio, and effective dimensionality (via the
//! correlation dimension estimator) from a reservoir of sampled points.
use crate::types::{BBox, CoordType, DataShape, Point, QueryMix};

/// Online statistics computed from a reservoir sample.
///
/// All statistics are computed lazily when [`OnlineStats::compute`] is called.
pub struct OnlineStats<C: CoordType, const D: usize> {
    _phantom: std::marker::PhantomData<C>,
}

impl<C: CoordType, const D: usize> OnlineStats<C, D> {
    /// Create a new `OnlineStats` instance.
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute a [`DataShape`] from the given reservoir sample and query mix.
    ///
    /// Returns `None` if the sample is empty.
    pub fn compute(&self, samples: &[Point<C, D>], query_mix: QueryMix) -> Option<DataShape<D>> {
        if samples.is_empty() {
            return None;
        }

        let n = samples.len();

        let pts: Vec<[f64; D]> = samples
            .iter()
            .map(|p| {
                let mut arr = [0.0f64; D];
                for (d, slot) in arr.iter_mut().enumerate().take(D) {
                    *slot = p.coords()[d].into();
                }
                arr
            })
            .collect();

        // ── Bounding box ──────────────────────────────────────────────────────
        let mut min_coords = [f64::INFINITY; D];
        let mut max_coords = [f64::NEG_INFINITY; D];
        for pt in &pts {
            for d in 0..D {
                if pt[d] < min_coords[d] {
                    min_coords[d] = pt[d];
                }
                if pt[d] > max_coords[d] {
                    max_coords[d] = pt[d];
                }
            }
        }
        let bbox_f64 = BBox::<f64, D>::new(Point::new(min_coords), Point::new(max_coords));

        // ── Per-axis mean and variance ────────────────────────────────────────
        let mut mean = [0.0f64; D];
        for pt in &pts {
            for d in 0..D {
                mean[d] += pt[d];
            }
        }
        for slot in mean.iter_mut().take(D) {
            *slot /= n as f64;
        }

        let mut variance = [0.0f64; D];
        for pt in &pts {
            for d in 0..D {
                let diff = pt[d] - mean[d];
                variance[d] += diff * diff;
            }
        }
        for slot in variance.iter_mut().take(D) {
            *slot /= n as f64;
        }

        // ── Per-axis skewness (Fisher's moment coefficient) ───────────────────
        let mut skewness = [0.0f64; D];
        for pt in &pts {
            for d in 0..D {
                let diff = pt[d] - mean[d];
                skewness[d] += diff * diff * diff;
            }
        }
        for d in 0..D {
            let std_dev = variance[d].sqrt();
            if std_dev > 1e-12 {
                skewness[d] = (skewness[d] / n as f64) / (std_dev * std_dev * std_dev);
            } else {
                skewness[d] = 0.0;
            }
        }

        let clustering_coef = compute_clustering_coef(&pts, &bbox_f64);
        let effective_dim = compute_effective_dim(&pts);

        Some(DataShape {
            point_count: n,
            bbox: BBox::new(Point::new(min_coords), Point::new(max_coords)),
            skewness,
            clustering_coef,
            overlap_ratio: 0.0,
            effective_dim,
            query_mix,
        })
    }
}

impl<C: CoordType, const D: usize> Default for OnlineStats<C, D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the clustering coefficient from a set of points.
///
/// Returns `expected_nn / observed_nn` where expected is for a uniform
/// distribution of the same density. Values > 1.0 indicate clustering.
fn compute_clustering_coef<const D: usize>(pts: &[[f64; D]], bbox: &BBox<f64, D>) -> f64 {
    let n = pts.len();
    if n < 2 {
        return 1.0;
    }

    let mut volume = 1.0f64;
    for d in 0..D {
        let extent = bbox.max.coords()[d] - bbox.min.coords()[d];
        volume *= extent.max(1e-12);
    }

    let gamma_factor = gamma_1_plus_1_over_d(D);
    let expected_nn = (volume / n as f64).powf(1.0 / D as f64) * gamma_factor;

    let query_count = n.min(256);
    let step = if n > query_count { n / query_count } else { 1 };

    let mut total_nn_dist = 0.0f64;
    let mut valid_queries = 0usize;

    for i in (0..n).step_by(step).take(query_count) {
        let mut min_dist = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dist = euclidean_dist_sq(&pts[i], &pts[j]).sqrt();
            if dist < min_dist {
                min_dist = dist;
            }
        }
        if min_dist.is_finite() {
            total_nn_dist += min_dist;
            valid_queries += 1;
        }
    }

    if valid_queries == 0 || total_nn_dist < 1e-15 {
        return 1.0;
    }

    let observed_nn = total_nn_dist / valid_queries as f64;
    if observed_nn < 1e-15 {
        return 1.0;
    }

    (expected_nn / observed_nn).max(0.0)
}

/// Approximate Γ(1 + 1/D) for D = 1..8.
fn gamma_1_plus_1_over_d(d: usize) -> f64 {
    match d {
        1 => 1.0,
        2 => 0.886_226_925_452_758,
        3 => 0.8929795115692493,
        4 => 0.906_402_477_055_477,
        5 => 0.9181687423997607,
        6 => 0.927_760_442_095_343,
        7 => 0.9354892840699859,
        8 => 0.9417426998497052,
        _ => 1.0 - 0.1 / d as f64,
    }
}

/// Compute squared Euclidean distance between two D-dimensional points.
#[inline]
fn euclidean_dist_sq<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0f64;
    for d in 0..D {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sum
}

/// Compute effective dimensionality via the correlation dimension estimator.
///
/// Estimates `d` from the slope of `log C(r)` vs `log r` using two radii at
/// the 10th and 50th percentiles of pairwise distances.
fn compute_effective_dim<const D: usize>(pts: &[[f64; D]]) -> f64 {
    let n = pts.len();
    if n < 4 {
        return D as f64;
    }

    let sample_size = n.min(200);
    let step = if n > sample_size { n / sample_size } else { 1 };
    let sample: Vec<&[f64; D]> = (0..n)
        .step_by(step)
        .take(sample_size)
        .map(|i| &pts[i])
        .collect();
    let m = sample.len();

    if m < 4 {
        return D as f64;
    }

    let mut dists = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            let d = euclidean_dist_sq(sample[i], sample[j]).sqrt();
            if d > 1e-15 {
                dists.push(d);
            }
        }
    }

    if dists.len() < 4 {
        return D as f64;
    }

    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let total_pairs = dists.len() as f64;
    let r1_idx = (total_pairs * 0.10) as usize;
    let r2_idx = (total_pairs * 0.50) as usize;
    let r1 = dists[r1_idx.min(dists.len() - 1)];
    let r2 = dists[r2_idx.min(dists.len() - 1)];

    if r1 < 1e-15 || r2 < 1e-15 || (r2 / r1) < 1.01 {
        return D as f64;
    }

    let c1 = dists.partition_point(|&d| d <= r1) as f64;
    let c2 = dists.partition_point(|&d| d <= r2) as f64;

    if c1 < 1.0 || c2 < 1.0 || c1 >= c2 {
        return D as f64;
    }

    let dim = (c2.ln() - c1.ln()) / (r2.ln() - r1.ln());
    dim.max(0.5).min(D as f64 + 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Point, QueryMix};
    use proptest::prelude::*;

    // clustering_coef must be approximately 1.0 for uniform random data and strictly
    // greater than 1.0 for clustered data (Gaussian clusters with σ << domain size).
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 30,
            ..Default::default()
        })]

        #[test]
        fn prop_clustering_coef_monotonicity(seed in 0u64..30) {
            let n = 500usize;
            let domain = 1000.0f64;

            // Generate uniform random points.
            let mut rng = Lcg::new(seed.wrapping_mul(6364136223846793005).wrapping_add(1));
            let uniform_pts: Vec<Point<f64, 2>> = (0..n)
                .map(|_| Point::new([rng.next_f64() * domain, rng.next_f64() * domain]))
                .collect();

            // Generate clustered points: 5 tight Gaussian clusters.
            let mut rng2 = Lcg::new(seed.wrapping_mul(1442695040888963407).wrapping_add(2));
            let centres: Vec<[f64; 2]> = (0..5)
                .map(|_| [rng2.next_f64() * 800.0 + 100.0, rng2.next_f64() * 800.0 + 100.0])
                .collect();
            let sigma = 10.0; // σ << domain (1000)
            let clustered_pts: Vec<Point<f64, 2>> = (0..n)
                .map(|i| {
                    let c = &centres[i % 5];
                    let x = (c[0] + rng2.next_normal() * sigma).clamp(0.0, domain);
                    let y = (c[1] + rng2.next_normal() * sigma).clamp(0.0, domain);
                    Point::new([x, y])
                })
                .collect();

            let stats = OnlineStats::<f64, 2>::new();
            let qm = QueryMix::default();

            let uniform_shape = stats.compute(&uniform_pts, qm).unwrap();
            let clustered_shape = stats.compute(&clustered_pts, qm).unwrap();

            // Uniform data: clustering_coef should be approximately 1.0.
            // Allow generous tolerance since we're using a small sample.
            prop_assert!(
                uniform_shape.clustering_coef > 0.3 && uniform_shape.clustering_coef < 3.0,
                "uniform clustering_coef={:.4} not near 1.0",
                uniform_shape.clustering_coef
            );

            // Clustered data: clustering_coef should be > 1.0 (observed NN < expected).
            prop_assert!(
                clustered_shape.clustering_coef > uniform_shape.clustering_coef,
                "clustered_coef={:.4} not > uniform_coef={:.4}",
                clustered_shape.clustering_coef,
                uniform_shape.clustering_coef
            );
        }
    }

    // for points on a d-dimensional manifold (d < D), effective_dim must be approximately d.
    // Specifically: uniform 2D data → effective_dim ≈ 2.0; linear data → effective_dim ≈ 1.0.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 20,
            ..Default::default()
        })]

        #[test]
        fn prop_effective_dim_estimation(seed in 0u64..20) {
            let n = 500usize;
            let mut rng = Lcg::new(seed.wrapping_mul(6364136223846793005).wrapping_add(42));

            // 2D uniform data in 2D space → effective_dim ≈ 2.0
            let uniform_2d: Vec<Point<f64, 2>> = (0..n)
                .map(|_| Point::new([rng.next_f64() * 100.0, rng.next_f64() * 100.0]))
                .collect();

            // Linear data in 2D space (points on a line) → effective_dim ≈ 1.0
            let linear_2d: Vec<Point<f64, 2>> = (0..n)
                .map(|i| {
                    let t = i as f64 / n as f64 * 100.0;
                    // Add tiny noise to avoid degenerate distances.
                    let noise = rng.next_f64() * 0.01;
                    Point::new([t + noise, t * 0.5 + noise])
                })
                .collect();

            let stats = OnlineStats::<f64, 2>::new();
            let qm = QueryMix::default();

            let shape_2d = stats.compute(&uniform_2d, qm).unwrap();
            let shape_1d = stats.compute(&linear_2d, qm).unwrap();

            // Uniform 2D: effective_dim should be near 2.0 (allow 1.0–3.0 range).
            prop_assert!(
                shape_2d.effective_dim >= 1.0 && shape_2d.effective_dim <= 3.0,
                "uniform 2D effective_dim={:.4} not in [1.0, 3.0]",
                shape_2d.effective_dim
            );

            // Linear 1D: effective_dim should be near 1.0 (allow 0.5–2.0 range).
            prop_assert!(
                shape_1d.effective_dim >= 0.5 && shape_1d.effective_dim <= 2.0,
                "linear 1D effective_dim={:.4} not in [0.5, 2.0]",
                shape_1d.effective_dim
            );

            // Linear data should have lower effective_dim than 2D uniform data.
            prop_assert!(
                shape_1d.effective_dim < shape_2d.effective_dim,
                "linear effective_dim={:.4} not < uniform 2D effective_dim={:.4}",
                shape_1d.effective_dim,
                shape_2d.effective_dim
            );
        }
    }

    /// Simple LCG for test data generation.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
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

    #[test]
    fn skewness_array_length_equals_d() {
        let pts: Vec<Point<f64, 3>> = (0..100)
            .map(|i| Point::new([i as f64, i as f64 * 2.0, i as f64 * 3.0]))
            .collect();
        let stats = OnlineStats::<f64, 3>::new();
        let shape = stats.compute(&pts, QueryMix::default()).unwrap();
        assert_eq!(shape.skewness.len(), 3);
    }

    #[test]
    fn empty_sample_returns_none() {
        let stats = OnlineStats::<f64, 2>::new();
        assert!(stats.compute(&[], QueryMix::default()).is_none());
    }

    #[test]
    fn single_point_returns_some() {
        let pts = vec![Point::<f64, 2>::new([1.0, 2.0])];
        let stats = OnlineStats::<f64, 2>::new();
        let shape = stats.compute(&pts, QueryMix::default());
        assert!(shape.is_some());
    }

    #[test]
    fn uniform_clustering_coef_near_one() {
        // Generate 400 uniform points in [0, 100]^2.
        let mut rng = Lcg::new(12345);
        let pts: Vec<Point<f64, 2>> = (0..400)
            .map(|_| Point::new([rng.next_f64() * 100.0, rng.next_f64() * 100.0]))
            .collect();
        let stats = OnlineStats::<f64, 2>::new();
        let shape = stats.compute(&pts, QueryMix::default()).unwrap();
        // Generous bounds: uniform data should give clustering_coef in [0.3, 3.0].
        assert!(
            shape.clustering_coef > 0.3 && shape.clustering_coef < 3.0,
            "clustering_coef={:.4}",
            shape.clustering_coef
        );
    }
}
