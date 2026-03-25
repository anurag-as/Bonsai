use std::ops::{Add, Div, Mul, Sub};
use std::time::Instant;

/// Trait bound for coordinate scalars. Implemented for `f32` and `f64`.
pub trait CoordType:
    Copy
    + PartialOrd
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Into<f64>
    + From<f32>
{
    fn zero() -> Self;
    fn infinity() -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
}

impl CoordType for f32 {
    #[inline]
    fn zero() -> Self {
        0.0_f32
    }
    #[inline]
    fn infinity() -> Self {
        f32::INFINITY
    }
    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl CoordType for f64 {
    #[inline]
    fn zero() -> Self {
        0.0_f64
    }
    #[inline]
    fn infinity() -> Self {
        f64::INFINITY
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

/// A point in D-dimensional space with coordinate type `C`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point<C = f64, const D: usize = 2>([C; D]);

impl<C: CoordType, const D: usize> Point<C, D> {
    /// Construct a point from a fixed-size coordinate array.
    #[inline]
    pub fn new(coords: [C; D]) -> Self {
        Self(coords)
    }

    /// Return a reference to the underlying coordinate array.
    #[inline]
    pub fn coords(&self) -> &[C; D] {
        &self.0
    }

    /// Return a mutable reference to the underlying coordinate array.
    #[inline]
    pub fn coords_mut(&mut self) -> &mut [C; D] {
        &mut self.0
    }
}

/// Axis-aligned bounding box in D dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox<C = f64, const D: usize = 2> {
    pub min: Point<C, D>,
    pub max: Point<C, D>,
}

impl<C: CoordType, const D: usize> BBox<C, D> {
    /// Construct a bounding box from min and max points.
    #[inline]
    pub fn new(min: Point<C, D>, max: Point<C, D>) -> Self {
        Self { min, max }
    }

    /// Returns `true` iff `point` lies within this box on all D axes (inclusive).
    pub fn contains_point(&self, point: &Point<C, D>) -> bool {
        for d in 0..D {
            let coord = point.coords()[d];
            if coord < self.min.coords()[d] || coord > self.max.coords()[d] {
                return false;
            }
        }
        true
    }

    /// Returns `true` iff this bounding box overlaps `other` on all D axes.
    pub fn intersects(&self, other: &BBox<C, D>) -> bool {
        for d in 0..D {
            if self.max.coords()[d] < other.min.coords()[d]
                || other.max.coords()[d] < self.min.coords()[d]
            {
                return false;
            }
        }
        true
    }
}

/// Unique identifier for an index entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryId(pub u64);

/// Discriminant for the four spatial index backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    RTree,
    KDTree,
    Quadtree,
    Grid,
}

/// Snapshot of the query workload distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryMix {
    pub range_frac: f64,
    pub knn_frac: f64,
    pub join_frac: f64,
    pub mean_selectivity: f64,
}

impl Default for QueryMix {
    fn default() -> Self {
        Self {
            range_frac: 1.0,
            knn_frac: 0.0,
            join_frac: 0.0,
            mean_selectivity: 0.01,
        }
    }
}

/// Statistical snapshot of the current dataset and query workload.
#[derive(Debug, Clone)]
pub struct DataShape<const D: usize> {
    pub point_count: usize,
    pub bbox: BBox<f64, D>,
    /// Per-axis skewness; statically sized to D.
    pub skewness: [f64; D],
    /// Ratio of observed to uniform nearest-neighbour distance.
    pub clustering_coef: f64,
    /// Mean bounding-box overlap ratio.
    pub overlap_ratio: f64,
    /// Intrinsic dimensionality via correlation dimension.
    pub effective_dim: f64,
    pub query_mix: QueryMix,
}

/// Snapshot of index state — readable without acquiring any lock.
#[derive(Debug, Clone)]
pub struct Stats<const D: usize> {
    pub backend: BackendKind,
    pub point_count: usize,
    pub migrations: u64,
    pub last_migration_at: Option<Instant>,
    pub query_count: u64,
    pub data_shape: DataShape<D>,
    pub migrating: bool,
    pub dimensions: usize,
}

/// Error type for all fallible Bonsai operations.
#[derive(Debug)]
pub enum BonsaiError {
    NotFound(EntryId),
    Frozen,
    MigrationInProgress,
    Serialisation(String),
    Config(String),
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for BonsaiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BonsaiError::NotFound(id) => write!(f, "entry {:?} not found", id),
            BonsaiError::Frozen => {
                write!(
                    f,
                    "index is frozen — call unfreeze() to re-enable adaptation"
                )
            }
            BonsaiError::MigrationInProgress => write!(f, "migration already in progress"),
            BonsaiError::Serialisation(msg) => write!(f, "serialisation error: {}", msg),
            BonsaiError::Config(msg) => write!(f, "invalid configuration: {}", msg),
            BonsaiError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for BonsaiError {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Point coordinate array length equals D
    proptest! {
        #[test]
        fn prop_point_coord_len_d2(coords in prop::array::uniform2(-1.0e9_f64..1.0e9)) {
            let p = Point::<f64, 2>::new(coords);
            prop_assert_eq!(p.coords().len(), 2);
        }
    }

    proptest! {
        #[test]
        fn prop_point_coord_len_d3(
            a in -1.0e9_f64..1.0e9,
            b in -1.0e9_f64..1.0e9,
            c in -1.0e9_f64..1.0e9,
        ) {
            let p = Point::<f64, 3>::new([a, b, c]);
            prop_assert_eq!(p.coords().len(), 3);
        }
    }

    // DataShape skewness array length equals D
    proptest! {
        #[test]
        fn prop_datashape_skewness_len_d2(s0 in -10.0_f64..10.0, s1 in -10.0_f64..10.0) {
            let shape = DataShape::<2> {
                point_count: 0,
                bbox: BBox::new(Point::new([0.0, 0.0]), Point::new([1.0, 1.0])),
                skewness: [s0, s1],
                clustering_coef: 1.0,
                overlap_ratio: 0.0,
                effective_dim: 2.0,
                query_mix: QueryMix::default(),
            };
            prop_assert_eq!(shape.skewness.len(), 2);
        }
    }

    proptest! {
        #[test]
        fn prop_datashape_skewness_len_d3(
            s0 in -10.0_f64..10.0,
            s1 in -10.0_f64..10.0,
            s2 in -10.0_f64..10.0,
        ) {
            let shape = DataShape::<3> {
                point_count: 0,
                bbox: BBox::new(Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 1.0, 1.0])),
                skewness: [s0, s1, s2],
                clustering_coef: 1.0,
                overlap_ratio: 0.0,
                effective_dim: 3.0,
                query_mix: QueryMix::default(),
            };
            prop_assert_eq!(shape.skewness.len(), 3);
        }
    }

    // BBox point-in-region correctness
    proptest! {
        #[test]
        fn prop_bbox_contains_point_d2(
            min0 in -1.0e6_f64..0.0, min1 in -1.0e6_f64..0.0,
            max0 in 0.0_f64..1.0e6,  max1 in 0.0_f64..1.0e6,
            px in -1.5e6_f64..1.5e6, py in -1.5e6_f64..1.5e6,
        ) {
            let bbox = BBox::<f64, 2>::new(Point::new([min0, min1]), Point::new([max0, max1]));
            let point = Point::<f64, 2>::new([px, py]);
            let expected = px >= min0 && px <= max0 && py >= min1 && py <= max1;
            prop_assert_eq!(bbox.contains_point(&point), expected);
        }
    }

    proptest! {
        #[test]
        fn prop_bbox_contains_point_d3(
            min0 in -1.0e6_f64..0.0, min1 in -1.0e6_f64..0.0, min2 in -1.0e6_f64..0.0,
            max0 in 0.0_f64..1.0e6,  max1 in 0.0_f64..1.0e6,  max2 in 0.0_f64..1.0e6,
            px in -1.5e6_f64..1.5e6, py in -1.5e6_f64..1.5e6, pz in -1.5e6_f64..1.5e6,
        ) {
            let bbox = BBox::<f64, 3>::new(
                Point::new([min0, min1, min2]),
                Point::new([max0, max1, max2]),
            );
            let point = Point::<f64, 3>::new([px, py, pz]);
            let expected = px >= min0 && px <= max0
                && py >= min1 && py <= max1
                && pz >= min2 && pz <= max2;
            prop_assert_eq!(bbox.contains_point(&point), expected);
        }
    }

    #[test]
    fn bbox_contains_point_basic() {
        let bbox = BBox::<f64, 2>::new(Point::new([0.0, 0.0]), Point::new([1.0, 1.0]));
        assert!(bbox.contains_point(&Point::new([0.5, 0.5])));
        assert!(bbox.contains_point(&Point::new([0.0, 0.0]))); // min boundary
        assert!(bbox.contains_point(&Point::new([1.0, 1.0]))); // max boundary
        assert!(!bbox.contains_point(&Point::new([1.1, 0.5])));
        assert!(!bbox.contains_point(&Point::new([-0.1, 0.5])));
    }

    #[test]
    fn bbox_intersects_basic() {
        let a = BBox::<f64, 2>::new(Point::new([0.0, 0.0]), Point::new([2.0, 2.0]));
        let b = BBox::<f64, 2>::new(Point::new([1.0, 1.0]), Point::new([3.0, 3.0]));
        let c = BBox::<f64, 2>::new(Point::new([3.0, 3.0]), Point::new([4.0, 4.0]));
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn entry_id_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(EntryId(1));
        set.insert(EntryId(2));
        set.insert(EntryId(1));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn bonsai_error_display() {
        let e = BonsaiError::NotFound(EntryId(42));
        assert!(e.to_string().contains("42"));
        let e2 = BonsaiError::DimensionMismatch {
            expected: 3,
            got: 2,
        };
        assert!(e2.to_string().contains('3'));
    }
}
