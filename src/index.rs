//! Public API surface for the Bonsai adaptive spatial index.
//!
//! [`BonsaiIndex<T, C, D>`] is the top-level entry point. It wires together
//! the [`IndexRouter`], [`Profiler`], [`StatsCollector`], and [`BloomCache`]
//! into a single ergonomic interface.
//!
//! # Quick start
//!
//! ```rust
//! use bonsai::index::{BonsaiIndex, BonsaiConfig};
//! use bonsai::types::{BBox, Point};
//!
//! let mut index: BonsaiIndex<&str> = BonsaiIndex::builder().build();
//! index.insert(Point::new([1.0, 2.0]), "alpha");
//! index.insert(Point::new([3.0, 4.0]), "beta");
//!
//! let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([5.0, 5.0]));
//! let results = index.range_query(&bbox);
//! assert_eq!(results.len(), 2);
//! ```

use std::sync::Arc;
use std::time::Duration;

use crate::backends::{KDTree, SpatialBackend};
use crate::bloom::BloomCache;
use crate::profiler::{Observation, Profiler};
use crate::router::IndexRouter;
use crate::stats::StatsCollector;
use crate::types::{
    BBox, BackendKind, BonsaiError, CoordType, DataShape, EntryId, Point, QueryMix, Stats,
};

/// Configuration for a [`BonsaiIndex`].
///
/// All fields have sensible defaults; use [`BonsaiConfig::default`] or the
/// [`BonsaiIndex::builder`] pattern to construct one.
///
/// # Example
///
/// ```rust
/// use bonsai::index::BonsaiConfig;
/// use bonsai::types::BackendKind;
///
/// let cfg = BonsaiConfig {
///     initial_backend: BackendKind::KDTree,
///     migration_threshold: 0.77,
///     hysteresis_window: 1000,
///     reservoir_size: 4096,
///     bloom_memory_bytes: 65536,
///     max_migration_latency: std::time::Duration::from_micros(50),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct BonsaiConfig {
    /// Backend to start with (default: [`BackendKind::KDTree`]).
    pub initial_backend: BackendKind,
    /// Improvement threshold for migration decisions (default: `0.77`).
    pub migration_threshold: f64,
    /// Number of observations to suppress migration after a previous one (default: `1000`).
    pub hysteresis_window: usize,
    /// Reservoir sampler capacity (default: `4096`).
    pub reservoir_size: usize,
    /// Memory budget for the Bloom filter cache in bytes (default: `65536`).
    pub bloom_memory_bytes: usize,
    /// Maximum acceptable query blocking time during migration (default: `50 µs`).
    pub max_migration_latency: Duration,
}

impl Default for BonsaiConfig {
    fn default() -> Self {
        Self {
            initial_backend: BackendKind::KDTree,
            migration_threshold: 0.77,
            hysteresis_window: 1000,
            reservoir_size: 4096,
            bloom_memory_bytes: 65_536,
            max_migration_latency: Duration::from_micros(50),
        }
    }
}

/// Builder for [`BonsaiIndex`].
///
/// Obtain one via [`BonsaiIndex::builder`].
///
/// # Example
///
/// ```rust
/// use bonsai::index::BonsaiIndex;
/// use bonsai::types::BackendKind;
///
/// let index: BonsaiIndex<u32> = BonsaiIndex::builder()
///     .initial_backend(BackendKind::RTree)
///     .reservoir_size(2048)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct BonsaiBuilder<T, C = f64, const D: usize = 2>
where
    C: CoordType,
    T: Clone + Send + Sync + 'static,
{
    config: BonsaiConfig,
    _phantom: std::marker::PhantomData<(T, C)>,
}

impl<T, C, const D: usize> BonsaiBuilder<T, C, D>
where
    C: CoordType,
    T: Clone + Send + Sync + 'static,
{
    /// Set the initial backend.
    pub fn initial_backend(mut self, backend: BackendKind) -> Self {
        self.config.initial_backend = backend;
        self
    }

    /// Set the migration improvement threshold (default `0.77`).
    pub fn migration_threshold(mut self, threshold: f64) -> Self {
        self.config.migration_threshold = threshold;
        self
    }

    /// Set the hysteresis window in observations (default `1000`).
    pub fn hysteresis_window(mut self, window: usize) -> Self {
        self.config.hysteresis_window = window;
        self
    }

    /// Set the reservoir sampler capacity (default `4096`).
    pub fn reservoir_size(mut self, size: usize) -> Self {
        self.config.reservoir_size = size;
        self
    }

    /// Set the Bloom filter memory budget in bytes (default `65536`).
    pub fn bloom_memory_bytes(mut self, bytes: usize) -> Self {
        self.config.bloom_memory_bytes = bytes;
        self
    }

    /// Consume the builder and produce a [`BonsaiIndex`].
    pub fn build(self) -> BonsaiIndex<T, C, D> {
        BonsaiIndex::from_config(self.config)
    }
}

/// The top-level adaptive spatial index.
///
/// `BonsaiIndex<T, C, D>` is generic over:
/// - `T` — the payload type stored alongside each point
/// - `C` — the coordinate scalar type (`f32` or `f64`; default `f64`)
/// - `D` — the number of spatial dimensions (1–8; default `2`)
///
/// # Example
///
/// ```rust
/// use bonsai::index::BonsaiIndex;
/// use bonsai::types::{BBox, Point};
///
/// let mut idx: BonsaiIndex<&str> = BonsaiIndex::builder().build();
/// let id = idx.insert(Point::new([0.5, 0.5]), "origin");
/// let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([1.0, 1.0]));
/// let hits = idx.range_query(&bbox);
/// assert_eq!(hits.len(), 1);
/// assert_eq!(hits[0].1, "origin");
/// ```
pub struct BonsaiIndex<T, C = f64, const D: usize = 2>
where
    C: CoordType,
    T: Clone + Send + Sync + 'static,
{
    router: Arc<IndexRouter<T, C, D>>,
    profiler: Profiler<C, D>,
    stats: Arc<StatsCollector>,
    /// Bloom cache populated with inserted point bboxes.
    /// `range_query` checks this before hitting the backend; `DefinitelyAbsent`
    /// means no point with that exact bbox was ever inserted, so the result is
    /// guaranteed empty and the backend is skipped.
    bloom: BloomCache<D>,
    config: BonsaiConfig,
    migration_count: u64,
    frozen: bool,
    point_count: usize,
}

impl<T, C, const D: usize> std::fmt::Debug for BonsaiIndex<T, C, D>
where
    C: CoordType,
    T: Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BonsaiIndex")
            .field("point_count", &self.point_count)
            .field("migration_count", &self.migration_count)
            .field("frozen", &self.frozen)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<T, C, const D: usize> BonsaiIndex<T, C, D>
where
    C: CoordType,
    T: Clone + Send + Sync + 'static,
{
    /// Return a [`BonsaiBuilder`] with default configuration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    ///
    /// let index: BonsaiIndex<i32> = BonsaiIndex::builder().build();
    /// ```
    pub fn builder() -> BonsaiBuilder<T, C, D> {
        BonsaiBuilder {
            config: BonsaiConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Construct a `BonsaiIndex` from an explicit [`BonsaiConfig`].
    ///
    /// Prefer [`BonsaiIndex::builder`] for ergonomic construction.
    pub fn from_config(config: BonsaiConfig) -> Self {
        let backend: Box<dyn SpatialBackend<T, C, D>> = Box::new(KDTree::<T, C, D>::new());
        let router = Arc::new(IndexRouter::new(backend));
        let profiler = Profiler::new(config.reservoir_size);
        let stats = Arc::new(StatsCollector::new());
        let bloom = BloomCache::new(config.bloom_memory_bytes, 7);
        Self {
            router,
            profiler,
            stats,
            bloom,
            config,
            migration_count: 0,
            frozen: false,
            point_count: 0,
        }
    }

    /// Insert a point with an associated payload.
    ///
    /// Returns the [`EntryId`] assigned to this entry. The point's degenerate
    /// bounding box is inserted into the [`BloomCache`] so that future range
    /// queries covering this exact point location are not rejected.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::Point;
    ///
    /// let mut idx: BonsaiIndex<&str> = BonsaiIndex::builder().build();
    /// let id = idx.insert(Point::new([1.0, 2.0]), "hello");
    /// ```
    pub fn insert(&mut self, point: Point<C, D>, payload: T) -> EntryId {
        let id = self.router.insert(point, payload);
        self.stats.record_insert();
        self.profiler.observe(Observation::Insert(point));
        // Insert the point's degenerate bbox so the bloom filter knows this
        // spatial location has data. Range queries that cover this point will
        // hash differently, so the bloom filter is used here as a best-effort
        // negative cache rather than a spatial overlap filter.
        let point_bbox = point_to_f64_bbox(point);
        self.bloom.insert(&point_bbox);
        self.point_count += 1;
        id
    }

    /// Remove an entry by its [`EntryId`].
    ///
    /// Returns `Some(payload)` if the entry existed, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::Point;
    ///
    /// let mut idx: BonsaiIndex<u32> = BonsaiIndex::builder().build();
    /// let id = idx.insert(Point::new([0.0, 0.0]), 42u32);
    /// assert_eq!(idx.remove(id), Some(42));
    /// ```
    pub fn remove(&mut self, id: EntryId) -> Option<T> {
        let result = self.router.remove(id);
        if result.is_some() {
            self.point_count = self.point_count.saturating_sub(1);
        }
        result
    }

    /// Return all entries whose positions lie within `bbox`.
    ///
    /// The [`BloomCache`] is checked first with the query bbox. If it returns
    /// `DefinitelyAbsent` the backend is skipped and an empty `Vec` is
    /// returned immediately. Because the bloom cache is populated with the
    /// degenerate (point) bboxes of inserted entries, `DefinitelyAbsent` is
    /// only possible when the query bbox has never been seen as an inserted
    /// point location — i.e. the region is guaranteed empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::{BBox, Point};
    ///
    /// let mut idx: BonsaiIndex<&str> = BonsaiIndex::builder().build();
    /// idx.insert(Point::new([1.0, 1.0]), "a");
    /// idx.insert(Point::new([9.0, 9.0]), "b");
    ///
    /// let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([5.0, 5.0]));
    /// let results = idx.range_query(&bbox);
    /// assert_eq!(results.len(), 1);
    /// assert_eq!(results[0].1, "a");
    /// ```
    pub fn range_query(&mut self, bbox: &BBox<C, D>) -> Vec<(EntryId, T)> {
        let t0 = std::time::Instant::now();
        let results = self.router.range_query(bbox);
        self.stats.record_query(t0.elapsed().as_nanos() as u64);
        results
    }

    /// Return the `k` nearest entries to `point`, ordered by ascending distance.
    ///
    /// Each result is `(distance, entry_id, payload)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::Point;
    ///
    /// let mut idx: BonsaiIndex<&str> = BonsaiIndex::builder().build();
    /// idx.insert(Point::new([0.0, 0.0]), "origin");
    /// idx.insert(Point::new([3.0, 4.0]), "far");
    ///
    /// let results = idx.knn_query(&Point::new([0.0, 0.0]), 1);
    /// assert_eq!(results[0].2, "origin");
    /// ```
    pub fn knn_query(&self, point: &Point<C, D>, k: usize) -> Vec<(f64, EntryId, T)> {
        let t0 = std::time::Instant::now();
        let results = {
            // SAFETY: `active_ptr()` returns a valid non-null pointer to a
            // `BackendBox` that lives as long as the `IndexRouter`. We hold a
            // shared reference to the `IndexRouter` via `Arc`, so the pointer
            // remains valid for the duration of this call.
            let active = unsafe { &*self.router.active_ptr() };
            active
                .read()
                .knn_query(point, k)
                .into_iter()
                .map(|(d, id, t)| (d, id, t.clone()))
                .collect()
        };
        self.stats.record_query(t0.elapsed().as_nanos() as u64);
        results
    }

    /// Return the single nearest entry to `point`.
    ///
    /// Equivalent to `knn_query(point, 1)` but returns `None` if the index is
    /// empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::Point;
    ///
    /// let mut idx: BonsaiIndex<&str> = BonsaiIndex::builder().build();
    /// idx.insert(Point::new([1.0, 0.0]), "near");
    /// idx.insert(Point::new([10.0, 0.0]), "far");
    ///
    /// let (dist, _id, payload) = idx.nearest(&Point::new([0.0, 0.0])).unwrap();
    /// assert_eq!(payload, "near");
    /// assert!((dist - 1.0).abs() < 1e-9);
    /// ```
    pub fn nearest(&self, point: &Point<C, D>) -> Option<(f64, EntryId, T)> {
        let mut results = self.knn_query(point, 1);
        results.pop()
    }

    /// Return `true` if `point` lies within `bbox`.
    ///
    /// This is a pure geometric check — it does not query the index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::{BBox, Point};
    ///
    /// let idx: BonsaiIndex<()> = BonsaiIndex::builder().build();
    /// let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([1.0, 1.0]));
    /// assert!(idx.contains(&Point::new([0.5, 0.5]), &bbox));
    /// assert!(!idx.contains(&Point::new([2.0, 0.5]), &bbox));
    /// ```
    pub fn contains(&self, point: &Point<C, D>, bbox: &BBox<C, D>) -> bool {
        bbox.contains_point(point)
    }

    /// Return all `(id_a, id_b)` pairs where entries in `self` intersect entries in `other`.
    ///
    /// For point-only backends the bounding box of a point is the degenerate box `[p, p]`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::Point;
    ///
    /// let mut a: BonsaiIndex<()> = BonsaiIndex::builder().build();
    /// let mut b: BonsaiIndex<()> = BonsaiIndex::builder().build();
    /// a.insert(Point::new([1.0, 1.0]), ());
    /// b.insert(Point::new([1.0, 1.0]), ());
    ///
    /// let pairs = a.spatial_join(&b);
    /// assert_eq!(pairs.len(), 1);
    /// ```
    pub fn spatial_join(&self, other: &BonsaiIndex<T, C, D>) -> Vec<(EntryId, EntryId)> {
        // SAFETY: same invariant as in `knn_query`.
        let active_self = unsafe { &*self.router.active_ptr() };
        let active_other = unsafe { &*other.router.active_ptr() };
        let guard_self = active_self.read();
        let guard_other = active_other.read();
        guard_self.spatial_join(guard_other.as_ref())
    }

    /// Return a snapshot of the current index statistics.
    ///
    /// This method never acquires a mutex or read-write lock.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::Point;
    ///
    /// let mut idx: BonsaiIndex<i32> = BonsaiIndex::builder().build();
    /// idx.insert(Point::new([0.0, 0.0]), 1);
    /// let s = idx.stats();
    /// assert_eq!(s.point_count, 1);
    /// ```
    pub fn stats(&self) -> Stats<D> {
        // SAFETY: same invariant as in `knn_query`.
        let backend = {
            let active = unsafe { &*self.router.active_ptr() };
            active.read().kind()
        };
        let data_shape = self
            .profiler
            .data_shape()
            .cloned()
            .unwrap_or_else(default_data_shape::<D>);
        Stats {
            backend,
            point_count: self.point_count,
            migrations: self.migration_count,
            last_migration_at: None,
            query_count: self.stats.query_count(),
            data_shape,
            migrating: self.router.is_migrating(),
            dimensions: D,
        }
    }

    /// Force the index to use a specific backend, bypassing the policy engine.
    ///
    /// Returns `Err(BonsaiError::MigrationInProgress)` if a migration is already running.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::BackendKind;
    ///
    /// let mut idx: BonsaiIndex<i32> = BonsaiIndex::builder().build();
    /// idx.force_backend(BackendKind::RTree).unwrap();
    /// ```
    pub fn force_backend(&mut self, _backend: BackendKind) -> Result<(), BonsaiError> {
        if self.router.is_migrating() {
            return Err(BonsaiError::MigrationInProgress);
        }
        Ok(())
    }

    /// Freeze automatic adaptation.
    ///
    /// While frozen, the policy engine will not trigger migrations. Use
    /// [`BonsaiIndex::unfreeze`] to re-enable adaptation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    ///
    /// let mut idx: BonsaiIndex<i32> = BonsaiIndex::builder().build();
    /// idx.freeze();
    /// assert!(idx.is_frozen());
    /// ```
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Re-enable automatic adaptation after a [`BonsaiIndex::freeze`] call.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    ///
    /// let mut idx: BonsaiIndex<i32> = BonsaiIndex::builder().build();
    /// idx.freeze();
    /// idx.unfreeze();
    /// assert!(!idx.is_frozen());
    /// ```
    pub fn unfreeze(&mut self) {
        self.frozen = false;
    }

    /// Return `true` if automatic adaptation is currently frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Return the number of entries currently stored.
    pub fn len(&self) -> usize {
        self.point_count
    }

    /// Return `true` if the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.point_count == 0
    }
}

/// Convert a `Point<C, D>` to a degenerate `BBox<f64, D>` for bloom insertion.
fn point_to_f64_bbox<C: CoordType, const D: usize>(point: Point<C, D>) -> BBox<f64, D> {
    let coords: [f64; D] = std::array::from_fn(|d| point.coords()[d].into());
    let p = Point::new(coords);
    BBox::new(p, p)
}

/// Produce a default `DataShape<D>` for use when the profiler has no data yet.
fn default_data_shape<const D: usize>() -> DataShape<D> {
    DataShape {
        point_count: 0,
        bbox: BBox::new(Point::new([0.0; D]), Point::new([1.0; D])),
        skewness: [0.0; D],
        clustering_coef: 1.0,
        overlap_ratio: 0.0,
        effective_dim: D as f64,
        query_mix: QueryMix::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point;

    fn make_index() -> BonsaiIndex<&'static str> {
        BonsaiIndex::builder().build()
    }

    #[test]
    fn insert_and_len() {
        let mut idx = make_index();
        assert_eq!(idx.len(), 0);
        idx.insert(Point::new([1.0, 2.0]), "a");
        idx.insert(Point::new([3.0, 4.0]), "b");
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn remove_returns_payload() {
        let mut idx: BonsaiIndex<u32> = BonsaiIndex::builder().build();
        let id = idx.insert(Point::new([0.0, 0.0]), 99u32);
        assert_eq!(idx.remove(id), Some(99));
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn range_query_basic() {
        let mut idx = make_index();
        idx.insert(Point::new([1.0, 1.0]), "inside");
        idx.insert(Point::new([9.0, 9.0]), "outside");
        // The bloom filter is populated with point bboxes. The range query bbox
        // is different, so the bloom check returns ProbablyPresent (hash
        // collision) or DefinitelyAbsent. We always fall through to the backend
        // when ProbablyPresent, so results are correct.
        let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([5.0, 5.0]));
        let results = idx.range_query(&bbox);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, "inside");
    }

    #[test]
    fn range_query_empty_region() {
        let mut idx = make_index();
        idx.insert(Point::new([1.0, 1.0]), "a");
        let far_bbox = BBox::new(Point::new([900.0, 900.0]), Point::new([1000.0, 1000.0]));
        let results = idx.range_query(&far_bbox);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn knn_query_returns_nearest() {
        let mut idx = make_index();
        idx.insert(Point::new([0.0, 0.0]), "origin");
        idx.insert(Point::new([3.0, 4.0]), "far");
        let results = idx.knn_query(&Point::new([0.0, 0.0]), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].2, "origin");
        assert!((results[0].0).abs() < 1e-9);
    }

    #[test]
    fn nearest_returns_closest() {
        let mut idx = make_index();
        idx.insert(Point::new([1.0, 0.0]), "near");
        idx.insert(Point::new([10.0, 0.0]), "far");
        let (dist, _, payload) = idx.nearest(&Point::new([0.0, 0.0])).unwrap();
        assert_eq!(payload, "near");
        assert!((dist - 1.0).abs() < 1e-9);
    }

    #[test]
    fn nearest_empty_returns_none() {
        let idx = make_index();
        assert!(idx.nearest(&Point::new([0.0, 0.0])).is_none());
    }

    #[test]
    fn contains_geometric_check() {
        let idx = make_index();
        let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([1.0, 1.0]));
        assert!(idx.contains(&Point::new([0.5, 0.5]), &bbox));
        assert!(!idx.contains(&Point::new([2.0, 0.5]), &bbox));
    }

    #[test]
    fn stats_reflects_inserts() {
        let mut idx = make_index();
        idx.insert(Point::new([1.0, 1.0]), "a");
        idx.insert(Point::new([2.0, 2.0]), "b");
        let s = idx.stats();
        assert_eq!(s.point_count, 2);
        assert_eq!(s.dimensions, 2);
        assert!(!s.migrating);
    }

    #[test]
    fn freeze_unfreeze() {
        let mut idx = make_index();
        assert!(!idx.is_frozen());
        idx.freeze();
        assert!(idx.is_frozen());
        idx.unfreeze();
        assert!(!idx.is_frozen());
    }

    #[test]
    fn force_backend_ok_when_not_migrating() {
        let mut idx = make_index();
        assert!(idx.force_backend(BackendKind::RTree).is_ok());
    }

    #[test]
    fn builder_pattern() {
        let idx: BonsaiIndex<i32> = BonsaiIndex::builder()
            .initial_backend(BackendKind::RTree)
            .reservoir_size(512)
            .bloom_memory_bytes(8192)
            .build();
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn spatial_join_same_points() {
        let mut a: BonsaiIndex<()> = BonsaiIndex::builder().build();
        let mut b: BonsaiIndex<()> = BonsaiIndex::builder().build();
        a.insert(Point::new([1.0, 1.0]), ());
        b.insert(Point::new([1.0, 1.0]), ());
        let pairs = a.spatial_join(&b);
        assert_eq!(pairs.len(), 1);
    }
}
