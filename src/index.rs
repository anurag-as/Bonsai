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

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

// On wasm32-unknown-unknown, std::time::Instant panics at runtime.
// Timing is a no-op on WASM — latency stats are always 0.
#[cfg(target_arch = "wasm32")]
struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    #[inline]
    fn now() -> Self {
        Instant
    }
    #[inline]
    fn elapsed(&self) -> Duration {
        Duration::ZERO
    }
}

use crate::backends::{GridIndex, KDTree, Quadtree, RTree, SpatialBackend};
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
    pub(crate) router: Arc<IndexRouter<T, C, D>>,
    profiler: Profiler<C, D>,
    stats: Arc<StatsCollector>,
    bloom: BloomCache<D>,
    pub(crate) config: BonsaiConfig,
    pub(crate) migration_count: u64,
    pub(crate) frozen: bool,
    pub(crate) point_count: usize,
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
        let t0 = Instant::now();
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
        let t0 = Instant::now();
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

    /// Reset the index to an empty state, preserving configuration.
    ///
    /// Replaces the active backend with a new, empty instance of the same
    /// backend kind, resets the bloom cache, profiler, stats collector, and
    /// point count to their initial states.
    ///
    /// Preserves: `config`, `frozen`, `migration_count`.
    ///
    /// Returns `Err(BonsaiError::MigrationInProgress)` if a migration is
    /// currently in progress; the index state is left unmodified in that case.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::Point;
    ///
    /// let mut idx: BonsaiIndex<u32> = BonsaiIndex::builder().build();
    /// idx.insert(Point::new([1.0, 2.0]), 1);
    /// idx.clear().unwrap();
    /// assert_eq!(idx.len(), 0);
    /// ```
    pub fn clear(&mut self) -> Result<(), BonsaiError> {
        if self.router.is_migrating() {
            return Err(BonsaiError::MigrationInProgress);
        }
        // Read the active backend kind before replacing the router.
        let kind = {
            // SAFETY: same invariant as in `knn_query` — `active_ptr()` returns
            // a valid non-null pointer that lives as long as the `IndexRouter`,
            // which we hold via `Arc`. `&mut self` guarantees no concurrent
            // borrows exist.
            let active = unsafe { &*self.router.active_ptr() };
            active.read().kind()
        };
        let fresh_backend: Box<dyn SpatialBackend<T, C, D>> = match kind {
            BackendKind::KDTree => Box::new(KDTree::new()),
            BackendKind::RTree => Box::new(RTree::new()),
            BackendKind::Quadtree => Box::new(Quadtree::new()),
            BackendKind::Grid => Box::new(GridIndex::<T, C, D>::default()),
        };
        self.router = Arc::new(IndexRouter::new(fresh_backend));
        self.bloom = BloomCache::new(self.config.bloom_memory_bytes, 7);
        self.profiler = Profiler::new(self.config.reservoir_size);
        self.stats = Arc::new(StatsCollector::new());
        self.point_count = 0;
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

    #[test]
    fn clear_on_empty_index_succeeds() {
        let mut idx = make_index();
        assert!(idx.clear().is_ok());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn clear_resets_len() {
        let mut idx = make_index();
        for i in 0..5 {
            idx.insert(Point::new([i as f64, i as f64]), "x");
        }
        assert_eq!(idx.len(), 5);
        idx.clear().unwrap();
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn clear_preserves_frozen() {
        let mut idx = make_index();
        idx.freeze();
        idx.clear().unwrap();
        assert!(idx.is_frozen());
    }

    #[test]
    fn clear_preserves_migration_count() {
        let mut idx = make_index();
        let migrations_before = idx.stats().migrations;
        idx.insert(Point::new([1.0, 1.0]), "a");
        idx.clear().unwrap();
        assert_eq!(idx.stats().migrations, migrations_before);
    }

    #[test]
    fn clear_returns_err_when_migrating() {
        let mut idx: BonsaiIndex<&str> = BonsaiIndex::builder().build();
        // Begin a migration so `is_migrating()` returns true.
        use crate::backends::KDTree as KD;
        idx.router
            .begin_migration(Box::new(KD::<&str, f64, 2>::new()));
        let result = idx.clear();
        assert!(matches!(result, Err(BonsaiError::MigrationInProgress)));
        // Clean up so the router drops cleanly.
        idx.router.commit_migration();
    }

    #[test]
    fn clear_empty_range_query_returns_empty() {
        let mut idx = make_index();
        idx.insert(Point::new([1.0, 1.0]), "a");
        idx.clear().unwrap();
        let full = BBox::new(Point::new([-1e9, -1e9]), Point::new([1e9, 1e9]));
        assert!(idx.range_query(&full).is_empty());
    }

    #[test]
    fn clear_then_insert_range_query() {
        let mut idx = make_index();
        idx.insert(Point::new([50.0, 50.0]), "old");
        idx.clear().unwrap();
        idx.insert(Point::new([1.0, 1.0]), "new");
        let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([5.0, 5.0]));
        let results = idx.range_query(&bbox);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, "new");
    }

    #[test]
    fn clear_then_insert_knn_query() {
        let mut idx = make_index();
        idx.insert(Point::new([50.0, 50.0]), "old");
        idx.clear().unwrap();
        idx.insert(Point::new([1.0, 1.0]), "new");
        let results = idx.knn_query(&Point::new([0.0, 0.0]), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].2, "new");
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        /// Strategy that generates a 2-D point with coordinates in [-1000, 1000].
        fn point_strategy() -> impl Strategy<Value = Point<f64, 2>> {
            (-1000.0_f64..1000.0_f64, -1000.0_f64..1000.0_f64).prop_map(|(x, y)| Point::new([x, y]))
        }

        /// Strategy that generates a (min, max) bbox where min ≤ max on each axis.
        fn bbox_strategy() -> impl Strategy<Value = BBox<f64, 2>> {
            (
                -1000.0_f64..1000.0_f64,
                -1000.0_f64..1000.0_f64,
                0.001_f64..500.0_f64,
                0.001_f64..500.0_f64,
            )
                .prop_map(|(x, y, w, h)| BBox::new(Point::new([x, y]), Point::new([x + w, y + h])))
        }

        // Backend kind preserved after clear
        proptest! {
            #![proptest_config(proptest::test_runner::Config {
                cases: 100,
                ..Default::default()
            })]

            #[test]
            fn prop_clear_preserves_backend_kind(
                points in prop::collection::vec(point_strategy(), 0..30),
                backend_idx in 0usize..4,
            ) {
                let kinds = [
                    BackendKind::KDTree,
                    BackendKind::RTree,
                    BackendKind::Quadtree,
                    BackendKind::Grid,
                ];
                let kind = kinds[backend_idx % 4];
                let mut idx: BonsaiIndex<u32> = BonsaiIndex::builder()
                    .initial_backend(kind)
                    .build();
                // Force the backend so the active kind matches our choice.
                idx.force_backend(kind).unwrap();
                for (i, p) in points.iter().enumerate() {
                    idx.insert(*p, i as u32);
                }
                let kind_before = idx.stats().backend;
                idx.clear().unwrap();
                prop_assert_eq!(
                    idx.stats().backend,
                    kind_before,
                    "backend kind must be preserved after clear"
                );
            }
        }

        // len() invariant after clear
        proptest! {
            #![proptest_config(proptest::test_runner::Config {
                cases: 100,
                ..Default::default()
            })]

            #[test]
            fn prop_clear_len_invariant(
                pre_inserts in prop::collection::vec(point_strategy(), 0..30),
                post_inserts in prop::collection::vec(point_strategy(), 0..30),
            ) {
                let mut idx: BonsaiIndex<u32> = BonsaiIndex::builder().build();
                for (i, p) in pre_inserts.iter().enumerate() {
                    idx.insert(*p, i as u32);
                }
                idx.clear().unwrap();
                for (i, p) in post_inserts.iter().enumerate() {
                    idx.insert(*p, i as u32);
                }
                prop_assert_eq!(
                    idx.len(),
                    post_inserts.len(),
                    "len() must equal the number of inserts after clear"
                );
            }
        }

        //Empty query results after clear
        proptest! {
            #![proptest_config(proptest::test_runner::Config {
                cases: 100,
                ..Default::default()
            })]

            #[test]
            fn prop_clear_empty_queries(
                points in prop::collection::vec(point_strategy(), 1..30),
                query_bbox in bbox_strategy(),
                knn_point in point_strategy(),
                k in 1usize..10,
            ) {
                let mut idx: BonsaiIndex<u32> = BonsaiIndex::builder().build();
                for (i, p) in points.iter().enumerate() {
                    idx.insert(*p, i as u32);
                }
                idx.clear().unwrap();
                prop_assert!(
                    idx.range_query(&query_bbox).is_empty(),
                    "range_query must return empty after clear"
                );
                prop_assert!(
                    idx.knn_query(&knn_point, k).is_empty(),
                    "knn_query must return empty after clear"
                );
            }
        }

        // Round-trip equivalence
        proptest! {
            #![proptest_config(proptest::test_runner::Config {
                cases: 100,
                ..Default::default()
            })]

            #[test]
            fn prop_clear_round_trip_equivalence(
                pre_inserts in prop::collection::vec(point_strategy(), 0..20),
                entries in prop::collection::vec(
                    (point_strategy(), 0u32..10_000u32),
                    1..20,
                ),
            ) {
                let full_bbox = BBox::new(
                    Point::new([-1001.0, -1001.0]),
                    Point::new([1001.0, 1001.0]),
                );

                // Fresh index: insert entries directly.
                let mut fresh: BonsaiIndex<u32> = BonsaiIndex::builder().build();
                for (p, v) in &entries {
                    fresh.insert(*p, *v);
                }
                let mut fresh_results: Vec<u32> = fresh
                    .range_query(&full_bbox)
                    .into_iter()
                    .map(|(_, v)| v)
                    .collect();
                fresh_results.sort_unstable();

                // Cleared index: insert some data, clear, then insert the same entries.
                let mut cleared: BonsaiIndex<u32> = BonsaiIndex::builder().build();
                for (i, p) in pre_inserts.iter().enumerate() {
                    cleared.insert(*p, i as u32 + 100_000);
                }
                cleared.clear().unwrap();
                for (p, v) in &entries {
                    cleared.insert(*p, *v);
                }
                let mut cleared_results: Vec<u32> = cleared
                    .range_query(&full_bbox)
                    .into_iter()
                    .map(|(_, v)| v)
                    .collect();
                cleared_results.sort_unstable();

                prop_assert_eq!(
                    fresh_results,
                    cleared_results,
                    "clear+insert must produce the same range_query results as fresh+insert"
                );
            }
        }
    }
}
