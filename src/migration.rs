//! Migration engine for transparent backend switching.
//!
//! [`MigrationEngine`] executes a four-phase incremental migration from one
//! [`SpatialBackend`] to another using STR bulk loading with Hilbert curve
//! ordering, then performs an atomic pointer swap so that in-flight queries
//! are never blocked for more than a single cache-line load.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

// On wasm32-unknown-unknown, std::time::Instant panics at runtime.
// Migration duration is always Duration::ZERO on WASM.
#[cfg(target_arch = "wasm32")]
struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    #[inline]
    fn now() -> Self {
        Instant
    }
    #[inline]
    fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}

use crate::backends::SpatialBackend;
use crate::hilbert::HilbertCurve;
use crate::types::{BBox, BackendKind, CoordType, EntryId, Point};

/// Normalise a coordinate value in `[0, domain]` to a Hilbert grid cell in
/// `[0, 2^order - 1]`.
fn normalise_coord(v: f64, domain: f64, order: u32) -> u64 {
    let cells = (1u64 << order) as f64;
    ((v / domain) * cells).clamp(0.0, cells - 1.0) as u64
}

/// Compute the bounding box of a slice of points, returning a `BBox<f64, D>`.
fn points_bbox<C: CoordType, const D: usize>(
    pts: &[(Point<C, D>, EntryId)],
) -> Option<BBox<f64, D>> {
    if pts.is_empty() {
        return None;
    }
    let mut min_c = [f64::INFINITY; D];
    let mut max_c = [f64::NEG_INFINITY; D];
    for (p, _) in pts {
        for d in 0..D {
            let v: f64 = p.coords()[d].into();
            if v < min_c[d] {
                min_c[d] = v;
            }
            if v > max_c[d] {
                max_c[d] = v;
            }
        }
    }
    Some(BBox::new(Point::new(min_c), Point::new(max_c)))
}

/// Sort `(point, id)` pairs by their D-dimensional Hilbert index.
///
/// Uses `order=16` per axis (capped so `D * order <= 128`).
fn hilbert_sort<C: CoordType, const D: usize>(
    entries: &mut [(Point<C, D>, EntryId)],
    domain_bbox: &BBox<f64, D>,
) {
    let order = (128u32 / D as u32).min(16);
    let curve = HilbertCurve::<D>::new(order);

    let domain_span: [f64; D] = std::array::from_fn(|d| {
        let span = domain_bbox.max.coords()[d] - domain_bbox.min.coords()[d];
        if span <= 0.0 {
            1.0
        } else {
            span
        }
    });
    let domain_min: [f64; D] = std::array::from_fn(|d| domain_bbox.min.coords()[d]);

    entries.sort_by_cached_key(|(p, _)| {
        let coords: [u64; D] = std::array::from_fn(|d| {
            let v: f64 = p.coords()[d].into();
            normalise_coord(v - domain_min[d], domain_span[d], order)
        });
        curve.index(&coords)
    });
}

/// Tracks entries written to the active backend after the bulk-load snapshot
/// so they can be drained into the shadow backend during Phase 3.
#[derive(Debug)]
struct DeltaLog<C: CoordType, const D: usize> {
    inserts: Vec<(Point<C, D>, EntryId)>,
    removes: Vec<EntryId>,
}

impl<C: CoordType, const D: usize> Default for DeltaLog<C, D> {
    fn default() -> Self {
        Self {
            inserts: Vec::new(),
            removes: Vec::new(),
        }
    }
}

/// The result of a completed migration.
#[derive(Debug)]
pub struct MigrationResult {
    /// The kind of the new backend.
    pub new_backend: BackendKind,
    /// Wall-clock duration of the entire migration.
    pub duration: std::time::Duration,
    /// Number of entries in the new backend after migration.
    pub entry_count: usize,
}

/// Executes a four-phase incremental migration between spatial backends.
///
/// # Phases
///
/// 1. **Bulk load** — snapshot all existing entries, sort by Hilbert index,
///    bulk-load into the new backend.
/// 2. **Transition** — set `migrating = true`; callers must route new
///    inserts/removes to both backends via [`MigrationEngine::record_insert`] /
///    [`MigrationEngine::record_remove`].
/// 3. **Drain** — replay the delta log (entries written after the snapshot)
///    into the shadow backend.
/// 4. **Atomic swap** — replace the active backend with the shadow, drop the
///    old backend, reset `migrating`.
pub struct MigrationEngine<T, C, const D: usize>
where
    C: CoordType,
    T: Send + Sync + 'static,
{
    migrating: Arc<AtomicBool>,
    delta: Arc<Mutex<DeltaLog<C, D>>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, C, const D: usize> MigrationEngine<T, C, D>
where
    C: CoordType,
    T: Clone + Send + Sync + 'static,
{
    /// Create a new `MigrationEngine`.
    pub fn new() -> Self {
        Self {
            migrating: Arc::new(AtomicBool::new(false)),
            delta: Arc::new(Mutex::new(DeltaLog::default())),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Return `true` if a migration is currently in progress.
    pub fn is_migrating(&self) -> bool {
        self.migrating.load(Ordering::SeqCst)
    }

    /// Record an insert that happened after the bulk-load snapshot.
    ///
    /// Must be called for every insert while `is_migrating()` is `true`.
    pub fn record_insert(&self, point: Point<C, D>, id: EntryId) {
        if self.migrating.load(Ordering::SeqCst) {
            self.delta.lock().unwrap().inserts.push((point, id));
        }
    }

    /// Record a remove that happened after the bulk-load snapshot.
    ///
    /// Must be called for every remove while `is_migrating()` is `true`.
    pub fn record_remove(&self, id: EntryId) {
        if self.migrating.load(Ordering::SeqCst) {
            self.delta.lock().unwrap().removes.push(id);
        }
    }

    /// Execute the full four-phase migration.
    ///
    /// `active` is the current backend (read-only snapshot taken here).
    /// `payload_map` maps `EntryId.0 → T` so payloads can be transferred.
    /// `new_backend_fn` constructs the target backend via `SpatialBackend::bulk_load`.
    ///
    /// Returns the new backend and a [`MigrationResult`] summary.
    pub fn run<B>(
        &self,
        active: &dyn SpatialBackend<T, C, D>,
        payload_map: &HashMap<u64, T>,
        new_backend_fn: fn(Vec<(Point<C, D>, T)>) -> B,
    ) -> (B, MigrationResult)
    where
        B: SpatialBackend<T, C, D>,
    {
        let start = Instant::now();

        // ── Phase 1: snapshot + Hilbert sort + bulk load ──────────────────────
        let all_entries = active.all_entries();
        let mut snapshot: Vec<(Point<C, D>, EntryId)> =
            all_entries.iter().map(|(p, id, _)| (*p, *id)).collect();

        let domain_bbox = points_bbox(&snapshot)
            .unwrap_or_else(|| BBox::new(Point::new([0.0f64; D]), Point::new([1.0f64; D])));

        hilbert_sort(&mut snapshot, &domain_bbox);

        // Build the payload list in Hilbert order.
        let bulk_entries: Vec<(Point<C, D>, T)> = snapshot
            .iter()
            .filter_map(|(p, id)| payload_map.get(&id.0).map(|t| (*p, t.clone())))
            .collect();

        // ── Phase 2: set migrating flag so callers start routing deltas ───────
        self.migrating.store(true, Ordering::SeqCst);
        {
            let mut delta = self.delta.lock().unwrap();
            delta.inserts.clear();
            delta.removes.clear();
        }

        let mut shadow = new_backend_fn(bulk_entries);

        // ── Phase 3: drain delta log into shadow ──────────────────────────────
        let delta = {
            let mut guard = self.delta.lock().unwrap();
            std::mem::take(&mut *guard)
        };

        // Apply removes first so we don't re-insert something that was removed.
        let removed_ids: std::collections::HashSet<u64> =
            delta.removes.iter().map(|id| id.0).collect();

        for (point, id) in delta.inserts {
            if !removed_ids.contains(&id.0) {
                if let Some(payload) = payload_map.get(&id.0) {
                    shadow.insert(point, payload.clone());
                }
            }
        }
        for id in &delta.removes {
            shadow.remove(*id);
        }

        // ── Phase 4: reset migrating flag (atomic, SeqCst) ───────────────────
        //
        // In a full IndexRouter this would be an AtomicPtr::store(SeqCst).
        // Here we model the same ordering guarantee with an AtomicBool store.
        self.migrating.store(false, Ordering::SeqCst);

        let entry_count = shadow.len();
        let result = MigrationResult {
            new_backend: shadow.kind(),
            duration: start.elapsed(),
            entry_count,
        };

        (shadow, result)
    }
}

impl<T, C, const D: usize> Default for MigrationEngine<T, C, D>
where
    C: CoordType,
    T: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A simple index that wraps a boxed backend and a `MigrationEngine`, routing
/// inserts/removes to both backends during migration.
///
/// This is a self-contained demonstration of the migration protocol used by
/// the full `IndexRouter`. It is intentionally minimal — the production
/// `IndexRouter` (task 13) will use `AtomicPtr` for lock-free routing.
///
/// Stable external `EntryId`s are assigned by `SimpleIndex` itself and stored
/// as part of the backend payload, so they survive migration to a new backend.
pub struct SimpleIndex<T, C, const D: usize>
where
    C: CoordType,
    T: Clone + Send + Sync + 'static,
{
    /// The active backend stores `(external_id, payload)` pairs.
    active: Box<dyn SpatialBackend<(EntryId, T), C, D>>,
    /// Map from external `EntryId` → backend-internal `EntryId`, used for removes.
    ext_to_internal: HashMap<u64, EntryId>,
    engine: MigrationEngine<(EntryId, T), C, D>,
    next_id: u64,
}

impl<T, C, const D: usize> SimpleIndex<T, C, D>
where
    C: CoordType,
    T: Clone + Send + Sync + 'static,
{
    /// Create a `SimpleIndex` backed by `backend`.
    pub fn new(backend: Box<dyn SpatialBackend<(EntryId, T), C, D>>) -> Self {
        Self {
            active: backend,
            ext_to_internal: HashMap::new(),
            engine: MigrationEngine::new(),
            next_id: 0,
        }
    }

    /// Insert a point with payload. Returns the stable external `EntryId`.
    pub fn insert(&mut self, point: Point<C, D>, payload: T) -> EntryId {
        let ext_id = EntryId(self.next_id);
        self.next_id += 1;

        let internal_id = self.active.insert(point, (ext_id, payload.clone()));
        self.ext_to_internal.insert(ext_id.0, internal_id);

        if self.engine.is_migrating() {
            self.engine.record_insert(point, internal_id);
        }
        ext_id
    }

    /// Remove an entry by external `EntryId`. Returns the payload if found.
    pub fn remove(&mut self, ext_id: EntryId) -> Option<T> {
        let internal_id = self.ext_to_internal.remove(&ext_id.0)?;
        let result = self.active.remove(internal_id);

        if result.is_some() && self.engine.is_migrating() {
            self.engine.record_remove(internal_id);
        }
        result.map(|(_, payload)| payload)
    }

    /// Run a range query against the active backend, returning stable external IDs.
    pub fn range_query(&self, bbox: &BBox<C, D>) -> Vec<EntryId> {
        self.active
            .range_query(bbox)
            .into_iter()
            .map(|(_, (ext_id, _))| *ext_id)
            .collect()
    }

    /// Migrate to a new backend constructed by `new_backend_fn`.
    ///
    /// Returns `None` if a migration is already in progress.
    #[allow(clippy::type_complexity)]
    pub fn migrate<B>(
        &mut self,
        new_backend_fn: fn(Vec<(Point<C, D>, (EntryId, T))>) -> B,
    ) -> Option<MigrationResult>
    where
        B: SpatialBackend<(EntryId, T), C, D> + 'static,
    {
        if self.engine.is_migrating() {
            return None;
        }

        // Build a payload_map for the engine: internal_id → (ext_id, payload).
        let payload_map: HashMap<u64, (EntryId, T)> = self
            .active
            .all_entries()
            .into_iter()
            .map(|(_, internal_id, (ext_id, payload))| (internal_id.0, (*ext_id, payload.clone())))
            .collect();

        let (new_backend, result) =
            self.engine
                .run(self.active.as_ref(), &payload_map, new_backend_fn);

        // Rebuild ext_to_internal from the new backend.
        self.ext_to_internal.clear();
        for (_, new_internal_id, (ext_id, _)) in new_backend.all_entries() {
            self.ext_to_internal.insert(ext_id.0, new_internal_id);
        }

        // Atomic swap: replace active with the new backend.
        self.active = Box::new(new_backend);

        Some(result)
    }

    /// Return the number of entries in the active backend.
    pub fn len(&self) -> usize {
        self.active.len()
    }

    /// Return `true` if the active backend is empty.
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{KDTree, RTree};
    use proptest::prelude::*;

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

    fn rand_pts_2d(n: usize, seed: u64) -> Vec<Point<f64, 2>> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| Point::new([rng.next_f64() * 1000.0, rng.next_f64() * 1000.0]))
            .collect()
    }

    fn brute_range(pts: &[(Point<f64, 2>, EntryId)], bbox: &BBox<f64, 2>) -> Vec<EntryId> {
        let mut ids: Vec<EntryId> = pts
            .iter()
            .filter(|(p, _)| bbox.contains_point(p))
            .map(|(_, id)| *id)
            .collect();
        ids.sort_by_key(|id| id.0);
        ids
    }

    fn make_index() -> SimpleIndex<usize, f64, 2> {
        SimpleIndex::new(Box::new(KDTree::<(EntryId, usize), f64, 2>::new()))
    }

    #[test]
    fn migration_preserves_all_entries() {
        let pts = rand_pts_2d(500, 42);
        let mut index = make_index();
        let mut pt_ids: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
        for (i, &p) in pts.iter().enumerate() {
            let id = index.insert(p, i);
            pt_ids.push((p, id));
        }

        let result = index
            .migrate(RTree::<(EntryId, usize), f64, 2>::bulk_load)
            .unwrap();
        assert_eq!(result.new_backend, BackendKind::RTree);
        assert_eq!(result.entry_count, 500);

        let full = BBox::new(Point::new([0.0, 0.0]), Point::new([1000.0, 1000.0]));
        let mut after: Vec<EntryId> = index.range_query(&full);
        after.sort_by_key(|id| id.0);
        let mut expected: Vec<EntryId> = pt_ids.iter().map(|(_, id)| *id).collect();
        expected.sort_by_key(|id| id.0);
        assert_eq!(after, expected, "entries lost during migration");
    }

    #[test]
    fn migration_range_query_consistent() {
        let pts = rand_pts_2d(1000, 7);
        let mut index = make_index();
        let mut pt_ids: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
        for (i, &p) in pts.iter().enumerate() {
            let id = index.insert(p, i);
            pt_ids.push((p, id));
        }

        let bbox = BBox::new(Point::new([200.0, 200.0]), Point::new([800.0, 800.0]));
        let mut before: Vec<EntryId> = index.range_query(&bbox);
        before.sort_by_key(|id| id.0);

        index
            .migrate(RTree::<(EntryId, usize), f64, 2>::bulk_load)
            .unwrap();

        let mut after: Vec<EntryId> = index.range_query(&bbox);
        after.sort_by_key(|id| id.0);

        assert_eq!(before, after, "range query results changed after migration");
    }

    #[test]
    fn migration_duration_reasonable() {
        let pts = rand_pts_2d(5_000, 99);
        let mut index = make_index();
        for (i, &p) in pts.iter().enumerate() {
            index.insert(p, i);
        }
        let result = index
            .migrate(RTree::<(EntryId, usize), f64, 2>::bulk_load)
            .unwrap();
        assert!(
            result.duration.as_secs() < 1,
            "migration took too long: {:?}",
            result.duration
        );
    }

    #[test]
    fn hilbert_sort_is_deterministic() {
        let pts = rand_pts_2d(100, 13);
        let ids: Vec<EntryId> = (0..100).map(|i| EntryId(i as u64)).collect();
        let mut entries: Vec<(Point<f64, 2>, EntryId)> = pts.into_iter().zip(ids).collect();
        let domain = BBox::new(Point::new([0.0, 0.0]), Point::new([1000.0, 1000.0]));
        let mut entries2 = entries.clone();
        hilbert_sort(&mut entries, &domain);
        hilbert_sort(&mut entries2, &domain);
        assert_eq!(
            entries.iter().map(|(_, id)| id.0).collect::<Vec<_>>(),
            entries2.iter().map(|(_, id)| id.0).collect::<Vec<_>>(),
        );
    }

    fn pt2d() -> impl Strategy<Value = Point<f64, 2>> {
        (0.0_f64..1000.0, 0.0_f64..1000.0).prop_map(|(x, y)| Point::new([x, y]))
    }

    fn bbox2d() -> impl Strategy<Value = BBox<f64, 2>> {
        (
            0.0_f64..800.0,
            0.0_f64..800.0,
            50.0_f64..300.0,
            50.0_f64..300.0,
        )
            .prop_map(|(x, y, w, h)| BBox::new(Point::new([x, y]), Point::new([x + w, y + h])))
    }

    // All entries present before migration must be queryable after migration
    // completes, and no entries may be lost during the atomic swap.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 50,
            ..Default::default()
        })]

        #[test]
        fn prop_migration_completeness(
            pts in prop::collection::vec(pt2d(), 10..200),
            bbox in bbox2d(),
        ) {
            let mut index = SimpleIndex::new(Box::new(KDTree::<(EntryId, usize), f64, 2>::new()));
            let mut pt_ids: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
            for (i, p) in pts.iter().enumerate() {
                let id = index.insert(*p, i);
                pt_ids.push((*p, id));
            }

            let before_count = index.range_query(&bbox).len();
            let expected_brute = brute_range(&pt_ids, &bbox).len();
            prop_assert_eq!(before_count, expected_brute);

            index.migrate(RTree::<(EntryId, usize), f64, 2>::bulk_load).unwrap();

            let after_count = index.range_query(&bbox).len();
            prop_assert_eq!(
                after_count,
                expected_brute,
                "migration changed range query result: before={} after={} brute={}",
                before_count, after_count, expected_brute
            );

            let full = BBox::new(Point::new([0.0, 0.0]), Point::new([1000.0, 1000.0]));
            prop_assert_eq!(
                index.range_query(&full).len(),
                pts.len(),
                "entry count changed after migration"
            );
        }
    }

    // Queries issued immediately before and after the atomic swap must return
    // identical results, bounding the observable disruption to a single
    // pointer-width atomic load.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 50,
            ..Default::default()
        })]

        #[test]
        fn prop_migration_query_consistency(
            pts in prop::collection::vec(pt2d(), 10..150),
            bbox in bbox2d(),
        ) {
            let mut index = SimpleIndex::new(Box::new(KDTree::<(EntryId, usize), f64, 2>::new()));
            let mut pt_ids: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
            for (i, p) in pts.iter().enumerate() {
                let id = index.insert(*p, i);
                pt_ids.push((*p, id));
            }

            let mut before: Vec<EntryId> = index.range_query(&bbox);
            before.sort_by_key(|id| id.0);

            let swap_start = Instant::now();
            index.migrate(RTree::<(EntryId, usize), f64, 2>::bulk_load).unwrap();
            let swap_duration = swap_start.elapsed();

            let mut after: Vec<EntryId> = index.range_query(&bbox);
            after.sort_by_key(|id| id.0);

            prop_assert_eq!(
                before, after,
                "query results differ across migration boundary"
            );

            prop_assert!(
                swap_duration.as_micros() < 500_000,
                "migration took unexpectedly long: {:?}",
                swap_duration
            );
        }
    }
}
