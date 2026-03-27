//! Spatial backend trait and implementations.
//!
//! All four spatial index backends implement the [`SpatialBackend`] trait,
//! which provides a uniform interface for insert, remove, range query, kNN
//! query, spatial join, and bulk loading.

use crate::types::{BBox, BackendKind, CoordType, EntryId, Point};

pub mod grid;
pub mod kdtree;
pub mod quadtree;
pub mod rtree;

pub use grid::GridIndex;
pub use kdtree::KDTree;
pub use quadtree::Quadtree;
pub use rtree::RTree;

/// Uniform interface implemented by all spatial index backends.
///
/// # Type Parameters
/// - `T`: payload type stored alongside each point
/// - `C`: coordinate scalar type (must implement [`CoordType`])
/// - `D`: number of spatial dimensions (const generic)
pub trait SpatialBackend<T, C, const D: usize>: Send + Sync
where
    C: CoordType,
{
    /// Insert a point with an associated payload. Returns the unique [`EntryId`]
    /// assigned to this entry.
    fn insert(&mut self, point: Point<C, D>, payload: T) -> EntryId;

    /// Remove the entry with the given [`EntryId`]. Returns the payload if the
    /// entry existed, or `None` otherwise.
    fn remove(&mut self, id: EntryId) -> Option<T>;

    /// Return all entries whose positions are contained within `bbox`.
    fn range_query(&self, bbox: &BBox<C, D>) -> Vec<(EntryId, &T)>;

    /// Return the `k` nearest entries to `point`, ordered by ascending
    /// Euclidean distance. Each result is `(distance, id, payload_ref)`.
    fn knn_query(&self, point: &Point<C, D>, k: usize) -> Vec<(f64, EntryId, &T)>;

    /// Return all pairs `(id_a, id_b)` where the bounding box of entry `id_a`
    /// in `self` intersects the bounding box of entry `id_b` in `other`.
    ///
    /// For point-only backends the "bounding box" of a point is the degenerate
    /// box `[p, p]`.
    fn spatial_join(&self, other: &dyn SpatialBackend<T, C, D>) -> Vec<(EntryId, EntryId)>;

    /// Bulk-load a set of `(point, payload)` pairs into a new backend instance.
    ///
    /// Implementations may sort entries (e.g. by Hilbert index) for better
    /// spatial locality.
    fn bulk_load(entries: Vec<(Point<C, D>, T)>) -> Self
    where
        Self: Sized;

    /// Return the number of entries currently stored.
    fn len(&self) -> usize;

    /// Return `true` if the backend contains no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the [`BackendKind`] discriminant for this backend.
    fn kind(&self) -> BackendKind;

    /// Return all `(point, id, payload)` triples — used by `spatial_join`
    /// implementations that need to iterate over the other backend's entries.
    fn all_entries(&self) -> Vec<(Point<C, D>, EntryId, &T)>;
}
