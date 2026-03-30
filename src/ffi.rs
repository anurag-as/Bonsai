//! C FFI layer for `BonsaiIndex` (feature = "ffi").
//!
//! Exposes a 2-D, f64-coordinate, opaque-payload index to C callers.
//! All functions follow C naming conventions and use caller-provided output
//! buffers so that no heap allocation is required on the C side.
//!
//! # Safety contract
//!
//! Every `unsafe` block in this module carries a `// SAFETY:` comment.
//! Callers must uphold the following invariants:
//!
//! - `BonsaiHandle` pointers must originate from `bonsai_new` and must not be
//!   used after `bonsai_free`.
//! - Output buffer pointers (`out_ids`, `out_dist`) must point to valid,
//!   writable memory of at least `capacity` / `k` elements respectively.

#![cfg(feature = "ffi")]

use std::ffi::c_char;
use std::os::raw::c_int;

use crate::index::{BonsaiConfig, BonsaiIndex};
use crate::types::{BBox, BackendKind, EntryId, Point};

/// Opaque handle to a heap-allocated `BonsaiIndex<OpaquePayload, f64, 2>`.
pub struct BonsaiHandle(BonsaiIndex<OpaquePayload, f64, 2>);

// SAFETY: `*mut c_void` is `Send + Sync` under the C API contract: the caller
// is responsible for external synchronisation of the payload pointer.
unsafe impl Send for BonsaiHandle {}
unsafe impl Sync for BonsaiHandle {}

/// Newtype wrapping `*mut c_void` with `Send + Sync + Clone`.
///
/// Bonsai stores the pointer as-is and never dereferences it; lifetime
/// management is the caller's responsibility.
#[derive(Clone, Copy)]
pub struct OpaquePayload(#[allow(dead_code)] *mut std::ffi::c_void);

// SAFETY: same contract as `BonsaiHandle` above.
unsafe impl Send for OpaquePayload {}
unsafe impl Sync for OpaquePayload {}

/// Snapshot of index statistics returned to C callers.
#[repr(C)]
pub struct BonsaiStats {
    pub point_count: u64,
    pub query_count: u64,
    pub migration_count: u64,
    pub migrating: c_int,
    pub backend_kind: c_int,
}

/// Allocate a new `BonsaiIndex` and return an opaque handle.
///
/// The caller is responsible for freeing the handle with `bonsai_free`.
///
/// # Safety
///
/// The returned pointer must be freed exactly once via `bonsai_free` and must
/// not be used after that call.
#[no_mangle]
pub unsafe extern "C" fn bonsai_new() -> *mut BonsaiHandle {
    let index: BonsaiIndex<OpaquePayload, f64, 2> =
        BonsaiIndex::from_config(BonsaiConfig::default());
    Box::into_raw(Box::new(BonsaiHandle(index)))
}

/// Free a `BonsaiIndex` handle previously returned by `bonsai_new`.
///
/// Calling with a null pointer is a no-op.
///
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by `bonsai_new`
/// that has not yet been freed.
#[no_mangle]
pub unsafe extern "C" fn bonsai_free(handle: *mut BonsaiHandle) {
    if handle.is_null() {
        return;
    }
    // SAFETY: `handle` is non-null and was produced by `Box::into_raw` in
    // `bonsai_new`. We are the sole owner and may drop it.
    drop(Box::from_raw(handle));
}

/// Insert a 2-D point with an opaque payload pointer.
///
/// Returns the `EntryId` (a `u64`) assigned to the new entry, or `u64::MAX`
/// on error (null handle).
///
/// # Safety
///
/// `handle` must be a valid, non-null pointer returned by `bonsai_new`.
/// `payload` may be null; it is stored as-is and never dereferenced by Bonsai.
#[no_mangle]
pub unsafe extern "C" fn bonsai_insert_2d(
    handle: *mut BonsaiHandle,
    x: f64,
    y: f64,
    payload: *mut std::ffi::c_void,
) -> u64 {
    if handle.is_null() {
        return u64::MAX;
    }
    // SAFETY: `handle` is non-null and valid for the duration of this call.
    let index = &mut (*handle).0;
    let id = index.insert(Point::new([x, y]), OpaquePayload(payload));
    id.0
}

/// Remove an entry by its `EntryId`.
///
/// Returns `1` if the entry was found and removed, `0` otherwise.
///
/// # Safety
///
/// `handle` must be a valid, non-null pointer returned by `bonsai_new`.
#[no_mangle]
pub unsafe extern "C" fn bonsai_remove(handle: *mut BonsaiHandle, entry_id: u64) -> c_int {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: `handle` is non-null and valid for the duration of this call.
    let index = &mut (*handle).0;
    match index.remove(EntryId(entry_id)) {
        Some(_) => 1,
        None => 0,
    }
}

/// Run a 2-D range query and write matching entry IDs into a caller-provided buffer.
///
/// Returns the number of results written (≤ `capacity`). If `handle` is null
/// or `out_ids` is null, returns `0`.
///
/// # Safety
///
/// - `handle` must be a valid, non-null pointer returned by `bonsai_new`.
/// - `out_ids` must point to a writable array of at least `capacity` `u64`s.
#[no_mangle]
pub unsafe extern "C" fn bonsai_range_query_2d(
    handle: *mut BonsaiHandle,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    out_ids: *mut u64,
    capacity: usize,
) -> usize {
    if handle.is_null() || out_ids.is_null() {
        return 0;
    }
    // SAFETY: `handle` is non-null and valid for the duration of this call.
    let index = &mut (*handle).0;
    let bbox = BBox::new(Point::new([min_x, min_y]), Point::new([max_x, max_y]));
    let results = index.range_query(&bbox);
    let count = results.len().min(capacity);
    // SAFETY: `out_ids` points to a writable array of at least `capacity`
    // elements as required by the caller contract. We write at most `count`
    // elements, which is ≤ `capacity`.
    for (i, (id, _payload)) in results.iter().take(count).enumerate() {
        *out_ids.add(i) = id.0;
    }
    count
}
///
/// Writes up to `k` results. Each result slot `i` receives:
/// - `out_ids[i]`  — the `EntryId` as `u64`
/// - `out_dist[i]` — the Euclidean distance as `f64`
///
/// Returns the number of results written. Returns `0` if `handle`, `out_ids`,
/// or `out_dist` is null.
///
/// # Safety
///
/// - `handle` must be a valid, non-null pointer returned by `bonsai_new`.
/// - `out_ids` must point to a writable array of at least `k` `u64`s.
/// - `out_dist` must point to a writable array of at least `k` `f64`s.
#[no_mangle]
pub unsafe extern "C" fn bonsai_knn_query_2d(
    handle: *const BonsaiHandle,
    qx: f64,
    qy: f64,
    k: usize,
    out_ids: *mut u64,
    out_dist: *mut f64,
) -> usize {
    if handle.is_null() || out_ids.is_null() || out_dist.is_null() {
        return 0;
    }
    // SAFETY: `handle` is non-null and valid for the duration of this call.
    let index = &(*handle).0;
    let results = index.knn_query(&Point::new([qx, qy]), k);
    let count = results.len().min(k);
    // SAFETY: `out_ids` and `out_dist` each point to writable arrays of at
    // least `k` elements; we write at most `count` ≤ `k` elements.
    for (i, (dist, id, _payload)) in results.iter().take(count).enumerate() {
        *out_ids.add(i) = id.0;
        *out_dist.add(i) = *dist;
    }
    count
}

/// Fill `*out` with a statistics snapshot for the given index.
///
/// Returns `1` on success, `0` if `handle` or `out` is null.
///
/// # Safety
///
/// - `handle` must be a valid, non-null pointer returned by `bonsai_new`.
/// - `out` must point to a writable `BonsaiStats`.
#[no_mangle]
pub unsafe extern "C" fn bonsai_stats(handle: *const BonsaiHandle, out: *mut BonsaiStats) -> c_int {
    if handle.is_null() || out.is_null() {
        return 0;
    }
    // SAFETY: `handle` and `out` are non-null and valid for the duration of this call.
    let index = &(*handle).0;
    let s = index.stats();
    let backend_kind = match s.backend {
        BackendKind::RTree => 0,
        BackendKind::KDTree => 1,
        BackendKind::Quadtree => 2,
        BackendKind::Grid => 3,
    };
    *out = BonsaiStats {
        point_count: s.point_count as u64,
        query_count: s.query_count,
        migration_count: s.migrations,
        migrating: s.migrating as c_int,
        backend_kind,
    };
    1
}

/// Force the index to use a specific backend, bypassing the policy engine.
///
/// `backend` is an integer code:
/// - `0` → R-tree
/// - `1` → KD-tree
/// - `2` → Quadtree
/// - `3` → Grid
///
/// Returns `1` on success, `0` on error (null handle, unknown code, or
/// migration already in progress).
///
/// # Safety
///
/// `handle` must be a valid, non-null pointer returned by `bonsai_new`.
#[no_mangle]
pub unsafe extern "C" fn bonsai_force_backend(handle: *mut BonsaiHandle, backend: c_int) -> c_int {
    if handle.is_null() {
        return 0;
    }
    let kind = match backend {
        0 => BackendKind::RTree,
        1 => BackendKind::KDTree,
        2 => BackendKind::Quadtree,
        3 => BackendKind::Grid,
        _ => return 0,
    };
    // SAFETY: `handle` is non-null and valid for the duration of this call.
    let index = &mut (*handle).0;
    match index.force_backend(kind) {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

/// Write the null-terminated backend name into `out_buf` (at most `buf_len`
/// bytes including the null terminator).
///
/// Returns the number of bytes written (excluding the null terminator), or
/// `usize::MAX` on error.
///
/// # Safety
///
/// - `handle` must be a valid, non-null pointer returned by `bonsai_new`.
/// - `out_buf` must point to a writable buffer of at least `buf_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn bonsai_backend_name(
    handle: *const BonsaiHandle,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if handle.is_null() || out_buf.is_null() || buf_len == 0 {
        return usize::MAX;
    }
    // SAFETY: `handle` is non-null and valid for the duration of this call.
    let index = &(*handle).0;
    let name = match index.stats().backend {
        BackendKind::RTree => "rtree",
        BackendKind::KDTree => "kdtree",
        BackendKind::Quadtree => "quadtree",
        BackendKind::Grid => "grid",
    };
    let bytes = name.as_bytes();
    let copy_len = bytes.len().min(buf_len - 1);
    // SAFETY: `out_buf` points to a writable buffer of at least `buf_len` bytes;
    // we write at most `copy_len` bytes plus a null terminator, which is ≤ `buf_len`.
    std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, out_buf, copy_len);
    *out_buf.add(copy_len) = 0;
    copy_len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonsai_free_null_is_noop() {
        // SAFETY: null pointer — documented no-op.
        unsafe { bonsai_free(std::ptr::null_mut()) };
    }

    #[test]
    fn round_trip_insert_range_query() {
        unsafe {
            let h = bonsai_new();
            assert!(!h.is_null());

            let id0 = bonsai_insert_2d(h, 1.0, 1.0, std::ptr::null_mut());
            let id1 = bonsai_insert_2d(h, 5.0, 5.0, std::ptr::null_mut());
            let _id2 = bonsai_insert_2d(h, 9.0, 9.0, std::ptr::null_mut());

            let mut out_ids = [0u64; 8];
            let count =
                bonsai_range_query_2d(h, 0.0, 0.0, 6.0, 6.0, out_ids.as_mut_ptr(), out_ids.len());
            assert_eq!(count, 2);
            let found: std::collections::HashSet<u64> = out_ids[..count].iter().copied().collect();
            assert!(found.contains(&id0));
            assert!(found.contains(&id1));

            bonsai_free(h);
        }
    }

    #[test]
    fn round_trip_knn_query() {
        unsafe {
            let h = bonsai_new();

            let id0 = bonsai_insert_2d(h, 0.0, 0.0, std::ptr::null_mut());
            let _id1 = bonsai_insert_2d(h, 10.0, 10.0, std::ptr::null_mut());

            let mut out_ids = [0u64; 2];
            let mut out_dist = [0.0f64; 2];
            let count =
                bonsai_knn_query_2d(h, 0.0, 0.0, 1, out_ids.as_mut_ptr(), out_dist.as_mut_ptr());
            assert_eq!(count, 1);
            assert_eq!(out_ids[0], id0);
            assert!(out_dist[0].abs() < 1e-9);

            bonsai_free(h);
        }
    }

    #[test]
    fn remove_entry() {
        unsafe {
            let h = bonsai_new();

            let id = bonsai_insert_2d(h, 3.0, 3.0, std::ptr::null_mut());
            assert_eq!(bonsai_remove(h, id), 1);

            let mut out_ids = [0u64; 4];
            let count =
                bonsai_range_query_2d(h, 0.0, 0.0, 10.0, 10.0, out_ids.as_mut_ptr(), out_ids.len());
            assert_eq!(count, 0);

            bonsai_free(h);
        }
    }

    #[test]
    fn stats_round_trip() {
        unsafe {
            let h = bonsai_new();

            bonsai_insert_2d(h, 1.0, 2.0, std::ptr::null_mut());
            bonsai_insert_2d(h, 3.0, 4.0, std::ptr::null_mut());

            let mut s = BonsaiStats {
                point_count: 0,
                query_count: 0,
                migration_count: 0,
                migrating: 0,
                backend_kind: 0,
            };
            let ok = bonsai_stats(h, &mut s as *mut BonsaiStats);
            assert_eq!(ok, 1);
            assert_eq!(s.point_count, 2);

            bonsai_free(h);
        }
    }

    #[test]
    fn force_backend_ok() {
        unsafe {
            let h = bonsai_new();
            assert_eq!(bonsai_force_backend(h, 0), 1);
            bonsai_free(h);
        }
    }

    #[test]
    fn null_handle_returns_safe_defaults() {
        unsafe {
            assert_eq!(
                bonsai_insert_2d(std::ptr::null_mut(), 0.0, 0.0, std::ptr::null_mut()),
                u64::MAX
            );
            assert_eq!(bonsai_remove(std::ptr::null_mut(), 0), 0);

            let mut out_ids = [0u64; 4];
            assert_eq!(
                bonsai_range_query_2d(
                    std::ptr::null_mut(),
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    out_ids.as_mut_ptr(),
                    4,
                ),
                0
            );

            let mut out_ids2 = [0u64; 1];
            let mut out_dist = [0.0f64; 1];
            assert_eq!(
                bonsai_knn_query_2d(
                    std::ptr::null(),
                    0.0,
                    0.0,
                    1,
                    out_ids2.as_mut_ptr(),
                    out_dist.as_mut_ptr(),
                ),
                0
            );

            assert_eq!(bonsai_force_backend(std::ptr::null_mut(), 0), 0);
        }
    }
}
