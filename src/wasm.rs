//! WASM bindings for `BonsaiIndex` (feature = "wasm").
//!
//! Exposes a 2-D, f64-coordinate index to JavaScript via `wasm-bindgen`.
//! The payload type is `u32` for simplicity and JS compatibility.
//!
//! # Usage from JavaScript (Node.js / CommonJS)
//!
//! ```js
//! const { WasmBonsaiIndex } = require('./pkg/bonsai.js');
//!
//! const idx = new WasmBonsaiIndex();
//! idx.insert(1.0, 2.0, 42);
//! const results = idx.range_query(0.0, 0.0, 5.0, 5.0);
//! // results is a JS Array with layout [entry_id, payload, ...]
//! idx.free();
//! ```

use wasm_bindgen::prelude::*;

use crate::index::{BonsaiConfig, BonsaiIndex};
use crate::types::{BBox, BackendKind, Point};

/// A 2-D spatial index exposed to JavaScript.
///
/// Uses `f64` coordinates and `u32` payloads. All methods are callable from JS.
#[wasm_bindgen]
pub struct WasmBonsaiIndex {
    inner: BonsaiIndex<u32, f64, 2>,
}

#[wasm_bindgen]
impl WasmBonsaiIndex {
    /// Create a new `WasmBonsaiIndex` with default configuration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmBonsaiIndex {
        WasmBonsaiIndex {
            inner: BonsaiIndex::from_config(BonsaiConfig::default()),
        }
    }

    /// Insert a 2-D point with a `u32` payload.
    ///
    /// Returns the `EntryId` (as `f64` for JS compatibility, since JS numbers
    /// are 64-bit floats and `u64` would lose precision for large IDs).
    pub fn insert(&mut self, x: f64, y: f64, payload: u32) -> f64 {
        let id = self.inner.insert(Point::new([x, y]), payload);
        id.0 as f64
    }

    /// Run a range query and return results as a `Float64Array`.
    ///
    /// The returned array is a flat sequence of `[entry_id, payload, ...]`
    /// pairs — two `f64` values per result. `entry_id` is the numeric ID and
    /// `payload` is the stored `u32` cast to `f64`.
    pub fn range_query(&mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<f64> {
        let bbox = BBox::new(Point::new([min_x, min_y]), Point::new([max_x, max_y]));
        let results = self.inner.range_query(&bbox);
        let mut out = Vec::with_capacity(results.len() * 2);
        for (id, payload) in results {
            out.push(id.0 as f64);
            out.push(payload as f64);
        }
        out
    }

    /// Run a k-nearest-neighbour query and return results as a `Float64Array`.
    ///
    /// The returned array is a flat sequence of `[distance, entry_id, payload, ...]`
    /// triples — three `f64` values per result.
    pub fn knn_query(&self, x: f64, y: f64, k: usize) -> Vec<f64> {
        let results = self.inner.knn_query(&Point::new([x, y]), k);
        let mut out = Vec::with_capacity(results.len() * 3);
        for (dist, id, payload) in results {
            out.push(dist);
            out.push(id.0 as f64);
            out.push(payload as f64);
        }
        out
    }

    /// Return index statistics as a flat JS array.
    ///
    /// Layout: `[point_count, query_count, migration_count, migrating (0/1), backend_kind]`
    ///
    /// `backend_kind` codes: 0=RTree, 1=KDTree, 2=Quadtree, 3=Grid.
    pub fn stats(&self) -> Vec<f64> {
        let s = self.inner.stats();
        let backend_kind = match s.backend {
            BackendKind::RTree => 0.0,
            BackendKind::KDTree => 1.0,
            BackendKind::Quadtree => 2.0,
            BackendKind::Grid => 3.0,
        };
        vec![
            s.point_count as f64,
            s.query_count as f64,
            s.migrations as f64,
            if s.migrating { 1.0 } else { 0.0 },
            backend_kind,
        ]
    }

    /// Return the number of entries currently stored.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return `true` if the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for WasmBonsaiIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_insert_and_len() {
        let mut idx = WasmBonsaiIndex::new();
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
        idx.insert(1.0, 2.0, 10);
        idx.insert(3.0, 4.0, 20);
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
    }

    #[test]
    fn wasm_range_query_basic() {
        let mut idx = WasmBonsaiIndex::new();
        idx.insert(1.0, 1.0, 1);
        idx.insert(9.0, 9.0, 2);
        let results = idx.range_query(0.0, 0.0, 5.0, 5.0);
        // one result: two f64 values [entry_id, payload]
        assert_eq!(results.len(), 2);
        assert_eq!(results[1] as u32, 1);
    }

    #[test]
    fn wasm_range_query_empty() {
        let mut idx = WasmBonsaiIndex::new();
        idx.insert(1.0, 1.0, 42);
        let results = idx.range_query(900.0, 900.0, 1000.0, 1000.0);
        assert!(results.is_empty());
    }

    #[test]
    fn wasm_knn_query_basic() {
        let mut idx = WasmBonsaiIndex::new();
        idx.insert(0.0, 0.0, 1);
        idx.insert(10.0, 10.0, 2);
        let results = idx.knn_query(0.0, 0.0, 1);
        // one result: three f64 values [distance, entry_id, payload]
        assert_eq!(results.len(), 3);
        assert!(results[0].abs() < 1e-9);
        assert_eq!(results[2] as u32, 1);
    }

    #[test]
    fn wasm_stats_layout() {
        let mut idx = WasmBonsaiIndex::new();
        idx.insert(1.0, 2.0, 99);
        let s = idx.stats();
        assert_eq!(s.len(), 5);
        assert_eq!(s[0] as usize, 1);
    }

    #[test]
    fn wasm_insert_returns_unique_ids() {
        let mut idx = WasmBonsaiIndex::new();
        let id0 = idx.insert(0.0, 0.0, 0);
        let id1 = idx.insert(1.0, 1.0, 1);
        assert_ne!(id0 as u64, id1 as u64);
    }

    #[test]
    fn wasm_range_query_all_five_points() {
        let mut idx = WasmBonsaiIndex::new();
        let points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0), (9.0, 10.0)];
        for (i, (x, y)) in points.iter().enumerate() {
            idx.insert(*x, *y, i as u32);
        }
        let results = idx.range_query(0.0, 0.0, 10.0, 11.0);
        // five results: ten f64 values
        assert_eq!(results.len(), 10);
    }
}
