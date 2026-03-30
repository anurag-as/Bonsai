use ::serde::{Deserialize, Serialize};

use crate::index::{BonsaiConfig, BonsaiIndex};
use crate::types::{BackendKind, BonsaiError, CoordType, Point};

const MAGIC: u32 = 0xB0_45_A1_00;
const VERSION: u8 = 1;

#[derive(Serialize, Deserialize)]
struct SerializedConfig {
    initial_backend: u8,
    migration_threshold: f64,
    hysteresis_window: u64,
    reservoir_size: u64,
    bloom_memory_bytes: u64,
    max_migration_latency_micros: u64,
}

#[derive(Serialize, Deserialize)]
struct SerializedEntry<T> {
    coords: Vec<f64>,
    payload: T,
}

// `point_count` is intentionally omitted — it is always `entries.len()` and
// is reconstructed by re-inserting on decode.
// `active_backend` records which backend was active at serialisation time.
// The restored index always starts as KDTree (the default); the field is
// preserved in the format for future use (e.g. restoring the exact backend).
#[derive(Serialize, Deserialize)]
struct SerializedIndex<T> {
    magic: u32,
    version: u8,
    dimensions: u64,
    active_backend: u8,
    config: SerializedConfig,
    entries: Vec<SerializedEntry<T>>,
    migration_count: u64,
    frozen: bool,
}

fn backend_to_u8(k: BackendKind) -> u8 {
    match k {
        BackendKind::KDTree => 0,
        BackendKind::RTree => 1,
        BackendKind::Quadtree => 2,
        BackendKind::Grid => 3,
    }
}

fn u8_to_backend(v: u8) -> Option<BackendKind> {
    match v {
        0 => Some(BackendKind::KDTree),
        1 => Some(BackendKind::RTree),
        2 => Some(BackendKind::Quadtree),
        3 => Some(BackendKind::Grid),
        _ => None,
    }
}

impl<T, C, const D: usize> BonsaiIndex<T, C, D>
where
    C: CoordType,
    T: Clone + Send + Sync + 'static + Serialize + for<'de> Deserialize<'de>,
{
    /// Encode the full index state to bytes.
    ///
    /// The format includes a magic header, version, dimensionality, active
    /// backend kind, configuration, and all entries. Use [`Self::from_bytes`] to
    /// restore.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::Point;
    ///
    /// let mut idx: BonsaiIndex<u32> = BonsaiIndex::builder().build();
    /// idx.insert(Point::new([1.0, 2.0]), 42u32);
    /// let bytes = idx.to_bytes();
    /// assert!(!bytes.is_empty());
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        // SAFETY: `active_ptr()` returns a valid non-null pointer to a
        // `BackendBox` that lives as long as the `IndexRouter`. We hold a
        // shared reference to the `IndexRouter` via `Arc`, so the pointer
        // remains valid for the duration of this call.
        let active = unsafe { &*self.router.active_ptr() };
        let guard = active.read();

        let active_backend = guard.kind();
        let entries: Vec<SerializedEntry<T>> = guard
            .all_entries()
            .into_iter()
            .map(|(p, _id, payload)| SerializedEntry {
                coords: p.coords().iter().map(|&c| c.into()).collect(),
                payload: payload.clone(),
            })
            .collect();

        let cfg = &self.config;
        let serialized = SerializedIndex {
            magic: MAGIC,
            version: VERSION,
            dimensions: D as u64,
            active_backend: backend_to_u8(active_backend),
            config: SerializedConfig {
                initial_backend: backend_to_u8(cfg.initial_backend),
                migration_threshold: cfg.migration_threshold,
                hysteresis_window: cfg.hysteresis_window as u64,
                reservoir_size: cfg.reservoir_size as u64,
                bloom_memory_bytes: cfg.bloom_memory_bytes as u64,
                max_migration_latency_micros: cfg.max_migration_latency.as_micros() as u64,
            },
            entries,
            migration_count: self.migration_count,
            frozen: self.frozen,
        };

        bincode::serialize(&serialized).expect("bincode serialization is infallible for valid data")
    }

    /// Decode bytes produced by [`Self::to_bytes`] back into a `BonsaiIndex`.
    ///
    /// Returns `Err(BonsaiError::Serialisation)` on any malformed input —
    /// never panics.
    ///
    /// Note: coordinates are stored as `f64` and reconstructed via `From<f32>`
    /// (the only conversion available on `CoordType`). For `f64` indices this
    /// involves a `f64 → f32 → f64` round-trip, which may lose up to ~7
    /// significant decimal digits of precision. For `f32` indices there is no
    /// loss.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bonsai::index::BonsaiIndex;
    /// use bonsai::types::Point;
    ///
    /// let mut idx: BonsaiIndex<u32> = BonsaiIndex::builder().build();
    /// idx.insert(Point::new([1.0, 2.0]), 42u32);
    /// let bytes = idx.to_bytes();
    /// let restored = BonsaiIndex::<u32>::from_bytes(&bytes).unwrap();
    /// assert_eq!(restored.len(), 1);
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BonsaiError> {
        let decoded: SerializedIndex<T> =
            bincode::deserialize(bytes).map_err(|e| BonsaiError::Serialisation(e.to_string()))?;

        if decoded.magic != MAGIC {
            return Err(BonsaiError::Serialisation(
                "invalid magic bytes".to_string(),
            ));
        }
        if decoded.version != VERSION {
            return Err(BonsaiError::Serialisation(format!(
                "unsupported version {}",
                decoded.version
            )));
        }
        if decoded.dimensions != D as u64 {
            return Err(BonsaiError::Serialisation(format!(
                "dimension mismatch: encoded {} but index is {}",
                decoded.dimensions, D
            )));
        }

        let initial_backend = u8_to_backend(decoded.config.initial_backend)
            .ok_or_else(|| BonsaiError::Serialisation("unknown backend kind".to_string()))?;

        let config = BonsaiConfig {
            initial_backend,
            migration_threshold: decoded.config.migration_threshold,
            hysteresis_window: decoded.config.hysteresis_window as usize,
            reservoir_size: decoded.config.reservoir_size as usize,
            bloom_memory_bytes: decoded.config.bloom_memory_bytes as usize,
            max_migration_latency: std::time::Duration::from_micros(
                decoded.config.max_migration_latency_micros,
            ),
        };

        let mut index = BonsaiIndex::<T, C, D>::from_config(config);

        for entry in decoded.entries {
            if entry.coords.len() != D {
                return Err(BonsaiError::Serialisation(format!(
                    "entry has {} coordinates but index has D={}",
                    entry.coords.len(),
                    D
                )));
            }
            let coords: [C; D] = std::array::from_fn(|i| C::from(entry.coords[i] as f32));
            index.insert(Point::new(coords), entry.payload);
        }

        index.frozen = decoded.frozen;
        index.migration_count = decoded.migration_count;

        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use crate::index::BonsaiIndex;
    use crate::types::{BBox, BonsaiError, Point};
    use proptest::prelude::*;

    fn pt2d() -> impl Strategy<Value = Point<f64, 2>> {
        (0.0_f64..1000.0, 0.0_f64..1000.0).prop_map(|(x, y)| Point::new([x, y]))
    }

    fn bbox2d() -> impl Strategy<Value = BBox<f64, 2>> {
        (
            0.0_f64..800.0,
            0.0_f64..800.0,
            50.0_f64..200.0,
            50.0_f64..200.0,
        )
            .prop_map(|(x, y, w, h)| BBox::new(Point::new([x, y]), Point::new([x + w, y + h])))
    }

    // Serialise then deserialise must produce identical range query results.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 50,
            ..Default::default()
        })]

        #[test]
        fn prop_serialisation_round_trip(
            pts in prop::collection::vec(pt2d(), 1..80),
            bbox in bbox2d(),
        ) {
            let mut original: BonsaiIndex<u32> = BonsaiIndex::builder().build();
            for (i, &p) in pts.iter().enumerate() {
                original.insert(p, i as u32);
            }

            let bytes = original.to_bytes();
            let mut restored = BonsaiIndex::<u32>::from_bytes(&bytes)
                .expect("round-trip decode must succeed");

            let mut orig_results = original.range_query(&bbox);
            let mut rest_results = restored.range_query(&bbox);

            // Sort by payload (stable across re-insertion) rather than EntryId
            // (which is reassigned on decode).
            orig_results.sort_by_key(|(_, p)| *p);
            rest_results.sort_by_key(|(_, p)| *p);

            let orig_payloads: Vec<u32> = orig_results.iter().map(|(_, p)| *p).collect();
            let rest_payloads: Vec<u32> = rest_results.iter().map(|(_, p)| *p).collect();

            prop_assert_eq!(
                orig_payloads, rest_payloads,
                "payload sets differ after round-trip",
            );
        }
    }

    // Malformed bytes must return BonsaiError::Serialisation — never panic.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_serialisation_error_on_malformed(
            garbage in prop::collection::vec(any::<u8>(), 0..256),
        ) {
            match BonsaiIndex::<u32>::from_bytes(&garbage) {
                Err(BonsaiError::Serialisation(_)) | Ok(_) => {}
                Err(other) => prop_assert!(false, "unexpected error variant: {:?}", other),
            }
        }
    }

    #[test]
    fn round_trip_preserves_entry_count() {
        let mut idx: BonsaiIndex<u32> = BonsaiIndex::builder().build();
        for i in 0..100u32 {
            idx.insert(Point::new([i as f64, i as f64 * 2.0]), i);
        }
        let bytes = idx.to_bytes();
        let restored = BonsaiIndex::<u32>::from_bytes(&bytes).unwrap();
        assert_eq!(restored.len(), 100);
    }

    #[test]
    fn empty_index_round_trips() {
        let idx: BonsaiIndex<u32> = BonsaiIndex::builder().build();
        let bytes = idx.to_bytes();
        let restored = BonsaiIndex::<u32>::from_bytes(&bytes).unwrap();
        assert_eq!(restored.len(), 0);
    }

    #[test]
    fn malformed_bytes_return_error() {
        assert!(matches!(
            BonsaiIndex::<u32>::from_bytes(b"not valid bytes at all"),
            Err(BonsaiError::Serialisation(_))
        ));
    }

    #[test]
    fn wrong_magic_returns_error() {
        let mut idx: BonsaiIndex<u32> = BonsaiIndex::builder().build();
        idx.insert(Point::new([1.0, 2.0]), 42u32);
        let mut bytes = idx.to_bytes();
        if bytes.len() >= 4 {
            bytes[0] ^= 0xFF;
        }
        assert!(matches!(
            BonsaiIndex::<u32>::from_bytes(&bytes),
            Err(BonsaiError::Serialisation(_))
        ));
    }
}
