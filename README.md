# Bonsai

A zero-dependency, embeddable Rust library providing a self-tuning spatial index. Bonsai continuously profiles your data and query workload at runtime and transparently migrates between index backends — R-tree, KD-tree, Quadtree, and Grid — to maintain optimal performance without developer intervention.

Generic over dimensionality (`const D: usize`, D=1–8, default D=2) and coordinate type (`f32`/`f64`). Ships a C FFI layer, a WASM target, and a CLI binary. The core crate has zero mandatory dependencies.

---

## Crate
The package is available on [crate bonsai-index](https://crates.io/crates/bonsai-index)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
bonsai-index = "1.0"
```

### Feature Flags

| Feature | Description | Adds dependencies |
|---------|-------------|-------------------|
| `serde` | Serialise/deserialise full index state to/from `Vec<u8>` | `serde`, `bincode` |
| `wasm` | `wasm-bindgen` bindings for browser and Node.js | `wasm-bindgen` |
| `ffi` | C-compatible `extern "C"` API + `bonsai.h` header | none |
| `full` | All of the above | all of the above |

```toml
# Enable serialisation only
bonsai-index = { version = "1.0", features = ["serde"] }

# Enable everything
bonsai-index = { version = "1.0", features = ["full"] }
```

---

## Usage

### Basic 2D example (10 lines)

```rust
use bonsai::BonsaiIndex;
use bonsai::types::{BBox, Point};

let mut index = BonsaiIndex::<&str>::builder().build();

index.insert(Point::new([1.0, 2.0]), "alpha");
index.insert(Point::new([3.0, 4.0]), "beta");
index.insert(Point::new([5.0, 6.0]), "gamma");
index.insert(Point::new([7.0, 8.0]), "delta");
index.insert(Point::new([9.0, 0.0]), "epsilon");

let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([6.0, 6.0]));
let results = index.range_query(&bbox);
println!("range hits: {}", results.len()); // alpha, beta, gamma

let nearest = index.knn_query(&Point::new([5.0, 5.0]), 2);
println!("nearest: {:?}", nearest.iter().map(|(_, _, p)| p).collect::<Vec<_>>());

let s = index.stats();
println!("backend={:?}  points={}", s.backend, s.point_count);
```

### 3D example

```rust
use bonsai::BonsaiIndex;
use bonsai::types::{BBox, Point};

let mut index = BonsaiIndex::<u32, f64, 3>::builder().build();

index.insert(Point::new([1.0, 2.0, 3.0]), 1);
index.insert(Point::new([4.0, 5.0, 6.0]), 2);
index.insert(Point::new([7.0, 8.0, 9.0]), 3);

let bbox = BBox::new(Point::new([0.0, 0.0, 0.0]), Point::new([5.0, 6.0, 7.0]));
let hits = index.range_query(&bbox);
println!("3D range hits: {}", hits.len()); // 2
```

### 6D example (robotics phase-space)

```rust
use bonsai::BonsaiIndex;
use bonsai::types::{BBox, Point};

// 6D: [x, y, z, roll, pitch, yaw]
let mut index = BonsaiIndex::<String, f64, 6>::builder()
    .initial_backend(bonsai::types::BackendKind::KDTree)
    .reservoir_size(2048)
    .build();

index.insert(Point::new([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]), "pose_a".to_string());
index.insert(Point::new([4.0, 5.0, 6.0, 0.4, 0.5, 0.6]), "pose_b".to_string());

let nearest = index.knn_query(&Point::new([1.1, 2.1, 3.1, 0.1, 0.2, 0.3]), 1);
println!("nearest pose: {:?}", nearest[0].2);
```

### Builder configuration reference

```rust
use std::time::Duration;
use bonsai::{BonsaiIndex, types::BackendKind};

let index = BonsaiIndex::<String>::builder()
    // Starting backend (default: KDTree)
    .initial_backend(BackendKind::RTree)
    // Alternative must be ≥23% cheaper to trigger migration (default: 0.77)
    .migration_threshold(0.77)
    // Suppress re-migration for N observations after a migration (default: 1000)
    .hysteresis_window(1000)
    // Reservoir sampler capacity for profiling (default: 4096)
    .reservoir_size(4096)
    // Bloom filter memory budget in bytes (default: 65536)
    .bloom_memory_bytes(65_536)
    .build();
```

### Escape hatches

```rust
// Disable automatic adaptation
index.freeze();

// Force a specific backend, bypassing the policy engine
let _ = index.force_backend(BackendKind::RTree);

// Re-enable automatic adaptation
index.unfreeze();

// Reset the index to an empty state, preserving config, frozen flag, and migration count.
// Returns Err(BonsaiError::MigrationInProgress) if a migration is currently running.
index.clear().unwrap();
```

### Serialisation (feature = "serde")

```rust
// Snapshot full index state to bytes
let bytes = index.to_bytes();

// Restore from bytes — returns BonsaiError::Serialisation on malformed input
let restored = BonsaiIndex::<String>::from_bytes(&bytes)?;
```

---

## Using Bonsai from Other Languages

Bonsai exposes a C-compatible FFI layer (feature = `ffi`) that works from any language with a C FFI bridge.

First, build the shared library:

```sh
cargo build --release --features ffi
# produces: target/release/libbonsai.dylib  (macOS)
#           target/release/libbonsai.so     (Linux)
#           target/release/bonsai.dll       (Windows)
```

The generated header is at `bonsai.h`.

### C

```c
#include <stdio.h>
#include "bonsai.h"

int main(void) {
    BonsaiHandle *idx = bonsai_new();

    uint64_t id0 = bonsai_insert_2d(idx, 1.0, 2.0, NULL);
    uint64_t id1 = bonsai_insert_2d(idx, 5.0, 5.0, NULL);
    uint64_t id2 = bonsai_insert_2d(idx, 9.0, 8.0, NULL);

    /* Range query — [0,6]^2 should return id0 and id1 */
    uint64_t out_ids[16];
    size_t count = bonsai_range_query_2d(idx, 0.0, 0.0, 6.0, 6.0, out_ids, 16);
    printf("range hits: %zu\n", count);
    for (size_t i = 0; i < count; i++) {
        printf("  id=%llu\n", (unsigned long long)out_ids[i]);
    }

    /* kNN — 2 nearest to (5, 5) */
    uint64_t knn_ids[2];
    double   knn_dist[2];
    size_t k = bonsai_knn_query_2d(idx, 5.0, 5.0, 2, knn_ids, knn_dist);
    printf("kNN(k=2) from (5,5):\n");
    for (size_t i = 0; i < k; i++) {
        printf("  id=%llu  dist=%.4f\n",
               (unsigned long long)knn_ids[i], knn_dist[i]);
    }

    /* Stats */
    BonsaiStats stats;
    bonsai_stats(idx, &stats);
    printf("points=%llu  backend=%d\n",
           (unsigned long long)stats.point_count, stats.backend_kind);

    bonsai_free(idx);
    return 0;
}
```

Compile and run:

```sh
# macOS
cc -o demo demo.c -I. -L target/release -lbonsai -rpath @loader_path/target/release
./demo

# Linux
cc -o demo demo.c -I. -L target/release -lbonsai -Wl,-rpath,'$ORIGIN/target/release'
./demo
```

### Python

Uses the standard library `ctypes` — no extra dependencies required.

```python
import ctypes, sys

# Load the shared library
if sys.platform == "darwin":
    lib = ctypes.CDLL("target/release/libbonsai.dylib")
elif sys.platform == "win32":
    lib = ctypes.CDLL("target/release/bonsai.dll")
else:
    lib = ctypes.CDLL("target/release/libbonsai.so")

# --- type annotations ---
lib.bonsai_new.restype  = ctypes.c_void_p
lib.bonsai_free.argtypes = [ctypes.c_void_p]

lib.bonsai_insert_2d.restype  = ctypes.c_uint64
lib.bonsai_insert_2d.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_void_p]

lib.bonsai_remove.restype  = ctypes.c_int
lib.bonsai_remove.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

lib.bonsai_range_query_2d.restype  = ctypes.c_size_t
lib.bonsai_range_query_2d.argtypes = [
    ctypes.c_void_p,
    ctypes.c_double, ctypes.c_double,   # min_x, min_y
    ctypes.c_double, ctypes.c_double,   # max_x, max_y
    ctypes.POINTER(ctypes.c_uint64),    # out_ids
    ctypes.c_size_t,                    # capacity
]

lib.bonsai_knn_query_2d.restype  = ctypes.c_size_t
lib.bonsai_knn_query_2d.argtypes = [
    ctypes.c_void_p,
    ctypes.c_double, ctypes.c_double,   # qx, qy
    ctypes.c_size_t,                    # k
    ctypes.POINTER(ctypes.c_uint64),    # out_ids
    ctypes.POINTER(ctypes.c_double),    # out_dist
]

class BonsaiStats(ctypes.Structure):
    _fields_ = [
        ("point_count",     ctypes.c_uint64),
        ("query_count",     ctypes.c_uint64),
        ("migration_count", ctypes.c_uint64),
        ("migrating",       ctypes.c_int),
        ("backend_kind",    ctypes.c_int),
    ]

lib.bonsai_stats.restype  = ctypes.c_int
lib.bonsai_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(BonsaiStats)]

BACKEND_NAMES = {0: "rtree", 1: "kdtree", 2: "quadtree", 3: "grid"}

# --- usage ---
idx = lib.bonsai_new()

id0 = lib.bonsai_insert_2d(idx, 1.0, 2.0, None)
id1 = lib.bonsai_insert_2d(idx, 5.0, 5.0, None)
id2 = lib.bonsai_insert_2d(idx, 9.0, 8.0, None)

# Range query — [0, 6]^2
out_ids  = (ctypes.c_uint64 * 16)()
count = lib.bonsai_range_query_2d(idx, 0.0, 0.0, 6.0, 6.0, out_ids, 16)
print(f"range hits: {count}")
for i in range(count):
    print(f"  id={out_ids[i]}")

# kNN — 2 nearest to (5, 5)
knn_ids  = (ctypes.c_uint64 * 2)()
knn_dist = (ctypes.c_double * 2)()
k = lib.bonsai_knn_query_2d(idx, 5.0, 5.0, 2, knn_ids, knn_dist)
print(f"kNN(k=2) from (5, 5):")
for i in range(k):
    print(f"  id={knn_ids[i]}  dist={knn_dist[i]:.4f}")

# Stats
stats = BonsaiStats()
lib.bonsai_stats(idx, ctypes.byref(stats))
print(f"points={stats.point_count}  backend={BACKEND_NAMES[stats.backend_kind]}")

lib.bonsai_free(idx)
```

Run:

```sh
python3 bonsai_demo.py
```

---

## Development

### Build

```sh
cargo build
cargo build --all-features
```

### Test

```sh
# All tests including property-based and integration tests
cargo test --all-features

# Core tests only (no optional features)
cargo test
```

### Lint and format

```sh
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

### Documentation

```sh
cargo doc --all-features --no-deps --open
```

### Benchmarks

```sh
# Run all benchmark groups
cargo bench --all-features

# Run a specific group
cargo bench --bench range_query_latency
cargo bench --bench insert_throughput
cargo bench --bench knn_latency
cargo bench --bench migration_cost
cargo bench --bench bloom_cache_impact
cargo bench --bench hilbert_vs_natural
cargo bench --bench adaptation_convergence
```

Benchmark groups cover: `insert_throughput`, `range_query_latency`, `knn_latency`, `migration_cost`, `bloom_cache_impact`, `hilbert_vs_natural`, `adaptation_convergence`. Datasets include uniform 2D/3D/6D, clustered 2D, OSM-style 2D, and robotics 6D phase-space.

### CLI

Build and run the `bonsai` CLI (requires the `serde` feature):

```sh
cargo build --features serde

# Load a CSV file into the index
bonsai load data.csv

# Query a bounding box
bonsai query range 0 0 100 100

# k-nearest neighbours
bonsai query knn 50 50 5

# Print index statistics
bonsai stats

# Live-updating TUI (backend, data shape, cost estimates, throughput)
bonsai watch

# Run benchmark suite and print p50/p95/p99 latencies
bonsai bench

# ASCII art visualisation of the index structure
bonsai visualise

# Force migration to a specific backend
bonsai migrate rtree
```

### WASM demo

```sh
# Install wasm-pack if needed
cargo install wasm-pack

# Build the WASM package
wasm-pack build --features wasm

# Run the Node.js demo
node examples/wasm_demo.js
```

---

## Demos

Run the full end-to-end demo with:

```sh
cargo run --example demo_bonsai
cargo run --example demo_bonsai --features serde      # includes serialisation
cargo run --example demo_bonsai --features serde,ffi  # includes C FFI
```

The demo runs 16 sections in sequence, each building on the previous, using a single dataset of 2 000 clustered 2D points:

| Section | What it shows |
|---------|---------------|
| 1. Hilbert sort | Sort 2 000 clustered points into cache-friendly Hilbert order |
| 2. BloomCache | Populate from Hilbert-sorted insertions; O(1) empty-result rejection |
| 3. All 4 backends | Bulk-load R-tree, KD-tree, Quadtree, Grid in Hilbert order |
| 4. Range query | Bloom gates all four backends; brute-force correctness check |
| 5. kNN(k=5) | All four backends compared; distances must match |
| 6. Insert-remove | Remove 50 of 100 entries; verify no ghost results |
| 7. Profiler | Feed dataset; print `DataShape` (clustering_coef, effective_dim, skewness) |
| 8. CostModel | Rank all four backends by estimated range query cost |
| 9. PolicyEngine | Tick on DataShape; watch migration decision fire after hysteresis window |
| 10. Migration | KD-tree → winner backend; range query results identical before and after |
| 11. IndexRouter + StatsCollector | 4 insert threads + 2 query threads; lock-free stats |
| 12. BonsaiIndex | Full public API: insert, range query, kNN, stats |
| 13. Serialisation | `to_bytes` / `from_bytes` round-trip; query results match (feature = `serde`) |
| 14. C FFI | `bonsai_new`, `bonsai_insert_2d`, `bonsai_range_query_2d`, `bonsai_free` (feature = `ffi`) |
| 15. CLI | `load`, `stream`, `stats`, `query range`, `query knn`, `visualise` |
| 16. Latency table | 100k-point backends; 1 000 range queries each; p50/p95/p99 per backend |

---

## Use Cases

### Games & Simulation

**Boids flocking simulator** — thousands of agents querying their local neighbourhood every frame; data density shifts constantly as flocks form and disperse, perfect for triggering Bonsai's adaptive switching.

**Procedural dungeon with dynamic enemies** — spatial queries for line-of-sight, aggro radius, pathfinding; a great stress test for mixed range + kNN workloads.

**N-body gravity simulator** — 3D point data with extreme clustering (solar systems) and vast voids; the kind of non-uniform distribution that destroys fixed-choice indexes.

### Mapping & Geospatial

**Offline postcode/address search engine** — load the entire Royal Mail PAF dataset, run proximity and region queries with zero network dependency; a direct real-world use case.

**GPS trace analyser** — feed raw GPX files, detect stops, simplify routes, find points of interest clusters; data shape changes dramatically between motorway and city driving.

**Isochrone map generator** — given a point and travel time, compute the reachable region; heavy on range queries over road network nodes.

### Data & Analytics

**Astronomical catalogue explorer** — query the Gaia DR3 star catalogue (1.8 billion stars) by sky region or find k nearest stars to any coordinate; 3D with extreme non-uniformity.

**Anomaly detector for sensor networks** — IoT sensors reporting 3D position + readings; find spatial outliers in real time using kNN distance as an anomaly score.

**Real estate comparable finder** — given a property, find the k most spatially similar sold properties; 4D or 5D index over lat, lon, size, age, price range.

### Robotics & Physics

**Collision avoidance system** — 3D bounding box queries for a drone fleet or robot arm; Bonsai's lock-free migration means zero jitter during index restructuring.

**Particle physics event display** — visualise detector hits from CERN open data; millions of 3D points with extreme clustering near interaction vertices.

---

## Architecture

Bonsai is built around a lock-free `IndexRouter` holding an atomic pointer to the active backend. A `Profiler` pipeline (reservoir sampler → online stats → cost model → policy engine) runs in the background and triggers a `MigrationEngine` when a better backend is identified. A `BloomCache` short-circuits empty-result range queries in O(1).

```
BonsaiIndex<T, C, D>
  ├── IndexRouter       — atomic ptr to active backend
  ├── Profiler          — ReservoirSampler → OnlineStats → CostModel → PolicyEngine
  ├── MigrationEngine   — STR bulk load → incremental drain → atomic swap
  ├── BloomCache        — probabilistic negative cache (zero false negatives)
  └── StatsCollector    — lock-free query/insert counters
```

Migration is transparent: queries are never blocked for more than 50 µs (one atomic pointer load). During migration, writes are routed to both the active and shadow backends; after the drain phase, an atomic SeqCst swap promotes the shadow to active.

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
