/**
 * Bonsai WASM demo — Node.js
 *
 * This demo mirrors the story told in demo_bonsai.rs, but from JavaScript via
 * the WASM boundary. It uses the same conceptual dataset (clustered 2-D points
 * spread across a 1000×1000 domain) and the same query region ([300,700]^2)
 * so the results are directly comparable to the Rust demo output.
 *
 * Prerequisites
 * -------------
 * 1. Install wasm-pack:
 *      cargo install wasm-pack
 *
 * 2. Build the WASM package targeting Node.js (from the workspace root):
 *      wasm-pack build --target nodejs --features wasm
 *
 *    This produces a `pkg/` directory containing `bonsai.js` and `bonsai_bg.wasm`.
 *
 * Running
 * -------
 *      node examples/wasm_demo.js
 *
 * Requires Node.js 14+ (CommonJS `require` support).
 */

const { WasmBonsaiIndex } = require('../pkg/bonsai.js');

// ── Dataset ──────────────────────────────────────────────────────────────────
// Five representative points drawn from the same 1000×1000 domain used in the
// Rust demo: two inside the [300,700]^2 query region, one on the boundary, and
// two outside. Payloads are integer labels matching their position in the array.
const POINTS = [
  { x: 350.0, y: 420.0, label: 0 },  // inside  [300,700]^2
  { x: 580.0, y: 610.0, label: 1 },  // inside  [300,700]^2
  { x: 700.0, y: 700.0, label: 2 },  // on boundary (inclusive)
  { x: 150.0, y: 200.0, label: 3 },  // outside
  { x: 850.0, y: 900.0, label: 4 },  // outside
];

const QUERY = { minX: 300.0, minY: 300.0, maxX: 700.0, maxY: 700.0 };
const KNN_ORIGIN = { x: 500.0, y: 500.0 };

// ── Insert ────────────────────────────────────────────────────────────────────
const idx = new WasmBonsaiIndex();

console.log('Inserting 5 clustered 2-D points (same domain as Rust demo):');
for (const { x, y, label } of POINTS) {
  const id = idx.insert(x, y, label);
  console.log(`  insert(${x}, ${y}, payload=${label}) → id=${id}`);
}
console.log(`\nIndex size: ${idx.len()} points`);

// ── Range query ───────────────────────────────────────────────────────────────
// The same [300,700]^2 bbox used throughout the Rust demo (sections 4, 10–14).
// Expected: points 0, 1, 2 (inside or on boundary).
const raw = idx.range_query(QUERY.minX, QUERY.minY, QUERY.maxX, QUERY.maxY);

// wasm-bindgen returns a JS Array of numbers with layout [entry_id, payload, ...]
const rangeResults = [];
for (let i = 0; i < raw.length; i += 2) {
  rangeResults.push({ id: raw[i], payload: raw[i + 1] });
}

console.log(`\nRange query [${QUERY.minX},${QUERY.maxX}] × [${QUERY.minY},${QUERY.maxY}]:`);
console.log(`  ${rangeResults.length} result(s) (expected 3 — labels 0, 1, 2):`);
for (const { id, payload } of rangeResults) {
  console.log(`    entry_id=${id}  payload=${payload}`);
}

// ── kNN query ─────────────────────────────────────────────────────────────────
// Query from the centre of the domain (500, 500) — same as section 5 of the
// Rust demo. The two nearest points should be labels 1 and 2.
const knnRaw = idx.knn_query(KNN_ORIGIN.x, KNN_ORIGIN.y, 2);

// Layout: [distance, entry_id, payload, distance, entry_id, payload, ...]
const knnResults = [];
for (let i = 0; i < knnRaw.length; i += 3) {
  knnResults.push({ dist: knnRaw[i], id: knnRaw[i + 1], payload: knnRaw[i + 2] });
}

console.log(`\nkNN(k=2) from (${KNN_ORIGIN.x}, ${KNN_ORIGIN.y}):`);
for (let rank = 0; rank < knnResults.length; rank++) {
  const { dist, id, payload } = knnResults[rank];
  console.log(`  #${rank + 1}  dist=${dist.toFixed(4)}  entry_id=${id}  payload=${payload}`);
}

// ── Stats ─────────────────────────────────────────────────────────────────────
// Mirrors the stats() call in section 12 of the Rust demo.
const stats = idx.stats();
const BACKEND_NAMES = ['RTree', 'KDTree', 'Quadtree', 'Grid'];

console.log('\nIndex stats:');
console.log(`  point_count     = ${stats[0]}`);
console.log(`  query_count     = ${stats[1]}`);
console.log(`  migration_count = ${stats[2]}`);
console.log(`  migrating       = ${stats[3] === 1}`);
console.log(`  backend         = ${BACKEND_NAMES[stats[4]] ?? 'unknown'}`);
