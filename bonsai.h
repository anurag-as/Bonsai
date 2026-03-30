/* C header for the Bonsai spatial index FFI layer (feature = "ffi").
 *
 * All functions are safe to call from C, C++, Python (ctypes/cffi),
 * Go (cgo), and Swift.  See src/ffi.rs for the full safety contract.
 */

#ifndef BONSAI_H
#define BONSAI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to a heap-allocated BonsaiIndex.
 * Obtain via bonsai_new(); release via bonsai_free().
 */
typedef struct BonsaiHandle BonsaiHandle;

/**
 * Snapshot of index statistics.
 *
 * backend_kind:
 *   0 = R-tree, 1 = KD-tree, 2 = Quadtree, 3 = Grid
 */
typedef struct BonsaiStats {
    uint64_t point_count;
    uint64_t query_count;
    uint64_t migration_count;
    int      migrating;
    int      backend_kind;
} BonsaiStats;

/**
 * Allocate a new BonsaiIndex and return an opaque handle.
 * The caller must free the handle with bonsai_free().
 */
BonsaiHandle *bonsai_new(void);

/**
 * Free a BonsaiIndex handle.
 * Calling with NULL is a no-op.
 */
void bonsai_free(BonsaiHandle *handle);

/**
 * Insert a 2-D point with an opaque payload pointer.
 * Returns the EntryId (u64) assigned to the entry, or UINT64_MAX on error.
 * payload may be NULL; it is stored as-is and never dereferenced by Bonsai.
 */
uint64_t bonsai_insert_2d(BonsaiHandle *handle,
                          double x, double y,
                          void *payload);

/**
 * Remove an entry by its EntryId.
 * Returns 1 if found and removed, 0 otherwise.
 */
int bonsai_remove(BonsaiHandle *handle, uint64_t entry_id);

/**
 * Run a 2-D range query and write matching EntryIds into out_ids.
 * Returns the number of results written (<= capacity).
 * Returns 0 if handle or out_ids is NULL.
 */
size_t bonsai_range_query_2d(BonsaiHandle *handle,
                             double min_x, double min_y,
                             double max_x, double max_y,
                             uint64_t *out_ids, size_t capacity);

/**
 * Run a 2-D kNN query.
 * Writes up to k results into out_ids and out_dist.
 * Returns the number of results written.
 * Returns 0 if handle, out_ids, or out_dist is NULL.
 */
size_t bonsai_knn_query_2d(const BonsaiHandle *handle,
                           double qx, double qy, size_t k,
                           uint64_t *out_ids, double *out_dist);

/**
 * Fill *out with a statistics snapshot.
 * Returns 1 on success, 0 if handle or out is NULL.
 */
int bonsai_stats(const BonsaiHandle *handle, BonsaiStats *out);

/**
 * Force the index to use a specific backend.
 * backend: 0=RTree, 1=KDTree, 2=Quadtree, 3=Grid
 * Returns 1 on success, 0 on error.
 */
int bonsai_force_backend(BonsaiHandle *handle, int backend);

/**
 * Write the null-terminated backend name into out_buf.
 * Returns the number of bytes written (excluding null terminator),
 * or SIZE_MAX on error.
 */
size_t bonsai_backend_name(const BonsaiHandle *handle,
                           char *out_buf, size_t buf_len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BONSAI_H */
