//! Lock-free statistics accumulation.
//!
//! [`StatsCollector`] tracks query counts, insert counts, and latency
//! statistics using atomic operations only — no mutex is ever acquired on the
//! read path.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Lock-free accumulator of query and insert statistics.
///
/// All counters use `Relaxed` ordering for throughput; the `last_query_ns`
/// field uses `Release`/`Acquire` so that a reader who observes an updated
/// timestamp also sees the corresponding counter increments.
pub struct StatsCollector {
    query_count: AtomicU64,
    insert_count: AtomicU64,
    /// Sum of query latencies in nanoseconds.
    total_query_ns: AtomicU64,
    /// Timestamp of the last recorded query (nanoseconds since an arbitrary
    /// epoch — used only for relative comparisons).
    last_query_ns: AtomicU64,
    /// Epoch used to convert `Instant` to nanoseconds.
    epoch: Instant,
}

impl StatsCollector {
    /// Create a new `StatsCollector` with all counters at zero.
    pub fn new() -> Self {
        Self {
            query_count: AtomicU64::new(0),
            insert_count: AtomicU64::new(0),
            total_query_ns: AtomicU64::new(0),
            last_query_ns: AtomicU64::new(0),
            epoch: Instant::now(),
        }
    }

    /// Record a completed insert.
    #[inline]
    pub fn record_insert(&self) {
        self.insert_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a completed query with the given latency.
    #[inline]
    pub fn record_query(&self, latency_ns: u64) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.total_query_ns.fetch_add(latency_ns, Ordering::Relaxed);
        let now_ns = self.epoch.elapsed().as_nanos() as u64;
        self.last_query_ns.store(now_ns, Ordering::Release);
    }

    /// Total number of queries recorded.
    #[inline]
    pub fn query_count(&self) -> u64 {
        self.query_count.load(Ordering::Relaxed)
    }

    /// Total number of inserts recorded.
    #[inline]
    pub fn insert_count(&self) -> u64 {
        self.insert_count.load(Ordering::Relaxed)
    }

    /// Mean query latency in nanoseconds, or `0` if no queries have been
    /// recorded.
    pub fn mean_query_ns(&self) -> u64 {
        let count = self.query_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0;
        }
        self.total_query_ns.load(Ordering::Relaxed) / count
    }

    /// Nanoseconds since the epoch at which the last query was recorded, or
    /// `None` if no queries have been recorded yet.
    pub fn last_query_ns(&self) -> Option<u64> {
        let v = self.last_query_ns.load(Ordering::Acquire);
        if v == 0 {
            None
        } else {
            Some(v)
        }
    }
}

impl Default for StatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn initial_counts_are_zero() {
        let sc = StatsCollector::new();
        assert_eq!(sc.query_count(), 0);
        assert_eq!(sc.insert_count(), 0);
        assert_eq!(sc.mean_query_ns(), 0);
        assert!(sc.last_query_ns().is_none());
    }

    #[test]
    fn record_insert_increments_count() {
        let sc = StatsCollector::new();
        sc.record_insert();
        sc.record_insert();
        assert_eq!(sc.insert_count(), 2);
    }

    #[test]
    fn record_query_increments_count_and_latency() {
        let sc = StatsCollector::new();
        sc.record_query(100);
        sc.record_query(200);
        assert_eq!(sc.query_count(), 2);
        assert_eq!(sc.mean_query_ns(), 150);
        assert!(sc.last_query_ns().is_some());
    }

    #[test]
    fn concurrent_inserts_and_queries() {
        let sc = Arc::new(StatsCollector::new());
        let mut handles = Vec::new();

        for _ in 0..4 {
            let sc2 = Arc::clone(&sc);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    sc2.record_insert();
                }
            }));
        }
        for _ in 0..2 {
            let sc2 = Arc::clone(&sc);
            handles.push(thread::spawn(move || {
                for _ in 0..500 {
                    sc2.record_query(50);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(sc.insert_count(), 4000);
        assert_eq!(sc.query_count(), 1000);
    }
}
