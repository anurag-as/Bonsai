//! Profiler subsystem for the Bonsai adaptive spatial index.
//!
//! [`Profiler<C, D>`] orchestrates reservoir sampling, online statistics
//! computation, and query workload tracking. Observations are received via a
//! lock-free MPSC channel and processed in batches of 64.

pub mod cost_model;
pub mod policy;
pub mod reservoir;
pub mod stats;

use std::sync::mpsc::{self, Receiver, Sender};

use crate::types::{CoordType, DataShape, Point, QueryMix};
use reservoir::ReservoirSampler;
use stats::OnlineStats;

pub use cost_model::{CostEstimate, CostModel};
pub use policy::{MigrationDecision, PolicyEngine};

/// The type of query observation sent to the profiler.
#[derive(Debug, Clone, Copy)]
pub enum QueryKind {
    Range,
    Knn,
    Join,
}

/// An observation sent to the profiler via the MPSC channel.
#[derive(Debug, Clone)]
pub enum Observation<C: CoordType, const D: usize> {
    /// A point was inserted into the index.
    Insert(Point<C, D>),
    /// A query was executed.
    Query {
        kind: QueryKind,
        /// Fraction of total points returned (0.0–1.0).
        selectivity: f64,
        /// Whether the query returned any results.
        hit: bool,
    },
}

/// Snapshot of the query workload history.
#[derive(Debug, Clone)]
pub struct WorkloadHistory {
    pub range_count: u64,
    pub knn_count: u64,
    pub join_count: u64,
    pub range_hits: u64,
    pub knn_hits: u64,
    pub join_hits: u64,
    pub total_selectivity: f64,
    pub selectivity_count: u64,
}

impl WorkloadHistory {
    fn new() -> Self {
        Self {
            range_count: 0,
            knn_count: 0,
            join_count: 0,
            range_hits: 0,
            knn_hits: 0,
            join_hits: 0,
            total_selectivity: 0.0,
            selectivity_count: 0,
        }
    }

    /// Compute a [`QueryMix`] from the accumulated history.
    pub fn query_mix(&self) -> QueryMix {
        let total = (self.range_count + self.knn_count + self.join_count) as f64;
        if total < 1.0 {
            return QueryMix::default();
        }
        let mean_selectivity = if self.selectivity_count > 0 {
            self.total_selectivity / self.selectivity_count as f64
        } else {
            0.01
        };
        QueryMix {
            range_frac: self.range_count as f64 / total,
            knn_frac: self.knn_count as f64 / total,
            join_frac: self.join_count as f64 / total,
            mean_selectivity,
        }
    }

    /// Per-type hit rates as `(range_hit_rate, knn_hit_rate, join_hit_rate)`.
    pub fn hit_rates(&self) -> (f64, f64, f64) {
        let range_hr = if self.range_count > 0 {
            self.range_hits as f64 / self.range_count as f64
        } else {
            0.0
        };
        let knn_hr = if self.knn_count > 0 {
            self.knn_hits as f64 / self.knn_count as f64
        } else {
            0.0
        };
        let join_hr = if self.join_count > 0 {
            self.join_hits as f64 / self.join_count as f64
        } else {
            0.0
        };
        (range_hr, knn_hr, join_hr)
    }
}

/// The Profiler orchestrates reservoir sampling, statistics computation, and
/// query workload tracking.
///
/// Observations are sent via the [`Profiler::sender`] handle and processed
/// in batches of 64 when [`Profiler::flush`] or [`Profiler::process_pending`]
/// is called.
pub struct Profiler<C: CoordType, const D: usize> {
    sender: Sender<Observation<C, D>>,
    receiver: Receiver<Observation<C, D>>,
    sampler: ReservoirSampler<C, D>,
    stats: OnlineStats<C, D>,
    workload: WorkloadHistory,
    last_shape: Option<DataShape<D>>,
}

/// Batch size for processing observations.
const BATCH_SIZE: usize = 64;

impl<C: CoordType, const D: usize> Profiler<C, D> {
    /// Create a new `Profiler` with the given reservoir capacity.
    pub fn new(reservoir_capacity: usize) -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            sender,
            receiver,
            sampler: ReservoirSampler::new(reservoir_capacity),
            stats: OnlineStats::new(),
            workload: WorkloadHistory::new(),
            last_shape: None,
        }
    }

    /// Create a `Profiler` with the default reservoir capacity of 4096.
    pub fn default_capacity() -> Self {
        Self::new(4096)
    }

    /// Return a clone of the sender handle for submitting observations.
    ///
    /// Multiple senders can be created from this handle for multi-producer use.
    pub fn sender(&self) -> Sender<Observation<C, D>> {
        self.sender.clone()
    }

    /// Submit an observation directly (bypasses the channel).
    ///
    /// Convenience method for single-threaded use. Call [`Profiler::flush`] to
    /// force an immediate stats recompute.
    pub fn observe(&mut self, obs: Observation<C, D>) {
        self.process_observation(obs);
    }

    /// Process all pending observations from the channel in batches of 64.
    ///
    /// Returns the number of observations processed.
    pub fn process_pending(&mut self) -> usize {
        let mut total = 0;
        loop {
            let mut batch_count = 0;
            while batch_count < BATCH_SIZE {
                match self.receiver.try_recv() {
                    Ok(obs) => {
                        self.process_observation(obs);
                        batch_count += 1;
                    }
                    Err(_) => break,
                }
            }
            total += batch_count;
            if batch_count < BATCH_SIZE {
                break;
            }
        }
        if total > 0 {
            self.recompute_stats();
        }
        total
    }

    /// Flush all pending channel observations and recompute statistics.
    pub fn flush(&mut self) {
        self.process_pending();
        self.recompute_stats();
    }

    /// Return the last computed [`DataShape`], or `None` if no data has been observed.
    pub fn data_shape(&self) -> Option<&DataShape<D>> {
        self.last_shape.as_ref()
    }

    /// Return a reference to the query workload history.
    pub fn workload(&self) -> &WorkloadHistory {
        &self.workload
    }

    /// Return the number of points in the reservoir.
    pub fn reservoir_len(&self) -> usize {
        self.sampler.len()
    }

    /// Return the total number of points observed.
    pub fn total_observed(&self) -> usize {
        self.sampler.total_count()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn process_observation(&mut self, obs: Observation<C, D>) {
        match obs {
            Observation::Insert(point) => {
                self.sampler.update(point);
            }
            Observation::Query {
                kind,
                selectivity,
                hit,
            } => {
                self.workload.total_selectivity += selectivity;
                self.workload.selectivity_count += 1;
                match kind {
                    QueryKind::Range => {
                        self.workload.range_count += 1;
                        if hit {
                            self.workload.range_hits += 1;
                        }
                    }
                    QueryKind::Knn => {
                        self.workload.knn_count += 1;
                        if hit {
                            self.workload.knn_hits += 1;
                        }
                    }
                    QueryKind::Join => {
                        self.workload.join_count += 1;
                        if hit {
                            self.workload.join_hits += 1;
                        }
                    }
                }
            }
        }
    }

    fn recompute_stats(&mut self) {
        let query_mix = self.workload.query_mix();
        self.last_shape = self.stats.compute(self.sampler.samples(), query_mix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point;

    #[test]
    fn profiler_processes_inserts() {
        let mut profiler = Profiler::<f64, 2>::new(100);
        for i in 0..50 {
            profiler.observe(Observation::Insert(Point::new([i as f64, i as f64])));
        }
        assert_eq!(profiler.reservoir_len(), 50);
        assert_eq!(profiler.total_observed(), 50);
    }

    #[test]
    fn profiler_channel_processing() {
        let mut profiler = Profiler::<f64, 2>::new(100);
        let sender = profiler.sender();

        // Send observations via channel.
        for i in 0..200 {
            sender
                .send(Observation::Insert(Point::new([i as f64, 0.0])))
                .unwrap();
        }
        drop(sender);

        let processed = profiler.process_pending();
        assert_eq!(processed, 200);
        assert_eq!(profiler.reservoir_len(), 100); // capped at capacity
    }

    #[test]
    fn profiler_tracks_query_workload() {
        let mut profiler = Profiler::<f64, 2>::new(100);

        // Insert some points first.
        for i in 0..50 {
            profiler.observe(Observation::Insert(Point::new([i as f64, 0.0])));
        }

        // Record queries.
        for _ in 0..60 {
            profiler.observe(Observation::Query {
                kind: QueryKind::Range,
                selectivity: 0.01,
                hit: true,
            });
        }
        for _ in 0..40 {
            profiler.observe(Observation::Query {
                kind: QueryKind::Knn,
                selectivity: 0.001,
                hit: false,
            });
        }

        let mix = profiler.workload().query_mix();
        assert!((mix.range_frac - 0.6).abs() < 0.01);
        assert!((mix.knn_frac - 0.4).abs() < 0.01);
        assert!((mix.join_frac).abs() < 0.01);

        let (range_hr, knn_hr, _) = profiler.workload().hit_rates();
        assert!((range_hr - 1.0).abs() < 0.01);
        assert!((knn_hr - 0.0).abs() < 0.01);
    }

    #[test]
    fn profiler_data_shape_computed_after_flush() {
        let mut profiler = Profiler::<f64, 2>::new(100);
        for i in 0..100 {
            profiler.observe(Observation::Insert(Point::new([i as f64, i as f64])));
        }
        profiler.flush();
        assert!(profiler.data_shape().is_some());
        let shape = profiler.data_shape().unwrap();
        assert_eq!(shape.point_count, 100);
    }

    #[test]
    fn profiler_batch_size_is_64() {
        // Verify that process_pending processes in batches of 64.
        let mut profiler = Profiler::<f64, 2>::new(1000);
        let sender = profiler.sender();

        for i in 0..128 {
            sender
                .send(Observation::Insert(Point::new([i as f64, 0.0])))
                .unwrap();
        }
        drop(sender);

        let processed = profiler.process_pending();
        assert_eq!(processed, 128);
    }
}
