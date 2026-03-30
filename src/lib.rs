pub mod backends;
pub mod bloom;
pub mod hilbert;
pub mod index;
pub mod migration;
pub mod profiler;
pub mod router;
pub mod stats;
pub mod types;

pub use backends::{GridIndex, KDTree, Quadtree, RTree, SpatialBackend};
pub use index::{BonsaiConfig, BonsaiIndex};
pub use profiler::{
    CostEstimate, CostModel, MigrationDecision, Observation, PolicyEngine, Profiler, QueryKind,
    WorkloadHistory,
};
pub use router::IndexRouter;
pub use stats::StatsCollector;
pub use types::{
    BBox, BackendKind, BonsaiError, CoordType, DataShape, EntryId, Point, QueryMix, Stats,
};
