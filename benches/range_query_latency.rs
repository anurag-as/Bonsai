use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use bonsai::backends::{GridIndex, KDTree, Quadtree, RTree, SpatialBackend};
use bonsai::types::{BBox, Point};

mod datasets;

fn build_rtree_2d(pts: &[Point<f64, 2>]) -> RTree<usize, f64, 2> {
    let mut idx = RTree::new();
    for (i, &p) in pts.iter().enumerate() {
        idx.insert(p, i);
    }
    idx
}

fn build_kdtree_2d(pts: &[Point<f64, 2>]) -> KDTree<usize, f64, 2> {
    let mut idx = KDTree::new();
    for (i, &p) in pts.iter().enumerate() {
        idx.insert(p, i);
    }
    idx
}

fn build_quadtree_2d(pts: &[Point<f64, 2>]) -> Quadtree<usize, f64, 2> {
    let mut idx = Quadtree::new();
    for (i, &p) in pts.iter().enumerate() {
        idx.insert(p, i);
    }
    idx
}

fn build_grid_2d(pts: &[Point<f64, 2>]) -> GridIndex<usize, f64, 2> {
    let mut idx = GridIndex::new([10.0, 10.0], Point::new([0.0, 0.0]));
    for (i, &p) in pts.iter().enumerate() {
        idx.insert(p, i);
    }
    idx
}

fn bench_range_uniform_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_query_latency/uniform_2d");

    for &n in &[10_000usize, 100_000] {
        let pts = datasets::uniform_2d(n);
        let bbox = BBox::new(Point::new([300.0, 300.0]), Point::new([700.0, 700.0]));

        let rt = build_rtree_2d(&pts);
        group.bench_with_input(BenchmarkId::new("rtree", n), &rt, |b, idx| {
            b.iter(|| black_box(idx.range_query(&bbox)));
        });

        let kd = build_kdtree_2d(&pts);
        group.bench_with_input(BenchmarkId::new("kdtree", n), &kd, |b, idx| {
            b.iter(|| black_box(idx.range_query(&bbox)));
        });

        let qt = build_quadtree_2d(&pts);
        group.bench_with_input(BenchmarkId::new("quadtree", n), &qt, |b, idx| {
            b.iter(|| black_box(idx.range_query(&bbox)));
        });

        let gr = build_grid_2d(&pts);
        group.bench_with_input(BenchmarkId::new("grid", n), &gr, |b, idx| {
            b.iter(|| black_box(idx.range_query(&bbox)));
        });
    }

    group.finish();
}

fn bench_range_clustered_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_query_latency/clustered_2d");
    let n = 10_000usize;
    let pts = datasets::clustered_2d(n);
    let bbox = BBox::new(Point::new([300.0, 300.0]), Point::new([700.0, 700.0]));

    let rt = build_rtree_2d(&pts);
    group.bench_function("rtree", |b| {
        b.iter(|| black_box(rt.range_query(&bbox)));
    });

    let kd = build_kdtree_2d(&pts);
    group.bench_function("kdtree", |b| {
        b.iter(|| black_box(kd.range_query(&bbox)));
    });

    let qt = build_quadtree_2d(&pts);
    group.bench_function("quadtree", |b| {
        b.iter(|| black_box(qt.range_query(&bbox)));
    });

    let gr = build_grid_2d(&pts);
    group.bench_function("grid", |b| {
        b.iter(|| black_box(gr.range_query(&bbox)));
    });

    group.finish();
}

fn bench_range_osm_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_query_latency/osm_2d");
    let n = 10_000usize;
    let pts = datasets::osm_2d(n);
    let bbox = BBox::new(Point::new([200.0, 200.0]), Point::new([800.0, 800.0]));

    let rt = build_rtree_2d(&pts);
    group.bench_function("rtree", |b| {
        b.iter(|| black_box(rt.range_query(&bbox)));
    });

    let kd = build_kdtree_2d(&pts);
    group.bench_function("kdtree", |b| {
        b.iter(|| black_box(kd.range_query(&bbox)));
    });

    let gr = build_grid_2d(&pts);
    group.bench_function("grid", |b| {
        b.iter(|| black_box(gr.range_query(&bbox)));
    });

    group.finish();
}

fn bench_range_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_query_latency/uniform_3d");
    let n = 10_000usize;
    let pts = datasets::uniform_3d(n);
    let bbox = BBox::new(
        Point::new([300.0, 300.0, 300.0]),
        Point::new([700.0, 700.0, 700.0]),
    );

    let mut rt: RTree<usize, f64, 3> = RTree::new();
    for (i, &p) in pts.iter().enumerate() {
        rt.insert(p, i);
    }
    group.bench_function("rtree", |b| {
        b.iter(|| black_box(rt.range_query(&bbox)));
    });

    let mut kd: KDTree<usize, f64, 3> = KDTree::new();
    for (i, &p) in pts.iter().enumerate() {
        kd.insert(p, i);
    }
    group.bench_function("kdtree", |b| {
        b.iter(|| black_box(kd.range_query(&bbox)));
    });

    group.finish();
}

fn bench_range_6d(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_query_latency/uniform_6d");
    let n = 5_000usize;
    let pts = datasets::uniform_6d(n);
    let bbox = BBox::new(
        Point::new([300.0, 300.0, 300.0, 300.0, 300.0, 300.0]),
        Point::new([700.0, 700.0, 700.0, 700.0, 700.0, 700.0]),
    );

    let mut rt: RTree<usize, f64, 6> = RTree::new();
    for (i, &p) in pts.iter().enumerate() {
        rt.insert(p, i);
    }
    group.bench_function("rtree", |b| {
        b.iter(|| black_box(rt.range_query(&bbox)));
    });

    let mut kd: KDTree<usize, f64, 6> = KDTree::new();
    for (i, &p) in pts.iter().enumerate() {
        kd.insert(p, i);
    }
    group.bench_function("kdtree", |b| {
        b.iter(|| black_box(kd.range_query(&bbox)));
    });

    let pts6 = datasets::robotics_6d(n);
    let bbox6 = BBox::new(
        Point::new([-2.0, -2.0, -2.0, -1.0, -1.0, -1.0]),
        Point::new([2.0, 2.0, 2.0, 1.0, 1.0, 1.0]),
    );
    let mut kd6: KDTree<usize, f64, 6> = KDTree::new();
    for (i, &p) in pts6.iter().enumerate() {
        kd6.insert(p, i);
    }
    group.bench_function("kdtree_robotics", |b| {
        b.iter(|| black_box(kd6.range_query(&bbox6)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_range_uniform_2d,
    bench_range_clustered_2d,
    bench_range_osm_2d,
    bench_range_3d,
    bench_range_6d
);
criterion_main!(benches);
