use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use bonsai::backends::{GridIndex, KDTree, Quadtree, RTree, SpatialBackend};
use bonsai::types::Point;

mod datasets;

fn bench_knn_uniform_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_latency/uniform_2d");
    let query = Point::new([500.0, 500.0]);

    for &n in &[10_000usize, 100_000] {
        let pts = datasets::uniform_2d(n);

        let mut rt: RTree<usize, f64, 2> = RTree::new();
        for (i, &p) in pts.iter().enumerate() {
            rt.insert(p, i);
        }
        group.bench_with_input(BenchmarkId::new("rtree_k10", n), &rt, |b, idx| {
            b.iter(|| black_box(idx.knn_query(&query, 10)));
        });

        let mut kd: KDTree<usize, f64, 2> = KDTree::new();
        for (i, &p) in pts.iter().enumerate() {
            kd.insert(p, i);
        }
        group.bench_with_input(BenchmarkId::new("kdtree_k10", n), &kd, |b, idx| {
            b.iter(|| black_box(idx.knn_query(&query, 10)));
        });

        let mut qt: Quadtree<usize, f64, 2> = Quadtree::new();
        for (i, &p) in pts.iter().enumerate() {
            qt.insert(p, i);
        }
        group.bench_with_input(BenchmarkId::new("quadtree_k10", n), &qt, |b, idx| {
            b.iter(|| black_box(idx.knn_query(&query, 10)));
        });

        let mut gr: GridIndex<usize, f64, 2> = GridIndex::new([10.0, 10.0], Point::new([0.0, 0.0]));
        for (i, &p) in pts.iter().enumerate() {
            gr.insert(p, i);
        }
        group.bench_with_input(BenchmarkId::new("grid_k10", n), &gr, |b, idx| {
            b.iter(|| black_box(idx.knn_query(&query, 10)));
        });
    }

    group.finish();
}

fn bench_knn_clustered_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_latency/clustered_2d");
    let n = 10_000usize;
    let pts = datasets::clustered_2d(n);
    let query = Point::new([500.0, 500.0]);

    let mut rt: RTree<usize, f64, 2> = RTree::new();
    for (i, &p) in pts.iter().enumerate() {
        rt.insert(p, i);
    }
    group.bench_function("rtree_k10", |b| {
        b.iter(|| black_box(rt.knn_query(&query, 10)));
    });

    let mut kd: KDTree<usize, f64, 2> = KDTree::new();
    for (i, &p) in pts.iter().enumerate() {
        kd.insert(p, i);
    }
    group.bench_function("kdtree_k10", |b| {
        b.iter(|| black_box(kd.knn_query(&query, 10)));
    });

    group.finish();
}

fn bench_knn_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_latency/uniform_3d");
    let n = 10_000usize;
    let pts = datasets::uniform_3d(n);
    let query = Point::new([500.0, 500.0, 500.0]);

    let mut rt: RTree<usize, f64, 3> = RTree::new();
    for (i, &p) in pts.iter().enumerate() {
        rt.insert(p, i);
    }
    group.bench_function("rtree_k10", |b| {
        b.iter(|| black_box(rt.knn_query(&query, 10)));
    });

    let mut kd: KDTree<usize, f64, 3> = KDTree::new();
    for (i, &p) in pts.iter().enumerate() {
        kd.insert(p, i);
    }
    group.bench_function("kdtree_k10", |b| {
        b.iter(|| black_box(kd.knn_query(&query, 10)));
    });

    group.finish();
}

fn bench_knn_6d(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_latency/uniform_6d");
    let n = 5_000usize;
    let pts = datasets::uniform_6d(n);
    let query = Point::new([500.0, 500.0, 500.0, 500.0, 500.0, 500.0]);

    let mut rt: RTree<usize, f64, 6> = RTree::new();
    for (i, &p) in pts.iter().enumerate() {
        rt.insert(p, i);
    }
    group.bench_function("rtree_k10", |b| {
        b.iter(|| black_box(rt.knn_query(&query, 10)));
    });

    let mut kd: KDTree<usize, f64, 6> = KDTree::new();
    for (i, &p) in pts.iter().enumerate() {
        kd.insert(p, i);
    }
    group.bench_function("kdtree_k10", |b| {
        b.iter(|| black_box(kd.knn_query(&query, 10)));
    });

    let pts6 = datasets::robotics_6d(n);
    let query6 = Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let mut kd6: KDTree<usize, f64, 6> = KDTree::new();
    for (i, &p) in pts6.iter().enumerate() {
        kd6.insert(p, i);
    }
    group.bench_function("kdtree_robotics_k10", |b| {
        b.iter(|| black_box(kd6.knn_query(&query6, 10)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_knn_uniform_2d,
    bench_knn_clustered_2d,
    bench_knn_3d,
    bench_knn_6d
);
criterion_main!(benches);
