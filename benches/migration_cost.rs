use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use bonsai::backends::{GridIndex, KDTree, Quadtree, RTree, SpatialBackend};
use bonsai::types::Point;

mod datasets;

fn bench_bulk_load_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("migration_cost/bulk_load_2d");

    for &n in &[10_000usize, 100_000] {
        let pts = datasets::uniform_2d(n);
        let entries: Vec<(Point<f64, 2>, usize)> =
            pts.iter().enumerate().map(|(i, &p)| (p, i)).collect();

        group.bench_with_input(BenchmarkId::new("rtree", n), &entries, |b, e| {
            b.iter(|| black_box(RTree::<usize, f64, 2>::bulk_load(e.clone())));
        });

        group.bench_with_input(BenchmarkId::new("kdtree", n), &entries, |b, e| {
            b.iter(|| black_box(KDTree::<usize, f64, 2>::bulk_load(e.clone())));
        });

        group.bench_with_input(BenchmarkId::new("quadtree", n), &entries, |b, e| {
            b.iter(|| black_box(Quadtree::<usize, f64, 2>::bulk_load(e.clone())));
        });

        group.bench_with_input(BenchmarkId::new("grid", n), &entries, |b, e| {
            b.iter(|| black_box(GridIndex::<usize, f64, 2>::bulk_load(e.clone())));
        });
    }

    group.finish();
}

fn bench_bulk_load_clustered(c: &mut Criterion) {
    let mut group = c.benchmark_group("migration_cost/bulk_load_clustered_2d");
    let n = 10_000usize;
    let pts = datasets::clustered_2d(n);
    let entries: Vec<(Point<f64, 2>, usize)> =
        pts.iter().enumerate().map(|(i, &p)| (p, i)).collect();

    group.bench_function("rtree", |b| {
        b.iter(|| black_box(RTree::<usize, f64, 2>::bulk_load(entries.clone())));
    });

    group.bench_function("kdtree", |b| {
        b.iter(|| black_box(KDTree::<usize, f64, 2>::bulk_load(entries.clone())));
    });

    group.finish();
}

fn bench_bulk_load_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("migration_cost/bulk_load_3d");
    let n = 10_000usize;
    let pts = datasets::uniform_3d(n);
    let entries: Vec<(Point<f64, 3>, usize)> =
        pts.iter().enumerate().map(|(i, &p)| (p, i)).collect();

    group.bench_function("rtree", |b| {
        b.iter(|| black_box(RTree::<usize, f64, 3>::bulk_load(entries.clone())));
    });

    group.bench_function("kdtree", |b| {
        b.iter(|| black_box(KDTree::<usize, f64, 3>::bulk_load(entries.clone())));
    });

    group.finish();
}

fn bench_bulk_load_6d(c: &mut Criterion) {
    let mut group = c.benchmark_group("migration_cost/bulk_load_6d");
    let n = 5_000usize;
    let pts = datasets::uniform_6d(n);
    let entries: Vec<(Point<f64, 6>, usize)> =
        pts.iter().enumerate().map(|(i, &p)| (p, i)).collect();

    group.bench_function("rtree", |b| {
        b.iter(|| black_box(RTree::<usize, f64, 6>::bulk_load(entries.clone())));
    });

    group.bench_function("kdtree", |b| {
        b.iter(|| black_box(KDTree::<usize, f64, 6>::bulk_load(entries.clone())));
    });

    let pts6 = datasets::robotics_6d(n);
    let entries6: Vec<(Point<f64, 6>, usize)> =
        pts6.iter().enumerate().map(|(i, &p)| (p, i)).collect();
    group.bench_function("kdtree_robotics", |b| {
        b.iter(|| black_box(KDTree::<usize, f64, 6>::bulk_load(entries6.clone())));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bulk_load_2d,
    bench_bulk_load_clustered,
    bench_bulk_load_3d,
    bench_bulk_load_6d
);
criterion_main!(benches);
