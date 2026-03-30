use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use bonsai::backends::{GridIndex, KDTree, Quadtree, RTree, SpatialBackend};
use bonsai::types::Point;

mod datasets;

fn bench_insert_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput/2d");

    for &n in &[1_000usize, 10_000, 100_000] {
        let pts = datasets::uniform_2d(n);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("rtree", n), &pts, |b, pts| {
            b.iter(|| {
                let mut idx = RTree::<usize, f64, 2>::new();
                for (i, &p) in pts.iter().enumerate() {
                    black_box(idx.insert(p, i));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("kdtree", n), &pts, |b, pts| {
            b.iter(|| {
                let mut idx = KDTree::<usize, f64, 2>::new();
                for (i, &p) in pts.iter().enumerate() {
                    black_box(idx.insert(p, i));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("quadtree", n), &pts, |b, pts| {
            b.iter(|| {
                let mut idx = Quadtree::<usize, f64, 2>::new();
                for (i, &p) in pts.iter().enumerate() {
                    black_box(idx.insert(p, i));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("grid", n), &pts, |b, pts| {
            b.iter(|| {
                let mut idx = GridIndex::<usize, f64, 2>::new([10.0, 10.0], Point::new([0.0, 0.0]));
                for (i, &p) in pts.iter().enumerate() {
                    black_box(idx.insert(p, i));
                }
            });
        });
    }

    group.finish();
}

fn bench_insert_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput/3d");
    let n = 10_000usize;
    let pts = datasets::uniform_3d(n);
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("rtree", |b| {
        b.iter(|| {
            let mut idx = RTree::<usize, f64, 3>::new();
            for (i, &p) in pts.iter().enumerate() {
                black_box(idx.insert(p, i));
            }
        });
    });

    group.bench_function("kdtree", |b| {
        b.iter(|| {
            let mut idx = KDTree::<usize, f64, 3>::new();
            for (i, &p) in pts.iter().enumerate() {
                black_box(idx.insert(p, i));
            }
        });
    });

    group.bench_function("grid", |b| {
        b.iter(|| {
            let mut idx =
                GridIndex::<usize, f64, 3>::new([10.0, 10.0, 10.0], Point::new([0.0, 0.0, 0.0]));
            for (i, &p) in pts.iter().enumerate() {
                black_box(idx.insert(p, i));
            }
        });
    });

    group.finish();
}

fn bench_insert_6d(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput/6d");
    let n = 5_000usize;
    let pts = datasets::uniform_6d(n);
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("rtree", |b| {
        b.iter(|| {
            let mut idx = RTree::<usize, f64, 6>::new();
            for (i, &p) in pts.iter().enumerate() {
                black_box(idx.insert(p, i));
            }
        });
    });

    group.bench_function("kdtree", |b| {
        b.iter(|| {
            let mut idx = KDTree::<usize, f64, 6>::new();
            for (i, &p) in pts.iter().enumerate() {
                black_box(idx.insert(p, i));
            }
        });
    });

    group.bench_function("robotics_6d", |b| {
        let pts6 = datasets::robotics_6d(n);
        b.iter(|| {
            let mut idx = KDTree::<usize, f64, 6>::new();
            for (i, &p) in pts6.iter().enumerate() {
                black_box(idx.insert(p, i));
            }
        });
    });

    group.finish();
}

fn bench_insert_clustered(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput/clustered_2d");
    let n = 10_000usize;
    let pts = datasets::clustered_2d(n);
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("rtree", |b| {
        b.iter(|| {
            let mut idx = RTree::<usize, f64, 2>::new();
            for (i, &p) in pts.iter().enumerate() {
                black_box(idx.insert(p, i));
            }
        });
    });

    group.bench_function("kdtree", |b| {
        b.iter(|| {
            let mut idx = KDTree::<usize, f64, 2>::new();
            for (i, &p) in pts.iter().enumerate() {
                black_box(idx.insert(p, i));
            }
        });
    });

    group.bench_function("grid", |b| {
        b.iter(|| {
            let mut idx = GridIndex::<usize, f64, 2>::new([10.0, 10.0], Point::new([0.0, 0.0]));
            for (i, &p) in pts.iter().enumerate() {
                black_box(idx.insert(p, i));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_2d,
    bench_insert_3d,
    bench_insert_6d,
    bench_insert_clustered
);
criterion_main!(benches);
