use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use bonsai::index::BonsaiIndex;
use bonsai::types::{BBox, BackendKind, Point};

mod datasets;

fn bench_bonsai_insert_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptation_convergence/insert_uniform_2d");

    for &n in &[1_000usize, 10_000] {
        let pts = datasets::uniform_2d(n);

        group.bench_with_input(BenchmarkId::new("bonsai_default", n), &pts, |b, pts| {
            b.iter(|| {
                let mut idx: BonsaiIndex<usize> = BonsaiIndex::builder().build();
                for (i, &p) in pts.iter().enumerate() {
                    black_box(idx.insert(p, i));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("bonsai_rtree", n), &pts, |b, pts| {
            b.iter(|| {
                let mut idx: BonsaiIndex<usize> = BonsaiIndex::builder()
                    .initial_backend(BackendKind::RTree)
                    .build();
                for (i, &p) in pts.iter().enumerate() {
                    black_box(idx.insert(p, i));
                }
            });
        });
    }

    group.finish();
}

fn bench_bonsai_range_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptation_convergence/range_query_2d");

    for &n in &[1_000usize, 10_000] {
        let pts = datasets::uniform_2d(n);
        let bbox = BBox::new(Point::new([300.0, 300.0]), Point::new([700.0, 700.0]));

        group.bench_with_input(BenchmarkId::new("bonsai_default", n), &pts, |b, pts| {
            b.iter_batched(
                || {
                    let mut idx: BonsaiIndex<usize> = BonsaiIndex::builder().build();
                    for (i, &p) in pts.iter().enumerate() {
                        idx.insert(p, i);
                    }
                    idx
                },
                |mut idx| black_box(idx.range_query(&bbox)),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_bonsai_clustered(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptation_convergence/clustered_2d");
    let n = 5_000usize;
    let pts = datasets::clustered_2d(n);
    let bbox = BBox::new(Point::new([300.0, 300.0]), Point::new([700.0, 700.0]));

    group.bench_function("insert", |b| {
        b.iter(|| {
            let mut idx: BonsaiIndex<usize> = BonsaiIndex::builder().build();
            for (i, &p) in pts.iter().enumerate() {
                black_box(idx.insert(p, i));
            }
        });
    });

    group.bench_function("range_query", |b| {
        b.iter_batched(
            || {
                let mut idx: BonsaiIndex<usize> = BonsaiIndex::builder().build();
                for (i, &p) in pts.iter().enumerate() {
                    idx.insert(p, i);
                }
                idx
            },
            |mut idx| black_box(idx.range_query(&bbox)),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bonsai_insert_uniform,
    bench_bonsai_range_query,
    bench_bonsai_clustered
);
criterion_main!(benches);
