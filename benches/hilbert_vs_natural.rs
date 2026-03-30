use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use bonsai::backends::{KDTree, RTree, SpatialBackend};
use bonsai::hilbert::HilbertCurve;
use bonsai::types::{BBox, Point};

mod datasets;

fn normalise(v: f64, domain: f64, order: u32) -> u64 {
    let cells = (1u64 << order) as f64;
    ((v / domain) * cells).clamp(0.0, cells - 1.0) as u64
}

fn hilbert_sort(pts: &[Point<f64, 2>]) -> Vec<Point<f64, 2>> {
    let hilbert = HilbertCurve::<2>::new(10);
    let mut indexed: Vec<(u128, Point<f64, 2>)> = pts
        .iter()
        .map(|&p| {
            let hx = normalise(p.coords()[0], 1000.0, hilbert.order);
            let hy = normalise(p.coords()[1], 1000.0, hilbert.order);
            (hilbert.index(&[hx, hy]), p)
        })
        .collect();
    indexed.sort_by_key(|&(h, _)| h);
    indexed.into_iter().map(|(_, p)| p).collect()
}

fn bench_insert_order_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("hilbert_vs_natural/insert_2d");

    for &n in &[10_000usize, 100_000] {
        let pts_natural = datasets::uniform_2d(n);
        let pts_hilbert = hilbert_sort(&pts_natural);

        group.bench_with_input(
            BenchmarkId::new("kdtree_natural", n),
            &pts_natural,
            |b, pts| {
                b.iter(|| {
                    let mut idx = KDTree::<usize, f64, 2>::new();
                    for (i, &p) in pts.iter().enumerate() {
                        black_box(idx.insert(p, i));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("kdtree_hilbert", n),
            &pts_hilbert,
            |b, pts| {
                b.iter(|| {
                    let mut idx = KDTree::<usize, f64, 2>::new();
                    for (i, &p) in pts.iter().enumerate() {
                        black_box(idx.insert(p, i));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rtree_natural", n),
            &pts_natural,
            |b, pts| {
                b.iter(|| {
                    let mut idx = RTree::<usize, f64, 2>::new();
                    for (i, &p) in pts.iter().enumerate() {
                        black_box(idx.insert(p, i));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rtree_hilbert", n),
            &pts_hilbert,
            |b, pts| {
                b.iter(|| {
                    let mut idx = RTree::<usize, f64, 2>::new();
                    for (i, &p) in pts.iter().enumerate() {
                        black_box(idx.insert(p, i));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_range_after_insert_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("hilbert_vs_natural/range_after_insert_2d");
    let n = 10_000usize;
    let pts_natural = datasets::uniform_2d(n);
    let pts_hilbert = hilbert_sort(&pts_natural);
    let bbox = BBox::new(Point::new([300.0, 300.0]), Point::new([700.0, 700.0]));

    let mut kd_natural: KDTree<usize, f64, 2> = KDTree::new();
    for (i, &p) in pts_natural.iter().enumerate() {
        kd_natural.insert(p, i);
    }
    group.bench_function("kdtree_natural", |b| {
        b.iter(|| black_box(kd_natural.range_query(&bbox)));
    });

    let mut kd_hilbert: KDTree<usize, f64, 2> = KDTree::new();
    for (i, &p) in pts_hilbert.iter().enumerate() {
        kd_hilbert.insert(p, i);
    }
    group.bench_function("kdtree_hilbert", |b| {
        b.iter(|| black_box(kd_hilbert.range_query(&bbox)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_order_2d,
    bench_range_after_insert_order
);
criterion_main!(benches);
