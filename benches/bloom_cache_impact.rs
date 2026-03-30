use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use bonsai::backends::{KDTree, SpatialBackend};
use bonsai::bloom::{BloomCache, BloomResult};
use bonsai::types::{BBox, Point};

mod datasets;

fn build_kdtree(pts: &[Point<f64, 2>]) -> KDTree<usize, f64, 2> {
    let mut idx = KDTree::new();
    for (i, &p) in pts.iter().enumerate() {
        idx.insert(p, i);
    }
    idx
}

fn build_bloom(pts: &[Point<f64, 2>]) -> BloomCache<2> {
    let mut bloom = BloomCache::<2>::new(65_536, 7);
    for &p in pts {
        bloom.insert(&BBox::new(p, p));
    }
    bloom
}

fn bench_range_with_bloom(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_cache_impact");

    for &n in &[10_000usize, 100_000] {
        let pts = datasets::uniform_2d(n);
        let kd = build_kdtree(&pts);
        let bloom = build_bloom(&pts);

        let hit_bbox = BBox::new(Point::new([300.0, 300.0]), Point::new([700.0, 700.0]));
        // outside the [0, 1000] domain — guaranteed bloom miss
        let miss_bbox = BBox::new(
            Point::new([1_100.0, 1_100.0]),
            Point::new([1_200.0, 1_200.0]),
        );

        group.bench_with_input(BenchmarkId::new("no_bloom_hit", n), &kd, |b, idx| {
            b.iter(|| black_box(idx.range_query(&hit_bbox)));
        });

        group.bench_with_input(BenchmarkId::new("bloom_check_hit", n), &bloom, |b, bl| {
            b.iter(|| {
                let result = bl.check(&hit_bbox);
                if matches!(result, BloomResult::ProbablyPresent) {
                    black_box(kd.range_query(&hit_bbox))
                } else {
                    black_box(vec![])
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("bloom_check_miss", n), &bloom, |b, bl| {
            b.iter(|| {
                let result = bl.check(&miss_bbox);
                if matches!(result, BloomResult::ProbablyPresent) {
                    black_box(kd.range_query(&miss_bbox))
                } else {
                    black_box(vec![])
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_range_with_bloom);
criterion_main!(benches);
