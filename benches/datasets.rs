#![allow(dead_code)]

use bonsai::types::Point;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

pub fn uniform_2d(n: usize) -> Vec<Point<f64, 2>> {
    let mut rng = Lcg::new(1);
    (0..n)
        .map(|_| Point::new([rng.next_f64() * 1000.0, rng.next_f64() * 1000.0]))
        .collect()
}

pub fn uniform_3d(n: usize) -> Vec<Point<f64, 3>> {
    let mut rng = Lcg::new(2);
    (0..n)
        .map(|_| {
            Point::new([
                rng.next_f64() * 1000.0,
                rng.next_f64() * 1000.0,
                rng.next_f64() * 1000.0,
            ])
        })
        .collect()
}

pub fn uniform_6d(n: usize) -> Vec<Point<f64, 6>> {
    let mut rng = Lcg::new(3);
    (0..n)
        .map(|_| {
            Point::new([
                rng.next_f64() * 1000.0,
                rng.next_f64() * 1000.0,
                rng.next_f64() * 1000.0,
                rng.next_f64() * 1000.0,
                rng.next_f64() * 1000.0,
                rng.next_f64() * 1000.0,
            ])
        })
        .collect()
}

pub fn clustered_2d(n: usize) -> Vec<Point<f64, 2>> {
    let mut rng = Lcg::new(4);
    const CLUSTERS: usize = 20;
    let centres: Vec<[f64; 2]> = (0..CLUSTERS)
        .map(|_| [rng.next_f64() * 900.0 + 50.0, rng.next_f64() * 900.0 + 50.0])
        .collect();
    (0..n)
        .map(|i| {
            let c = &centres[i % CLUSTERS];
            let x = (c[0] + rng.next_normal() * 20.0).clamp(0.0, 1000.0);
            let y = (c[1] + rng.next_normal() * 20.0).clamp(0.0, 1000.0);
            Point::new([x, y])
        })
        .collect()
}

/// OSM-style 2D: road-network clustering with points distributed along linear segments.
pub fn osm_2d(n: usize) -> Vec<Point<f64, 2>> {
    let mut rng = Lcg::new(5);
    const ROADS: usize = 10;
    let road_starts: Vec<[f64; 2]> = (0..ROADS)
        .map(|_| [rng.next_f64() * 1000.0, rng.next_f64() * 1000.0])
        .collect();
    let road_dirs: Vec<[f64; 2]> = (0..ROADS)
        .map(|_| {
            let angle = rng.next_f64() * std::f64::consts::TAU;
            [angle.cos(), angle.sin()]
        })
        .collect();
    (0..n)
        .map(|i| {
            let r = i % ROADS;
            let t = rng.next_f64() * 500.0;
            let x = (road_starts[r][0] + road_dirs[r][0] * t + rng.next_normal() * 5.0)
                .clamp(0.0, 1000.0);
            let y = (road_starts[r][1] + road_dirs[r][1] * t + rng.next_normal() * 5.0)
                .clamp(0.0, 1000.0);
            Point::new([x, y])
        })
        .collect()
}

/// Robotics 6D phase-space: position (x, y, z) in [-5, 5] and velocity (vx, vy, vz) in [-2, 2].
pub fn robotics_6d(n: usize) -> Vec<Point<f64, 6>> {
    let mut rng = Lcg::new(6);
    (0..n)
        .map(|_| {
            Point::new([
                rng.next_f64() * 10.0 - 5.0,
                rng.next_f64() * 10.0 - 5.0,
                rng.next_f64() * 10.0 - 5.0,
                rng.next_f64() * 4.0 - 2.0,
                rng.next_f64() * 4.0 - 2.0,
                rng.next_f64() * 4.0 - 2.0,
            ])
        })
        .collect()
}
