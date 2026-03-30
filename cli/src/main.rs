use std::fs;
use std::io::{self, BufRead, Write};
use std::time::Instant;

use bonsai::index::BonsaiIndex;
use bonsai::types::{BBox, BackendKind, Point, Stats};

const STATE_FILE: &str = "bonsai_index.bin";
const META_FILE: &str = "bonsai_meta.txt";

fn load_state() -> Option<BonsaiIndex<String>> {
    let bytes = fs::read(STATE_FILE).ok()?;
    BonsaiIndex::<String>::from_bytes(&bytes).ok()
}

fn save_state(index: &BonsaiIndex<String>) {
    let bytes = index.to_bytes();
    fs::write(STATE_FILE, bytes).expect("failed to write index state");
}

fn load_meta() -> Option<(f64, f64, f64, f64)> {
    let s = fs::read_to_string(META_FILE).ok()?;
    let parts: Vec<f64> = s
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    if parts.len() >= 4 {
        Some((parts[0], parts[1], parts[2], parts[3]))
    } else {
        None
    }
}

fn save_meta(min_x: f64, min_y: f64, max_x: f64, max_y: f64) {
    let s = format!("{} {} {} {}", min_x, min_y, max_x, max_y);
    let _ = fs::write(META_FILE, s);
}

fn backend_name(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::KDTree => "KDTree",
        BackendKind::RTree => "RTree",
        BackendKind::Quadtree => "Quadtree",
        BackendKind::Grid => "Grid",
    }
}

fn parse_backend(s: &str) -> Result<BackendKind, String> {
    match s.to_lowercase().as_str() {
        "kdtree" | "kd" => Ok(BackendKind::KDTree),
        "rtree" | "rt" => Ok(BackendKind::RTree),
        "quadtree" | "quad" => Ok(BackendKind::Quadtree),
        "grid" => Ok(BackendKind::Grid),
        _ => Err(format!(
            "unknown backend '{}'; use kdtree, rtree, quadtree, or grid",
            s
        )),
    }
}

fn print_stats(s: &Stats<2>) {
    println!("backend:         {}", backend_name(s.backend));
    println!("point_count:     {}", s.point_count);
    println!("migrations:      {}", s.migrations);
    println!("query_count:     {}", s.query_count);
    println!("migrating:       {}", s.migrating);
    println!("dimensions:      {}", s.dimensions);
    let ds = &s.data_shape;
    println!("data_shape:");
    println!("  points:        {}", ds.point_count);
    println!(
        "  bbox:          [{:.3},{:.3}] – [{:.3},{:.3}]",
        ds.bbox.min.coords()[0],
        ds.bbox.min.coords()[1],
        ds.bbox.max.coords()[0],
        ds.bbox.max.coords()[1],
    );
    println!(
        "  skewness:      [{:.4}, {:.4}]",
        ds.skewness[0], ds.skewness[1]
    );
    println!("  clustering:    {:.4}", ds.clustering_coef);
    println!("  overlap_ratio: {:.4}", ds.overlap_ratio);
    println!("  effective_dim: {:.4}", ds.effective_dim);
    println!(
        "  query_mix:     range={:.2} knn={:.2} join={:.2} sel={:.4}",
        ds.query_mix.range_frac,
        ds.query_mix.knn_frac,
        ds.query_mix.join_frac,
        ds.query_mix.mean_selectivity,
    );
}

fn cmd_load(path: &str) {
    // Resume an existing index so that repeated loads accumulate data and the
    // profiler can observe the evolving distribution and trigger migrations.
    let mut index = load_state().unwrap_or_else(|| BonsaiIndex::<String>::builder().build());
    let before = index.len();

    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error reading '{}': {}", path, e);
        std::process::exit(1);
    });

    let is_geojson = path.ends_with(".geojson") || path.ends_with(".json");
    let (count, min_x, min_y, max_x, max_y) = if is_geojson {
        load_geojson(&mut index, &content)
    } else {
        load_csv(&mut index, &content)
    };

    // Expand the stored bbox to cover all data loaded so far.
    let (ax0, ay0, ax1, ay1) = if let Some((px0, py0, px1, py1)) = load_meta() {
        (
            px0.min(min_x),
            py0.min(min_y),
            px1.max(max_x),
            py1.max(max_y),
        )
    } else {
        (min_x, min_y, max_x, max_y)
    };

    save_state(&index);
    save_meta(ax0, ay0, ax1, ay1);
    println!(
        "loaded {} entries from '{}' (index now has {} total)",
        count,
        path,
        before + count,
    );
}

fn load_csv(index: &mut BonsaiIndex<String>, content: &str) -> (usize, f64, f64, f64, f64) {
    let mut count = 0usize;
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut first_data_line = true;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() < 2 {
            continue;
        }
        let x = match parts[0].trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => {
                // Skip header row (first non-blank, non-comment line that
                // doesn't parse as a number).
                if first_data_line {
                    first_data_line = false;
                    continue;
                }
                continue;
            }
        };
        first_data_line = false;
        let y = match parts[1].trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let label = if parts.len() > 2 {
            parts[2].trim().to_string()
        } else {
            format!("pt{}", count)
        };
        index.insert(Point::new([x, y]), label);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        count += 1;
    }
    if count == 0 {
        (0, 0.0, 0.0, 1.0, 1.0)
    } else {
        (count, min_x, min_y, max_x, max_y)
    }
}

fn load_geojson(index: &mut BonsaiIndex<String>, content: &str) -> (usize, f64, f64, f64, f64) {
    let mut count = 0usize;
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut remaining = content;
    while let Some(pos) = remaining.find("\"coordinates\"") {
        remaining = &remaining[pos + 13..];
        let trimmed = remaining.trim_start_matches([' ', '\t', '\n', '\r', ':']);
        if !trimmed.starts_with('[') {
            continue;
        }
        let inner = &trimmed[1..];
        if let Some(end) = inner.find(']') {
            let coord_str = &inner[..end];
            let parts: Vec<&str> = coord_str.split(',').collect();
            if parts.len() >= 2 {
                if let (Ok(x), Ok(y)) = (
                    parts[0].trim().parse::<f64>(),
                    parts[1].trim().parse::<f64>(),
                ) {
                    let label = format!("pt{}", count);
                    index.insert(Point::new([x, y]), label);
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        (0, 0.0, 0.0, 1.0, 1.0)
    } else {
        (count, min_x, min_y, max_x, max_y)
    }
}

fn cmd_stream(interval: usize) {
    let mut index = load_state().unwrap_or_else(|| BonsaiIndex::<String>::builder().build());
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    if let Some((px0, py0, px1, py1)) = load_meta() {
        min_x = px0;
        min_y = py0;
        max_x = px1;
        max_y = py1;
    }

    let mut count = 0usize;
    let mut first_data_line = true;
    let stdin = io::stdin();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() < 2 {
            continue;
        }
        let x = match parts[0].trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => {
                if first_data_line {
                    first_data_line = false;
                    continue;
                }
                continue;
            }
        };
        first_data_line = false;
        let y = match parts[1].trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let label = if parts.len() > 2 {
            parts[2].trim().to_string()
        } else {
            format!("pt{}", index.len())
        };

        index.insert(Point::new([x, y]), label);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        count += 1;

        if count.is_multiple_of(interval) {
            let s = index.stats();
            println!(
                "points={:>8}  backend={:<10}  migrations={}",
                s.point_count,
                backend_name(s.backend),
                s.migrations,
            );
            io::stdout().flush().ok();
            save_state(&index);
            save_meta(min_x, min_y, max_x, max_y);
        }
    }

    save_state(&index);
    save_meta(min_x, min_y, max_x, max_y);
    println!(
        "stream complete — inserted {} points (index now has {} total)",
        count,
        index.len(),
    );
}

fn cmd_query_range(min_x: f64, min_y: f64, max_x: f64, max_y: f64) {
    let mut index = load_state().unwrap_or_else(|| {
        eprintln!("no index found — run 'bonsai load <file>' first");
        std::process::exit(1);
    });
    let bbox = BBox::new(Point::new([min_x, min_y]), Point::new([max_x, max_y]));
    let results = index.range_query(&bbox);
    println!(
        "{} result(s) in [{},{},{},{}]:",
        results.len(),
        min_x,
        min_y,
        max_x,
        max_y
    );
    for (id, payload) in &results {
        println!("  id={} label={}", id.0, payload);
    }
}

fn cmd_query_knn(qx: f64, qy: f64, k: usize) {
    let index = load_state().unwrap_or_else(|| {
        eprintln!("no index found — run 'bonsai load <file>' first");
        std::process::exit(1);
    });
    let pt = Point::new([qx, qy]);
    let results = index.knn_query(&pt, k);
    println!("{} nearest to ({}, {}):", k, qx, qy);
    for (dist, id, payload) in &results {
        println!("  id={} dist={:.4} label={}", id.0, dist, payload);
    }
}

fn cmd_stats() {
    let index = load_state().unwrap_or_else(|| {
        eprintln!("no index found — run 'bonsai load <file>' first");
        std::process::exit(1);
    });
    let s = index.stats();
    print_stats(&s);
}

fn cmd_watch() {
    let mut last_query_count = 0u64;
    let mut last_time = Instant::now();
    let mut first = true;

    println!("watching index state (press Ctrl-C to stop)...");
    loop {
        let index = match load_state() {
            Some(idx) => idx,
            None => {
                println!("\r[no index — run 'bonsai load <file>' first]");
                std::thread::sleep(std::time::Duration::from_secs(1));
                continue;
            }
        };

        let s = index.stats();
        let elapsed = last_time.elapsed().as_secs_f64();
        let qps = if elapsed > 0.0 {
            (s.query_count.saturating_sub(last_query_count)) as f64 / elapsed
        } else {
            0.0
        };
        last_query_count = s.query_count;
        last_time = Instant::now();

        if !first {
            print!("\x1b[6A");
        }
        first = false;

        println!(
            "backend:     {:10}  points: {:8}  migrations: {}",
            backend_name(s.backend),
            s.point_count,
            s.migrations
        );
        println!(
            "queries:     {:10}  throughput: {:.1} q/s",
            s.query_count, qps
        );
        println!(
            "migrating:   {:10}  dimensions: {}",
            s.migrating, s.dimensions
        );
        println!(
            "data_shape:  clustering={:.3}  eff_dim={:.3}  skew=[{:.3},{:.3}]",
            s.data_shape.clustering_coef,
            s.data_shape.effective_dim,
            s.data_shape.skewness[0],
            s.data_shape.skewness[1],
        );
        println!(
            "cost_est:    range_frac={:.2}  knn_frac={:.2}  sel={:.4}",
            s.data_shape.query_mix.range_frac,
            s.data_shape.query_mix.knn_frac,
            s.data_shape.query_mix.mean_selectivity,
        );
        println!("─────────────────────────────────────────────────────");
        io::stdout().flush().ok();

        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

fn cmd_bench() {
    let mut index = load_state().unwrap_or_else(|| {
        eprintln!("no index found — run 'bonsai load <file>' first");
        std::process::exit(1);
    });

    let s = index.stats();
    let n = s.point_count;
    if n == 0 {
        eprintln!("index is empty");
        std::process::exit(1);
    }

    let domain = s.data_shape.bbox;
    let dx = domain.max.coords()[0] - domain.min.coords()[0];
    let dy = domain.max.coords()[1] - domain.min.coords()[1];
    let cx = domain.min.coords()[0] + dx / 2.0;
    let cy = domain.min.coords()[1] + dy / 2.0;

    const ITERS: usize = 1000;

    let mut range_ns: Vec<u64> = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        let frac = 0.1 + (i % 5) as f64 * 0.02;
        let bbox = BBox::new(
            Point::new([cx - dx * frac, cy - dy * frac]),
            Point::new([cx + dx * frac, cy + dy * frac]),
        );
        let t0 = Instant::now();
        let _ = index.range_query(&bbox);
        range_ns.push(t0.elapsed().as_nanos() as u64);
    }
    range_ns.sort_unstable();

    let mut knn_ns: Vec<u64> = Vec::with_capacity(ITERS);
    let query_pt = Point::new([cx, cy]);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let _ = index.knn_query(&query_pt, 10);
        knn_ns.push(t0.elapsed().as_nanos() as u64);
    }
    knn_ns.sort_unstable();

    fn percentile(sorted: &[u64], p: f64) -> u64 {
        let idx = ((sorted.len() as f64 * p / 100.0) as usize).min(sorted.len() - 1);
        sorted[idx]
    }

    println!("benchmark results ({} iterations, {} points):", ITERS, n);
    println!(
        "  range query:  p50={} µs  p95={} µs  p99={} µs",
        percentile(&range_ns, 50.0) / 1000,
        percentile(&range_ns, 95.0) / 1000,
        percentile(&range_ns, 99.0) / 1000,
    );
    println!(
        "  knn(k=10):    p50={} µs  p95={} µs  p99={} µs",
        percentile(&knn_ns, 50.0) / 1000,
        percentile(&knn_ns, 95.0) / 1000,
        percentile(&knn_ns, 99.0) / 1000,
    );
}

fn cmd_visualise() {
    let mut index = load_state().unwrap_or_else(|| {
        eprintln!("no index found — run 'bonsai load <file>' first");
        std::process::exit(1);
    });

    let s = index.stats();
    let n = s.point_count;
    if n == 0 {
        println!("(empty index)");
        return;
    }

    let (ax0, ay0, ax1, ay1) = if let Some((mx0, my0, mx1, my1)) = load_meta() {
        (mx0, my0, mx1, my1)
    } else {
        let ds_bbox = s.data_shape.bbox;
        let ds_dx = ds_bbox.max.coords()[0] - ds_bbox.min.coords()[0];
        let ds_dy = ds_bbox.max.coords()[1] - ds_bbox.min.coords()[1];
        if ds_dx > 1e-6 && ds_dy > 1e-6 {
            (
                ds_bbox.min.coords()[0],
                ds_bbox.min.coords()[1],
                ds_bbox.max.coords()[0],
                ds_bbox.max.coords()[1],
            )
        } else {
            (0.0f64, 0.0f64, 100.0f64, 100.0f64)
        }
    };

    const COLS: usize = 60;
    const ROWS: usize = 20;

    let mut grid = vec![vec![0u32; COLS]; ROWS];
    let dx = (ax1 - ax0).max(1e-9);
    let dy = (ay1 - ay0).max(1e-9);
    let cell_w = dx / COLS as f64;
    let cell_h = dy / ROWS as f64;

    for (row, grid_row) in grid.iter_mut().enumerate() {
        for (col, cell) in grid_row.iter_mut().enumerate() {
            let x0 = ax0 + col as f64 * cell_w;
            let y0 = ay0 + row as f64 * cell_h;
            let bbox = BBox::new(Point::new([x0, y0]), Point::new([x0 + cell_w, y0 + cell_h]));
            *cell = index.range_query(&bbox).len() as u32;
        }
    }

    let max_density = grid
        .iter()
        .flat_map(|r| r.iter())
        .copied()
        .max()
        .unwrap_or(1)
        .max(1);
    let chars = [' ', '·', ':', '+', '*', '#'];

    println!(
        "index visualisation — {} points, backend: {}",
        n,
        backend_name(s.backend)
    );
    println!("x: [{:.2}, {:.2}]  y: [{:.2}, {:.2}]", ax0, ax1, ay0, ay1);
    println!("┌{}┐", "─".repeat(COLS));
    for row in grid.iter().rev() {
        print!("│");
        for &cell in row {
            let idx = (cell as f64 / max_density as f64 * (chars.len() - 1) as f64) as usize;
            print!("{}", chars[idx.min(chars.len() - 1)]);
        }
        println!("│");
    }
    println!("└{}┘", "─".repeat(COLS));
    println!("density: ' '=0  '·'=low  ':'=med  '+'=high  '*'=very high  '#'=max");
}

fn cmd_migrate(backend_str: &str) {
    let mut index = load_state().unwrap_or_else(|| {
        eprintln!("no index found — run 'bonsai load <file>' first");
        std::process::exit(1);
    });

    let backend = parse_backend(backend_str).unwrap_or_else(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    });

    index.force_backend(backend).unwrap_or_else(|e| {
        eprintln!("migration failed: {}", e);
        std::process::exit(1);
    });

    save_state(&index);
    println!("migrated to {}", backend_name(backend));
}

fn print_usage() {
    eprintln!("usage: bonsai <command> [args]");
    eprintln!();
    eprintln!("commands:");
    eprintln!("  load <file>                       load CSV or GeoJSON into index");
    eprintln!("  stream [--interval N]             read x,y[,label] lines from stdin");
    eprintln!("  query range <x0> <y0> <x1> <y1>  range query");
    eprintln!("  query knn <x> <y> <k>             k-nearest-neighbour query");
    eprintln!("  stats                             print index statistics");
    eprintln!("  watch                             live-updating TUI");
    eprintln!("  bench                             benchmark p50/p95/p99 latencies");
    eprintln!("  visualise                         ASCII art of index structure");
    eprintln!("  migrate <backend>                 force backend (kdtree|rtree|quadtree|grid)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    match args[1].as_str() {
        "load" => {
            if args.len() < 3 {
                eprintln!("usage: bonsai load <file>");
                std::process::exit(1);
            }
            cmd_load(&args[2]);
        }
        "stream" => {
            let interval = args
                .windows(2)
                .find(|w| w[0] == "--interval")
                .and_then(|w| w[1].parse::<usize>().ok())
                .unwrap_or(100);
            cmd_stream(interval);
        }
        "query" => {
            if args.len() < 3 {
                eprintln!("usage: bonsai query range|knn ...");
                std::process::exit(1);
            }
            match args[2].as_str() {
                "range" => {
                    if args.len() < 7 {
                        eprintln!("usage: bonsai query range <x0> <y0> <x1> <y1>");
                        std::process::exit(1);
                    }
                    let coords: Vec<f64> = args[3..7]
                        .iter()
                        .map(|s| {
                            s.parse().unwrap_or_else(|_| {
                                eprintln!("invalid number: {}", s);
                                std::process::exit(1);
                            })
                        })
                        .collect();
                    cmd_query_range(coords[0], coords[1], coords[2], coords[3]);
                }
                "knn" => {
                    if args.len() < 6 {
                        eprintln!("usage: bonsai query knn <x> <y> <k>");
                        std::process::exit(1);
                    }
                    let x: f64 = args[3].parse().unwrap_or_else(|_| {
                        eprintln!("invalid x");
                        std::process::exit(1);
                    });
                    let y: f64 = args[4].parse().unwrap_or_else(|_| {
                        eprintln!("invalid y");
                        std::process::exit(1);
                    });
                    let k: usize = args[5].parse().unwrap_or_else(|_| {
                        eprintln!("invalid k");
                        std::process::exit(1);
                    });
                    cmd_query_knn(x, y, k);
                }
                other => {
                    eprintln!("unknown query subcommand '{}'; use range or knn", other);
                    std::process::exit(1);
                }
            }
        }
        "stats" => cmd_stats(),
        "watch" => cmd_watch(),
        "bench" => cmd_bench(),
        "visualise" | "visualize" => cmd_visualise(),
        "migrate" => {
            if args.len() < 3 {
                eprintln!("usage: bonsai migrate <backend>");
                std::process::exit(1);
            }
            cmd_migrate(&args[2]);
        }
        other => {
            eprintln!("unknown command '{}'\n", other);
            print_usage();
            std::process::exit(1);
        }
    }
}
