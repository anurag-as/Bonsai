//! Demo: Hilbert curve indices for a 4×4 grid of 2D points, showing
//! space-filling curve order, D=8 128-bit budget, and spatial locality.

use bonsai::hilbert::HilbertCurve;

fn main() {
    let curve = HilbertCurve::<2>::new(2);

    println!("=== 4×4 Hilbert curve (order=2) ===\n");

    let mut cells: Vec<(u128, u64, u64)> = (0u64..4)
        .flat_map(|y| (0u64..4).map(move |x| (curve.index(&[x, y]), x, y)))
        .collect();
    cells.sort_by_key(|&(idx, _, _)| idx);

    println!("Hilbert order (index → (x, y)):");
    for (idx, x, y) in &cells {
        println!("  {:2} → ({}, {})", idx, x, y);
    }

    println!("\nGrid view (Hilbert index at each cell):");
    println!("  y\\x  0   1   2   3");
    for y in (0u64..4).rev() {
        print!("   {}  ", y);
        for x in 0u64..4 {
            print!("{:3} ", curve.index(&[x, y]));
        }
        println!();
    }

    println!("\n=== D=8, order=16 (128-bit budget) ===");
    let c8 = HilbertCurve::<8>::new(16);
    let max = (1u64 << 16) - 1;
    println!("  index([0;8])   = {}", c8.index(&[0u64; 8]));
    println!("  index([max;8]) = {}", c8.index(&[max; 8]));
    println!("  u128::MAX      = {}", u128::MAX);

    println!("\n=== Spatial locality (2D, order=4, 16×16 grid) ===");
    let c16 = HilbertCurve::<2>::new(4);
    let base = [7u64, 7u64];
    let near = [8u64, 7u64];
    let far = [0u64, 15u64];
    let i_base = c16.index(&base);
    let i_near = c16.index(&near);
    let i_far = c16.index(&far);
    println!("  base ({:2},{:2}) → {:4}", base[0], base[1], i_base);
    println!("  near ({:2},{:2}) → {:4}  diff={}", near[0], near[1], i_near, i_base.abs_diff(i_near));
    println!("  far  ({:2},{:2}) → {:4}  diff={}", far[0], far[1], i_far, i_base.abs_diff(i_far));
}
