//! Front shapes and dense work per factorization level, root first.
//!
//!     cargo run --release --example level_shapes -- test_matrices/em_m4_sym1.sparta
//!
//! Work is counted in real flops of the complex LDL^T front factorization:
//! a pivot block of `nc` columns and a Schur update of the trailing `nr - nc`
//! rows, eight real flops per complex multiply-add.
mod common;

use common::System;
use sparta::Analysis;

fn main() {
    let path = std::env::args().nth(1).expect("usage: level_shapes FILE");
    let system = System::load(&path).unwrap_or_else(|e| panic!("{e}"));
    let plan = Analysis::ldlt(system.n, &system.col_ptr, &system.row_idx, &sparta::Config::default())
        .unwrap_or_else(|e| panic!("{e}"));
    let levels = plan.level_fronts();
    let total: f64 = levels.iter().flatten().map(|&(r, c)| flops(r, c)).sum();
    println!("total {:.1} GFLOP real", total / 1e9);
    let mut cumulative = 0.0;
    for (l, level) in levels.iter().enumerate().rev() {
        let work: f64 = level.iter().map(|&(r, c)| flops(r, c)).sum();
        cumulative += work;
        let widest = level.iter().max_by_key(|f| f.0).copied().unwrap_or((0, 0));
        println!(
            "L{l:2} fronts {:6}  widest {:5}x{:<5}  {:8.1} GFLOP  cum-from-top {:5.1}%",
            level.len(),
            widest.0,
            widest.1,
            work / 1e9,
            100.0 * cumulative / total
        );
    }
}

/// Real flops of one complex LDL^T front with `nr` rows and `nc` pivots.
fn flops(nr: usize, nc: usize) -> f64 {
    let (r, c) = (nr as f64, nc as f64);
    // Sum over pivots k of the (nr - k)^2 / 2 lower-triangle update: c * (r^2 - r*c + c^2/3).
    8.0 * 0.5 * c * (r * r - r * c + c * c / 3.0)
}
