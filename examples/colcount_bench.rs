//! Throughput measurement for the symbolic column-count routine.
//!
//!     cargo run --release --example colcount_bench
//!     cargo run --release --example colcount_bench -- 12 16 20 24
//!     cargo run --release --example colcount_bench -- --grid 30 40
//!
//! Driven by `measure_colcounts.sh`.
//!
//! The point is to time `column_counts_gnp` on input shaped the way SPARTA
//! actually shapes it. Timing it on a raw matrix pattern would be misleading:
//! the real call site in `symbolic_factorize` hands it a pattern that has
//! already been reordered by nested dissection and postordered, which changes
//! both the elimination tree's shape and the amount of fill being counted.
//! So this runs the whole preprocessing chain and times only the final call.
//!
//! Two workloads:
//!   curl-curl  the second-order Nedelec systems SPARTA targets (default)
//!   grid       7-point Laplacian on a k^3 grid, for a cheaper scaling sweep

use sparta::symbolic::column_counts::column_counts_gnp;
use sparta::symbolic::elimination_tree::EliminationTree;
use sparta::symbolic::postorder::postorder;
use sparta::symbolic::{order, permute_pattern, symmetric_pattern, CscPattern};
use std::time::Instant;

/// Mirror of the preprocessing in `symbolic_factorize`: fill-reducing order,
/// composed with a postorder of its elimination tree, applied to the pattern.
///
/// Returns the permuted pattern and its elimination tree — exactly the pair the
/// production call site passes to `column_counts_gnp`.
fn prepare(full: &CscPattern) -> (CscPattern, EliminationTree) {
    let n = full.n;
    let fill_order = order(full, &sparta::Config::default()).expect("ordering failed");
    let first = permute_pattern(full, &fill_order);
    let first_etree = EliminationTree::from_pattern(&first);
    let (post, post_inv) = postorder(&first_etree);
    let perm: Vec<usize> = post.iter().map(|&p| fill_order[p]).collect();
    let permuted = permute_pattern(full, &perm);
    let etree = EliminationTree {
        parent: (0..n)
            .map(|new| {
                let p = first_etree.parent[post[new]];
                if p == sparta::symbolic::elimination_tree::NO_PARENT {
                    p
                } else {
                    post_inv[p as usize] as u32
                }
            })
            .collect(),
        n,
    };
    (permuted, etree)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn grid3d(k: usize) -> (usize, Vec<usize>, Vec<usize>) {
    let n = k * k * k;
    let at = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
    let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let h = at(x, y, z);
                cols[h].push(h);
                let mut link = |a: usize, b: usize| {
                    cols[a].push(b);
                    cols[b].push(a);
                };
                if x + 1 < k {
                    link(h, at(x + 1, y, z));
                }
                if y + 1 < k {
                    link(h, at(x, y + 1, z));
                }
                if z + 1 < k {
                    link(h, at(x, y, z + 1));
                }
            }
        }
    }
    // Emit as a lower-triangle CSC, which is what symmetric_pattern expects.
    let mut col_ptr = vec![0usize];
    let mut row_idx = Vec::new();
    for (j, col) in cols.iter_mut().enumerate() {
        col.sort_unstable();
        col.dedup();
        row_idx.extend(col.iter().copied().filter(|&i| i >= j));
        col_ptr.push(row_idx.len());
    }
    (n, col_ptr, row_idx)
}

fn measure(label: &str, n: usize, col_ptr: &[usize], row_idx: &[usize]) {
    let full = symmetric_pattern(n, col_ptr, row_idx);
    let nnz_a = full.row_idx.len();

    let prep = Instant::now();
    let (pattern, etree) = prepare(&full);
    let prep_s = prep.elapsed().as_secs_f64();

    // Warm up, then take the median of repeated runs. The tree and pattern are
    // built once, outside the timed region: this measures counting, not setup.
    let counts = column_counts_gnp(&pattern, &etree);
    let nnz_l: usize = counts.iter().sum();

    let runs = if n > 100_000 {
        5
    } else if n > 20_000 {
        11
    } else {
        25
    };
    let samples: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            let out = column_counts_gnp(&pattern, &etree);
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(out);
            dt
        })
        .collect();
    let med = median(samples);

    // Throughput against the input, which is what the algorithm is linear in.
    let mnnz_per_s = nnz_a as f64 / med / 1e6;
    // Ratio of output size to input size: how much work the O(|L|) predecessor
    // would have had to do relative to this one.
    let fill_ratio = nnz_l as f64 / nnz_a as f64;

    println!(
        "{label:<18} {n:>9} {nnz_a:>12} {nnz_l:>14} {:>11.3} {mnnz_per_s:>12.1} {fill_ratio:>9.1}x {prep_s:>9.3}",
        med * 1e3
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let grid_mode = args.first().map(|a| a == "--grid").unwrap_or(false);
    let sizes: Vec<usize> = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(|a| a.parse().expect("sizes must be integers"))
        .collect();

    println!();
    println!(
        "{:<18} {:>9} {:>12} {:>14} {:>11} {:>12} {:>10} {:>9}",
        "system", "n", "nnz(A)", "nnz(L)", "count_ms", "Mnnz/s", "fill", "prep_s"
    );
    println!("{}", "-".repeat(101));

    if grid_mode {
        let ks = if sizes.is_empty() {
            vec![20, 30, 40]
        } else {
            sizes
        };
        for k in ks {
            let (n, col_ptr, row_idx) = grid3d(k);
            measure(&format!("grid_{k}^3"), n, &col_ptr, &row_idx);
        }
    } else {
        let ks = if sizes.is_empty() {
            vec![12, 16, 20, 24]
        } else {
            sizes
        };
        for k in ks {
            let (n, col_ptr, row_idx, _values) = sparta_matgen::curl_curl(&[k, k, k], 1.0, 0.1);
            measure(&format!("curl_curl_{k}^3"), n, &col_ptr, &row_idx);
        }
    }
    println!();
}
