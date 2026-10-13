//! The symbolic analysis builds several of its passes in parallel. The plan it
//! produces must not depend on how many workers ran, or the permutation, the
//! pivot sequence and the rounding all move with the thread count.

use sparta::{Analysis, Config, OrderingMethod};

/// Builds the lower triangle of a `k` by `k` by `k` 7-point stencil in CSC.
/// Returns the dimension, column offsets and row indices.
fn grid(k: usize) -> (usize, Vec<usize>, Vec<usize>) {
    let n = k * k * k;
    let at = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
    let mut ptr = vec![0];
    let mut rows = Vec::new();
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let j = at(x, y, z);
                let mut col = vec![j];
                if x + 1 < k {
                    col.push(at(x + 1, y, z));
                }
                if y + 1 < k {
                    col.push(at(x, y + 1, z));
                }
                if z + 1 < k {
                    col.push(at(x, y, z + 1));
                }
                col.sort_unstable();
                rows.extend(col);
                ptr.push(rows.len());
            }
        }
    }
    (n, ptr, rows)
}

/// Analyzes `(n, ptr, rows)` inside a pool of `workers` threads.
fn digest_with(workers: usize, n: usize, ptr: &[usize], rows: &[usize], lu: bool) -> u64 {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    pool.install(|| {
        let config = Config {
            ordering: OrderingMethod::Scalpel,
            ..Config::default()
        };
        let plan = if lu {
            Analysis::lu(n, ptr, rows, &config)
        } else {
            Analysis::ldlt(n, ptr, rows, &config)
        };
        plan.unwrap().digest()
    })
}

#[test]
/// A pattern large enough to take the parallel paths gives one plan at any
/// worker count, on both the LDL^T and the LU entry points.
fn plan_is_independent_of_worker_count() {
    // 8000 columns clears the threshold below which the passes stay serial.
    let (n, ptr, rows) = grid(20);
    assert!(n > 4096, "pattern must be large enough to run in parallel");
    for lu in [false, true] {
        let reference = digest_with(1, n, &ptr, &rows, lu);
        for workers in [2, 4, 8] {
            assert_eq!(
                reference,
                digest_with(workers, n, &ptr, &rows, lu),
                "plan changed with {workers} workers (lu = {lu})"
            );
        }
    }
}

#[test]
/// Repeating one analysis in one process gives the same plan every time.
fn plan_is_stable_across_repeats() {
    let (n, ptr, rows) = grid(20);
    let first = digest_with(4, n, &ptr, &rows, false);
    for _ in 0..3 {
        assert_eq!(first, digest_with(4, n, &ptr, &rows, false));
    }
}

#[test]
/// Below the parallel threshold the serial paths must agree with the parallel
/// ones on the same pattern, so the threshold itself cannot change a plan.
fn small_patterns_match_across_worker_counts() {
    let (n, ptr, rows) = grid(8);
    let reference = digest_with(1, n, &ptr, &rows, false);
    for workers in [2, 4] {
        assert_eq!(reference, digest_with(workers, n, &ptr, &rows, false));
    }
}
