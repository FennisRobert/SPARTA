//! Time the port matrix `Z = B^T A^{-1} B` of sparse ports
//! (`Solver::port_matrix`) against `B^T X` from one dense block solve of all
//! ports (`Solver::solve_values`), and the sparse solve
//! (`Solver::solve_sparse`) against the same dense solve, on CPU complex128
//! `LDL^T` factors.
//!
//!     cargo run --release --example port_matrix_timing -- [case] [size] [ports] [threads] [reps] [scattered|clustered]
//!
//! - `helmholtz` (default): the 2D 5-point operator `4 - k^2` on a
//!   `size x size` grid (default 400), `k^2 = 0.3 (1 + 0.02 i)`: complex
//!   symmetric, lossy, indefinite. Each port is six neighbouring grid nodes.
//! - `curl-curl`: `sparta_matgen::curl_curl` on a `size^3` grid (default 20),
//!   `omega = 1`, `sigma = 0.1`. Each port is six consecutive unknowns.
//!
//! Ports (default 200, weight 1/6 on each unknown) start at pseudo-random
//! places, anywhere (`scattered`) or within a window of 1% of the unknowns
//! (`clustered`). `threads` (default 1) is `Config::threads`. Times are the
//! median, minimum and maximum of `reps` runs (default 5) after one warm-up.
//!
//! A fixed calibration workload on the same number of threads runs first and
//! its time is printed, so runs on a shared machine can be compared: a case
//! whose calibration is well above the fastest seen ran on a busy machine.
use num_complex::Complex64 as C;
use rayon::prelude::*;
use sparta::{Analysis, Config, Solver, SparseRhs};
use std::time::Instant;

/// Lower-triangle CSC of the 2D Helmholtz operator.
fn helmholtz(m: usize, k2: C) -> (usize, Vec<usize>, Vec<usize>, Vec<C>) {
    let n = m * m;
    let (mut ptr, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
    for j in 0..n {
        rows.push(j);
        values.push(C::new(4.0, 0.0) - k2);
        if j % m + 1 < m {
            rows.push(j + 1);
            values.push(C::new(-1.0, 0.0));
        }
        if j + m < n {
            rows.push(j + m);
            values.push(C::new(-1.0, 0.0));
        }
        ptr.push(rows.len());
    }
    (n, ptr, rows, values)
}

/// `p` ports of six unknowns each, weight 1/6, starting at pseudo-random
/// unknowns in `first..first + span`; `next(i, k)` is the `k`-th unknown of
/// the port at `i`.
fn ports(
    n: usize,
    p: usize,
    (first, span): (usize, usize),
    next: impl Fn(usize, usize) -> usize,
) -> SparseRhs<C> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let columns: Vec<Vec<(usize, C)>> = (0..p)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let i = first + (state >> 33) as usize % span.max(1);
            (0..6)
                .map(|k| (next(i, k) % n, C::new(1.0 / 6.0, 0.0)))
                .collect()
        })
        .collect();
    SparseRhs::from_columns(n, &columns).unwrap()
}

/// Median, minimum and maximum seconds over `reps` runs after a warm-up.
fn time<R>(reps: usize, mut f: impl FnMut() -> R) -> ([f64; 3], R) {
    let mut out = f();
    let mut runs = Vec::with_capacity(reps);
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        out = f();
        runs.push(t.elapsed().as_secs_f64());
    }
    runs.sort_by(f64::total_cmp);
    let mid = runs.len() / 2;
    let median = if runs.len() % 2 == 1 {
        runs[mid]
    } else {
        (runs[mid - 1] + runs[mid]) / 2.0
    };
    ([median, runs[0], runs[runs.len() - 1]], out)
}

fn show([m, lo, hi]: [f64; 3]) -> String {
    format!("{m:.4} ({lo:.4}-{hi:.4}) s")
}

/// A fixed amount of cache-resident complex arithmetic on each of `threads`
/// threads; its wall time says how much of the machine was free.
fn calibrate(threads: usize) -> f64 {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let t = Instant::now();
    let sums: Vec<C> = pool.install(|| {
        (0..threads)
            .into_par_iter()
            .map(|seed| {
                let mut v: Vec<C> = (0..512)
                    .map(|i| C::new(1.0 + (i + seed) as f64 * 1e-6, 1e-3))
                    .collect();
                let w = C::new(0.999_999, 1e-6);
                for _ in 0..600_000 {
                    for x in &mut v {
                        *x = *x * w + C::new(1e-9, 0.0);
                    }
                }
                v.iter().sum()
            })
            .collect()
    });
    std::hint::black_box(sums);
    t.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).map_or("helmholtz", String::as_str);
    let arg = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let p = arg(3, 200);
    let threads = arg(4, 1);
    let reps = arg(5, 5);
    let clustered = args.get(6).is_some_and(|s| s == "clustered");
    let window = |n: usize| if clustered { (n / 2, n / 100) } else { (0, n) };
    let calibration = calibrate(threads);
    let ((n, ptr, rows, values), b) = match case {
        "curl-curl" => {
            let m = arg(2, 20);
            let a = sparta_matgen::curl_curl(&[m, m, m], 1.0, 0.1);
            let b = ports(a.0, p, window(a.0), |i, k| i + k);
            (a, b)
        }
        _ => {
            let m = arg(2, 400);
            let a = helmholtz(m, C::new(0.3, 0.006));
            let b = ports(a.0, p, window(a.0), move |i, k| i + (k % 3) + (k / 3) * m);
            (a, b)
        }
    };
    let config = Config {
        force_cpu: true,
        threads,
        ..Config::default()
    };
    let plan = Analysis::ldlt(n, &ptr, &rows, &config).unwrap();
    let nnz = plan.factor_nonzeros();
    let mut s = Solver::new(plan).unwrap();
    let t = Instant::now();
    s.factor_c128(&values).unwrap();
    println!(
        "{case}: n {n}, nnz(L) {nnz}, {p} {} ports, {threads} threads, factor {:.3} s, calibration {calibration:.4} s",
        if clustered { "clustered" } else { "scattered" },
        t.elapsed().as_secs_f64()
    );
    let (tz, z) = time(reps, || s.port_matrix(&b).unwrap());
    // `PORT_ONLY=1`: the port matrix alone, for the process's peak memory.
    if std::env::var_os("PORT_ONLY").is_some() {
        println!("port_matrix {}", show(tz));
        return;
    }
    let dense = b.to_dense();
    let (td, (zd, xd)) = time(reps, || {
        let x = s.solve_values(&dense, p).unwrap();
        (b.transpose_times(&x, p).unwrap(), x)
    });
    drop(dense);
    let (ts, xs) = time(reps, || s.solve_sparse(&b).unwrap());
    let largest = |v: &[C]| v.iter().map(|v| v.norm()).fold(0.0, f64::max);
    let xdiff = xs
        .iter()
        .zip(&xd)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max);
    let scale = zd.iter().map(|v| v.norm()).fold(0.0, f64::max);
    let diff = z
        .iter()
        .zip(&zd)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max);
    let reach: Vec<_> = (0..p)
        .map(|c| s.sparse_reach(b.column(c).0).unwrap())
        .collect();
    let mean = reach.iter().map(|r| r.factor_entries as f64).sum::<f64>() / p.max(1) as f64;
    println!(
        "port_matrix {} | {p} dense solves + B^T X {} ({:.1}x by medians) | \
         solve_sparse {} | max|dZ|/max|Z| {:.1e} | max|dX|/max|X| {:.1e} | mean reach {:.1}% of the factor",
        show(tz),
        show(td),
        td[0] / tz[0],
        show(ts),
        diff / scale,
        xdiff / largest(&xd),
        100.0 * mean / reach.first().map_or(1, |r| r.total_factor_entries) as f64
    );
}
