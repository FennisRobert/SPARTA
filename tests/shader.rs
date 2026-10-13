//! The GPU kernels compile, every pipeline they declare can be built, and a
//! factorization and solve through them is correct.
//!
//! Metal compiles its kernels at run time from source embedded by build.rs, and
//! Vulkan builds its pipelines from embedded SPIR-V, both when a `Solver` is
//! created. `cargo build` therefore says nothing about whether a pipeline
//! builds on this device -- a Metal syntax error or a Vulkan pipeline that
//! exceeds a device limit surfaces only when something first reaches the GPU.
//! These tests are the cheapest things that reach it, so the mistake surfaces
//! here rather than in a caller's first factorization. Without a GPU they pass
//! without doing anything.

use sparta::{Analysis, Complex32, Config, Solver};

/// Builds a tridiagonal pattern of `n` columns, which is enough to produce a
/// valid plan without spending time on ordering.
fn pattern(n: usize) -> (Vec<usize>, Vec<usize>) {
    let (mut ptr, mut rows) = (vec![0usize], Vec::new());
    for j in 0..n {
        for i in j..(j + 2).min(n) {
            rows.push(i);
        }
        ptr.push(rows.len());
    }
    (ptr, rows)
}

#[test]
/// Creating a solver compiles the kernels and builds every pipeline, on both
/// factorization paths. A failure here names the backend's own message.
fn every_pipeline_builds() {
    if !sparta::gpu_available() {
        return;
    }
    let config = Config::default();
    let (ptr, rows) = pattern(64);
    for (name, plan) in [
        ("ldlt", Analysis::ldlt(64, &ptr, &rows, &config)),
        ("lu", Analysis::lu(64, &ptr, &rows, &config)),
    ] {
        let plan = plan.unwrap_or_else(|e| panic!("{name}: analysis failed: {e}"));
        // Solver::new is what compiles the kernels and builds the pipelines;
        // its error carries the compiler's or the driver's diagnostics.
        if let Err(e) = Solver::new(plan) {
            panic!("{name}: the GPU kernels did not build: {e}");
        }
    }
}

/// `y = A x` for the matrix whose CSC is (`ptr`, `rows`, `values`); with
/// `lower` the CSC holds one triangle of a symmetric matrix.
fn product(
    ptr: &[usize],
    rows: &[usize],
    values: &[Complex32],
    lower: bool,
    x: &[Complex32],
) -> Vec<Complex32> {
    let mut y = vec![Complex32::new(0.0, 0.0); x.len()];
    for j in 0..ptr.len() - 1 {
        for p in ptr[j]..ptr[j + 1] {
            let i = rows[p];
            y[i] += values[p] * x[j];
            if lower && i != j {
                y[j] += values[p] * x[i];
            }
        }
    }
    y
}

/// Relative residual `||b - A x|| / ||b||` of a solve.
fn residual(
    ptr: &[usize],
    rows: &[usize],
    values: &[Complex32],
    lower: bool,
    x: &[Complex32],
    b: &[Complex32],
) -> f64 {
    let ax = product(ptr, rows, values, lower, x);
    let norm = |v: &mut dyn Iterator<Item = Complex32>| {
        v.map(|z| f64::from(z.norm_sqr())).sum::<f64>().sqrt()
    };
    norm(&mut ax.iter().zip(b).map(|(a, b)| b - a)) / norm(&mut b.iter().copied())
}

#[test]
/// A curl-curl grid, large enough for small, narrow and wide fronts, factors
/// and solves on the GPU to single-precision accuracy through the LDL^T path
/// and, from both triangles, the LU path.
fn factors_and_solves_on_the_gpu() {
    if !sparta::gpu_available() {
        return;
    }
    let (n, ptr, rows, values) = sparta_matgen::curl_curl(&[8, 8, 8], 1.0, 0.1);
    let lower: Vec<Complex32> = values
        .iter()
        .map(|v| Complex32::new(v.re as f32, v.im as f32))
        .collect();
    // The same matrix with both triangles, for the LU path.
    let mut columns: Vec<Vec<(usize, Complex32)>> = vec![Vec::new(); n];
    for j in 0..n {
        for p in ptr[j]..ptr[j + 1] {
            columns[j].push((rows[p], lower[p]));
            if rows[p] != j {
                columns[rows[p]].push((j, lower[p]));
            }
        }
    }
    let (mut full_ptr, mut full_rows, mut full) = (vec![0usize], Vec::new(), Vec::new());
    for mut column in columns {
        column.sort_by_key(|&(i, _)| i);
        for (i, v) in column {
            full_rows.push(i);
            full.push(v);
        }
        full_ptr.push(full_rows.len());
    }
    let b: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(1.0 + (i % 7) as f32, (i % 3) as f32 - 1.0))
        .collect();
    let config = Config::default();
    let cases = [
        (
            "ldlt",
            Analysis::ldlt(n, &ptr, &rows, &config),
            &ptr,
            &rows,
            &lower,
            true,
        ),
        (
            "lu",
            Analysis::lu(n, &full_ptr, &full_rows, &config),
            &full_ptr,
            &full_rows,
            &full,
            false,
        ),
    ];
    for (name, plan, ptr, rows, values, symmetric) in cases {
        let plan = plan.unwrap_or_else(|e| panic!("{name}: analysis failed: {e}"));
        let mut solver = Solver::new(plan).unwrap_or_else(|e| panic!("{name}: {e}"));
        solver
            .factor(values)
            .unwrap_or_else(|e| panic!("{name}: factor failed: {e}"));
        let x = solver
            .solve(&b)
            .unwrap_or_else(|e| panic!("{name}: solve failed: {e}"));
        let r = residual(ptr, rows, values, symmetric, &x, &b);
        assert!(r < 1e-4, "{name}: relative residual {r:.3e}");
    }
}
