//! The four value types end to end: real and complex, single and double, on
//! the CPU and the GPU, against a small indefinite grid with a known solution.
use sparta::{Analysis, Complex32, Complex64, Config, Kind, Scalar, ScalarKind, Solver};

/// Grid side; the system has `SIDE * SIDE` unknowns.
const SIDE: usize = 24;

/// The 5-point Laplacian on a `SIDE` by `SIDE` grid plus `shift` on the
/// diagonal, as CSC triplets `(row, col, value)`. `skew` adds an antisymmetric
/// convection term, which makes the matrix nonsymmetric.
fn grid(shift: f64, skew: f64) -> Vec<(usize, usize, f64)> {
    let mut entries = Vec::new();
    for y in 0..SIDE {
        for x in 0..SIDE {
            let i = y * SIDE + x;
            entries.push((i, i, 4.0 + shift));
            if x + 1 < SIDE {
                entries.push((i + 1, i, -1.0 + skew));
                entries.push((i, i + 1, -1.0 - skew));
            }
            if y + 1 < SIDE {
                entries.push((i + SIDE, i, -1.0 + skew));
                entries.push((i, i + SIDE, -1.0 - skew));
            }
        }
    }
    entries.sort_by_key(|&(r, c, _)| (c, r));
    entries
}

/// Plan and CSC values for `kind`: the lower triangle for LDL^T, everything
/// for LU. The LDL^T matrix is shifted by -2.5, which makes it indefinite so
/// that it needs 2x2 pivots. The LU one is shifted by +0.5 and carries
/// convection: LU pivots only within a front, so an indefinite nonsymmetric
/// matrix would meet near-zero pivots in every value type. Complex types get
/// the imaginary part `0.3 * value` on the diagonal.
fn system<T: Scalar>(kind: Kind, config: &Config) -> (Analysis, Vec<T>, Vec<(usize, usize, T)>) {
    let n = SIDE * SIDE;
    let (shift, skew) = if kind == Kind::Lu { (0.5, 0.3) } else { (-2.5, 0.0) };
    let entries: Vec<(usize, usize, T)> = grid(shift, skew)
        .into_iter()
        .filter(|&(r, c, _)| kind == Kind::Lu || r >= c)
        .map(|(r, c, v)| {
            let im = if r == c && T::KIND.is_complex() { 0.3 * v } else { 0.0 };
            (r, c, T::from_c64(Complex64::new(v, im)))
        })
        .collect();
    let mut ptr = vec![0; n + 1];
    for &(_, c, _) in &entries {
        ptr[c + 1] += 1;
    }
    for c in 0..n {
        ptr[c + 1] += ptr[c];
    }
    let rows: Vec<usize> = entries.iter().map(|e| e.0).collect();
    let plan = match kind {
        Kind::Lu => Analysis::lu(n, &ptr, &rows, config),
        Kind::Ldlt => Analysis::ldlt(n, &ptr, &rows, config),
    }
    .unwrap();
    let values = entries.iter().map(|e| e.2).collect();
    (plan, values, entries)
}

/// `A x` in complex128 from the stored entries; LDL^T entries are mirrored.
fn apply<T: Scalar>(kind: Kind, entries: &[(usize, usize, T)], x: &[T]) -> Vec<Complex64> {
    let mut y = vec![Complex64::new(0.0, 0.0); x.len()];
    for &(r, c, v) in entries {
        y[r] += v.to_c64() * x[c].to_c64();
        if kind == Kind::Ldlt && r != c {
            y[c] += v.to_c64() * x[r].to_c64();
        }
    }
    y
}

/// Factor and solve two right-hand sides in `T`, and return the worst
/// relative error against the known solution. `blas` sends every CPU front
/// through the blocked BLAS and LAPACK route.
fn solve<T: Scalar>(kind: Kind, force_cpu: bool, blas: bool) -> f64 {
    let mut config = Config {
        force_cpu,
        ..Config::default()
    };
    if blas {
        config.small_front_rows = 1;
        config.cpu_blas_min_rows = 1;
        config.cpu_panel = 4;
    }
    let (plan, values, entries) = system::<T>(kind, &config);
    let n = SIDE * SIDE;
    let expected: Vec<T> = (0..2 * n)
        .map(|i| {
            let t = i as f64;
            T::from_c64(Complex64::new((0.37 * t).sin(), (0.11 * t).cos()))
        })
        .collect();
    let mut b = Vec::new();
    for column in expected.chunks(n) {
        b.extend(apply(kind, &entries, column).into_iter().map(T::from_c64));
    }
    let mut solver = Solver::new(plan).unwrap();
    solver.factor_values(&values).unwrap();
    assert_eq!(solver.factored_kind(), Some(T::KIND));
    let x = solver.solve_values(&b, 2).unwrap();
    let scale = expected.iter().map(|v| v.to_c64().norm()).fold(0.0, f64::max);
    x.iter()
        .zip(&expected)
        .map(|(a, e)| (a.to_c64() - e.to_c64()).norm() / scale)
        .fold(0.0, f64::max)
}

#[test]
/// Every value type on the CPU, LDL^T and LU, each to its own precision, on
/// the default routes and on the BLAS route.
fn cpu_all_types() {
    for (kind, blas) in [(Kind::Ldlt, false), (Kind::Lu, false), (Kind::Ldlt, true), (Kind::Lu, true)] {
        let errors = [
            (ScalarKind::F32, solve::<f32>(kind, true, blas), 1e-3),
            (ScalarKind::F64, solve::<f64>(kind, true, blas), 1e-11),
            (ScalarKind::Complex32, solve::<Complex32>(kind, true, blas), 1e-3),
            (ScalarKind::Complex64, solve::<Complex64>(kind, true, blas), 1e-11),
        ];
        for (scalar, error, tol) in errors {
            eprintln!("cpu {kind:?} blas={blas} {scalar}: max error {error:.2e}");
            assert!(error < tol, "cpu {kind:?} {scalar}: error {error:.2e}");
        }
    }
}

#[test]
/// Single precision on the GPU; double precision is refused there.
fn gpu_single_precision() {
    if !sparta::gpu_available() {
        return;
    }
    for kind in [Kind::Ldlt, Kind::Lu] {
        for (scalar, error) in [
            (ScalarKind::F32, solve::<f32>(kind, false, false)),
            (ScalarKind::Complex32, solve::<Complex32>(kind, false, false)),
        ] {
            eprintln!("gpu {kind:?} {scalar}: max error {error:.2e}");
            assert!(error < 1e-3, "gpu {kind:?} {scalar}: error {error:.2e}");
        }
        let (plan, _, _) = system::<f64>(kind, &Config::default());
        let mut solver = Solver::new(plan).unwrap();
        let (_, values, _) = system::<f64>(kind, &Config::default());
        assert!(solver.factor_values(&values).is_err());
    }
}

#[test]
/// A complex right-hand side against real factors, and the refusals: a real
/// right-hand side against complex factors, and a precision mismatch.
fn mixed_right_hand_sides() {
    let config = Config {
        force_cpu: true,
        ..Config::default()
    };
    let (plan, values, entries) = system::<f64>(Kind::Ldlt, &config);
    let n = SIDE * SIDE;
    let expected: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64).sin(), (i as f64 * 0.5).cos()))
        .collect();
    let mut b = vec![Complex64::new(0.0, 0.0); n];
    for &(r, c, v) in &entries {
        b[r] += v * expected[c];
        if r != c {
            b[c] += v * expected[r];
        }
    }
    let mut solver = Solver::new(plan).unwrap();
    solver.factor_values(&values).unwrap();
    let x = solver.solve_values(&b, 1).unwrap();
    for (a, e) in x.iter().zip(&expected) {
        assert!((a - e).norm() < 1e-10, "{a} != {e}");
    }
    assert!(solver.solve_values(&vec![0.0f32; n], 1).is_err());
    let (plan, values, _) = system::<Complex64>(Kind::Ldlt, &config);
    let mut solver = Solver::new(plan).unwrap();
    solver.factor_values(&values).unwrap();
    assert!(solver.solve_values(&vec![0.0f64; n], 1).is_err());
}
