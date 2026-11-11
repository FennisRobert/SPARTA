//! Sparse right-hand sides and the port matrix `Z = B^T A^{-1} B` against
//! dense solves of the same factorization.
use sparta::{Analysis, Complex32, Complex64 as C, Config, Error, Kind, Scalar, Solver, SparseRhs};
use std::collections::BTreeMap;

/// A small deterministic generator, so every run sees the same matrices.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() * n as f64) as usize % n
    }
}

/// Lower-triangle CSC of a symmetric matrix given as `(row, col) -> value`
/// with `row >= col`; entries listed twice are summed.
struct Lower {
    n: usize,
    ptr: Vec<usize>,
    rows: Vec<usize>,
    values: Vec<C>,
}
fn lower(n: usize, entries: impl IntoIterator<Item = (usize, usize, C)>) -> Lower {
    let mut map: BTreeMap<(usize, usize), C> = BTreeMap::new();
    for (i, j, v) in entries {
        let key = (i.min(j), i.max(j));
        *map.entry(key).or_insert(C::new(0.0, 0.0)) += v;
    }
    let (mut ptr, mut rows, mut values) = (vec![0; n + 1], Vec::new(), Vec::new());
    for (&(col, row), &v) in &map {
        ptr[col + 1] += 1;
        rows.push(row);
        values.push(v);
    }
    for j in 0..n {
        ptr[j + 1] += ptr[j];
    }
    Lower {
        n,
        ptr,
        rows,
        values,
    }
}

/// Random sparse symmetric: `degree` couplings per row, diagonal `diag`
/// (complex symmetric when `complex`).
fn random(n: usize, degree: usize, diag: f64, complex: bool, seed: u64) -> Lower {
    let mut rng = Rng(seed);
    let mut e = Vec::new();
    for i in 0..n {
        e.push((
            i,
            i,
            C::new(
                diag + rng.next(),
                if complex { 0.5 * rng.next() } else { 0.0 },
            ),
        ));
        for _ in 0..degree {
            let j = rng.below(n);
            if j != i {
                let im = if complex { rng.next() - 0.5 } else { 0.0 };
                e.push((i, j, C::new(rng.next() - 0.5, im)));
            }
        }
    }
    lower(n, e)
}

/// The 5-point grid operator `4 - k2` on the diagonal, `-1` to each neighbour.
fn helmholtz(m: usize, k2: C) -> Lower {
    let mut e = Vec::new();
    for y in 0..m {
        for x in 0..m {
            let i = y * m + x;
            e.push((i, i, C::new(4.0, 0.0) - k2));
            if x + 1 < m {
                e.push((i + 1, i, C::new(-1.0, 0.0)));
            }
            if y + 1 < m {
                e.push((i + m, i, C::new(-1.0, 0.0)));
            }
        }
    }
    lower(m * m, e)
}

/// `p` columns of `k` random unknowns each, with random values.
fn ports<T: Scalar>(n: usize, p: usize, k: usize, seed: u64) -> SparseRhs<T> {
    let mut rng = Rng(seed);
    let columns: Vec<Vec<(usize, T)>> = (0..p)
        .map(|_| {
            (0..k)
                .map(|_| {
                    (
                        rng.below(n),
                        T::from_c64(C::new(rng.next() + 0.5, rng.next() - 0.5)),
                    )
                })
                .collect()
        })
        .collect();
    SparseRhs::from_columns(n, &columns).unwrap()
}

fn config(threads: usize) -> Config {
    Config {
        force_cpu: true,
        threads,
        nemin: 8,
        ..Config::default()
    }
}

/// A CPU solver factored on `a` in value type `T`.
fn factored<T: Scalar>(a: &Lower, config: &Config) -> Solver {
    let plan = Analysis::ldlt(a.n, &a.ptr, &a.rows, config).unwrap();
    let mut s = Solver::new(plan).unwrap();
    let values: Vec<T> = a.values.iter().map(|&v| T::from_c64(v)).collect();
    s.factor_values(&values).unwrap();
    s
}

/// `B^T X` from a dense solve of every column of `b`, and `X` itself.
fn reference<T: Scalar>(s: &mut Solver, b: &SparseRhs<T>) -> (Vec<T>, Vec<T>) {
    let x = s.solve_values(&b.to_dense(), b.ncols()).unwrap();
    (b.transpose_times(&x, b.ncols()).unwrap(), x)
}

/// The largest difference relative to the largest reference entry.
fn relative<T: Scalar>(got: &[T], want: &[T]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .map(|v| v.to_c64().norm())
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    got.iter()
        .zip(want)
        .map(|(a, b)| (a.to_c64() - b.to_c64()).norm())
        .fold(0.0, f64::max)
        / scale
}

fn assert_symmetric<T: Scalar>(z: &[T], p: usize) {
    for i in 0..p {
        for j in 0..p {
            assert_eq!(z[i + j * p], z[j + i * p], "Z[{i},{j}] is not its mirror");
        }
    }
}

/// `port_matrix` and `solve_sparse` against dense solves, to `tol`.
fn check<T: Scalar>(s: &mut Solver, b: &SparseRhs<T>, tol: f64) -> Vec<T> {
    let z = s.port_matrix(b).unwrap();
    let (want, x) = reference(s, b);
    assert_symmetric(&z, b.ncols());
    let dz = relative(&z, &want);
    assert!(dz <= tol, "Z differs by {dz:e}");
    let dx = relative(&s.solve_sparse(b).unwrap(), &x);
    assert!(dx <= tol, "X differs by {dx:e}");
    z
}

#[test]
/// Random real symmetric, definite and indefinite, in both precisions.
fn real_symmetric() {
    for (diag, seed) in [(8.0, 1), (0.0, 2)] {
        let a = random(600, 4, diag, false, seed);
        let mut s = factored::<f64>(&a, &config(0));
        check(&mut s, &ports::<f64>(a.n, 37, 5, seed + 10), 1e-11);
        let mut s = factored::<f32>(&a, &config(0));
        check(&mut s, &ports::<f32>(a.n, 37, 5, seed + 10), 2e-3);
    }
}

#[test]
/// Random complex symmetric, and a complex Helmholtz grid, in both
/// precisions. Nothing is conjugated.
fn complex_symmetric() {
    let a = random(700, 5, 1.0, true, 3);
    let mut s = factored::<C>(&a, &config(0));
    check(&mut s, &ports::<C>(a.n, 41, 6, 4), 1e-11);
    let mut s = factored::<Complex32>(&a, &config(0));
    check(&mut s, &ports::<Complex32>(a.n, 41, 6, 4), 2e-3);
    let a = helmholtz(40, C::new(1.3, 0.05));
    let mut s = factored::<C>(&a, &config(0));
    check(&mut s, &ports::<C>(a.n, 50, 6, 5), 1e-11);
}

#[test]
/// A complex `B` against real factors is taken as its real and imaginary
/// parts, as `solve_values` takes it.
fn complex_ports_on_real_factors() {
    let a = random(500, 4, 0.5, false, 6);
    let mut s = factored::<f64>(&a, &config(0));
    check(&mut s, &ports::<C>(a.n, 30, 4, 7), 1e-11);
    let mut s = factored::<f32>(&a, &config(0));
    check(&mut s, &ports::<Complex32>(a.n, 30, 4, 7), 2e-3);
}

#[test]
/// Pairs with a small diagonal and a strong coupling pivot as 2x2 blocks,
/// and so must pairs with a zero diagonal when they share a front. Unit
/// columns on every row make `Z` the whole inverse, so ports sit on one row
/// and on both rows of every pair.
fn two_by_two_pivots() {
    let pairs = 60;
    let n = 2 * pairs;
    for (diagonal, complex, splits) in [(0.01, false, true), (0.01, true, true), (0.0, true, false)]
    {
        let mut e = Vec::new();
        for i in 0..pairs {
            let (j, k) = (2 * i, 2 * i + 1);
            e.push((j, j, C::new(diagonal, 0.0)));
            e.push((k, k, C::new(diagonal, 0.0)));
            e.push((
                k,
                j,
                C::new(2.0 + (i % 5) as f64, if complex { 0.7 } else { 0.0 }),
            ));
            if i + 1 < pairs {
                e.push((j + 2, j, C::new(0.3, 0.0)));
                e.push((k + 2, j, C::new(-0.2, 0.0)));
            }
        }
        let a = lower(n, e);
        // One front, then many: split over fronts, the ordering can separate
        // a pair, and its rows pivot 1x1 in their own fronts.
        let nemins: &[usize] = if splits { &[n, 4] } else { &[n] };
        for &nemin in nemins {
            let cfg = Config {
                nemin,
                small_front_rows: 1,
                ..config(0)
            };
            let mut s = factored::<C>(&a, &cfg);
            // A zero diagonal pivots 1x1 only by being perturbed, so an
            // unperturbed factor of it pivoted 2x2.
            assert_eq!(s.perturbed_pivots(), 0);
            let unit: Vec<Vec<(usize, C)>> = (0..n).map(|i| vec![(i, C::new(1.0, 0.0))]).collect();
            let b = SparseRhs::from_columns(n, &unit).unwrap();
            check(&mut s, &b, 1e-11);
        }
    }
}

#[test]
/// A singular block is perturbed, and `Z` is that of the perturbed factor,
/// as every solve of it is.
fn perturbed_pivots() {
    let a = lower(
        2,
        [
            (0, 0, C::new(1.0, 0.0)),
            (1, 0, C::new(1.0, 0.0)),
            (1, 1, C::new(1.0, 0.0)),
        ],
    );
    for small in [1, 32] {
        let cfg = Config {
            small_front_rows: small,
            cpu_panel: 1,
            ..config(0)
        };
        let mut s = factored::<C>(&a, &cfg);
        assert_eq!(s.perturbed_pivots(), 1);
        let one = C::new(1.0, 0.0);
        let b = SparseRhs::from_columns(
            2,
            &[vec![(0, one)], vec![(1, one)], vec![(0, one), (1, -one)]],
        )
        .unwrap();
        check(&mut s, &b, 1e-8);
    }
}

#[test]
/// A lossless grid 1e-9 off its first eigenvalue: condition about 1e10.
fn near_singular() {
    let m = 30;
    let lambda = 4.0 - 4.0 * (std::f64::consts::PI / (m + 1) as f64).cos();
    let a = helmholtz(m, C::new(lambda + 1e-9, 0.0));
    let mut s = factored::<C>(&a, &config(0));
    check(&mut s, &ports::<C>(a.n, 20, 4, 8), 1e-4);
}

#[test]
/// Empty columns give zero rows and columns; a row listed twice counts with
/// the sum; identical columns give identical rows; nothing in the block at
/// all gives an empty `Z`.
fn empty_duplicate_and_identical_columns() {
    let a = helmholtz(25, C::new(0.9, 0.1));
    let n = a.n;
    let mut s = factored::<C>(&a, &config(0));
    let one = C::new(1.0, 0.0);
    let columns = vec![
        vec![(3, one), (300, C::new(0.0, 2.0))],
        vec![],
        vec![(3, one), (300, C::new(0.0, 1.0)), (300, C::new(0.0, 1.0))],
        vec![(3, one), (300, C::new(0.0, 2.0))],
        vec![(17, C::new(0.5, 0.5)), (17, C::new(0.5, -0.5))],
        vec![(17, one)],
        vec![],
    ];
    let b = SparseRhs::from_columns(n, &columns).unwrap();
    let z = check(&mut s, &b, 1e-12);
    let p = columns.len();
    for empty in [1, 6] {
        for k in 0..p {
            assert_eq!(z[empty + k * p], C::new(0.0, 0.0));
            assert_eq!(z[k + empty * p], C::new(0.0, 0.0));
        }
    }
    let close = |u: C, v: C| (u - v).norm() <= 1e-12 * (u.norm() + v.norm());
    for k in 0..p {
        for (c, d) in [(0, 2), (0, 3), (4, 5)] {
            assert!(
                close(z[k + c * p], z[k + d * p]),
                "columns {c} and {d}, row {k}"
            );
        }
    }
    let x = s.solve_sparse(&b).unwrap();
    assert!(x[n..2 * n].iter().all(|&v| v == C::new(0.0, 0.0)));
    let none = SparseRhs::<C>::from_columns(n, &[]).unwrap();
    assert!(s.port_matrix(&none).unwrap().is_empty());
    assert!(s.solve_sparse(&none).unwrap().is_empty());
}

#[test]
/// Solvers on pools of 1 and 3 threads agree with the global pool to
/// rounding: their factorizations may split work differently.
fn thread_counts() {
    let a = helmholtz(60, C::new(1.1, 0.02));
    let b = ports::<C>(a.n, 45, 6, 9);
    let mut s = factored::<C>(&a, &config(0));
    let global = check(&mut s, &b, 1e-11);
    for threads in [1, 3] {
        let mut s = factored::<C>(&a, &config(threads));
        let z = check(&mut s, &b, 1e-11);
        assert!(relative(&z, &global) <= 1e-11);
    }
}

#[test]
/// The reach of a port is a path to the root: never more than the factor,
/// growing with the ports' spread, and refused on bad rows.
fn reach() {
    let a = helmholtz(50, C::new(1.0, 0.1));
    let plan = Analysis::ldlt(a.n, &a.ptr, &a.rows, &config(0)).unwrap();
    let s = Solver::new(plan).unwrap();
    let one = s.sparse_reach(&[0]).unwrap();
    assert!(one.fronts >= 1 && one.fronts < one.total_fronts);
    assert!(one.factor_entries < one.total_factor_entries);
    let all: Vec<usize> = (0..a.n).collect();
    let every = s.sparse_reach(&all).unwrap();
    assert_eq!(every.fronts, every.total_fronts);
    assert_eq!(every.factor_entries, every.total_factor_entries);
    assert_eq!(every.columns, a.n);
    let two = s.sparse_reach(&[0, a.n - 1]).unwrap();
    assert!(two.fronts >= one.fronts);
    assert_eq!(s.sparse_reach(&[]).unwrap().fronts, 0);
    assert!(matches!(s.sparse_reach(&[a.n]), Err(Error::Invalid(_))));
}

#[test]
/// Bad blocks, unfactored or LU solvers and mismatched types are refused.
fn inputs_are_checked() {
    let one = C::new(1.0, 0.0);
    assert!(SparseRhs::<C>::new(0, vec![0], vec![], vec![]).is_err());
    assert!(SparseRhs::new(4, vec![0, 1], vec![4], vec![one]).is_err());
    assert!(SparseRhs::new(4, vec![1, 1], vec![0], vec![one]).is_err());
    assert!(SparseRhs::new(4, vec![0, 2, 1], vec![0, 1], vec![one, one]).is_err());
    assert!(SparseRhs::new(4, vec![0, 1], vec![0], vec![one, one]).is_err());
    assert!(SparseRhs::new(4, vec![0, 1], vec![0], vec![C::new(f64::NAN, 0.0)]).is_err());
    assert!(SparseRhs::<C>::new(4, vec![0, 0], vec![], vec![]).is_ok());

    let a = helmholtz(10, C::new(0.5, 0.1));
    let b = ports::<C>(a.n, 3, 2, 11);
    let plan = Analysis::ldlt(a.n, &a.ptr, &a.rows, &config(0)).unwrap();
    let mut s = Solver::new(plan).unwrap();
    assert!(matches!(s.port_matrix(&b), Err(Error::Invalid(_))));
    s.factor_c128(&a.values).unwrap();
    assert!(s.port_matrix(&ports::<C>(a.n + 1, 3, 2, 11)).is_err());
    assert!(s.port_matrix(&ports::<f64>(a.n, 3, 2, 11)).is_err());
    assert!(s.port_matrix(&ports::<Complex32>(a.n, 3, 2, 11)).is_err());
    assert!(s.port_matrix(&b).is_ok());
    s.destroy();
    assert!(s.port_matrix(&b).is_err());
    assert!(s.sparse_reach(&[0]).is_err());

    // LU factors have no `D`, and `B^T A^{-1} B` is not their structure.
    let (n, mut ptr, mut rows, mut values) = (a.n, vec![0], Vec::new(), Vec::new());
    for j in 0..n {
        for i in [j.wrapping_sub(1), j, j + 1] {
            if i < n {
                rows.push(i);
                values.push(if i == j {
                    C::new(4.0, 0.0)
                } else {
                    C::new(-1.0, 0.3)
                });
            }
        }
        ptr.push(rows.len());
    }
    let plan = Analysis::lu(n, &ptr, &rows, &config(0)).unwrap();
    assert_eq!(plan.kind(), Kind::Lu);
    let mut s = Solver::new(plan).unwrap();
    s.factor_c128(&values).unwrap();
    assert!(matches!(s.port_matrix(&b), Err(Error::Unsupported(_))));
    assert!(matches!(s.solve_sparse(&b), Err(Error::Unsupported(_))));
    assert!(matches!(s.sparse_reach(&[0]), Err(Error::Unsupported(_))));

    // GPU factors are single precision and are refused rather than used.
    if sparta::gpu_available() {
        let plan = Analysis::ldlt(a.n, &a.ptr, &a.rows, &Config::default()).unwrap();
        let mut s = Solver::new(plan).unwrap();
        let values: Vec<Complex32> = a
            .values
            .iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect();
        s.factor(&values).unwrap();
        let b32 = ports::<Complex32>(a.n, 3, 2, 11);
        assert!(matches!(s.port_matrix(&b32), Err(Error::Unsupported(_))));
        assert!(matches!(s.solve_sparse(&b32), Err(Error::Unsupported(_))));
        assert!(matches!(s.sparse_reach(&[0]), Err(Error::Unsupported(_))));
    }
}
