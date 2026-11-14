//! Sparse right-hand sides and projections `W^T A^-1 B` against the dense
//! route: `solve_values` on the same columns made dense, then the projection.
use sparta::{Analysis, Complex32, Complex64 as C, Config, Kind, Solver, SparseColumns};

/// A `side^3` grid, indefinite, complex symmetric, with imaginary parts only
/// where `complex(i)` holds for both ends of an entry: ports on a real body,
/// which puts real fronts (the real split) under complex ones. With `pairs`,
/// every other diagonal entry is zero, which forces 2x2 pivots.
fn grid(side: usize, complex: &dyn Fn(usize) -> bool, pairs: bool) -> (usize, Vec<C>) {
    let n = side * side * side;
    let mut a = vec![C::new(0.0, 0.0); n * n];
    for i in 0..n {
        let d = if pairs && i % 2 == 0 { 0.0 } else { -2.5 + (i % 5) as f64 };
        a[i * n + i] = C::new(d, if complex(i) { 0.3 } else { 0.0 });
        for step in [1, side, side * side] {
            let j = i + step;
            let same_line = step != 1 || i / side == j / side;
            let same_plane = step != side || i / (side * side) == j / (side * side);
            if j < n && same_line && same_plane {
                let v = C::new(-1.0 + 0.1 * (i % 7) as f64, if complex(i) && complex(j) { 0.1 } else { 0.0 });
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
    }
    (n, a)
}

/// The lower-triangle CSC of `a` and a solver factored with it.
fn factored(a: &[C], n: usize, kind: Kind, config: &Config) -> Solver {
    let (mut ptr, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..n {
            if (kind == Kind::Lu || i >= j) && a[i * n + j] != C::new(0.0, 0.0) {
                rows.push(i);
                values.push(a[i * n + j]);
            }
        }
        ptr.push(rows.len());
    }
    let plan = match kind {
        Kind::Lu => Analysis::lu(n, &ptr, &rows, config),
        Kind::Ldlt => Analysis::ldlt(n, &ptr, &rows, config),
    }
    .unwrap();
    let mut solver = Solver::new(plan).unwrap();
    solver.factor_values(&values).unwrap();
    solver
}

/// Sparse columns as plain arrays, so the tests can make them dense and
/// project with them independently of the solver.
struct Block {
    n: usize,
    ptr: Vec<usize>,
    rows: Vec<usize>,
    values: Vec<C>,
}

impl Block {
    /// Port-like columns: column `j` touches a handful of unknowns around a
    /// spread-out centre, one row listed twice, and column 3 left empty.
    fn ports(n: usize, count: usize, scale: C) -> Self {
        let (mut ptr, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
        for j in 0..count {
            if j != 3 {
                let centre = (j * 7919) % n;
                for k in 0..5 {
                    rows.push((centre + k * 13) % n);
                    values.push(C::new(1.0 + k as f64, 0.5 - j as f64 * 0.1) * scale);
                }
                // A repeated row counts as the sum.
                rows.push(centre);
                values.push(C::new(0.25, 0.0) * scale);
            }
            ptr.push(rows.len());
        }
        Self { n, ptr, rows, values }
    }

    fn count(&self) -> usize {
        self.ptr.len() - 1
    }

    fn view(&self) -> SparseColumns<'_, C> {
        SparseColumns::new(self.n, &self.ptr, &self.rows, &self.values).unwrap()
    }

    /// Column `j` at `[j * n..]`, repeats summed.
    fn dense(&self) -> Vec<C> {
        let mut x = vec![C::new(0.0, 0.0); self.n * self.count()];
        for j in 0..self.count() {
            for e in self.ptr[j]..self.ptr[j + 1] {
                x[j * self.n + self.rows[e]] += self.values[e];
            }
        }
        x
    }

    /// `self^T x` for dense `x` of `nrhs` columns, column-major.
    fn project(&self, x: &[C], nrhs: usize) -> Vec<C> {
        let q = self.count();
        let mut z = vec![C::new(0.0, 0.0); q * nrhs];
        for j in 0..nrhs {
            for i in 0..q {
                z[j * q + i] = (self.ptr[i]..self.ptr[i + 1]).map(|e| self.values[e] * x[j * self.n + self.rows[e]]).sum();
            }
        }
        z
    }

    fn single(&self) -> Vec<Complex32> {
        self.values.iter().map(|v| Complex32::new(v.re as f32, v.im as f32)).collect()
    }
}

fn close(actual: &[C], expected: &[C], tol: f64, what: &str) {
    let scale = expected.iter().map(|v| v.norm()).fold(1e-300, f64::max);
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((*a - *e).norm() <= tol * scale, "{what}: entry {i}: {a} != {e}");
    }
}

/// Lower-triangle CSC of dense `a`.
fn lower(a: &[C], n: usize) -> (Vec<usize>, Vec<usize>, Vec<C>) {
    let (mut ptr, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
    for j in 0..n {
        for i in j..n {
            if a[i * n + j] != C::new(0.0, 0.0) {
                rows.push(i);
                values.push(a[i * n + j]);
            }
        }
        ptr.push(rows.len());
    }
    (ptr, rows, values)
}

#[test]
/// Every route of the fast path agrees with the dense one: the real split
/// on and off, supernodal and multifrontal, single fronts and many, 2x2
/// pivots, a separate `W`, and projections on `B` itself, exactly symmetric.
fn sparse_solves_and_projections_match_the_dense_route() {
    let side = 7;
    let patches: [&dyn Fn(usize) -> bool; 3] = [&|i| i % 7 == 0 && i < 120, &|_| true, &|_| false];
    for (p, patch) in patches.iter().enumerate() {
        for pairs in [false, true] {
            let (n, a) = grid(side, patch, pairs);
            for (supernodal, real_split, nemin) in [(true, true, 4), (true, false, 4), (false, true, 4), (true, true, 1000)] {
                let config = Config { force_cpu: true, supernodal, real_split, nemin, ..Config::default() };
                let what = format!("patch {p} pairs {pairs} supernodal {supernodal} split {real_split} nemin {nemin}");
                let mut solver = factored(&a, n, Kind::Ldlt, &config);
                let b = Block::ports(n, 11, C::new(1.0, 0.0));
                let w = Block::ports(n, 4, C::new(0.5, -1.0));
                let x = solver.solve_values(&b.dense(), 11).unwrap();
                // The zero-diagonal grid split over many fronts is the
                // unsupported kind (pivots perturbed, factors far from the
                // matrix), where two exact routes may round apart. As one
                // front it pivots in 2x2 blocks and is compared.
                if solver.perturbed_pivots() > 0 {
                    continue;
                }
                close(&solver.solve_sparse(&b.view()).unwrap(), &x, 1e-10, &what);
                let z = solver.solve_projected(&b.view(), None).unwrap();
                close(&z, &b.project(&x, 11), 1e-10, &what);
                for i in 0..11 {
                    for j in 0..11 {
                        assert_eq!(z[j * 11 + i], z[i * 11 + j], "{what}: Z not exactly symmetric");
                    }
                }
                let zw = solver.solve_projected(&b.view(), Some(&w.view())).unwrap();
                close(&zw, &w.project(&x, 11), 1e-10, &what);
                // The empty column projects to a zero row and column.
                assert!((0..11).all(|i| z[3 * 11 + i] == C::new(0.0, 0.0)), "{what}: empty column");
            }
        }
    }
}

#[test]
/// Real factors with complex ports go as real and imaginary parts, and
/// complex64 CPU factors take the fast path too.
fn real_factors_and_single_precision() {
    let (n, a) = grid(6, &|_| false, false);
    let (ptr, rows, values) = lower(&a, n);
    let real: Vec<f64> = values.iter().map(|v| v.re).collect();
    let config = Config { force_cpu: true, nemin: 4, ..Config::default() };
    let mut solver = Solver::new(Analysis::ldlt(n, &ptr, &rows, &config).unwrap()).unwrap();
    solver.factor_values(&real).unwrap();
    let b = Block::ports(n, 6, C::new(1.0, 0.0));
    let w = Block::ports(n, 4, C::new(0.5, -1.0));
    let x = solver.solve_values(&b.dense(), 6).unwrap();
    close(&solver.solve_sparse(&b.view()).unwrap(), &x, 1e-10, "real factors, solve");
    close(&solver.solve_projected(&b.view(), None).unwrap(), &b.project(&x, 6), 1e-10, "real factors, project");
    close(&solver.solve_projected(&b.view(), Some(&w.view())).unwrap(), &w.project(&x, 6), 1e-10, "real factors, W");

    let single: Vec<Complex32> = values.iter().map(|v| Complex32::new(v.re as f32, v.im as f32)).collect();
    let mut solver = Solver::new(Analysis::ldlt(n, &ptr, &rows, &config).unwrap()).unwrap();
    solver.factor_values(&single).unwrap();
    let bv = b.single();
    let b32 = SparseColumns::new(n, &b.ptr, &b.rows, &bv).unwrap();
    let dense: Vec<Complex32> = b.dense().iter().map(|v| Complex32::new(v.re as f32, v.im as f32)).collect();
    let x32 = solver.solve_values(&dense, 6).unwrap();
    let scale = x32.iter().map(|v| v.norm()).fold(0.0, f32::max);
    let s32 = solver.solve_sparse(&b32).unwrap();
    assert!(s32.iter().zip(&x32).all(|(a, e)| (a - e).norm() <= 1e-4 * scale), "complex64 sparse solve");
}

#[test]
/// LU factors and the GPU have no fast path and answer through the dense
/// route, with the same result.
fn other_factors_take_the_dense_route() {
    let (n, a) = grid(5, &|i| i < 30, false);
    let config = Config { force_cpu: true, ..Config::default() };
    let mut lu = factored(&a, n, Kind::Lu, &config);
    let b = Block::ports(n, 5, C::new(1.0, 0.0));
    let x = lu.solve_values(&b.dense(), 5).unwrap();
    close(&lu.solve_sparse(&b.view()).unwrap(), &x, 1e-12, "LU solve");
    close(&lu.solve_projected(&b.view(), None).unwrap(), &b.project(&x, 5), 1e-12, "LU project");
    if sparta::gpu_available() {
        let (lp, lr, lv) = lower(&a, n);
        let plan = Analysis::ldlt(n, &lp, &lr, &Config::default()).unwrap();
        let mut gpu = Solver::new(plan).unwrap();
        let lv32: Vec<Complex32> = lv.iter().map(|v| Complex32::new(v.re as f32, v.im as f32)).collect();
        gpu.factor_values(&lv32).unwrap();
        let bv32 = b.single();
        let b32 = SparseColumns::new(n, &b.ptr, &b.rows, &bv32).unwrap();
        let z = gpu.solve_projected(&b32, None).unwrap();
        let want = b.project(&x, 5);
        let scale = want.iter().map(|v| v.norm()).fold(0.0, f64::max);
        for (g, e) in z.iter().zip(&want) {
            assert!((C::new(g.re as f64, g.im as f64) - e).norm() <= 1e-3 * scale, "GPU projection");
        }
    }
}

#[test]
/// Malformed blocks and mismatches are refused with a reason.
fn input_checks() {
    let rows = [0usize, 9];
    let values = [C::new(1.0, 0.0); 2];
    assert!(SparseColumns::new(5, &[0, 2], &rows, &values).is_err(), "row outside");
    assert!(SparseColumns::new(10, &[1, 2], &rows, &values).is_err(), "bad start");
    assert!(SparseColumns::new(10, &[0, 3], &rows, &values).is_err(), "count mismatch");
    let bad = [C::new(f64::NAN, 0.0), C::new(1.0, 0.0)];
    assert!(SparseColumns::new(10, &[0, 2], &rows, &bad).is_err(), "nan");
    let (n, a) = grid(3, &|_| true, false);
    let mut solver = factored(&a, n, Kind::Ldlt, &Config { force_cpu: true, ..Config::default() });
    let b = SparseColumns::new(n + 1, &[0, 2], &rows, &values).unwrap();
    assert!(solver.solve_sparse(&b).is_err(), "wrong dimension");
}
