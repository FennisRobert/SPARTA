//! CPU numerical regression tests using independent dense matrix products.
use sparta::{Analysis, Complex64 as C, Config, Kind, Solver};

/// Build CSC and a CPU plan for row-major `a` of size `n`, preserving structural zeros if `dense`.
fn plan(a: &[C], n: usize, kind: Kind, config: &Config, dense: bool) -> (Analysis, Vec<C>) {
    let (mut ptr, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..n {
            if (kind == Kind::Lu || i >= j) && (dense || a[i * n + j] != C::new(0.0, 0.0)) {
                rows.push(i);
                values.push(a[i * n + j]);
            }
        }
        ptr.push(rows.len());
    }
    let p = if kind == Kind::Lu {
        Analysis::lu(n, &ptr, &rows, config)
    } else {
        Analysis::ldlt(n, &ptr, &rows, config)
    }
    .unwrap();
    (p, values)
}
/// Multiply row-major `a` by each packed column of `x` at dimension `n`.
fn product(a: &[C], n: usize, x: &[C]) -> Vec<C> {
    x.chunks(n)
        .flat_map(|v| (0..n).map(move |i| (0..n).map(|j| a[i * n + j] * v[j]).sum()))
        .collect()
}
/// Assert componentwise scaled error between complex128 `actual` and `expected`.
fn close(actual: &[C], expected: &[C], tol: f64) {
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*a - *e).norm() <= tol * (1.0 + e.norm()),
            "entry {i}: {a} != {e}"
        );
    }
}
/// Deterministic complex dense matrix; `kind` selects transpose symmetry without conjugation.
fn matrix(n: usize, kind: Kind) -> Vec<C> {
    let mut a = vec![C::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let (r, c) = if kind == Kind::Ldlt {
                (i.max(j), i.min(j))
            } else {
                (i, j)
            };
            let t = (r * 73 + c * 29 + 7) as f64;
            a[i * n + j] = C::new(t.sin(), (t * 0.7).cos())
                + if i == j {
                    C::new(0.1, 0.0)
                } else {
                    C::new(0.0, 0.0)
                };
        }
    }
    a
}

#[test]
/// Exercise serial, one-column and blocked panels, including Accelerate and Rust updates.
fn dense_pivoting_and_multiple_rhs() {
    for kind in [Kind::Ldlt, Kind::Lu] {
        for n in [2, 9, 35, 67] {
            let a = matrix(n, kind);
            let expected: Vec<_> = (0..3 * n)
                .map(|i| C::new((i as f64).sin(), (i as f64 * 0.3).cos()))
                .collect();
            let b = product(&a, n, &expected);
            for (small, panel, blas, supernodal) in [
                (100, 7, 1, false),
                (1, 1, 1, false),
                (1, 3, usize::MAX, false),
                (1, 16, 1, false),
                (1, 32, 64, false),
                (100, 7, 1, true),
                (1, 32, 64, true),
            ] {
                let config = Config {
                    force_cpu: true,
                    supernodal,
                    small_front_rows: small,
                    cpu_panel: panel,
                    cpu_blas_min_rows: blas,
                    ..Config::default()
                };
                let (p, values) = plan(&a, n, kind, &config, true);
                let mut s = Solver::new(p).unwrap();
                assert!(s.is_cpu());
                s.factor_c128(&values).unwrap();
                let x = s.solve_many_c128(&b, 3).unwrap();
                close(&x, &expected, 2e-9);
                assert_eq!(s.perturbed_pivots(), 0);
                assert_eq!(s.factor_device_seconds(), 0.0);
            }
        }
    }
}

#[test]
/// A solver on a pool of its own, of one thread or several, solves exactly as
/// one on the global pool does, and an impossible count is refused.
fn thread_counts_agree() {
    let side = 16;
    let n = side * side;
    for kind in [Kind::Ldlt, Kind::Lu] {
        let mut a = vec![C::new(0.0, 0.0); n * n];
        for i in 0..n {
            a[i * n + i] = C::new(5.0, 0.5);
            for j in [i + 1, i + side] {
                if j < n && (j != i + 1 || i / side == j / side) {
                    a[i * n + j] = C::new(-0.8, 0.1);
                    a[j * n + i] = if kind == Kind::Lu { C::new(-0.3, 0.2) } else { a[i * n + j] };
                }
            }
        }
        let expected: Vec<_> = (0..2 * n).map(|i| C::new((i as f64).sin(), 0.3)).collect();
        let b = product(&a, n, &expected);
        for threads in [0, 1, 3] {
            let config = Config {
                force_cpu: true,
                threads,
                nemin: 2,
                small_front_rows: 1,
                cpu_blas_min_rows: 1,
                ..Config::default()
            };
            let (p, values) = plan(&a, n, kind, &config, false);
            let mut s = Solver::new(p).unwrap();
            s.factor_c128(&values).unwrap();
            close(&s.solve_many_c128(&b, 2).unwrap(), &expected, 1e-12);
        }
    }
    let config = Config { force_cpu: true, threads: 5000, ..Config::default() };
    assert!(Analysis::ldlt(1, &[0, 1], &[0], &config).is_err());
}

#[test]
/// Child Schur assembly, workspace reuse and local/global permutations agree on sparse grids.
fn sparse_multilevel_refactor() {
    let side = 12;
    let n = side * side;
    for kind in [Kind::Ldlt, Kind::Lu] {
        let mut a = vec![C::new(0.0, 0.0); n * n];
        for i in 0..n {
            a[i * n + i] = C::new(5.0, 0.5);
            for j in [i + 1, i + side] {
                if j < n && (j != i + 1 || i / side == j / side) {
                    a[i * n + j] = C::new(-0.8, 0.1);
                    a[j * n + i] = if kind == Kind::Lu {
                        C::new(-0.3, 0.2)
                    } else {
                        a[i * n + j]
                    };
                }
            }
        }
        let config = Config {
            force_cpu: true,
            nemin: 2,
            small_front_rows: 1,
            cpu_panel: 3,
            cpu_blas_min_rows: 1,
            ..Config::default()
        };
        let (p, values) = plan(&a, n, kind, &config, false);
        assert!(p.supernodes() > 3 && p.levels() > 2);
        let expected: Vec<_> = (0..n).map(|i| C::new((i as f64).sin(), 0.3)).collect();
        let mut s = Solver::new(p).unwrap();
        for scale in [1.0, 1.3, -0.7] {
            s.factor_c128(&values.iter().map(|&v| v * scale).collect::<Vec<_>>())
                .unwrap();
            let b: Vec<_> = product(&a, n, &expected)
                .iter()
                .map(|v| v * scale)
                .collect();
            close(&s.solve_c128(&b).unwrap(), &expected, 1e-12);
        }
    }
}

#[test]
/// Preserve matrix differences below float32 resolution without iterative refinement.
fn ill_conditioned_complex128_input_is_not_rounded() {
    let a = [
        C::new(1.0, 0.0),
        C::new(1.0, 0.0),
        C::new(1.0, 0.0),
        C::new(1.0 + 1e-10, 0.0),
    ];
    for kind in [Kind::Ldlt, Kind::Lu] {
        let (p, v) = plan(
            &a,
            2,
            kind,
            &Config {
                force_cpu: true,
                ..Config::default()
            },
            true,
        );
        let mut s = Solver::new(p).unwrap();
        s.factor_c128(&v).unwrap();
        let expected = [C::new(1.0, 0.0), C::new(-1.0, 0.0)];
        close(
            &s.solve_c128(&product(&a, 2, &expected)).unwrap(),
            &expected,
            2e-6,
        );
        assert_eq!(s.perturbed_pivots(), 0);
    }
}

#[test]
/// Indefinite zero-diagonal blocks require 2x2 pivots, even across one-column panels.
fn two_by_two_pivots_cross_panel_boundaries() {
    let n = 10;
    let mut a = vec![C::new(0.0, 0.0); n * n];
    for i in (0..n).step_by(2) {
        a[i * n + i + 1] = C::new(2.0, 3.0);
        a[(i + 1) * n + i] = C::new(2.0, 3.0);
    }
    for panel in [1, 3, 4] {
        let config = Config {
            force_cpu: true,
            small_front_rows: 1,
            cpu_panel: panel,
            cpu_blas_min_rows: 1,
            ..Config::default()
        };
        let (p, v) = plan(&a, n, Kind::Ldlt, &config, true);
        let mut s = Solver::new(p).unwrap();
        s.factor_c128(&v).unwrap();
        let x = vec![C::new(0.7, -0.4); n];
        close(&s.solve_c128(&product(&a, n, &x)).unwrap(), &x, 1e-14);
        assert_eq!(s.perturbed_pivots(), 0);
        let (lo, hi) = s.pivot_range().unwrap();
        assert!(lo > 3.0 && hi < 4.0);
    }
}

#[test]
/// Failed refactors invalidate factors, successful retries recover, and destroy is permanent.
fn lifecycle_and_validation() {
    let config = Config {
        force_cpu: true,
        ..Config::default()
    };
    let (p, v) = plan(&[C::new(2.0, 1.0)], 1, Kind::Ldlt, &config, true);
    let mut s = Solver::new(p).unwrap();
    assert!(s.solve_c128(&v).is_err());
    s.factor_c128(&v).unwrap();
    assert!(s.solve_many_c128(&v, usize::MAX).is_err());
    assert!(s.solve_c128(&[C::new(f64::NAN, 0.0)]).is_err());
    assert!(s.factor_c128(&[]).is_err());
    assert!(s.solve_c128(&v).is_err());
    assert!(s.pivot_range().is_none());
    s.factor_c128(&v).unwrap();
    assert!(s.factor_c128(&[C::new(0.0, 0.0)]).is_err());
    assert!(s.solve_c128(&v).is_err());
    s.factor_c128(&v).unwrap();
    s.destroy();
    s.destroy();
    assert!(s.factor_c128(&v).is_err());
    assert!(s.solve_c128(&v).is_err());
}

#[test]
/// CPU settings are validated and storage estimates count sixteen-byte complex values.
fn configuration_and_memory() {
    let c = Config {
        force_cpu: true,
        cpu_panel: 7,
        cpu_block_rows: 5,
        cpu_blas_min_rows: 3,
        ..Config::default()
    };
    assert_eq!(c.check(), Ok(()));
    assert!(Config {
        cpu_panel: 0,
        ..c.clone()
    }
    .check()
    .is_err());
    let a = matrix(9, Kind::Ldlt);
    let (p, _) = plan(&a, 9, Kind::Ldlt, &c, true);
    let (gpu, _) = plan(
        &a,
        9,
        Kind::Ldlt,
        &Config {
            force_cpu: false,
            ..c
        },
        true,
    );
    assert_eq!(p.front_bytes(), gpu.front_bytes() * 2);
    assert_eq!(p.factor_bytes(), gpu.factor_bytes() * 2);
    assert_eq!(p.panel_bytes(), 0);
}

#[test]
/// A singular nonzero block is reported as perturbed, rather than silently accepted as exact.
fn static_perturbations_are_counted() {
    let a = vec![C::new(1.0, 0.0); 4];
    for kind in [Kind::Ldlt, Kind::Lu] {
        for (small, supernodal) in [(1, false), (32, false), (1, true), (32, true)] {
            let config = Config {
                force_cpu: true,
                supernodal,
                small_front_rows: small,
                cpu_panel: 1,
                ..Config::default()
            };
            let (p, v) = plan(&a, 2, kind, &config, true);
            let mut s = Solver::new(p).unwrap();
            s.factor_c128(&v).unwrap();
            assert_eq!(s.perturbed_pivots(), 1);
            let b = vec![C::new(2.0, 0.0); 2];
            let x = s.solve_c128(&b).unwrap();
            close(&product(&a, 2, &x), &b, 1e-14);
            assert!(s.factor_c128(&[C::new(f64::NAN, 0.0); 3]).is_err());
            assert!(s.solve_c128(&b).is_err());
        }
    }
}

#[test]
/// The supernodal route factors a sparse indefinite 3D grid to the same
/// solution as the multifrontal one, through the serial and blocked routes,
/// with 2x2 pivots, on one thread and many, and again after refactoring.
fn supernodal_matches_multifrontal() {
    let side = 7;
    let n = side * side * side;
    let mut a = vec![C::new(0.0, 0.0); n * n];
    for i in 0..n {
        // Small and sign-alternating, so Bunch-Kaufman takes 2x2 pivots.
        a[i * n + i] = C::new(if i % 3 == 0 { 0.01 } else { -2.5 + (i % 5) as f64 }, 0.2);
        for step in [1, side, side * side] {
            let j = i + step;
            let same_line = step != 1 || i / side == j / side;
            let same_plane = step != side || i / (side * side) == j / (side * side);
            if j < n && same_line && same_plane {
                let v = C::new(-1.0 + 0.1 * (i % 7) as f64, 0.05 * (j % 3) as f64);
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
    }
    let expected: Vec<_> = (0..2 * n).map(|i| C::new((i as f64 * 0.37).sin(), (i as f64).cos())).collect();
    let b = product(&a, n, &expected);
    for (small, nemin, threads) in [(1, 1, 1), (1, 8, 0), (1000, 4, 0), (16, 2, 3)] {
        let solve = |supernodal: bool, scale: f64| {
            let config = Config {
                force_cpu: true,
                supernodal,
                small_front_rows: small,
                nemin,
                threads,
                cpu_block_rows: 5,
                cpu_blas_min_rows: 1,
                ..Config::default()
            };
            let (p, values) = plan(&a, n, Kind::Ldlt, &config, false);
            assert!(p.levels() > 2);
            let mut s = Solver::new(p).unwrap();
            for first in [0.5, scale] {
                s.factor_c128(&values.iter().map(|&v| v * first).collect::<Vec<_>>()).unwrap();
            }
            let rhs: Vec<_> = b.iter().map(|v| v * scale).collect();
            (s.solve_many_c128(&rhs, 2).unwrap(), s.perturbed_pivots(), s.pivot_range().unwrap())
        };
        let (multi, bumped, range) = solve(false, -1.3);
        let (sn, sn_bumped, sn_range) = solve(true, -1.3);
        close(&multi, &expected, 1e-9);
        close(&sn, &expected, 1e-9);
        close(&sn, &multi, 1e-11);
        assert_eq!(sn_bumped, bumped);
        assert!((sn_range.0 / range.0 - 1.0).abs() < 1e-8 && (sn_range.1 / range.1 - 1.0).abs() < 1e-8);
    }
}

/// 60 pairs `(2i, 2i+1)` with a zero diagonal, coupled inside each pair and
/// weakly to the next one: nonsingular and well conditioned, but each zero
/// pivot is only usable as a 2x2 block with its partner.
fn zero_diagonal_pairs(pairs: usize) -> Vec<C> {
    let n = 2 * pairs;
    let mut a = vec![C::new(0.0, 0.0); n * n];
    let mut set = |i: usize, j: usize, v: C| {
        a[i * n + j] = v;
        a[j * n + i] = v;
    };
    for i in 0..pairs {
        let (j, k) = (2 * i, 2 * i + 1);
        set(k, j, C::new(2.0 + (i % 5) as f64, 0.7));
        if i + 1 < pairs {
            set(j + 2, j, C::new(0.3, 0.0));
            set(k + 2, j, C::new(-0.2, 0.0));
        }
    }
    a
}

#[test]
/// Small fronts split the pairs of a zero-diagonal matrix, so a front holds
/// one member and its pivot is zero or tiny. Static pivoting keeps the factor
/// usable without refinement; with exact-zero rescue alone, tiny pivots were
/// divided by as found and the solution was off by about 1e22.
fn zero_diagonal_pairs_split_over_fronts() {
    let pairs = 60;
    let n = 2 * pairs;
    let a = zero_diagonal_pairs(pairs);
    let expected: Vec<_> = (0..n).map(|i| C::new((i as f64).sin(), 0.3)).collect();
    let b = product(&a, n, &expected);
    for supernodal in [true, false] {
        for nemin in [n, 4] {
            let config = Config {
                force_cpu: true,
                supernodal,
                nemin,
                ..Config::default()
            };
            // The nonzeros and the (zero) diagonal, as the matrix would arrive.
            let (mut ptr, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
            for j in 0..n {
                for i in j..n {
                    if i == j || a[i * n + j] != C::new(0.0, 0.0) {
                        rows.push(i);
                        values.push(a[i * n + j]);
                    }
                }
                ptr.push(rows.len());
            }
            let p = Analysis::ldlt(n, &ptr, &rows, &config).unwrap();
            let fronts = p.supernodes();
            let mut s = Solver::new(p).unwrap();
            s.factor_c128(&values).unwrap();
            let x = s.solve_c128(&b).unwrap();
            if fronts == 1 {
                assert_eq!(s.perturbed_pivots(), 0);
                close(&x, &expected, 1e-13);
            } else {
                assert!(s.perturbed_pivots() > 0, "{fronts} fronts");
                close(&x, &expected, 1e-3);
            }
        }
    }
}

#[test]
/// A 2x2 candidate with no usable inverse falls back to a 1x1 pivot. The
/// fallback counts as a perturbation only when it changes the pivot.
fn two_by_two_fallback_counts_only_real_perturbations() {
    // Unknowns 0 and 1 form the candidate [[1e-30, 1e-9], [1e-9, 1]], whose
    // scaled determinant is about -1e-18. Rows 3 and 4 are equal, so `?sytrf`
    // reports a zero pivot and the panel kernel refactors the front, rescuing
    // one pivot.
    let n = 5;
    let mut a = vec![C::new(0.0, 0.0); n * n];
    for (i, j, v) in [
        (0, 0, 1e-30),
        (1, 0, 1e-9),
        (1, 1, 1.0),
        (2, 1, 10.0),
        (2, 2, 1.0),
        (3, 2, 1.0),
        (4, 2, 1.0),
        (3, 3, 1.0),
        (4, 3, 1.0),
        (4, 4, 1.0),
    ] {
        a[i * n + j] = C::new(v, 0.0);
        a[j * n + i] = C::new(v, 0.0);
    }
    for supernodal in [true, false] {
        for small_front_rows in [1, 64] {
            // Without static pivoting 1e-30 is kept as found: one perturbation.
            // With it, the panel kernel's fallback takes 1e-30 while 1e-9 is
            // still below it, so it is raised too: two. The serial kernel
            // takes 1e-30 last, with nothing left below it to grow, and keeps it.
            let raised = if small_front_rows == 1 { 2 } else { 1 };
            for (static_pivot, perturbed) in [(0.0, 1), (1e-12, raised)] {
                let config = Config {
                    force_cpu: true,
                    supernodal,
                    small_front_rows,
                    static_pivot,
                    ..Config::default()
                };
                let (p, values) = plan(&a, n, Kind::Ldlt, &config, false);
                assert_eq!(p.supernodes(), 1);
                let mut s = Solver::new(p).unwrap();
                s.factor_c128(&values).unwrap();
                assert_eq!(
                    s.perturbed_pivots(),
                    perturbed,
                    "supernodal {supernodal}, small_front_rows {small_front_rows}, static_pivot {static_pivot}"
                );
            }
        }
    }
}
