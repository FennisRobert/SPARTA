//! Every routine, every type, every kernel family this CPU runs, against
//! plain triple loops computed in double precision.
//!
//! Errors are judged against the standard bound for a product of depth k,
//! `|ΔC| ≤ γ·k·ε·(|alpha|·|A|·|B| + |beta|·|C|)`, so a kernel that drops a term
//! or mis-strides fails by orders of magnitude while rounding never does.
#![allow(clippy::needless_range_loop)]
use sparta_blas::{force_isa, gemm_strided, getrf, threading, trsm_strided, Complex, Diag, Element, Isa, Real, Side, Threading, Uplo};
use std::sync::Mutex;

/// The forced family is process-wide; tests take turns.
static ISA: Mutex<()> = Mutex::new(());

/// Run `f` once per family this CPU runs, and per threading setting.
fn each_isa(mut f: impl FnMut(&str)) {
    let _guard = ISA.lock().unwrap_or_else(|e| e.into_inner());
    for isa in Isa::available() {
        force_isa(Some(isa)).unwrap();
        for threading in [Threading::Caller, Threading::Machine] {
            threading::set(threading);
            f(&format!("{} {:?}", isa.name(), threading));
        }
    }
    threading::set(Threading::Machine);
    force_isa(None).unwrap();
}

/// A small deterministic generator, so failures reproduce.
struct Rng(u64);
impl Rng {
    fn real(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn value<T: Element>(&mut self) -> T {
        let re = T::Real::from_f64(self.real());
        let im = T::Real::from_f64(if T::COMPLEX { self.real() } else { 0.0 });
        T::new(re, im)
    }
}

fn wide<T: Element>(x: T) -> Complex<f64> {
    Complex::new(x.re().to_f64(), x.im().to_f64())
}

fn eps<T: Element>() -> f64 {
    if std::mem::size_of::<T::Real>() == 4 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    }
}

/// A matrix stored in a buffer with explicit strides, possibly padded.
struct Mat<T> {
    data: Vec<T>,
    base: isize,
    rs: isize,
    cs: isize,
}

impl<T: Element> Mat<T> {
    /// `rows × cols` in one of four layouts: 0 column-major, 1 row-major,
    /// 2 column-major with padding, 3 row-major with both strides negative.
    fn new(rows: usize, cols: usize, layout: usize, rng: &mut Rng) -> Mat<T> {
        let (r, c) = (rows as isize, cols as isize);
        let (rs, cs, len, base) = match layout {
            0 => (1, r.max(1), rows * cols, 0),
            1 => (c.max(1), 1, rows * cols, 0),
            2 => (1, r + 3, (rows + 3) * cols.max(1), 0),
            _ => (-c.max(1), -1, rows * cols, (rows * cols) as isize - 1),
        };
        let data = (0..len.max(1)).map(|_| rng.value()).collect();
        Mat { data, base, rs, cs }
    }
    fn at(&self, i: usize, j: usize) -> T {
        self.data[(self.base + i as isize * self.rs + j as isize * self.cs) as usize]
    }
    fn ptr(&self) -> *const T {
        unsafe { self.data.as_ptr().offset(self.base) }
    }
    fn ptr_mut(&mut self) -> *mut T {
        unsafe { self.data.as_mut_ptr().offset(self.base) }
    }
}

fn check_gemm<T: Element>(label: &str, rng: &mut Rng) {
    let edges = [0usize, 1, 2, 3, 5, 7, 8, 13, 17, 33, 64, 97];
    let mut shapes: Vec<(usize, usize, usize)> = Vec::new();
    for &m in &edges {
        for &n in &[1usize, 6, 7, 13, 25] {
            for &k in &[0usize, 1, 9, 31] {
                shapes.push((m, n, k));
            }
        }
    }
    // Several blocks deep and wide, threaded splits, and the MC/KC edges.
    shapes.extend([(200, 190, 300), (385, 97, 257), (16, 900, 40), (500, 520, 64), (130, 8200, 3)]);
    let scalars: [(f64, f64, f64, f64); 4] = [(-1.0, 0.0, 1.0, 0.0), (-1.0, 0.0, 0.0, 0.0), (0.7, -0.3, 0.4, 0.9), (2.0, 0.0, -1.5, 0.0)];
    for (s, &(m, n, k)) in shapes.iter().enumerate() {
        let (ar, ai, br, bi) = scalars[s % scalars.len()];
        let alpha = T::new(T::Real::from_f64(ar), T::Real::from_f64(if T::COMPLEX { ai } else { 0.0 }));
        let beta = T::new(T::Real::from_f64(br), T::Real::from_f64(if T::COMPLEX { bi } else { 0.0 }));
        let (la, lb, lc) = (s % 4, (s / 4) % 4, (s / 2) % 4);
        let a = Mat::<T>::new(m, k, la, rng);
        let b = Mat::<T>::new(k, n, lb, rng);
        let mut c = Mat::<T>::new(m, n, lc, rng);
        // beta == 0 must overwrite: plant NaN where C will be written.
        if beta == T::zero() {
            for v in c.data.iter_mut() {
                *v = T::new(T::Real::from_f64(f64::NAN), T::Real::ZERO);
            }
        }
        let before: Vec<Vec<T>> = (0..m).map(|i| (0..n).map(|j| c.at(i, j)).collect()).collect();
        unsafe {
            gemm_strided(m, n, k, alpha, a.ptr(), a.rs, a.cs, b.ptr(), b.rs, b.cs, beta, c.ptr_mut(), c.rs, c.cs);
        }
        let (wa, wb) = (wide(alpha), wide(beta));
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex::new(0.0, 0.0);
                let mut mag = 0.0;
                for p in 0..k {
                    let t = wide(a.at(i, p)) * wide(b.at(p, j));
                    sum += t;
                    mag += t.norm();
                }
                let old = if beta == T::zero() { Complex::new(0.0, 0.0) } else { wide(before[i][j]) };
                let want = wa * sum + wb * old;
                let got = wide(c.at(i, j));
                let bound = 8.0 * (k as f64 + 2.0) * eps::<T>() * (wa.norm() * mag + wb.norm() * old.norm()) + 1e-300;
                let err = (got - want).norm();
                assert!(
                    err <= bound,
                    "gemm {label} {}: m={m} n={n} k={k} layouts=({la},{lb},{lc}) at ({i},{j}): got {got}, want {want}, err {err:e} > {bound:e}",
                    std::any::type_name::<T>()
                );
            }
        }
    }
}

fn check_trsm<T: Element>(label: &str, rng: &mut Rng) {
    let mut case = 0;
    for side in [Side::Left, Side::Right] {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for trans in [false, true] {
                for diag in [Diag::NonUnit, Diag::Unit] {
                    for &(m, n) in &[(1usize, 1usize), (5, 3), (33, 7), (31, 130), (97, 40), (150, 260)] {
                        case += 1;
                        let order = if side == Side::Left { m } else { n };
                        // A well-conditioned triangle: unit-sized off-diagonals
                        // scaled down, a dominant diagonal.
                        let mut a = Mat::<T>::new(order, order, case % 3, rng);
                        for i in 0..order {
                            for j in 0..order {
                                let idx = (a.base + i as isize * a.rs + j as isize * a.cs) as usize;
                                let v = a.data[idx];
                                let s = T::Real::from_f64(if i == j { 1.0 } else { 0.5 / (order as f64).sqrt() });
                                a.data[idx] = if i == j { T::new(T::Real::from_f64(2.0) + v.re() * s, v.im() * s) } else { T::new(v.re() * s, v.im() * s) };
                            }
                        }
                        let mut b = Mat::<T>::new(m, n, (case / 3) % 4, rng);
                        let rhs: Vec<Vec<Complex<f64>>> = (0..m).map(|i| (0..n).map(|j| wide(b.at(i, j))).collect()).collect();
                        let alpha = T::new(T::Real::from_f64(if case % 2 == 0 { 1.0 } else { -0.5 }), T::Real::ZERO);
                        unsafe {
                            trsm_strided(side, uplo, trans, diag, m, n, alpha, a.ptr(), a.rs, a.cs, b.ptr_mut(), b.rs, b.cs);
                        }
                        // op(A) entry (i, j) as the solve sees it.
                        let op = |i: usize, j: usize| -> Complex<f64> {
                            let (r, c) = if trans { (j, i) } else { (i, j) };
                            let inside = match uplo {
                                Uplo::Lower => r >= c,
                                Uplo::Upper => r <= c,
                            };
                            if !inside {
                                Complex::new(0.0, 0.0)
                            } else if r == c && diag == Diag::Unit {
                                Complex::new(1.0, 0.0)
                            } else {
                                wide(a.at(r, c))
                            }
                        };
                        let wa = wide(alpha);
                        let mut worst: f64 = 0.0;
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = Complex::new(0.0, 0.0);
                                match side {
                                    Side::Left => (0..m).for_each(|p| sum += op(i, p) * wide(b.at(p, j))),
                                    Side::Right => (0..n).for_each(|p| sum += wide(b.at(i, p)) * op(p, j)),
                                }
                                worst = worst.max((sum - wa * rhs[i][j]).norm());
                            }
                        }
                        let bound = 64.0 * order as f64 * eps::<T>() * 4.0;
                        assert!(
                            worst <= bound,
                            "trsm {label} {}: {side:?} {uplo:?} trans={trans} {diag:?} m={m} n={n}: residual {worst:e} > {bound:e}",
                            std::any::type_name::<T>()
                        );
                    }
                }
            }
        }
    }
}

fn check_getrf<T: Element>(label: &str, rng: &mut Rng) {
    for &(m, n) in &[(1usize, 1usize), (1, 5), (5, 1), (7, 7), (40, 33), (33, 40), (130, 130), (257, 200)] {
        let lda = m + 2;
        let mut a: Vec<T> = (0..lda * n).map(|_| rng.value()).collect();
        let orig = a.clone();
        let mut ipiv = vec![0i32; m.min(n)];
        let info = unsafe { getrf(m, n, a.as_mut_ptr(), lda, &mut ipiv) };
        assert_eq!(info, 0, "getrf {label}: random matrix reported a zero pivot");
        // P·A: the interchanges applied to the original, in order.
        let mut pa: Vec<Vec<Complex<f64>>> = (0..m).map(|i| (0..n).map(|j| wide(orig[j * lda + i])).collect()).collect();
        for (i, &p) in ipiv.iter().enumerate() {
            pa.swap(i, (p - 1) as usize);
        }
        let mut worst: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex::new(0.0, 0.0);
                for p in 0..=i.min(j).min(m.min(n) - 1) {
                    let l = if p == i { Complex::new(1.0, 0.0) } else { wide(a[p * lda + i]) };
                    if p <= j {
                        sum += l * wide(a[j * lda + p]);
                    }
                }
                worst = worst.max((sum - pa[i][j]).norm());
                scale = scale.max(pa[i][j].norm());
                // Pivoting on |re| + |im| bounds every multiplier's modulus by
                // sqrt(2): |l| <= |re| + |im| of the entry <= that of the pivot
                // <= sqrt(2)·|pivot|.
                if i > j && j < m.min(n) {
                    let l = wide(a[j * lda + i]).norm();
                    assert!(l <= 2f64.sqrt() * (1.0 + 1e-6), "getrf {label}: multiplier modulus {l} above sqrt(2)");
                }
            }
        }
        let bound = 32.0 * (m.max(n) as f64) * eps::<T>() * scale;
        assert!(worst <= bound, "getrf {label} {}: m={m} n={n}: |PA - LU| = {worst:e} > {bound:e}", std::any::type_name::<T>());
    }
    // An exactly zero column: info names it, and the factorization finishes.
    let n = 6;
    let mut a: Vec<T> = (0..n * n).map(|_| rng.value()).collect();
    for i in 0..n {
        a[2 * n + i] = T::zero();
    }
    let mut ipiv = vec![0i32; n];
    let info = unsafe { getrf(n, n, a.as_mut_ptr(), n, &mut ipiv) };
    assert_eq!(info, 3, "getrf {label}: zero column 3 not reported");
}

macro_rules! per_type {
    ($name:ident, $check:ident) => {
        #[test]
        fn $name() {
            let mut rng = Rng(42);
            each_isa(|label| {
                $check::<f64>(label, &mut rng);
                $check::<f32>(label, &mut rng);
                $check::<Complex<f64>>(label, &mut rng);
                $check::<Complex<f32>>(label, &mut rng);
            });
        }
    };
}
per_type!(gemm_matches_reference, check_gemm);
per_type!(trsm_matches_reference, check_trsm);
per_type!(getrf_matches_reference, check_getrf);

#[test]
fn abi_matches_strided_calls() {
    use sparta_blas::abi;
    let _guard = ISA.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = Rng(7);
    // Row-major C (3×4) -= A (3×2) · B^T, B stored 4×2: the call SPARTA makes most.
    let a: Vec<Complex<f64>> = (0..6).map(|_| rng.value()).collect();
    let b: Vec<Complex<f64>> = (0..8).map(|_| rng.value()).collect();
    let mut c: Vec<Complex<f64>> = (0..12).map(|_| rng.value()).collect();
    let mut want = c.clone();
    for i in 0..3 {
        for j in 0..4 {
            for p in 0..2 {
                want[i * 4 + j] -= a[i * 2 + p] * b[j * 2 + p];
            }
        }
    }
    let (minus, one) = (Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0));
    unsafe {
        abi::zgemm(
            abi::ROW_MAJOR,
            abi::NO_TRANS,
            abi::TRANS,
            3,
            4,
            2,
            (&minus as *const Complex<f64>).cast(),
            a.as_ptr().cast(),
            2,
            b.as_ptr().cast(),
            2,
            (&one as *const Complex<f64>).cast(),
            c.as_mut_ptr().cast(),
            4,
        );
    }
    for (g, w) in c.iter().zip(&want) {
        assert!((g - w).norm() < 1e-13, "{g} != {w}");
    }
}

/// Solve `A·x = b` from `sytrf`'s output with LAPACK's `?sytrs` algorithm
/// (lower), so the test checks the factor *in LAPACK's storage convention*,
/// which is what SPARTA's `adopt` reads.
fn sytrs<T: Element>(n: usize, a: &[T], lda: usize, ipiv: &[i32], b: &mut [T]) {
    let at = |i: usize, j: usize| a[i + j * lda];
    let mut k = 0;
    while k < n {
        if ipiv[k] > 0 {
            let kp = (ipiv[k] - 1) as usize;
            b.swap(k, kp);
            for i in k + 1..n {
                b[i] = b[i] - at(i, k) * b[k];
            }
            b[k] = b[k] / at(k, k);
            k += 1;
        } else {
            let kp = (-ipiv[k] - 1) as usize;
            b.swap(k + 1, kp);
            for i in k + 2..n {
                b[i] = b[i] - at(i, k) * b[k] - at(i, k + 1) * b[k + 1];
            }
            let akm1k = at(k + 1, k);
            let akm1 = at(k, k) / akm1k;
            let ak = at(k + 1, k + 1) / akm1k;
            let denom = akm1 * ak - T::one();
            let bkm1 = b[k] / akm1k;
            let bk = b[k + 1] / akm1k;
            b[k] = (ak * bkm1 - bk) / denom;
            b[k + 1] = (akm1 * bk - bkm1) / denom;
            k += 2;
        }
    }
    let mut k = n as isize - 1;
    while k >= 0 {
        let ku = k as usize;
        if ipiv[ku] > 0 {
            for i in ku + 1..n {
                b[ku] = b[ku] - at(i, ku) * b[i];
            }
            b.swap(ku, (ipiv[ku] - 1) as usize);
            k -= 1;
        } else {
            for i in ku + 1..n {
                b[ku] = b[ku] - at(i, ku) * b[i];
                b[ku - 1] = b[ku - 1] - at(i, ku - 1) * b[i];
            }
            b.swap(ku, (-ipiv[ku] - 1) as usize);
            k -= 2;
        }
    }
}

fn check_sytrf<T: Element>(label: &str, rng: &mut Rng) {
    // Sizes below, at and across the panel width, with weak diagonals so
    // 2 × 2 pivots and interchanges happen.
    for &(n, diag) in &[(1usize, 1.0), (2, 0.0), (7, 0.1), (63, 0.05), (64, 0.0), (65, 0.3), (130, 0.01), (300, 0.0), (311, 2.0)] {
        let lda = n + 1;
        let mut full = vec![T::zero(); n * n];
        for j in 0..n {
            for i in j..n {
                let mut v: T = rng.value();
                if i == j {
                    v = T::new(v.re() * T::Real::from_f64(diag), v.im() * T::Real::from_f64(diag));
                }
                full[i + j * n] = v;
                full[j + i * n] = v;
            }
        }
        let mut a = vec![T::zero(); lda * n];
        for j in 0..n {
            for i in j..n {
                a[i + j * lda] = full[i + j * n];
            }
        }
        let mut ipiv = vec![0i32; n];
        let info = unsafe { sparta_blas::sytrf(n, a.as_mut_ptr(), lda, &mut ipiv) };
        assert_eq!(info, 0, "sytrf {label}: n={n}: random matrix reported a zero pivot");
        let x: Vec<T> = (0..n).map(|_| rng.value()).collect();
        let mut b: Vec<T> = (0..n).map(|i| (0..n).fold(T::zero(), |s, j| s + full[i + j * n] * x[j])).collect();
        sytrs(n, &a, lda, &ipiv, &mut b);
        let err = b.iter().zip(&x).map(|(g, w)| wide(*g - *w).norm()).fold(0.0, f64::max);
        let scale = x.iter().map(|v| wide(*v).norm()).fold(0.0, f64::max);
        let bound = if std::mem::size_of::<T::Real>() == 4 { 1e-1 } else { 1e-8 };
        assert!(err <= bound * scale, "sytrf {label} {}: n={n} diag={diag}: solve error {err:e}", std::any::type_name::<T>());
    }
}
per_type!(sytrf_solves_through_lapack_convention, check_sytrf);
