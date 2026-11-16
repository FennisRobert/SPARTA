//! Symmetric indefinite `L·D·Lᵀ` with Bunch-Kaufman pivoting, lower triangle,
//! with LAPACK's conventions exactly (`?sytrf`, `UPLO = 'L'`).
//!
//! Complex matrices are complex *symmetric*: the transpose is plain, never
//! conjugated. Magnitudes in the pivot tests are `|re| + |im|`, as LAPACK's.
//!
//! The algorithm is LAPACK's, transliterated: panels of [`NB`] columns are
//! factored left to right (`?lasyf`), each column updated just before its
//! pivot test from the panel's earlier columns and `W = L·D`; then the rest of
//! the matrix is updated at once, `A22 -= L21·W21ᵀ`, by GEMM. The last panel
//! is factored unblocked (`?sytf2`).
//!
//! `ipiv`, one-based: `ipiv[k] > 0` is a 1 × 1 pivot, rows and columns `k` and
//! `ipiv[k]` interchanged; `ipiv[k] = ipiv[k+1] < 0` is a 2 × 2 pivot, rows and
//! columns `k+1` and `-ipiv[k]` interchanged. Each column of L is stored in the
//! row order of its own step: later interchanges are not applied to it.
use crate::element::{Element, Real};
use crate::gemm::gemm;
use crate::pack::View;
use crate::threading;

/// Panel width. Narrower than LAPACK's 64: most supernodes' diagonal blocks
/// are under 64 columns, and a narrower panel sends more of each to GEMM. On
/// the em_m1 EMerge system (M4 Pro, complex128, SME) the leading phase ran at
/// 51, 60, 70 and 68 GFLOP/s for 96, 64, 32 and 48; 16 and 24 matched 32
/// within noise. Accelerate's `zsytrf` ran it at 66.
const NB: usize = 32;

/// Bunch-Kaufman's growth bound, (1 + √17) / 8.
fn alpha<R: Real>() -> R {
    R::from_f64((1.0 + 17f64.sqrt()) / 8.0)
}

/// Column-major access to an `n × n` block.
#[derive(Clone, Copy)]
struct Mat<T> {
    p: *mut T,
    ld: usize,
}

impl<T: Element> Mat<T> {
    #[inline(always)]
    unsafe fn at(self, i: usize, j: usize) -> *mut T {
        self.p.add(i + j * self.ld)
    }
    #[inline(always)]
    unsafe fn get(self, i: usize, j: usize) -> T {
        *self.at(i, j)
    }
    #[inline(always)]
    unsafe fn set(self, i: usize, j: usize, v: T) {
        *self.at(i, j) = v
    }
    unsafe fn swap(self, i: usize, j: usize, k: usize, l: usize) {
        std::ptr::swap(self.at(i, j), self.at(k, l));
    }
    /// The submatrix starting at `(i, i)`.
    unsafe fn from(self, i: usize) -> Mat<T> {
        Mat { p: self.at(i, i), ld: self.ld }
    }
}

/// The first index in `range` of largest `|re| + |im|` among `f(i)`.
unsafe fn argmax<T: Element>(range: std::ops::Range<usize>, f: impl Fn(usize) -> T) -> usize {
    let mut best = range.start;
    let mut big = f(best).abs1();
    for i in range.start + 1..range.end {
        let v = f(i).abs1();
        if v > big {
            big = v;
            best = i;
        }
    }
    best
}

/// `L·D·Lᵀ` of the lower triangle of the `n × n` column-major block `a`,
/// threaded per [`threading::set`]. Returns LAPACK's `info`: zero, or the
/// one-based index of the first exactly zero diagonal of D (the factorization
/// still completes).
///
/// # Safety
/// `a` must hold `lda · (n - 1) + n` values; `ipiv` at least `n`.
pub unsafe fn sytrf<T: Element>(n: usize, a: *mut T, lda: usize, ipiv: &mut [i32]) -> i32 {
    assert!(ipiv.len() >= n, "sytrf: ipiv holds {} of {n} interchanges", ipiv.len());
    let threads = threading::machine();
    let a = Mat { p: a, ld: lda };
    let mut w = vec![T::zero(); n * NB];
    let mut info = 0;
    let mut k = 0;
    while k < n {
        let rest = n - k;
        let (kb, step) = if k + NB < n {
            lasyf(threads, rest, a.from(k), &mut w, &mut ipiv[k..])
        } else {
            (rest, sytf2(rest, a.from(k), &mut ipiv[k..]))
        };
        if info == 0 && step > 0 {
            info = step + k as i32;
        }
        for p in &mut ipiv[k..k + kb] {
            *p = if *p > 0 { *p + k as i32 } else { *p - k as i32 };
        }
        k += kb;
    }
    info
}

/// `A[j0.., j0..] -= L·Wᵀ` on the lower triangle of the `m × m` block at
/// `(j0, j0)`, where L is `A[j0.., 0..k]` and W is `W[j0.., 0..k]`.
///
/// Recursive, as a SYRK: the block halves into two triangles, recursed, and
/// the rectangle below the first, one GEMM. Most of the work then lands in a
/// few large products, which the kernels run near peak and thread well,
/// rather than in one `NB`-column GEMM per block column. On a 1286-column
/// `zsytrf` on 8 Zen 4 cores, the whole factorization went from 62 to 106
/// GFLOP/s threaded (OpenBLAS: 212), and on one core from 0.67-0.78 to
/// 0.68-0.85 times OpenBLAS's speed over 128 to 1286 columns.
/// Triangles of `NB` columns or fewer go through scratch: their whole square
/// by GEMM, then only the lower triangle back, since the upper is not
/// referenced.
unsafe fn update_lower<T: Element>(threads: bool, j0: usize, m: usize, a: Mat<T>, w: Mat<T>, k: usize) {
    let l = |row: usize| View { p: a.at(row, 0) as *const T, rs: 1, cs: a.ld as isize };
    let wt = |row: usize| View { p: w.at(row, 0) as *const T, rs: w.ld as isize, cs: 1 };
    if m <= NB {
        let mut block = vec![T::zero(); m * m];
        gemm(threads, m, m, k, -T::one(), l(j0), wt(j0), T::zero(), block.as_mut_ptr(), 1, m as isize);
        for jj in 0..m {
            for i in jj..m {
                a.set(j0 + i, j0 + jj, a.get(j0 + i, j0 + jj) + block[i + jj * m]);
            }
        }
        return;
    }
    let h = (m / 2).div_ceil(NB) * NB;
    update_lower(threads, j0, h, a, w, k);
    gemm(threads, m - h, h, k, -T::one(), l(j0 + h), wt(j0), T::one(), a.at(j0 + h, j0), 1, a.ld as isize);
    update_lower(threads, j0 + h, m - h, a, w, k);
}

crate::isa::multiversion! {
    /// `out[0..len] -= Σ_p src[0..len, p]·s[p]` for `p < k`: one column of the
    /// panel brought up to date by the panel's earlier columns. `src` is
    /// column-major with leading dimension `lds`; `s[p]` is at `s + p·ss`.
    /// This is the panel's serial work, a matrix-vector product, so it is
    /// compiled per kernel family.
    unsafe fn panel_update<T: Element>(len: usize, k: usize, out: *mut T, src: *const T, lds: usize, s: *const T, ss: usize) => panel_update_body
}

/// The loop of [`panel_update`], four source columns per pass so `out` is
/// read and written a quarter as often. With the AVX copies, `zsytrf` on one
/// Zen 4 core went from 0.68-0.85 to 0.91-1.03 times OpenBLAS's speed (128 to
/// 1286 columns).
#[inline(always)]
unsafe fn panel_update_body<T: Element>(len: usize, k: usize, out: *mut T, src: *const T, lds: usize, s: *const T, ss: usize) {
    let out = std::slice::from_raw_parts_mut(out, len);
    let col = |p: usize| std::slice::from_raw_parts(src.add(p * lds), len);
    let mut p = 0;
    while p + 4 <= k {
        let (s0, s1, s2, s3) = (*s.add(p * ss), *s.add((p + 1) * ss), *s.add((p + 2) * ss), *s.add((p + 3) * ss));
        let (c0, c1, c2, c3) = (col(p), col(p + 1), col(p + 2), col(p + 3));
        for i in 0..len {
            out[i] = out[i] - (c0[i] * s0 + c1[i] * s1 + c2[i] * s2 + c3[i] * s3);
        }
        p += 4;
    }
    while p < k {
        let s0 = *s.add(p * ss);
        for (o, &x) in out.iter_mut().zip(col(p)) {
            *o = *o - x * s0;
        }
        p += 1;
    }
}

/// One blocked panel (`?lasyf`, lower) of the `n × n` trailing matrix `a`.
/// Returns (columns factored, `info` relative to the panel).
unsafe fn lasyf<T: Element>(threads: bool, n: usize, a: Mat<T>, w: &mut [T], ipiv: &mut [i32]) -> (usize, i32) {
    let w = Mat { p: w.as_mut_ptr(), ld: n };
    let al = alpha::<T::Real>();
    let zero = T::Real::ZERO;
    let mut info = 0;
    let mut k = 0;
    // Column k of the trailing matrix, updated by the panel so far, into
    // column `col` of W: W[k.., col] -= A[k.., 0..k]·W[row, 0..k]^T.
    let update = |k: usize, col: usize, row: usize| panel_update(n - k, k, w.at(k, col), a.at(k, 0), a.ld, w.at(row, 0), w.ld);
    loop {
        if (k + 1 >= NB && NB < n) || k >= n {
            break;
        }
        for i in k..n {
            w.set(i, k, a.get(i, k));
        }
        update(k, k, k);
        let mut kstep = 1;
        let absakk = w.get(k, k).abs1();
        let (imax, colmax) = if k + 1 < n {
            let i = argmax(k + 1..n, |i| w.get(i, k));
            (i, w.get(i, k).abs1())
        } else {
            (k, zero)
        };
        let kp;
        if absakk.max_r(colmax) == zero {
            if info == 0 {
                info = k as i32 + 1;
            }
            kp = k;
        } else {
            if absakk >= al * colmax {
                kp = k;
            } else {
                // Column imax, updated, into column k+1 of W.
                for i in 0..imax - k {
                    w.set(k + i, k + 1, a.get(imax, k + i));
                }
                for i in imax..n {
                    w.set(i, k + 1, a.get(i, imax));
                }
                update(k, k + 1, imax);
                let jmax = k + argmax(0..imax - k, |i| w.get(k + i, k + 1));
                let mut rowmax = w.get(jmax, k + 1).abs1();
                if imax + 1 < n {
                    let j = argmax(imax + 1..n, |i| w.get(i, k + 1));
                    rowmax = rowmax.max_r(w.get(j, k + 1).abs1());
                }
                if absakk >= al * colmax * (colmax / rowmax) {
                    kp = k;
                } else if w.get(imax, k + 1).abs1() >= al * rowmax {
                    kp = imax;
                    for i in k..n {
                        w.set(i, k, w.get(i, k + 1));
                    }
                } else {
                    kp = imax;
                    kstep = 2;
                }
            }
            let kk = k + kstep - 1;
            if kp != kk {
                // The updated column kp is already in column kk of W; copy the
                // not yet updated column kk of A to column kp.
                a.set(kp, kp, a.get(kk, kk));
                for i in 0..kp - kk - 1 {
                    a.set(kp, kk + 1 + i, a.get(kk + 1 + i, kk));
                }
                for i in kp + 1..n {
                    a.set(i, kp, a.get(i, kk));
                }
                for j in 0..kk {
                    a.swap(kk, j, kp, j);
                }
                for j in 0..=kk {
                    w.swap(kk, j, kp, j);
                }
            }
            if kstep == 1 {
                for i in k..n {
                    a.set(i, k, w.get(i, k));
                }
                if k + 1 < n {
                    let r1 = T::one() / a.get(k, k);
                    for i in k + 1..n {
                        a.set(i, k, a.get(i, k) * r1);
                    }
                }
            } else {
                if k + 2 < n {
                    let d21 = w.get(k + 1, k);
                    let d11 = w.get(k + 1, k + 1) / d21;
                    let d22 = w.get(k, k) / d21;
                    let t = T::one() / (d11 * d22 - T::one());
                    let d21 = t / d21;
                    for j in k + 2..n {
                        a.set(j, k, d21 * (d11 * w.get(j, k) - w.get(j, k + 1)));
                        a.set(j, k + 1, d21 * (d22 * w.get(j, k + 1) - w.get(j, k)));
                    }
                }
                a.set(k, k, w.get(k, k));
                a.set(k + 1, k, w.get(k + 1, k));
                a.set(k + 1, k + 1, w.get(k + 1, k + 1));
            }
        }
        if kstep == 1 {
            ipiv[k] = kp as i32 + 1;
        } else {
            ipiv[k] = -(kp as i32 + 1);
            ipiv[k + 1] = -(kp as i32 + 1);
        }
        k += kstep;
    }

    // A22 -= L21·W21^T, lower triangle.
    update_lower(threads, k, n - k, a, w, k);

    // Put L21 back in LAPACK's storage order: undo, in the earlier columns,
    // the row interchanges the panel applied to them.
    let mut j = k as isize - 1;
    while j >= 0 {
        let jj = j as usize;
        let mut jp = ipiv[jj];
        if jp < 0 {
            jp = -jp;
            j -= 1;
        }
        j -= 1;
        let jp = (jp - 1) as usize;
        if jp != jj && j >= 0 {
            for c in 0..=(j as usize) {
                a.swap(jp, c, jj, c);
            }
        }
    }
    (k, info)
}

/// The unblocked factorization (`?sytf2`, lower) of the `n × n` block `a`.
/// Returns `info` relative to the block.
unsafe fn sytf2<T: Element>(n: usize, a: Mat<T>, ipiv: &mut [i32]) -> i32 {
    let al = alpha::<T::Real>();
    let zero = T::Real::ZERO;
    let mut info = 0;
    let mut k = 0;
    while k < n {
        let mut kstep = 1;
        let absakk = a.get(k, k).abs1();
        let (imax, colmax) = if k + 1 < n {
            let i = argmax(k + 1..n, |i| a.get(i, k));
            (i, a.get(i, k).abs1())
        } else {
            (k, zero)
        };
        let kp;
        // `absakk != absakk` is NaN, which LAPACK treats as a zero column.
        #[allow(clippy::eq_op)]
        let nan = absakk != absakk;
        if absakk.max_r(colmax) == zero || nan {
            if info == 0 {
                info = k as i32 + 1;
            }
            kp = k;
        } else {
            if absakk >= al * colmax {
                kp = k;
            } else {
                let jmax = k + argmax(0..imax - k, |i| a.get(imax, k + i));
                let mut rowmax = a.get(imax, jmax).abs1();
                if imax + 1 < n {
                    let j = argmax(imax + 1..n, |i| a.get(i, imax));
                    rowmax = rowmax.max_r(a.get(j, imax).abs1());
                }
                if absakk >= al * colmax * (colmax / rowmax) {
                    kp = k;
                } else if a.get(imax, imax).abs1() >= al * rowmax {
                    kp = imax;
                } else {
                    kp = imax;
                    kstep = 2;
                }
            }
            let kk = k + kstep - 1;
            if kp != kk {
                for i in kp + 1..n {
                    a.swap(i, kk, i, kp);
                }
                for j in kk + 1..kp {
                    a.swap(j, kk, kp, j);
                }
                a.swap(kk, kk, kp, kp);
                if kstep == 2 {
                    a.swap(k + 1, k, kp, k);
                }
            }
            if kstep == 1 {
                if k + 1 < n {
                    let d11 = T::one() / a.get(k, k);
                    for j in k + 1..n {
                        let xj = d11 * a.get(j, k);
                        for i in j..n {
                            a.set(i, j, a.get(i, j) - a.get(i, k) * xj);
                        }
                    }
                    for i in k + 1..n {
                        a.set(i, k, a.get(i, k) * d11);
                    }
                }
            } else if k + 2 < n {
                let d21 = a.get(k + 1, k);
                let d11 = a.get(k + 1, k + 1) / d21;
                let d22 = a.get(k, k) / d21;
                let t = T::one() / (d11 * d22 - T::one());
                let d21 = t / d21;
                for j in k + 2..n {
                    let wk = d21 * (d11 * a.get(j, k) - a.get(j, k + 1));
                    let wkp1 = d21 * (d22 * a.get(j, k + 1) - a.get(j, k));
                    for i in j..n {
                        a.set(i, j, a.get(i, j) - a.get(i, k) * wk - a.get(i, k + 1) * wkp1);
                    }
                    a.set(j, k, wk);
                    a.set(j, k + 1, wkp1);
                }
            }
        }
        if kstep == 1 {
            ipiv[k] = kp as i32 + 1;
        } else {
            ipiv[k] = -(kp as i32 + 1);
            ipiv[k + 1] = -(kp as i32 + 1);
        }
        k += kstep;
    }
    info
}

/// `max` on [`Real`], which has only `PartialOrd`: the larger, or `self` when
/// either is NaN.
trait MaxR {
    fn max_r(self, other: Self) -> Self;
}
impl<R: Real> MaxR for R {
    fn max_r(self, other: R) -> R {
        if other > self {
            other
        } else {
            self
        }
    }
}
