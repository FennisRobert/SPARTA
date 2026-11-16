//! Triangular solves with many right-hand sides.
//!
//! Every case is brought to one: `L·X = B` with L lower triangular, solved in
//! place in B. A right-side solve `X·op(A) = B` is the left-side solve
//! `op(A)ᵀ·Xᵀ = Bᵀ`, which only swaps B's strides. A transposed triangle swaps
//! A's strides. An upper triangle becomes a lower one by reversing the order of
//! the unknowns, which negates the strides. Nothing is copied for any of it.
//!
//! The lower solve is recursive: solve the top half, subtract its effect from
//! the bottom half with one GEMM, solve the bottom half. Nearly all the work
//! lands in GEMMs whose depth is half the triangle, which is where the packed
//! kernels run at full speed; only leaves of [`LEAF`] rows are solved directly.
use crate::element::{Element, Real};
use crate::gemm::{gemm, Shared};
use crate::pack::View;
use crate::threading;
use rayon::prelude::*;

/// Which side of the unknown the triangle is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// `op(A)·X = alpha·B`.
    Left,
    /// `X·op(A) = alpha·B`.
    Right,
}

/// Which triangle of A holds the matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uplo {
    Upper,
    Lower,
}

/// Whether the diagonal is stored or taken as ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diag {
    NonUnit,
    Unit,
}

/// Rows below which a triangle is solved directly.
const LEAF: usize = 32;

/// Right-hand sides per direct-solve pass: they are copied into a
/// contiguous row-major scratch block of `LEAF × WIDTH`.
const WIDTH: usize = 128;

/// Right-hand sides from which a leaf inverts its triangle and applies the
/// inverse with one GEMM rather than substituting; see [`leaf_inverse`].
const INVERT_MIN: usize = 16;

/// Columns of B per task when a solve spreads over the pool.
const CHUNK: usize = 64;

/// Solve `op(A)·X = alpha·B` (left) or `X·op(A) = alpha·B` (right) in place
/// in B, threaded per [`threading::set`].
///
/// B is `m × n` with element `(i, j)` at `b + i·rsb + j·csb`; A is `m × m`
/// (left) or `n × n` (right) with element `(i, j)` at `a + i·rsa + j·csa`.
/// `trans` solves with `Aᵀ` (plain transpose, never conjugate). Only the
/// `uplo` triangle of A is read, and not its diagonal when `diag` is `Unit`.
///
/// # Safety
/// Both views must be valid for their shapes and strides and must not overlap.
#[allow(clippy::too_many_arguments)]
pub unsafe fn trsm_strided<T: Element>(
    side: Side,
    uplo: Uplo,
    trans: bool,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *mut T,
    rsb: isize,
    csb: isize,
) {
    trsm(threading::machine(), side, uplo, trans, diag, m, n, alpha, View { p: a, rs: rsa, cs: csa }, b, rsb, csb)
}

/// [`trsm_strided`] with the threading given.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn trsm<T: Element>(
    threads: bool,
    side: Side,
    uplo: Uplo,
    trans: bool,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: View<T>,
    b: *mut T,
    rsb: isize,
    csb: isize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha != T::one() {
        for j in 0..n {
            for i in 0..m {
                let p = b.offset(i as isize * rsb + j as isize * csb);
                *p = if alpha == T::zero() { T::zero() } else { alpha * *p };
            }
        }
        if alpha == T::zero() {
            return;
        }
    }
    // To the left side: X·op(A) = B is op(A)^T·X^T = B^T.
    let (m, n, rsb, csb, trans) = match side {
        Side::Left => (m, n, rsb, csb, trans),
        Side::Right => (n, m, csb, rsb, !trans),
    };
    // op(A) as a plain view, and whether it is lower.
    let a = if trans { View { p: a.p, rs: a.cs, cs: a.rs } } else { a };
    let lower = (uplo == Uplo::Lower) != trans;
    // An upper triangle, read from the last unknown back, is lower.
    let last = (m - 1) as isize;
    let (a, b, rsb) = if lower {
        (a, b, rsb)
    } else {
        (View { p: a.p.offset(last * (a.rs + a.cs)), rs: -a.rs, cs: -a.cs }, b.offset(last * rsb), -rsb)
    };
    let unit = diag == Diag::Unit;
    if threads && rayon::current_num_threads() > 1 && n >= 2 * CHUNK && m >= LEAF {
        // Columns of B are independent: one serial solve per chunk.
        let solve = Shared(|j0: usize| {
            lower_solve(false, true, unit, m, CHUNK.min(n - j0), a, b.offset(j0 as isize * csb), rsb, csb);
        });
        (0..n.div_ceil(CHUNK)).into_par_iter().for_each(|c| solve.get()(c * CHUNK));
    } else {
        lower_solve(threads, true, unit, m, n, a, b, rsb, csb);
    }
}

/// `L·X = B` in place, L lower `m × m`, recursively.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn lower_solve<T: Element>(
    threads: bool,
    invert: bool,
    unit: bool,
    m: usize,
    n: usize,
    a: View<T>,
    b: *mut T,
    rsb: isize,
    csb: isize,
) {
    if m <= LEAF {
        return leaf(invert, unit, m, n, a, b, rsb, csb);
    }
    let m1 = m / 2;
    lower_solve(threads, invert, unit, m1, n, a, b, rsb, csb);
    // B2 -= L21·X1
    let l21 = View { p: a.at(m1, 0), rs: a.rs, cs: a.cs };
    let x1 = View { p: b as *const T, rs: rsb, cs: csb };
    let b2 = b.offset(m1 as isize * rsb);
    gemm(threads, m - m1, n, m1, -T::one(), l21, x1, T::one(), b2, rsb, csb);
    lower_solve(threads, invert, unit, m - m1, n, View { p: a.at(m1, m1), rs: a.rs, cs: a.cs }, b2, rsb, csb);
}

/// `L·X = B` directly for `m ≤ LEAF`: forward substitution on a contiguous
/// copy of up to [`WIDTH`] columns at a time, row operations along the copy.
#[allow(clippy::too_many_arguments)]
unsafe fn leaf<T: Element>(invert: bool, unit: bool, m: usize, n: usize, a: View<T>, b: *mut T, rsb: isize, csb: isize) {
    if invert && unit && n >= INVERT_MIN && m > 1 {
        return leaf_inverse(unit, m, n, a, b, rsb, csb);
    }
    if T::COMPLEX {
        return leaf_split(unit, m, n, a, b, rsb, csb);
    }
    let mut inv = [T::zero(); LEAF];
    if !unit {
        for (i, v) in inv.iter_mut().enumerate().take(m) {
            *v = T::one() / *a.at(i, i);
        }
    }
    let mut work = vec![T::zero(); m * WIDTH.min(n)];
    for j0 in (0..n).step_by(WIDTH) {
        let w = WIDTH.min(n - j0);
        for i in 0..m {
            for j in 0..w {
                work[i * w + j] = *b.offset(i as isize * rsb + (j0 + j) as isize * csb);
            }
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..m {
            let (done, rest) = work.split_at_mut(i * w);
            let row = &mut rest[..w];
            for p in 0..i {
                let l = *a.at(i, p);
                if l != T::zero() {
                    let src = &done[p * w..p * w + w];
                    for (x, &y) in row.iter_mut().zip(src) {
                        *x = *x - l * y;
                    }
                }
            }
            if !unit {
                for x in row.iter_mut() {
                    *x = *x * inv[i];
                }
            }
        }
        for i in 0..m {
            for j in 0..w {
                *b.offset(i as isize * rsb + (j0 + j) as isize * csb) = work[i * w + j];
            }
        }
    }
}

/// [`leaf`] for complex values, on a copy split into a real plane and an
/// imaginary plane. Each row operation is then four real multiply-adds per
/// value on contiguous reals, which vectorize fully; on interleaved values the
/// complex product needs shuffles and the loops barely vectorize. On em_m4
/// (M4 Pro, complex128) the off-diagonal phase of the small fronts fell from
/// 1395 to TBD cpu ms.
unsafe fn leaf_split<T: Element>(unit: bool, m: usize, n: usize, a: View<T>, b: *mut T, rsb: isize, csb: isize) {
    let zero = T::Real::ZERO;
    let mut inv = [(zero, zero); LEAF];
    if !unit {
        for (i, v) in inv.iter_mut().enumerate().take(m) {
            let d = T::one() / *a.at(i, i);
            *v = (d.re(), d.im());
        }
    }
    let width = WIDTH.min(n);
    let mut re = vec![zero; m * width];
    let mut im = vec![zero; m * width];
    for j0 in (0..n).step_by(WIDTH) {
        let w = WIDTH.min(n - j0);
        for i in 0..m {
            for j in 0..w {
                let v = *b.offset(i as isize * rsb + (j0 + j) as isize * csb);
                re[i * w + j] = v.re();
                im[i * w + j] = v.im();
            }
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..m {
            let (re_done, re_rest) = re.split_at_mut(i * w);
            let (im_done, im_rest) = im.split_at_mut(i * w);
            let (xr, xi) = (&mut re_rest[..w], &mut im_rest[..w]);
            for p in 0..i {
                let l = *a.at(i, p);
                if l == T::zero() {
                    continue;
                }
                let (lr, li) = (l.re(), l.im());
                let (yr, yi) = (&re_done[p * w..p * w + w], &im_done[p * w..p * w + w]);
                for j in 0..w {
                    xr[j] = xr[j] - lr * yr[j] + li * yi[j];
                    xi[j] = xi[j] - lr * yi[j] - li * yr[j];
                }
            }
            if !unit {
                let (dr, di) = inv[i];
                for j in 0..w {
                    let (r, s) = (xr[j], xi[j]);
                    xr[j] = r * dr - s * di;
                    xi[j] = r * di + s * dr;
                }
            }
        }
        for i in 0..m {
            for j in 0..w {
                *b.offset(i as isize * rsb + (j0 + j) as isize * csb) = T::new(re[i * w + j], im[i * w + j]);
            }
        }
    }
}

/// [`leaf`] by explicit inverse: `L⁻¹` (at most `LEAF × LEAF`, built by
/// substitution on the identity) applied to a copy of B with one GEMM, which
/// runs on the packed kernels instead of scalar row operations.
///
/// Only unit triangles take this route. Those of a pivoted factorization
/// have bounded multipliers, and inverting them costs nothing measurable;
/// a non-unit triangle (LU's U) can be poorly conditioned, and its inverse
/// did cost accuracy: the em_m3 LU residual rose from 1.8e-13 to 5.9e-13.
/// On the small fronts' off-diagonal solves (unit, from LDLᵀ) `ztrsm` went
/// from 0.4 to 0.8-1.1 times Accelerate's speed (M4 Pro, 300 to 800 rows,
/// 16 to 100 columns).
#[allow(clippy::needless_range_loop)]
unsafe fn leaf_inverse<T: Element>(unit: bool, m: usize, n: usize, a: View<T>, b: *mut T, rsb: isize, csb: isize) {
    // inv, column-major m × m: column j solves L·x = e_j, forward.
    let mut inv = vec![T::zero(); m * m];
    for j in 0..m {
        let x = &mut inv[j * m..j * m + m];
        x[j] = T::one();
        for p in j..m {
            if !unit {
                x[p] = x[p] / *a.at(p, p);
            }
            let xp = x[p];
            if xp != T::zero() {
                for i in p + 1..m {
                    x[i] = x[i] - *a.at(i, p) * xp;
                }
            }
        }
    }
    // B's copy, column-major m × n, then B = inv · copy.
    let mut copy = vec![T::zero(); m * n];
    for j in 0..n {
        for i in 0..m {
            copy[i + j * m] = *b.offset(i as isize * rsb + j as isize * csb);
        }
    }
    let inv_v = View { p: inv.as_ptr(), rs: 1, cs: m as isize };
    let copy_v = View { p: copy.as_ptr(), rs: 1, cs: m as isize };
    gemm(false, m, n, m, T::one(), inv_v, copy_v, T::zero(), b, rsb, csb);
}
