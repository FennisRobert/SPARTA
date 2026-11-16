//! LU with partial pivoting, recursively (Toledo; LAPACK's `?getrf2`).
//!
//! Split the columns in half; factor the left half; apply its interchanges to
//! the right half; solve for the top of the right half with the unit lower
//! triangle; update the bottom with one GEMM; factor it; apply its
//! interchanges back to the left half. Like the recursive `trsm`, nearly all
//! the work ends up in GEMMs of half the remaining width.
//!
//! The results match LAPACK's conventions exactly: `ipiv` is one-based, the
//! pivot is the first entry of largest `|re| + |im|` in its column, and the
//! returned `info` is zero, or the one-based index of the first exactly zero
//! pivot (the factorization still completes, as LAPACK's does).
use crate::element::{Element, Real};
use crate::gemm::gemm;
use crate::pack::View;
use crate::threading;
use crate::trsm::lower_solve;

/// LU of the `m × n` column-major matrix `a` with leading dimension `lda`,
/// threaded per [`threading::set`]. Returns LAPACK's `info`.
///
/// - `ipiv`: receives `min(m, n)` one-based row interchanges.
///
/// # Safety
/// `a` must hold `lda · (n - 1) + m` values, with `lda ≥ m`.
pub unsafe fn getrf<T: Element>(m: usize, n: usize, a: *mut T, lda: usize, ipiv: &mut [i32]) -> i32 {
    assert!(ipiv.len() >= m.min(n), "getrf: ipiv holds {} of {} interchanges", ipiv.len(), m.min(n));
    recurse(threading::machine(), m, n, a, 1, lda as isize, ipiv.as_mut_ptr())
}

/// The recursion on a strided view; `ipiv` indices are relative to `a`.
unsafe fn recurse<T: Element>(threads: bool, m: usize, n: usize, a: *mut T, rs: isize, cs: isize, ipiv: *mut i32) -> i32 {
    let mn = m.min(n);
    if mn == 0 {
        return 0;
    }
    let at = |i: usize, j: usize| a.offset(i as isize * rs + j as isize * cs);
    if m == 1 {
        *ipiv = 1;
        return (*a == T::zero()) as i32;
    }
    if n == 1 {
        let (mut p, mut best) = (0, (*a).abs1());
        for i in 1..m {
            let v = (*at(i, 0)).abs1();
            if v > best {
                best = v;
                p = i;
            }
        }
        *ipiv = p as i32 + 1;
        if *at(p, 0) == T::zero() {
            return 1;
        }
        if p != 0 {
            std::ptr::swap(at(0, 0), at(p, 0));
        }
        let pivot = *a;
        let scale = pivot.re().abs() >= T::Real::MIN_NORMAL || pivot.im().abs() >= T::Real::MIN_NORMAL;
        let inv = T::one() / pivot;
        for i in 1..m {
            let x = at(i, 0);
            *x = if scale { *x * inv } else { *x / pivot };
        }
        return 0;
    }

    let n1 = mn / 2;
    let n2 = n - n1;
    let left = recurse(threads, m, n1, a, rs, cs, ipiv);
    // The left half's interchanges, on the right half.
    swap_rows(n2, at(0, n1), rs, cs, 0, n1, ipiv);
    // A12 = L11^-1 A12
    let l11 = View { p: a as *const T, rs, cs };
    // Substitution throughout, as LAPACK's: no inverted leaves.
    lower_solve(threads, false, true, n1, n2, l11, at(0, n1), rs, cs);
    // A22 -= A21 A12
    let a21 = View { p: at(n1, 0) as *const T, rs, cs };
    let a12 = View { p: at(0, n1) as *const T, rs, cs };
    gemm(threads, m - n1, n2, n1, -T::one(), a21, a12, T::one(), at(n1, n1), rs, cs);
    let right = recurse(threads, m - n1, n2, at(n1, n1), rs, cs, ipiv.add(n1));
    for i in n1..mn {
        *ipiv.add(i) += n1 as i32;
    }
    // The right half's interchanges, back on the left half.
    swap_rows(n1, a, rs, cs, n1, mn, ipiv);
    if left != 0 {
        left
    } else if right != 0 {
        right + n1 as i32
    } else {
        0
    }
}

/// Apply interchanges `ipiv[k1..k2]` (one-based, relative to `a`) to `ncols`
/// columns of `a`, in order, as LAPACK's `?laswp` does.
unsafe fn swap_rows<T: Element>(ncols: usize, a: *mut T, rs: isize, cs: isize, k1: usize, k2: usize, ipiv: *const i32) {
    for i in k1..k2 {
        let p = (*ipiv.add(i) - 1) as usize;
        if p != i {
            for j in 0..ncols {
                let col = a.offset(j as isize * cs);
                std::ptr::swap(col.offset(i as isize * rs), col.offset(p as isize * rs));
            }
        }
    }
}
