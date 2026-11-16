//! The routines in the CBLAS and LAPACK (f2c) signatures, for callers written
//! against a vendor BLAS. Function pointers only: nothing is exported under
//! the standard names, so SpartaBLAS never shadows another BLAS in the same
//! process.
//!
//! Supported: `?gemm` and `?trsm` without conjugation (`CblasConjTrans` is
//! refused), row- or column-major; `?getrf`; `?sytrf` for `UPLO = 'L'`.
use crate::{gemm_strided, getrf, sytrf, trsm_strided, Complex, Diag, Element, Side, Uplo};
use crate::Real;
use std::ffi::c_void;

pub const ROW_MAJOR: i32 = 101;
pub const COL_MAJOR: i32 = 102;
pub const NO_TRANS: i32 = 111;
pub const TRANS: i32 = 112;
pub const UPPER: i32 = 121;
pub const LOWER: i32 = 122;
pub const NON_UNIT: i32 = 131;
pub const UNIT: i32 = 132;
pub const LEFT: i32 = 141;
pub const RIGHT: i32 = 142;

/// (row stride, column stride) of `op(X)` for a matrix stored with leading
/// dimension `ld` in `order`.
fn strides(order: i32, trans: i32, ld: i32) -> (isize, isize) {
    assert!(trans == NO_TRANS || trans == TRANS, "SpartaBLAS: unsupported transpose code {trans}");
    let ld = ld as isize;
    let plain = if order == ROW_MAJOR { (ld, 1) } else { (1, ld) };
    if trans == NO_TRANS {
        plain
    } else {
        (plain.1, plain.0)
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn gemm<T: Element>(
    order: i32,
    ta: i32,
    tb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: T,
    a: *const c_void,
    lda: i32,
    b: *const c_void,
    ldb: i32,
    beta: T,
    c: *mut c_void,
    ldc: i32,
) {
    let (rsa, csa) = strides(order, ta, lda);
    let (rsb, csb) = strides(order, tb, ldb);
    let (rsc, csc) = strides(order, NO_TRANS, ldc);
    gemm_strided(
        m as usize,
        n as usize,
        k as usize,
        alpha,
        a.cast(),
        rsa,
        csa,
        b.cast(),
        rsb,
        csb,
        beta,
        c.cast(),
        rsc,
        csc,
    );
}

#[allow(clippy::too_many_arguments)]
unsafe fn trsm<T: Element>(
    order: i32,
    side: i32,
    uplo: i32,
    trans: i32,
    diag: i32,
    m: i32,
    n: i32,
    alpha: T,
    a: *const c_void,
    lda: i32,
    b: *mut c_void,
    ldb: i32,
) {
    let (rsa, csa) = strides(order, NO_TRANS, lda);
    let (rsb, csb) = strides(order, NO_TRANS, ldb);
    assert!(trans == NO_TRANS || trans == TRANS, "SpartaBLAS: unsupported transpose code {trans}");
    trsm_strided(
        if side == LEFT { Side::Left } else { Side::Right },
        if uplo == UPPER { Uplo::Upper } else { Uplo::Lower },
        trans == TRANS,
        if diag == UNIT { Diag::Unit } else { Diag::NonUnit },
        m as usize,
        n as usize,
        alpha,
        a.cast(),
        rsa,
        csa,
        b.cast(),
        rsb,
        csb,
    );
}

macro_rules! complex_abi {
    ($gemm:ident, $trsm:ident, $getrf:ident, $t:ty) => {
        /// CBLAS `?gemm`; `alpha` and `beta` by pointer.
        ///
        /// # Safety
        /// As the CBLAS routine.
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $gemm(
            order: i32,
            ta: i32,
            tb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const c_void,
            a: *const c_void,
            lda: i32,
            b: *const c_void,
            ldb: i32,
            beta: *const c_void,
            c: *mut c_void,
            ldc: i32,
        ) {
            let (alpha, beta) = (*alpha.cast::<Complex<$t>>(), *beta.cast::<Complex<$t>>());
            gemm(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        }
        /// CBLAS `?trsm`; `alpha` by pointer.
        ///
        /// # Safety
        /// As the CBLAS routine.
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $trsm(
            order: i32,
            side: i32,
            uplo: i32,
            trans: i32,
            diag: i32,
            m: i32,
            n: i32,
            alpha: *const c_void,
            a: *const c_void,
            lda: i32,
            b: *mut c_void,
            ldb: i32,
        ) {
            let alpha = *alpha.cast::<Complex<$t>>();
            trsm(order, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
        }
        getrf_abi!($getrf, Complex<$t>);
    };
}

macro_rules! real_abi {
    ($gemm:ident, $trsm:ident, $getrf:ident, $t:ty) => {
        /// CBLAS `?gemm`.
        ///
        /// # Safety
        /// As the CBLAS routine.
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $gemm(
            order: i32,
            ta: i32,
            tb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: $t,
            a: *const c_void,
            lda: i32,
            b: *const c_void,
            ldb: i32,
            beta: $t,
            c: *mut c_void,
            ldc: i32,
        ) {
            gemm(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        }
        /// CBLAS `?trsm`.
        ///
        /// # Safety
        /// As the CBLAS routine.
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $trsm(
            order: i32,
            side: i32,
            uplo: i32,
            trans: i32,
            diag: i32,
            m: i32,
            n: i32,
            alpha: $t,
            a: *const c_void,
            lda: i32,
            b: *mut c_void,
            ldb: i32,
        ) {
            trsm(order, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
        }
        getrf_abi!($getrf, $t);
    };
}

macro_rules! sytrf_abi {
    ($sytrf:ident, $t:ty) => {
        /// LAPACK `?sytrf`, f2c calling convention, lower triangle only. A
        /// workspace query (`lwork = -1`) answers `n`; the workspace given is
        /// not used, the routine allocates its own.
        ///
        /// # Safety
        /// As the LAPACK routine.
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $sytrf(
            uplo: *const u8,
            n: *const i32,
            a: *mut c_void,
            lda: *const i32,
            ipiv: *mut i32,
            work: *mut c_void,
            lwork: *const i32,
            info: *mut i32,
        ) {
            let n = *n as usize;
            if *lwork == -1 {
                *work.cast::<$t>() = <$t as Element>::new(<$t as Element>::Real::from_f64(n.max(1) as f64), <$t as Element>::Real::ZERO);
                *info = 0;
                return;
            }
            assert!(*uplo == b'L' || *uplo == b'l', "SpartaBLAS: ?sytrf supports the lower triangle only");
            let ipiv = std::slice::from_raw_parts_mut(ipiv, n);
            *info = sytrf::<$t>(n, a.cast(), *lda as usize, ipiv);
        }
    };
}

macro_rules! getrf_abi {
    ($getrf:ident, $t:ty) => {
        /// LAPACK `?getrf`, f2c calling convention (every argument by pointer).
        ///
        /// # Safety
        /// As the LAPACK routine.
        pub unsafe extern "C" fn $getrf(m: *const i32, n: *const i32, a: *mut c_void, lda: *const i32, ipiv: *mut i32, info: *mut i32) {
            let (m, n) = (*m as usize, *n as usize);
            let ipiv = std::slice::from_raw_parts_mut(ipiv, m.min(n));
            *info = getrf::<$t>(m, n, a.cast(), *lda as usize, ipiv);
        }
    };
}

complex_abi!(zgemm, ztrsm, zgetrf, f64);
sytrf_abi!(zsytrf, Complex<f64>);
sytrf_abi!(csytrf, Complex<f32>);
sytrf_abi!(dsytrf, f64);
sytrf_abi!(ssytrf, f32);
complex_abi!(cgemm, ctrsm, cgetrf, f32);
real_abi!(dgemm, dtrsm, dgetrf, f64);
real_abi!(sgemm, strsm, sgetrf, f32);

/// `BLASSetThreading`-style control: 0 for the machine, 1 for the caller.
/// Returns 0.
pub extern "C" fn set_threading(threading: u32) -> i32 {
    crate::threading::set(if threading == 1 { crate::Threading::Caller } else { crate::Threading::Machine });
    0
}
