//! BLAS and LAPACK entry points for the dense frontal algebra.
//!
//! Nothing here is specific to one vendor. Fronts are row-major, so the BLAS
//! wrappers pass `CblasRowMajor` and let CBLAS do the index arithmetic; LAPACK
//! is column-major only, so [`sytrf`] takes an explicitly packed column-major
//! block. Both interfaces are the standard ones that Accelerate, OpenBLAS and
//! MKL all export, and `crate::blas` decides which library the calls go to:
//! Accelerate on Apple, a library linked through `SPARTA_BLAS_LIB`, or one
//! loaded at run time.
//!
//! With no library, [`available`] reports false and every caller keeps to the
//! portable Rust loops in `panel.rs` and `update.rs`.
use super::C;
use crate::blas::{self, Library, PerThread};
use crate::scalar::{Scalar, ScalarKind};
use std::ffi::c_void;

/// Which of the four copies of the backend this is; every BLAS and LAPACK
/// call below picks its `s`, `d`, `c` or `z` routine from it.
const KIND: ScalarKind = <C as Scalar>::KIND;

/// Reports whether the CPU dense kernels have a BLAS and LAPACK to call.
pub(super) fn available() -> bool {
    blas::library().is_some()
}

/// The library, for the routines below, which only run once [`available`]
/// said there is one.
fn library() -> &'static Library {
    blas::library().expect("a dense routine ran without a BLAS; callers check `available` first")
}

/// How many threads BLAS may use for the calls this thread makes next.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Threading {
    /// The library decides, and will spread one large product over the machine.
    Machine,
    /// One thread, for a front that is itself one of many running at once.
    Caller,
}

/// Choose the BLAS threading model for this thread, reporting whether a
/// per-thread control exists at all.
///
/// SPARTA wants Rayon to own every level of parallelism, which needs a control
/// that is *per thread*; a global one would have workers fighting over it.
/// Two libraries offer one: Accelerate's `BLASSetThreading` (macOS 15) and
/// MKL's `MKL_Set_Num_Threads_Local`.
///
/// OpenBLAS deliberately is not: `openblas_set_num_threads` is process-wide,
/// so calling it from a Rayon worker would change the setting under every
/// other worker. It is set between levels instead, by [`set_global_threading`].
/// When neither control exists -- macOS 13 and 14, or a BLAS that is neither --
/// `level::schedule` keeps wide fronts off the spread path, which costs
/// parallelism but never correctness.
pub(super) fn set_threading(threading: Threading) -> bool {
    match blas::library().and_then(|library| library.per_thread) {
        // SAFETY: each arm calls the symbol it resolved, with one of the two
        // values that library documents.
        Some(PerThread::Accelerate(set)) => unsafe {
            set(match threading {
                Threading::Machine => 0,
                Threading::Caller => 1,
            }) == 0
        },
        // SAFETY: as above. MKL returns the previous count, not a status, so
        // the call itself is the success.
        Some(PerThread::Mkl(set)) => unsafe {
            set(match threading {
                Threading::Machine => 0,
                Threading::Caller => 1,
            });
            true
        },
        None => false,
    }
}

/// Whether the library has a thread control of either kind, the per-thread
/// one or OpenBLAS's process-wide one. A query: unlike calling either setter,
/// it changes no thread's setting.
pub(super) fn controllable() -> bool {
    blas::library().is_some_and(|library| library.per_thread.is_some() || library.global.is_some())
}

/// Choose the BLAS threading model for the whole process, for a library whose
/// only control is global, reporting whether it has one.
///
/// This is OpenBLAS's `openblas_set_num_threads`. `Machine` is the size of the
/// Rayon pool it is called on, so `RAYON_NUM_THREADS` or `Config::threads`
/// bounds BLAS as well, and an
/// `OPENBLAS_NUM_THREADS` set for some other library in the process (EMerge
/// sets it to 1 for numpy) does not leave the root front on one thread;
/// `Caller` is one thread. Because it changes the setting under every thread
/// at once, it may only be called while no other thread is inside BLAS:
/// `Solver::factor_all` calls it between levels, after
/// the previous level has joined, and the solve calls it before and after its
/// sweeps. A library that has a per-thread control ([`set_threading`]) is left
/// alone, and this reports false for it.
///
/// - `threading`: `Machine` for one front that should own every core, `Caller`
///   for a level of fronts that Rayon spreads across the cores itself.
pub(super) fn set_global_threading(threading: Threading) -> bool {
    match blas::library().and_then(|library| library.global) {
        // SAFETY: the symbol resolved in `crate::blas`, called with a positive count.
        Some(set) => unsafe {
            set(match threading {
                // The pool this runs on, read per call: a solver with its
                // own pool (`Config::threads`) bounds BLAS to that pool.
                Threading::Machine => rayon::current_num_threads().max(1) as i32,
                Threading::Caller => 1,
            });
            true
        },
        None => false,
    }
}

/// CBLAS enumerators, which are ABI constants rather than a header import.
mod ffi {
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;
    pub const UPPER: i32 = 121;
    pub const LOWER: i32 = 122;
    pub const NON_UNIT: i32 = 131;
    pub const UNIT: i32 = 132;
    pub const LEFT: i32 = 141;
    pub const RIGHT: i32 = 142;
}

/// `c = alpha * a * op(b) + beta * c`, row-major, through this copy's GEMM.
///
/// # Safety
/// The operands must be valid for the given shapes and strides, and `c` must
/// not overlap `a` or `b`.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm(
    trans_b: i32,
    (m, n, k): (usize, usize, usize),
    alpha: C,
    a: *const C,
    lda: usize,
    b: *const C,
    ldb: usize,
    beta: C,
    c: *mut C,
    ldc: usize,
) {
    let (m, n, k, lda, ldb, ldc) = (m as i32, n as i32, k as i32, lda as i32, ldb as i32, ldc as i32);
    let (a, b, c) = (a.cast::<c_void>(), b.cast::<c_void>(), c.cast::<c_void>());
    let (order, no) = (ffi::ROW_MAJOR, ffi::NO_TRANS);
    let (pa, pb) = ((&alpha as *const C).cast(), (&beta as *const C).cast());
    match KIND {
        ScalarKind::Complex64 => (library().zgemm)(order, no, trans_b, m, n, k, pa, a, lda, b, ldb, pb, c, ldc),
        // The legacy symbol on purpose. The `$NEWLAPACK` `cgemm` loses the
        // imaginary part when it accumulates into C: on C -= A B^T at
        // 600x600x1200, with imaginary parts a hundredth of the real ones, its
        // imaginary error is 4.5e-6 relative against the legacy 1.3e-7, the
        // real part unchanged; it tripled the residual of the 887k EMerge
        // system factored in complex64 (6.2e-4 to 1.9e-3). `zgemm`, `sgemm`
        // and `dgemm` show no such loss.
        ScalarKind::Complex32 => (library().cgemm)(order, no, trans_b, m, n, k, pa, a, lda, b, ldb, pb, c, ldc),
        ScalarKind::F64 => {
            let (alpha, beta) = (alpha.re_f64(), beta.re_f64());
            (library().dgemm)(order, no, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        }
        ScalarKind::F32 => {
            let (alpha, beta) = (alpha.re_f64() as f32, beta.re_f64() as f32);
            (library().sgemm)(order, no, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        }
    }
}

/// Side of a triangular solve: the triangular factor left or right of the unknown.
#[derive(Clone, Copy)]
pub(super) enum Side {
    Left,
    Right,
}

/// Which triangle of the stored block the solve reads, and whether its diagonal is implicit.
#[derive(Clone, Copy)]
pub(super) enum Tri {
    /// Unit upper triangle: a column-major unit lower factor read row-major.
    UnitUpper,
    /// Unit lower triangle, read row-major: LU's `L11`.
    UnitLower,
    /// Upper triangle with its diagonal, read row-major: LU's `U11`.
    Upper,
}

/// The CBLAS side, triangle and diagonal enumerators for a solve.
fn solve_codes(side: Side, tri: Tri) -> (i32, i32, i32) {
    let side = match side {
        Side::Left => ffi::LEFT,
        Side::Right => ffi::RIGHT,
    };
    let (uplo, diag) = match tri {
        Tri::UnitUpper => (ffi::UPPER, ffi::UNIT),
        Tri::UnitLower => (ffi::LOWER, ffi::UNIT),
        Tri::Upper => (ffi::UPPER, ffi::NON_UNIT),
    };
    (side, uplo, diag)
}

/// This copy's `?trsm` on raw operands, with `alpha = 1`.
///
/// # Safety
/// `t` must hold the triangle and `b` the `m` by `n` unknown, row-major with
/// strides `ldt` and `ldb`, and the two must not overlap.
#[allow(clippy::too_many_arguments)]
unsafe fn trsm_raw(side: Side, tri: Tri, m: usize, n: usize, t: *const C, ldt: usize, b: *mut C, ldb: usize) {
    let (side, uplo, diag) = solve_codes(side, tri);
    let (order, no, m, n, ldt, ldb) = (ffi::ROW_MAJOR, ffi::NO_TRANS, m as i32, n as i32, ldt as i32, ldb as i32);
    let one = C::from_re(1.0);
    let (t, b) = (t.cast::<c_void>(), b.cast::<c_void>());
    let pone = (&one as *const C).cast();
    match KIND {
        ScalarKind::Complex64 => (library().ztrsm)(order, side, uplo, no, diag, m, n, pone, t, ldt, b, ldb),
        ScalarKind::Complex32 => (library().ctrsm)(order, side, uplo, no, diag, m, n, pone, t, ldt, b, ldb),
        ScalarKind::F64 => (library().dtrsm)(order, side, uplo, no, diag, m, n, 1.0, t, ldt, b, ldb),
        ScalarKind::F32 => (library().strsm)(order, side, uplo, no, diag, m, n, 1.0, t, ldt, b, ldb),
    }
}

/// `c[m][n] -= a[m][k] * b[n][k]^T`, all row-major with the given row strides.
///
/// `lda`, `ldb` and `ldc` are element strides between rows and must be at
/// least `k`, `k` and `n`. The three operands must be distinct.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_nt(
    m: usize,
    n: usize,
    k: usize,
    a: &[C],
    lda: usize,
    b: &[C],
    ldb: usize,
    c: &mut [C],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!(lda >= k && ldb >= k && ldc >= n, "row strides too short");
    assert!(a.len() >= (m - 1) * lda + k, "left operand out of bounds");
    assert!(b.len() >= (n - 1) * ldb + k, "right operand out of bounds");
    assert!(c.len() >= (m - 1) * ldc + n, "target out of bounds");
    let (minus, one) = (C::from_re(-1.0), C::from_re(1.0));
    // SAFETY: the dimensions are checked above and fit i32 because Analysis
    // bounds every front to u32 elements. The operands are distinct slices and
    // the library completes before returning.
    unsafe {
        gemm(ffi::TRANS, (m, n, k), minus, a.as_ptr(), lda, b.as_ptr(), ldb, one, c.as_mut_ptr(), ldc);
    }
}

/// `c[m][n] -= a[m][k] * b[k][n]`, all row-major with the given row strides.
///
/// `lda`, `ldb` and `ldc` are element strides between rows and must be at
/// least `k`, `n` and `n`. The three operands must be distinct.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_nn(
    m: usize,
    n: usize,
    k: usize,
    a: &[C],
    lda: usize,
    b: &[C],
    ldb: usize,
    c: &mut [C],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!(lda >= k && ldb >= n && ldc >= n, "row strides too short");
    assert!(a.len() >= (m - 1) * lda + k, "left operand out of bounds");
    assert!(b.len() >= (k - 1) * ldb + n, "right operand out of bounds");
    assert!(c.len() >= (m - 1) * ldc + n, "target out of bounds");
    let (minus, one) = (C::from_re(-1.0), C::from_re(1.0));
    // SAFETY: as for `gemm_nt`.
    unsafe {
        gemm(ffi::NO_TRANS, (m, n, k), minus, a.as_ptr(), lda, b.as_ptr(), ldb, one, c.as_mut_ptr(), ldc);
    }
}

/// `c[m][n] -= a[m][k] * b[n][k]^T` with the left operand and the target inside
/// one row-major buffer: `a_at` and `c_at` are element offsets with row stride
/// `ld`. The left operand spans columns `0..k` of its rows and the target
/// columns `c_at - a_at ..`, so `a_at + k <= c_at` keeps them disjoint.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_nt_front(
    m: usize,
    n: usize,
    k: usize,
    front: &mut [C],
    a_at: usize,
    c_at: usize,
    ld: usize,
    b: &[C],
    ldb: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!(a_at + k <= c_at, "gemm operands overlap");
    assert!(c_at + (m - 1) * ld + n <= front.len(), "target out of bounds");
    assert!(b.len() >= (n - 1) * ldb + k, "right operand out of bounds");
    let (minus, one) = (C::from_re(-1.0), C::from_re(1.0));
    let base = front.as_mut_ptr();
    // SAFETY: both rectangles lie in `front`, asserted above, and are disjoint
    // because each row's left-operand columns end where the target's begin.
    // Accelerate reads one and writes the other, and completes before it
    // returns, so the aliasing `&mut front` would imply never materialises.
    unsafe {
        gemm(ffi::TRANS, (m, n, k), minus, base.add(a_at), ld, b.as_ptr(), ldb, one, base.add(c_at), ld);
    }
}

/// Solve a unit or explicit triangular system in place: `b <- b * t^-1` for
/// [`Side::Right`], `b <- t^-1 * b` for [`Side::Left`].
///
/// `b` is row-major `m` by `n` with row stride `ldb`. `t` is read row-major
/// with row stride `ldt` and order `n` (right) or `m` (left).
#[allow(clippy::too_many_arguments)]
pub(super) fn trsm(
    side: Side,
    tri: Tri,
    m: usize,
    n: usize,
    t: &[C],
    ldt: usize,
    b: &mut [C],
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    // SAFETY: dimensions fit i32 as in `gemm_nt`; `t` and `b` are distinct
    // allocations and the triangular order matches the requested side.
    unsafe { trsm_raw(side, tri, m, n, t.as_ptr(), ldt, b.as_mut_ptr(), ldb) }
}

/// [`trsm`] with the triangle and the unknown inside one row-major buffer of
/// row stride `ld`, at element offsets `t_at` and `b_at`. The triangle has
/// order `m` (left) or `n` (right).
#[allow(clippy::too_many_arguments)]
pub(super) fn trsm_front(side: Side, tri: Tri, m: usize, n: usize, front: &mut [C], t_at: usize, b_at: usize, ld: usize) {
    if m == 0 || n == 0 {
        return;
    }
    let order = match side {
        Side::Left => m,
        Side::Right => n,
    };
    assert!(t_at + (order - 1) * ld + order <= front.len(), "triangle out of bounds");
    assert!(b_at + (m - 1) * ld + n <= front.len(), "unknown out of bounds");
    let base = front.as_mut_ptr();
    // SAFETY: both rectangles lie in `front`, asserted above. The callers pass
    // the fully summed block as the triangle and a block of other rows or
    // other columns as the unknown, which never overlap; the library reads one,
    // writes the other, and completes before it returns.
    unsafe { trsm_raw(side, tri, m, n, base.add(t_at), ld, base.add(b_at), ld) }
}

/// [`trsm_front`] spread over the Rayon pool: the unknown is cut into blocks
/// of rows (right side) or columns (left side), which a triangular solve
/// treats independently, and each block is one single-threaded solve.
///
/// Accelerate's legacy `?trsm` does not thread at all, and its `$NEWLAPACK`
/// one threads less well than this: on a 2680 by 996 complex64 unknown, the
/// legacy `ctrsm` takes 59 ms threaded and 62 ms not, the new one 6.4 ms, and
/// ten row blocks of the new one on ten threads 5.3 ms (M4 Pro). On the 86k
/// EMerge system factored as LU, spreading takes the factorization from 1.47
/// to 1.15 s in complex128 and from 0.59 to 0.50 s in complex64. The calling
/// thread must have [`Threading::Caller`] set, as every pool thread does.
#[allow(clippy::too_many_arguments)]
pub(super) fn trsm_front_spread(side: Side, tri: Tri, m: usize, n: usize, front: &mut [C], t_at: usize, b_at: usize, ld: usize) {
    use rayon::prelude::*;
    if m == 0 || n == 0 {
        return;
    }
    let order = match side {
        Side::Left => m,
        Side::Right => n,
    };
    assert!(t_at + (order - 1) * ld + order <= front.len(), "triangle out of bounds");
    assert!(b_at + (m - 1) * ld + n <= front.len(), "unknown out of bounds");
    // Two blocks per thread, so stealing can even out a slow one.
    let width = match side {
        Side::Left => n,
        Side::Right => m,
    };
    let blocks = 2 * rayon::current_num_threads();
    let step = width.div_ceil(blocks).next_multiple_of(16).max(64);
    /// The front as a pointer the tasks can share; see the SAFETY note below.
    #[derive(Clone, Copy)]
    struct Base(*mut C);
    // SAFETY: carries no rights of its own; the tasks write disjoint blocks.
    unsafe impl Send for Base {}
    unsafe impl Sync for Base {}
    impl Base {
        /// Element `at` of the front. A method, so the tasks capture the
        /// whole `Send` wrapper rather than the bare pointer inside it.
        fn at(self, at: usize) -> *mut C {
            // SAFETY: every caller's offset lies inside `front`, asserted above.
            unsafe { self.0.add(at) }
        }
    }
    let base = Base(front.as_mut_ptr());
    (0..width.div_ceil(step)).into_par_iter().for_each(|block| {
        let (first, count) = (block * step, step.min(width - block * step));
        // SAFETY: inside `front` as asserted above. The blocks partition the
        // unknown, so no two tasks write the same element, and the triangle
        // they all read is disjoint from the unknown, as for `trsm_front`.
        unsafe {
            let t = base.at(t_at);
            match side {
                Side::Left => trsm_raw(side, tri, m, count, t, ld, base.at(b_at + first), ld),
                Side::Right => trsm_raw(side, tri, count, n, t, ld, base.at(b_at + first * ld), ld),
            }
        }
    });
}

/// `c[m][n] -= a[m][k] * b[k][n]` with all three operands inside one row-major
/// buffer of row stride `ld`, at element offsets `a_at`, `b_at` and `c_at`.
/// The caller guarantees the target overlaps neither operand.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_nn_front(m: usize, n: usize, k: usize, front: &mut [C], a_at: usize, b_at: usize, c_at: usize, ld: usize) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!(a_at + (m - 1) * ld + k <= front.len(), "left operand out of bounds");
    assert!(b_at + (k - 1) * ld + n <= front.len(), "right operand out of bounds");
    assert!(c_at + (m - 1) * ld + n <= front.len(), "target out of bounds");
    let (minus, one) = (C::from_re(-1.0), C::from_re(1.0));
    let base = front.as_mut_ptr();
    // SAFETY: the three rectangles lie in `front`, asserted above; the target
    // is disjoint from both operands by the caller's contract, and the library
    // completes before it returns.
    unsafe {
        gemm(ffi::NO_TRANS, (m, n, k), minus, base.add(a_at), ld, base.add(b_at), ld, one, base.add(c_at), ld);
    }
}

/// [`gemm_nn_front`] spread over the Rayon pool in blocks of target rows,
/// each one single-threaded product. The calling thread must have
/// [`Threading::Caller`] set, as every pool thread does.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_nn_front_spread(m: usize, n: usize, k: usize, front: &mut [C], a_at: usize, b_at: usize, c_at: usize, ld: usize) {
    use rayon::prelude::*;
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!(a_at + (m - 1) * ld + k <= front.len(), "left operand out of bounds");
    assert!(b_at + (k - 1) * ld + n <= front.len(), "right operand out of bounds");
    assert!(c_at + (m - 1) * ld + n <= front.len(), "target out of bounds");
    let step = m.div_ceil(2 * rayon::current_num_threads()).max(64);
    /// The front as a pointer the tasks can share; see the SAFETY note below.
    #[derive(Clone, Copy)]
    struct Base(*mut C);
    // SAFETY: carries no rights of its own; the tasks write disjoint rows.
    unsafe impl Send for Base {}
    unsafe impl Sync for Base {}
    impl Base {
        /// Element `at` of the front, by method so tasks capture the wrapper.
        fn at(self, at: usize) -> *mut C {
            // SAFETY: every caller's offset lies inside `front`, asserted above.
            unsafe { self.0.add(at) }
        }
    }
    let base = Base(front.as_mut_ptr());
    let (minus, one) = (C::from_re(-1.0), C::from_re(1.0));
    (0..m.div_ceil(step)).into_par_iter().for_each(|block| {
        let (first, rows) = (block * step, step.min(m - block * step));
        // SAFETY: inside `front` as asserted above. The blocks partition the
        // target rows, and the target overlaps neither operand, as for
        // `gemm_nn_front`.
        unsafe {
            gemm(ffi::NO_TRANS, (rows, n, k), minus, base.at(a_at + first * ld), ld, base.at(b_at), ld, one, base.at(c_at + first * ld), ld);
        }
    });
}

/// LU with partial pivoting of the column-major `n` by `n` block `a`, leading
/// dimension `lda`. Returns LAPACK's `info`: zero on success, `k > 0` when
/// `U(k,k)` is exactly zero. `ipiv` receives `n` one-based interchanges.
pub(super) fn getrf(n: usize, a: &mut [C], lda: usize, ipiv: &mut [i32]) -> i32 {
    debug_assert!(n == 0 || (a.len() >= (n - 1) * lda + n && ipiv.len() >= n));
    let (order, stride, mut info) = (n as i32, lda as i32, 0);
    let getrf = library().getrf(KIND);
    // SAFETY: an `n` by `n` column-major block with stride `lda` and room for
    // `n` interchanges, both sized above.
    unsafe { getrf(&order, &order, a.as_mut_ptr().cast(), &stride, ipiv.as_mut_ptr(), &mut info) };
    info
}

/// Bunch-Kaufman `LDL^T` of the lower triangle of column-major `a`, order `n`.
///
/// Returns LAPACK's `info`: zero on success, `k > 0` when `D(k,k)` is exactly
/// zero. `ipiv` holds `n` interchanges and `work` is resized as LAPACK asks.
pub(super) fn sytrf(n: usize, a: &mut [C], lda: usize, ipiv: &mut [i32], work: &mut [C]) -> i32 {
    debug_assert!(a.len() >= (n - 1) * lda + n && ipiv.len() >= n && !work.is_empty());
    let (order, stride, lwork) = (n as i32, lda as i32, work.len() as i32);
    let mut info = 0;
    // SAFETY: LAPACK reads the lower triangle of an `n` by `n` column-major
    // block with stride `lda` and writes `n` interchanges, both sized above.
    // The legacy f2c entry point takes `uplo` as a pointer to one character;
    // it compares through `lsame_`, which never reads the hidden length.
    unsafe {
        library().sytrf(KIND)(
            b"L".as_ptr(),
            &order,
            a.as_mut_ptr().cast(),
            &stride,
            ipiv.as_mut_ptr(),
            work.as_mut_ptr().cast(),
            &lwork,
            &mut info,
        );
    }
    info
}

/// Ask LAPACK for `zsytrf`'s preferred workspace length at order `n`.
pub(super) fn sytrf_workspace(n: usize) -> usize {
    let (order, stride, query) = (n as i32, n.max(1) as i32, -1i32);
    let (mut best, mut info) = (C::from_re(0.0), 0);
    // SAFETY: a workspace query touches neither `a` nor `ipiv`, which LAPACK
    // declares nullable, and writes one complex length into `best`.
    unsafe {
        library().sytrf(KIND)(
            b"L".as_ptr(),
            &order,
            std::ptr::null_mut(),
            &stride,
            std::ptr::null_mut(),
            (&mut best as *mut C).cast(),
            &query,
            &mut info,
        );
    }
    (best.re() as usize).max(n)
}
