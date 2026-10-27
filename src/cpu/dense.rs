//! BLAS and LAPACK entry points for the dense frontal algebra.
//!
//! Nothing here is specific to one vendor. Fronts are row-major, so the BLAS
//! wrappers pass `CblasRowMajor` and let CBLAS do the index arithmetic; LAPACK
//! is column-major only, so [`sytrf`] takes an explicitly packed column-major
//! block. Both interfaces are the standard ones that Accelerate, OpenBLAS and
//! MKL all export, and `build.rs` decides which library the names resolve to:
//! Accelerate on Apple, otherwise whatever `SPARTA_BLAS_LIB` named.
//!
//! On Apple the names link to the legacy unsuffixed symbols, because the
//! `$NEWLAPACK` interface needs macOS 13.3 and SPARTA supports macOS 13. Each
//! routine then switches to its `$NEWLAPACK` symbol at run time where the OS
//! has it ([`newest!`] and [`zsytrf`]), which is several times faster. The
//! one routine kept on the legacy symbol is complex64 GEMM; see [`gemm`].
//!
//! With no library linked, [`available`] reports false and every caller keeps
//! to the portable Rust loops in `panel.rs` and `update.rs`.
use super::C;
#[cfg(any(sparta_accelerate, sparta_blas))]
use crate::scalar::{Scalar, ScalarKind};
#[cfg(any(sparta_accelerate, sparta_blas))]
use std::ffi::c_void;

/// Which of the four copies of the backend this is; every BLAS and LAPACK
/// call below picks its `s`, `d`, `c` or `z` routine from it.
#[cfg(any(sparta_accelerate, sparta_blas))]
const KIND: ScalarKind = <C as Scalar>::KIND;

/// Reports whether this build links a BLAS and LAPACK.
pub(super) const fn available() -> bool {
    cfg!(any(sparta_accelerate, sparta_blas))
}

/// How many threads BLAS may use for the calls this thread makes next.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Threading {
    /// The library decides, and will spread one large product over the machine.
    Machine,
    /// One thread, for a front that is itself one of many running at once.
    Caller,
}

/// A BLAS vendor whose thread control SPARTA knows how to call.
#[cfg(any(sparta_accelerate, sparta_blas))]
#[derive(Clone, Copy)]
enum Vendor {
    Accelerate,
    Mkl,
    OpenBlas,
}

/// Whether SPARTA's own BLAS calls can go to `vendor`, judged from the library
/// name `build.rs` linked.
///
/// The thread controls are looked up across every module in the process, and
/// another one may carry a different vendor's: a Python process that loaded
/// MKL for PARDISO exports `MKL_Set_Num_Threads_Local` while SPARTA calls
/// OpenBLAS. Trusting it would spread levels across Rayon while OpenBLAS still
/// threads every call. A library whose name names none of the three may be any
/// of them, so every control is tried for it.
///
/// - `vendor`: the library whose control is about to be looked up.
#[cfg(any(sparta_accelerate, sparta_blas))]
fn linked(vendor: Vendor) -> bool {
    let name = env!("SPARTA_DENSE_LIBRARY").to_ascii_lowercase();
    let named = |part: &str| name.contains(part);
    if !(named("accelerate") || named("mkl") || named("openblas")) {
        return true;
    }
    match vendor {
        Vendor::Accelerate => named("accelerate"),
        Vendor::Mkl => named("mkl"),
        Vendor::OpenBlas => named("openblas"),
    }
}

/// Choose the BLAS threading model for this thread, reporting whether a
/// per-thread control exists at all.
///
/// SPARTA wants Rayon to own every level of parallelism, which needs a control
/// that is *per thread*; a global one would have workers fighting over it.
/// Two libraries offer one, and they are tried in that order:
///
/// - Accelerate's `BLASSetThreading`, which arrived in macOS 15.
/// - MKL's `MKL_Set_Num_Threads_Local`. The lowercase `mkl_set_num_threads_local`
///   it also exports is the Fortran interface, which takes its argument by
///   reference.
///
/// OpenBLAS deliberately is not: `openblas_set_num_threads` is process-wide,
/// so calling it from a Rayon worker would change the setting under every
/// other worker. It is set between levels instead, by [`set_global_threading`].
/// When neither control exists -- macOS 13 and 14, or a BLAS that is neither --
/// `level::schedule` keeps wide fronts off the spread path, which costs
/// parallelism but never correctness.
#[cfg(any(sparta_accelerate, sparta_blas))]
pub(super) fn set_threading(threading: Threading) -> bool {
    use std::sync::OnceLock;

    /// A resolved per-thread control, and the convention its argument follows.
    #[derive(Clone, Copy)]
    enum Control {
        /// `BLASSetThreading(0 = machine, 1 = caller)`, zero on success.
        Accelerate(unsafe extern "C" fn(u32) -> i32),
        /// `MKL_Set_Num_Threads_Local(0 = restore global, 1 = caller)`,
        /// returning the previous value rather than a status.
        Mkl(unsafe extern "C" fn(i32) -> i32),
    }

    static ENTRY: OnceLock<Option<Control>> = OnceLock::new();
    let entry = ENTRY.get_or_init(|| {
        // MKL's `mkl_rt` binds each routine on its first call, and that binding
        // is not safe to race: Rayon workers making their first calls at once
        // abort with "Error on loading function". Every worker passes through
        // here before its first dense call and waits for this to finish, so
        // call each routine once now, on one thread.
        let mut unit = [C::from_re(1.0)];
        gemm_nt(1, 1, 1, &[C::from_re(0.0)], 1, &[C::from_re(0.0)], 1, &mut unit, 1);
        trsm(Side::Right, Tri::UnitUpper, 1, 1, &[C::from_re(1.0)], 1, &mut unit, 1);
        let mut work = vec![C::from_re(0.0); sytrf_workspace(1).max(1)];
        sytrf(1, &mut unit, 1, &mut [0], &mut work);
        let find = |vendor: Vendor, name: &[u8]| {
            if linked(vendor) { ffi::loaded_symbol(name) } else { std::ptr::null_mut() }
        };
        let accelerate = find(Vendor::Accelerate, b"BLASSetThreading\0");
        if !accelerate.is_null() {
            // SAFETY: the resolved symbol has Accelerate's documented signature.
            return Some(Control::Accelerate(unsafe {
                std::mem::transmute::<*mut (), unsafe extern "C" fn(u32) -> i32>(accelerate)
            }));
        }
        let mkl = find(Vendor::Mkl, b"MKL_Set_Num_Threads_Local\0");
        (!mkl.is_null()).then(|| {
            // SAFETY: the resolved symbol has MKL's documented signature.
            Control::Mkl(unsafe {
                std::mem::transmute::<*mut (), unsafe extern "C" fn(i32) -> i32>(mkl)
            })
        })
    });
    match entry {
        // SAFETY: each arm calls the symbol it resolved, with one of the two
        // values that library documents.
        Some(Control::Accelerate(set)) => unsafe {
            set(match threading {
                Threading::Machine => 0,
                Threading::Caller => 1,
            }) == 0
        },
        // SAFETY: as above. MKL returns the previous count, not a status, so
        // the call itself is the success.
        Some(Control::Mkl(set)) => unsafe {
            set(match threading {
                Threading::Machine => 0,
                Threading::Caller => 1,
            });
            true
        },
        None => false,
    }
}

#[cfg(not(any(sparta_accelerate, sparta_blas)))]
pub(super) fn set_threading(_: Threading) -> bool {
    false
}

/// Choose the BLAS threading model for the whole process, for a library whose
/// only control is global, reporting whether it has one.
///
/// This is OpenBLAS's `openblas_set_num_threads`. `Machine` is the size of
/// SPARTA's Rayon pool, so `RAYON_NUM_THREADS` bounds BLAS as well, and an
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
#[cfg(any(sparta_accelerate, sparta_blas))]
pub(super) fn set_global_threading(threading: Threading) -> bool {
    use std::sync::OnceLock;

    /// `openblas_set_num_threads` and the size of the Rayon pool.
    #[derive(Clone, Copy)]
    struct Control {
        set: unsafe extern "C" fn(i32),
        machine: i32,
    }

    static ENTRY: OnceLock<Option<Control>> = OnceLock::new();
    let entry = ENTRY.get_or_init(|| {
        // `set_threading` also resolves the per-thread controls and makes the
        // first call of each routine on one thread; a library with a per-thread
        // control does not need this one.
        if set_threading(Threading::Machine) || !linked(Vendor::OpenBlas) {
            return None;
        }
        let set = ffi::loaded_symbol(b"openblas_set_num_threads\0");
        if set.is_null() {
            return None;
        }
        // SAFETY: the symbol has OpenBLAS's documented signature.
        Some(Control {
            set: unsafe { std::mem::transmute::<*mut (), unsafe extern "C" fn(i32)>(set) },
            machine: rayon::current_num_threads().max(1) as i32,
        })
    });
    match entry {
        // SAFETY: the symbol resolved above, called with a positive count.
        Some(control) => unsafe {
            (control.set)(match threading {
                Threading::Machine => control.machine,
                Threading::Caller => 1,
            });
            true
        },
        None => false,
    }
}

#[cfg(not(any(sparta_accelerate, sparta_blas)))]
pub(super) fn set_global_threading(_: Threading) -> bool {
    false
}

#[cfg(any(sparta_accelerate, sparta_blas))]
mod ffi {
    use std::ffi::c_void;
    // CBLAS enumerators, which are ABI constants rather than a header import.
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;
    pub const UPPER: i32 = 121;
    pub const LOWER: i32 = 122;
    pub const NON_UNIT: i32 = 131;
    pub const UNIT: i32 = 132;
    pub const LEFT: i32 = 141;
    pub const RIGHT: i32 = 142;

    /// Address of `name` (null-terminated) in any image already loaded into
    /// this process, or null when none exports it.
    ///
    /// `RTLD_DEFAULT` is `-2` on Apple and `0` on glibc and musl.
    #[cfg(unix)]
    pub fn loaded_symbol(name: &[u8]) -> *mut () {
        const RTLD_DEFAULT: *mut () = if cfg!(target_vendor = "apple") {
            -2isize as *mut ()
        } else {
            std::ptr::null_mut()
        };
        extern "C" {
            fn dlsym(handle: *mut (), symbol: *const u8) -> *mut ();
        }
        // SAFETY: `name` is null-terminated and `dlsym` returns null for a
        // symbol no loaded image has.
        unsafe { dlsym(RTLD_DEFAULT, name.as_ptr()) }
    }

    /// Address of `name` (null-terminated) in any module already loaded into
    /// this process, or null when none exports it.
    ///
    /// Windows has no `RTLD_DEFAULT`, so each module is asked in turn.
    #[cfg(windows)]
    pub fn loaded_symbol(name: &[u8]) -> *mut () {
        extern "system" {
            fn GetCurrentProcess() -> *mut ();
            fn K32EnumProcessModules(
                process: *mut (),
                modules: *mut *mut (),
                bytes: u32,
                needed: *mut u32,
            ) -> i32;
            fn GetProcAddress(module: *mut (), name: *const u8) -> *mut ();
        }
        let mut modules = [std::ptr::null_mut(); 1024];
        let mut needed = 0u32;
        // SAFETY: the buffer and its size in bytes describe the same array,
        // `needed` receives the byte count actually required, and `name` is
        // null-terminated.
        unsafe {
            if K32EnumProcessModules(
                GetCurrentProcess(),
                modules.as_mut_ptr(),
                std::mem::size_of_val(&modules) as u32,
                &mut needed,
            ) == 0
            {
                return std::ptr::null_mut();
            }
            let count = (needed as usize / std::mem::size_of::<*mut ()>()).min(modules.len());
            modules[..count]
                .iter()
                .map(|&module| GetProcAddress(module, name.as_ptr()))
                .find(|symbol| !symbol.is_null())
                .unwrap_or(std::ptr::null_mut())
        }
    }

    // `build.rs` emits the link directive naming the library these resolve in.
    // One set per scalar type. The complex routines take `alpha` and `beta`
    // by pointer, the real ones by value.
    macro_rules! routines {
        ($types:ident, $gemm:ident, $trsm:ident, $sytrf:ident, $getrf:ident, $scale:ty) => {
            /// The signatures below as pointer types, which [`super::newest`]
            /// resolves the `$NEWLAPACK` symbols into.
            #[allow(dead_code)]
            pub mod $types {
                use std::ffi::c_void;
                pub type Gemm = unsafe extern "C" fn(
                    i32, i32, i32, i32, i32, i32, $scale, *const c_void, i32, *const c_void, i32, $scale, *mut c_void, i32,
                );
                pub type Trsm = unsafe extern "C" fn(
                    i32, i32, i32, i32, i32, i32, i32, $scale, *const c_void, i32, *mut c_void, i32,
                );
                pub type Getrf = unsafe extern "C" fn(*const i32, *const i32, *mut c_void, *const i32, *mut i32, *mut i32);
            }
            extern "C" {
                #[allow(clippy::too_many_arguments)]
                pub fn $gemm(
                    order: i32,
                    trans_a: i32,
                    trans_b: i32,
                    m: i32,
                    n: i32,
                    k: i32,
                    alpha: $scale,
                    a: *const c_void,
                    lda: i32,
                    b: *const c_void,
                    ldb: i32,
                    beta: $scale,
                    c: *mut c_void,
                    ldc: i32,
                );
                #[allow(clippy::too_many_arguments)]
                pub fn $trsm(
                    order: i32,
                    side: i32,
                    uplo: i32,
                    trans_a: i32,
                    diag: i32,
                    m: i32,
                    n: i32,
                    alpha: $scale,
                    a: *const c_void,
                    lda: i32,
                    b: *mut c_void,
                    ldb: i32,
                );
                pub fn $getrf(
                    m: *const i32,
                    n: *const i32,
                    a: *mut c_void,
                    lda: *const i32,
                    ipiv: *mut i32,
                    info: *mut i32,
                );
                pub fn $sytrf(
                    uplo: *const u8,
                    n: *const i32,
                    a: *mut c_void,
                    lda: *const i32,
                    ipiv: *mut i32,
                    work: *mut c_void,
                    lwork: *const i32,
                    info: *mut i32,
                );
            }
        };
    }
    routines!(z, cblas_zgemm, cblas_ztrsm, zsytrf_, zgetrf_, *const c_void);
    routines!(c, cblas_cgemm, cblas_ctrsm, csytrf_, cgetrf_, *const c_void);
    routines!(d, cblas_dgemm, cblas_dtrsm, dsytrf_, dgetrf_, f64);
    routines!(s, cblas_sgemm, cblas_strsm, ssytrf_, sgetrf_, f32);
}

/// `$legacy` as the pointer type `$kind`, or, where this OS exports it,
/// Accelerate's `$NEWLAPACK` symbol of the same routine. Resolved once.
///
/// The new interface arrived in macOS 13.3 and SPARTA supports macOS 13, so it
/// is looked up at run time rather than linked, as [`zsytrf`] does. Both take
/// the same 32-bit arguments; the new one is simply faster, and on M4 Pro by
/// a wide margin: `ctrsm` on a 2680 by 996 unknown takes 6.4 ms against 58 ms,
/// and threads where the legacy one does not; `cgetrf` at order 1286 runs at
/// 418 GFLOP/s against 191; `zgemm` at 200 by 200 by 64 at 341 against 163.
/// The complex128 LU of the 86k EMerge system went from 1.31 to 1.15 s and
/// its complex64 LU from 0.63 to 0.50 s.
#[cfg(any(sparta_accelerate, sparta_blas))]
macro_rules! newest {
    ($legacy:path, $kind:ty, $name:literal) => {{
        static ENTRY: std::sync::OnceLock<$kind> = std::sync::OnceLock::new();
        *ENTRY.get_or_init(|| {
            let found = if cfg!(sparta_accelerate) {
                ffi::loaded_symbol(concat!($name, "$NEWLAPACK\0").as_bytes())
            } else {
                std::ptr::null_mut()
            };
            if found.is_null() {
                $legacy as $kind
            } else {
                // SAFETY: Accelerate declares the `$NEWLAPACK` symbol with the
                // legacy routine's signature, which `$kind` spells out. Some
                // call sites already sit inside an `unsafe` block.
                #[allow(unused_unsafe)]
                let routine = unsafe { std::mem::transmute::<*mut (), $kind>(found) };
                routine
            }
        })
    }};
}

/// `c = alpha * a * op(b) + beta * c`, row-major, through this copy's GEMM.
///
/// # Safety
/// The operands must be valid for the given shapes and strides, and `c` must
/// not overlap `a` or `b`.
#[cfg(any(sparta_accelerate, sparta_blas))]
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
        ScalarKind::Complex64 => newest!(ffi::cblas_zgemm, ffi::z::Gemm, "cblas_zgemm")(order, no, trans_b, m, n, k, pa, a, lda, b, ldb, pb, c, ldc),
        // The legacy symbol on purpose. The `$NEWLAPACK` `cgemm` loses the
        // imaginary part when it accumulates into C: on C -= A B^T at
        // 600x600x1200, with imaginary parts a hundredth of the real ones, its
        // imaginary error is 4.5e-6 relative against the legacy 1.3e-7, the
        // real part unchanged; it tripled the residual of the 887k EMerge
        // system factored in complex64 (6.2e-4 to 1.9e-3). `zgemm`, `sgemm`
        // and `dgemm` show no such loss.
        ScalarKind::Complex32 => ffi::cblas_cgemm(order, no, trans_b, m, n, k, pa, a, lda, b, ldb, pb, c, ldc),
        ScalarKind::F64 => {
            let (alpha, beta) = (alpha.re_f64(), beta.re_f64());
            newest!(ffi::cblas_dgemm, ffi::d::Gemm, "cblas_dgemm")(order, no, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        }
        ScalarKind::F32 => {
            let (alpha, beta) = (alpha.re_f64() as f32, beta.re_f64() as f32);
            newest!(ffi::cblas_sgemm, ffi::s::Gemm, "cblas_sgemm")(order, no, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
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
#[cfg(any(sparta_accelerate, sparta_blas))]
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
#[cfg(any(sparta_accelerate, sparta_blas))]
#[allow(clippy::too_many_arguments)]
unsafe fn trsm_raw(side: Side, tri: Tri, m: usize, n: usize, t: *const C, ldt: usize, b: *mut C, ldb: usize) {
    let (side, uplo, diag) = solve_codes(side, tri);
    let (order, no, m, n, ldt, ldb) = (ffi::ROW_MAJOR, ffi::NO_TRANS, m as i32, n as i32, ldt as i32, ldb as i32);
    let one = C::from_re(1.0);
    let (t, b) = (t.cast::<c_void>(), b.cast::<c_void>());
    let pone = (&one as *const C).cast();
    match KIND {
        ScalarKind::Complex64 => newest!(ffi::cblas_ztrsm, ffi::z::Trsm, "cblas_ztrsm")(order, side, uplo, no, diag, m, n, pone, t, ldt, b, ldb),
        ScalarKind::Complex32 => newest!(ffi::cblas_ctrsm, ffi::c::Trsm, "cblas_ctrsm")(order, side, uplo, no, diag, m, n, pone, t, ldt, b, ldb),
        ScalarKind::F64 => newest!(ffi::cblas_dtrsm, ffi::d::Trsm, "cblas_dtrsm")(order, side, uplo, no, diag, m, n, 1.0, t, ldt, b, ldb),
        ScalarKind::F32 => newest!(ffi::cblas_strsm, ffi::s::Trsm, "cblas_strsm")(order, side, uplo, no, diag, m, n, 1.0, t, ldt, b, ldb),
    }
}

/// `c[m][n] -= a[m][k] * b[n][k]^T`, all row-major with the given row strides.
///
/// `lda`, `ldb` and `ldc` are element strides between rows and must be at
/// least `k`, `k` and `n`. The three operands must be distinct.
#[cfg(any(sparta_accelerate, sparta_blas))]
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
#[cfg(any(sparta_accelerate, sparta_blas))]
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
#[cfg(any(sparta_accelerate, sparta_blas))]
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
#[cfg(any(sparta_accelerate, sparta_blas))]
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
#[cfg(any(sparta_accelerate, sparta_blas))]
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
#[cfg(any(sparta_accelerate, sparta_blas))]
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
#[cfg(any(sparta_accelerate, sparta_blas))]
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
#[cfg(any(sparta_accelerate, sparta_blas))]
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
#[cfg(any(sparta_accelerate, sparta_blas))]
pub(super) fn getrf(n: usize, a: &mut [C], lda: usize, ipiv: &mut [i32]) -> i32 {
    debug_assert!(n == 0 || (a.len() >= (n - 1) * lda + n && ipiv.len() >= n));
    let (order, stride, mut info) = (n as i32, lda as i32, 0);
    let getrf = match KIND {
        ScalarKind::Complex64 => newest!(ffi::zgetrf_, ffi::z::Getrf, "zgetrf"),
        ScalarKind::Complex32 => newest!(ffi::cgetrf_, ffi::c::Getrf, "cgetrf"),
        ScalarKind::F64 => newest!(ffi::dgetrf_, ffi::d::Getrf, "dgetrf"),
        ScalarKind::F32 => newest!(ffi::sgetrf_, ffi::s::Getrf, "sgetrf"),
    };
    // SAFETY: an `n` by `n` column-major block with stride `lda` and room for
    // `n` interchanges, both sized above.
    unsafe { getrf(&order, &order, a.as_mut_ptr().cast(), &stride, ipiv.as_mut_ptr(), &mut info) };
    info
}

/// `?sytrf`'s signature, which every type's legacy and `$NEWLAPACK` symbols share.
#[cfg(any(sparta_accelerate, sparta_blas))]
type Sytrf = unsafe extern "C" fn(
    *const u8,
    *const i32,
    *mut c_void,
    *const i32,
    *mut i32,
    *mut c_void,
    *const i32,
    *mut i32,
);

/// The `?sytrf` to call: Accelerate's `$NEWLAPACK` symbol where the OS has it,
/// otherwise the legacy one.
///
/// The new symbol arrived in macOS 13.3 and SPARTA supports macOS 13, so it is
/// looked up at run time rather than linked. Both return the same interchange
/// convention, which `Front::adopt` depends on. The workspace query must go to
/// the same symbol as the factorization, which this single lookup guarantees.
///
/// Apple M4 Pro (10P + 4E cores), macOS 26.6.2, `--curl-curl 30` with
/// `--force-cpu`, five interleaved runs each, load average 2 to 6: the
/// `leading` phase fell from 806 to 613 cpu ms and the factorization from
/// 3.70 s to 3.46 s wall (medians).
#[cfg(sparta_accelerate)]
fn zsytrf() -> Sytrf {
    use std::sync::OnceLock;
    static ENTRY: OnceLock<Sytrf> = OnceLock::new();
    *ENTRY.get_or_init(|| {
        // Null when this macOS does not have the symbol.
        let new = ffi::loaded_symbol(match KIND {
            ScalarKind::Complex64 => b"zsytrf$NEWLAPACK\0",
            ScalarKind::Complex32 => b"csytrf$NEWLAPACK\0",
            ScalarKind::F64 => b"dsytrf$NEWLAPACK\0",
            ScalarKind::F32 => b"ssytrf$NEWLAPACK\0",
        });
        if new.is_null() {
            legacy_sytrf()
        } else {
            // SAFETY: Accelerate's lapack.h declares this symbol with the
            // legacy signature, 32-bit integers and no hidden string length.
            unsafe { std::mem::transmute::<*mut (), Sytrf>(new) }
        }
    })
}

#[cfg(all(sparta_blas, not(sparta_accelerate)))]
fn zsytrf() -> Sytrf {
    legacy_sytrf()
}

/// This copy's unsuffixed `?sytrf`.
#[cfg(any(sparta_accelerate, sparta_blas))]
fn legacy_sytrf() -> Sytrf {
    match KIND {
        ScalarKind::Complex64 => ffi::zsytrf_,
        ScalarKind::Complex32 => ffi::csytrf_,
        ScalarKind::F64 => ffi::dsytrf_,
        ScalarKind::F32 => ffi::ssytrf_,
    }
}

/// Bunch-Kaufman `LDL^T` of the lower triangle of column-major `a`, order `n`.
///
/// Returns LAPACK's `info`: zero on success, `k > 0` when `D(k,k)` is exactly
/// zero. `ipiv` holds `n` interchanges and `work` is resized as LAPACK asks.
#[cfg(any(sparta_accelerate, sparta_blas))]
pub(super) fn sytrf(n: usize, a: &mut [C], lda: usize, ipiv: &mut [i32], work: &mut [C]) -> i32 {
    debug_assert!(a.len() >= (n - 1) * lda + n && ipiv.len() >= n && !work.is_empty());
    let (order, stride, lwork) = (n as i32, lda as i32, work.len() as i32);
    let mut info = 0;
    // SAFETY: LAPACK reads the lower triangle of an `n` by `n` column-major
    // block with stride `lda` and writes `n` interchanges, both sized above.
    // The legacy f2c entry point takes `uplo` as a pointer to one character;
    // it compares through `lsame_`, which never reads the hidden length.
    unsafe {
        zsytrf()(
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
#[cfg(any(sparta_accelerate, sparta_blas))]
pub(super) fn sytrf_workspace(n: usize) -> usize {
    let (order, stride, query) = (n as i32, n.max(1) as i32, -1i32);
    let (mut best, mut info) = (C::from_re(0.0), 0);
    // SAFETY: a workspace query touches neither `a` nor `ipiv`, which LAPACK
    // declares nullable, and writes one complex length into `best`.
    unsafe {
        zsytrf()(
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

#[cfg(not(any(sparta_accelerate, sparta_blas)))]
mod unavailable {
    use super::{Side, Tri, C};
    const NO: &str = "this build links no BLAS; see build.rs";

    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn gemm_nt(
        _: usize,
        _: usize,
        _: usize,
        _: &[C],
        _: usize,
        _: &[C],
        _: usize,
        _: &mut [C],
        _: usize,
    ) {
        unreachable!("{NO}")
    }
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn gemm_nn(
        _: usize,
        _: usize,
        _: usize,
        _: &[C],
        _: usize,
        _: &[C],
        _: usize,
        _: &mut [C],
        _: usize,
    ) {
        unreachable!("{NO}")
    }
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn gemm_nt_front(
        _: usize,
        _: usize,
        _: usize,
        _: &mut [C],
        _: usize,
        _: usize,
        _: usize,
        _: &[C],
        _: usize,
    ) {
        unreachable!("{NO}")
    }
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn trsm(
        _: Side,
        _: Tri,
        _: usize,
        _: usize,
        _: &[C],
        _: usize,
        _: &mut [C],
        _: usize,
    ) {
        unreachable!("{NO}")
    }
    pub(in super::super) fn sytrf(_: usize, _: &mut [C], _: usize, _: &mut [i32], _: &mut [C]) -> i32 {
        unreachable!("{NO}")
    }
    pub(in super::super) fn sytrf_workspace(_: usize) -> usize {
        unreachable!("{NO}")
    }
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn trsm_front(_: Side, _: Tri, _: usize, _: usize, _: &mut [C], _: usize, _: usize, _: usize) {
        unreachable!("{NO}")
    }
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn trsm_front_spread(_: Side, _: Tri, _: usize, _: usize, _: &mut [C], _: usize, _: usize, _: usize) {
        unreachable!("{NO}")
    }
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn gemm_nn_front(_: usize, _: usize, _: usize, _: &mut [C], _: usize, _: usize, _: usize, _: usize) {
        unreachable!("{NO}")
    }
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn gemm_nn_front_spread(_: usize, _: usize, _: usize, _: &mut [C], _: usize, _: usize, _: usize, _: usize) {
        unreachable!("{NO}")
    }
    pub(in super::super) fn getrf(_: usize, _: &mut [C], _: usize, _: &mut [i32]) -> i32 {
        unreachable!("{NO}")
    }
}
#[cfg(not(any(sparta_accelerate, sparta_blas)))]
pub(super) use unavailable::{
    gemm_nn, gemm_nn_front, gemm_nn_front_spread, gemm_nt, gemm_nt_front, getrf, sytrf, sytrf_workspace, trsm,
    trsm_front, trsm_front_spread,
};
