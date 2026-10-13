//! BLAS and LAPACK entry points for the dense frontal algebra.
//!
//! Nothing here is specific to one vendor. Fronts are row-major, so the BLAS
//! wrappers pass `CblasRowMajor` and let CBLAS do the index arithmetic; LAPACK
//! is column-major only, so [`sytrf`] takes an explicitly packed column-major
//! block. Both interfaces are the standard ones that Accelerate, OpenBLAS and
//! MKL all export, and `build.rs` decides which library the names resolve to:
//! Accelerate on Apple, otherwise whatever `SPARTA_BLAS_LIB` named.
//!
//! On Apple these are the legacy unsuffixed symbols, not the `$NEWLAPACK`
//! ones: the new interface needs macOS 13.3 and SPARTA supports macOS 13. The
//! one exception is `zsytrf`, which [`sytrf`] resolves at run time and calls
//! through the new symbol when the OS has it; see [`zsytrf`] for the numbers.
//!
//! With no library linked, [`available`] reports false and every caller keeps
//! to the portable Rust loops in `panel.rs` and `update.rs`.
use crate::Complex64 as C;

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
/// other worker. When this returns false -- OpenBLAS, or macOS 13 and 14 --
/// `level::schedule` keeps wide fronts off the spread path instead, which
/// costs parallelism but never correctness.
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
        let mut unit = [C::new(1.0, 0.0)];
        gemm_nt(1, 1, 1, &[C::new(0.0, 0.0)], 1, &[C::new(0.0, 0.0)], 1, &mut unit, 1);
        trsm(Side::Right, Tri::UnitUpper, 1, 1, &[C::new(1.0, 0.0)], 1, &mut unit, 1);
        let mut work = vec![C::new(0.0, 0.0); sytrf_workspace(1).max(1)];
        sytrf(1, &mut unit, 1, &mut [0], &mut work);
        let find = ffi::loaded_symbol;
        let accelerate = find(b"BLASSetThreading\0");
        if !accelerate.is_null() {
            // SAFETY: the resolved symbol has Accelerate's documented signature.
            return Some(Control::Accelerate(unsafe {
                std::mem::transmute::<*mut (), unsafe extern "C" fn(u32) -> i32>(accelerate)
            }));
        }
        let mkl = find(b"MKL_Set_Num_Threads_Local\0");
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

#[cfg(any(sparta_accelerate, sparta_blas))]
mod ffi {
    use super::C;
    // CBLAS enumerators, which are ABI constants rather than a header import.
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;
    pub const UPPER: i32 = 121;
    pub const UNIT: i32 = 132;
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
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_zgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const C,
            a: *const C,
            lda: i32,
            b: *const C,
            ldb: i32,
            beta: *const C,
            c: *mut C,
            ldc: i32,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_ztrsm(
            order: i32,
            side: i32,
            uplo: i32,
            trans_a: i32,
            diag: i32,
            m: i32,
            n: i32,
            alpha: *const C,
            a: *const C,
            lda: i32,
            b: *mut C,
            ldb: i32,
        );
        pub fn zsytrf_(
            uplo: *const u8,
            n: *const i32,
            a: *mut C,
            lda: *const i32,
            ipiv: *mut i32,
            work: *mut C,
            lwork: *const i32,
            info: *mut i32,
        );
    }
}

/// Side of a triangular solve: the triangular factor left or right of the unknown.
#[derive(Clone, Copy)]
pub(super) enum Side {
    Right,
}

/// Which triangle of the stored block the solve reads, and whether its diagonal is implicit.
#[derive(Clone, Copy)]
pub(super) enum Tri {
    /// Unit upper triangle: a column-major unit lower factor read row-major.
    UnitUpper,
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
    let (minus, one) = (C::new(-1.0, 0.0), C::new(1.0, 0.0));
    // SAFETY: the dimensions are checked above and fit i32 because Analysis
    // bounds every front to u32 elements. The operands are distinct slices and
    // the library completes before returning.
    unsafe {
        ffi::cblas_zgemm(
            ffi::ROW_MAJOR,
            ffi::NO_TRANS,
            ffi::TRANS,
            m as i32,
            n as i32,
            k as i32,
            &minus,
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            ldb as i32,
            &one,
            c.as_mut_ptr(),
            ldc as i32,
        );
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
    let (minus, one) = (C::new(-1.0, 0.0), C::new(1.0, 0.0));
    let base = front.as_mut_ptr();
    // SAFETY: both rectangles lie in `front`, asserted above, and are disjoint
    // because each row's left-operand columns end where the target's begin.
    // Accelerate reads one and writes the other, and completes before it
    // returns, so the aliasing `&mut front` would imply never materialises.
    unsafe {
        ffi::cblas_zgemm(
            ffi::ROW_MAJOR,
            ffi::NO_TRANS,
            ffi::TRANS,
            m as i32,
            n as i32,
            k as i32,
            &minus,
            base.add(a_at),
            ld as i32,
            b.as_ptr(),
            ldb as i32,
            &one,
            base.add(c_at),
            ld as i32,
        );
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
    let one = C::new(1.0, 0.0);
    let (uplo, diag) = match tri {
        Tri::UnitUpper => (ffi::UPPER, ffi::UNIT),
    };
    // SAFETY: dimensions fit i32 as in `gemm_nt`; `t` and `b` are distinct
    // allocations and the triangular order matches the requested side.
    unsafe {
        ffi::cblas_ztrsm(
            ffi::ROW_MAJOR,
            match side {
                Side::Right => ffi::RIGHT,
            },
            uplo,
            ffi::NO_TRANS,
            diag,
            m as i32,
            n as i32,
            &one,
            t.as_ptr(),
            ldt as i32,
            b.as_mut_ptr(),
            ldb as i32,
        );
    }
}

/// `zsytrf`'s signature, which the legacy and `$NEWLAPACK` symbols share.
#[cfg(any(sparta_accelerate, sparta_blas))]
type Sytrf = unsafe extern "C" fn(
    *const u8,
    *const i32,
    *mut C,
    *const i32,
    *mut i32,
    *mut C,
    *const i32,
    *mut i32,
);

/// The `zsytrf` to call: Accelerate's `$NEWLAPACK` symbol where the OS has it,
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
        let new = ffi::loaded_symbol(b"zsytrf$NEWLAPACK\0");
        if new.is_null() {
            ffi::zsytrf_
        } else {
            // SAFETY: Accelerate's lapack.h declares this symbol with the
            // legacy signature, 32-bit integers and no hidden string length.
            unsafe { std::mem::transmute::<*mut (), Sytrf>(new) }
        }
    })
}

#[cfg(all(sparta_blas, not(sparta_accelerate)))]
fn zsytrf() -> Sytrf {
    ffi::zsytrf_
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
            a.as_mut_ptr(),
            &stride,
            ipiv.as_mut_ptr(),
            work.as_mut_ptr(),
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
    let (mut best, mut info) = (C::new(0.0, 0.0), 0);
    // SAFETY: a workspace query touches neither `a` nor `ipiv`, which LAPACK
    // declares nullable, and writes one complex length into `best`.
    unsafe {
        zsytrf()(
            b"L".as_ptr(),
            &order,
            std::ptr::null_mut(),
            &stride,
            std::ptr::null_mut(),
            &mut best,
            &query,
            &mut info,
        );
    }
    (best.re as usize).max(n)
}

#[cfg(not(any(sparta_accelerate, sparta_blas)))]
mod unavailable {
    use super::{Side, Tri, C};
    const NO: &str = "this build links no BLAS; see build.rs";

    #[allow(clippy::too_many_arguments)]
    pub(in crate::cpu) fn gemm_nt(
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
    pub(in crate::cpu) fn gemm_nt_front(
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
    pub(in crate::cpu) fn trsm(
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
    pub(in crate::cpu) fn sytrf(_: usize, _: &mut [C], _: usize, _: &mut [i32], _: &mut [C]) -> i32 {
        unreachable!("{NO}")
    }
    pub(in crate::cpu) fn sytrf_workspace(_: usize) -> usize {
        unreachable!("{NO}")
    }
}
#[cfg(not(any(sparta_accelerate, sparta_blas)))]
pub(super) use unavailable::{gemm_nt, gemm_nt_front, sytrf, sytrf_workspace, trsm};
