//! The BLAS and LAPACK library behind the CPU dense kernels: which one it is,
//! its entry points, and its thread control.
//!
//! The four CPU copies (`cpu_z`, `cpu_c`, `cpu_d`, `cpu_s`) share one library,
//! so the choice lives here, once per process, rather than in `cpu/dense.rs`.
//! There are three ways to get one, decided by `build.rs`:
//!
//! - **Accelerate** on Apple, linked. Each routine switches to its
//!   `$NEWLAPACK` symbol at run time where the OS has it.
//! - **A library named at build time** by `SPARTA_BLAS_LIB`, linked.
//! - **A library loaded at run time** everywhere else, which is the default
//!   build off Apple: [`load`] opens a shared library by path and resolves
//!   every routine in it, plain (`cblas_zgemm`, `zsytrf_`, as OpenBLAS and MKL
//!   export them) or with the `scipy_` prefix of the `scipy-openblas32` wheel.
//!   The Python package calls it on import with the library it finds; other
//!   callers can, or can set `SPARTA_BLAS` to the library's path.
//!
//! Loading needs no compiler, headers or import library, so a build from
//! source takes no flags and still factors through an optimised BLAS. Without
//! any library, [`library`] is `None` and the CPU copies keep to their portable
//! Rust loops.
use crate::scalar::ScalarKind;
use std::ffi::c_void;
use std::sync::OnceLock;

/// `?gemm` through CBLAS. `S` is how `alpha` and `beta` are passed: by
/// pointer for the complex routines, by value for the real ones.
pub(crate) type Gemm<S> =
    unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, S, *const c_void, i32, *const c_void, i32, S, *mut c_void, i32);
/// `?trsm` through CBLAS, `alpha` passed as for [`Gemm`].
pub(crate) type Trsm<S> = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, i32, S, *const c_void, i32, *mut c_void, i32);
/// `?getrf`, the f2c-style LAPACK entry point every type shares.
pub(crate) type Getrf = unsafe extern "C" fn(*const i32, *const i32, *mut c_void, *const i32, *mut i32, *mut i32);
/// `?sytrf`, likewise. `uplo` is a pointer to one character; the routine
/// compares it through `lsame_`, which never reads the hidden length.
pub(crate) type Sytrf =
    unsafe extern "C" fn(*const u8, *const i32, *mut c_void, *const i32, *mut i32, *mut c_void, *const i32, *mut i32);

/// A complex scalar passed by pointer.
type ByRef = *const c_void;

/// A per-thread BLAS thread control, and the convention its argument follows.
#[derive(Clone, Copy)]
pub(crate) enum PerThread {
    /// `BLASSetThreading(0 = machine, 1 = caller)`, zero on success.
    #[cfg_attr(not(sparta_accelerate), allow(dead_code))]
    Accelerate(unsafe extern "C" fn(u32) -> i32),
    /// `MKL_Set_Num_Threads_Local(0 = restore global, 1 = caller)`,
    /// returning the previous value rather than a status.
    Mkl(unsafe extern "C" fn(i32) -> i32),
}

/// One library's entry points, one per routine and value type.
pub(crate) struct Library {
    /// What [`describe`] reports: the vendor and, for a loaded one, its path.
    pub name: String,
    pub zgemm: Gemm<ByRef>,
    pub cgemm: Gemm<ByRef>,
    pub dgemm: Gemm<f64>,
    pub sgemm: Gemm<f32>,
    pub ztrsm: Trsm<ByRef>,
    pub ctrsm: Trsm<ByRef>,
    pub dtrsm: Trsm<f64>,
    pub strsm: Trsm<f32>,
    /// Indexed by [`slot`].
    getrf: [Getrf; 4],
    /// Indexed by [`slot`].
    sytrf: [Sytrf; 4],
    /// Accelerate's or MKL's per-thread control; see `dense::set_threading`.
    pub per_thread: Option<PerThread>,
    /// OpenBLAS's process-wide `openblas_set_num_threads`, only for a library
    /// without a per-thread one; see `dense::set_global_threading`.
    pub global: Option<unsafe extern "C" fn(i32)>,
    /// Keeps a loaded library mapped for as long as its pointers are used,
    /// which is the life of the process.
    #[cfg(not(any(sparta_accelerate, sparta_blas)))]
    _handle: Option<libloading::Library>,
}

impl Library {
    /// The `?getrf` for `kind`.
    pub fn getrf(&self, kind: ScalarKind) -> Getrf {
        self.getrf[slot(kind)]
    }

    /// The `?sytrf` for `kind`.
    pub fn sytrf(&self, kind: ScalarKind) -> Sytrf {
        self.sytrf[slot(kind)]
    }
}

/// Position of `kind` in the `[z, c, d, s]` tables.
fn slot(kind: ScalarKind) -> usize {
    match kind {
        ScalarKind::Complex64 => 0,
        ScalarKind::Complex32 => 1,
        ScalarKind::F64 => 2,
        ScalarKind::F32 => 3,
    }
}

/// The process's library: `None` until it is decided, then `Some(None)` for
/// the portable loops or `Some(Some(_))` for a library.
static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();

/// The library the CPU dense kernels call, decided on first use: the linked
/// one, else whatever [`load`] loaded before, else `SPARTA_BLAS`, else none.
pub(crate) fn library() -> Option<&'static Library> {
    LIBRARY.get_or_init(default).as_ref()
}

/// The CPU dense kernels' provider in words: "OpenBLAS (C:\...\libscipy_openblas.dll)",
/// "Accelerate", or "portable loops" when there is none.
///
/// A query only: before the first factorization it decides nothing, so a
/// [`load`] after it still works.
pub fn describe() -> String {
    let current = match LIBRARY.get() {
        Some(current) => current.as_ref(),
        None if linked_build() => library(),
        None => None,
    };
    current.map_or_else(|| "portable loops".to_string(), |library| library.name.clone())
}

/// Load the BLAS and LAPACK at `path` for the CPU dense kernels, and return
/// how [`describe`] will name it.
///
/// Every routine SPARTA calls must resolve in that one library, under its
/// plain name or with the `scipy_` prefix, so a partial or 64-bit-integer
/// build (NumPy's `scipy_openblas64_`, whose names end in `64_`) is refused rather than
/// half-used. It is refused too once the choice is made: by an earlier load, by
/// a library linked at build time, or by a CPU factorization that already ran
/// without one. The library is then fixed for the life of the process.
///
/// - `path`: the shared library, as `LoadLibrary` or `dlopen` takes it.
pub fn load(path: &str) -> Result<String, String> {
    if let Some(current) = LIBRARY.get().or_else(|| linked_build().then(|| LIBRARY.get_or_init(default))) {
        return Err(match current {
            Some(library) => format!("the CPU BLAS is already {}", library.name),
            None => "the CPU solver already ran without a BLAS; load one before the first CPU factorization".into(),
        });
    }
    let library = open(path)?;
    warm(&library);
    let name = library.name.clone();
    LIBRARY.set(Some(library)).map_err(|_| "another thread chose the CPU BLAS first".to_string())?;
    Ok(name)
}

/// Whether `build.rs` linked a library, which then always wins.
const fn linked_build() -> bool {
    cfg!(any(sparta_accelerate, sparta_blas))
}

/// The library when nothing was loaded before first use.
fn default() -> Option<Library> {
    let library = default_library()?;
    warm(&library);
    Some(library)
}

#[cfg(any(sparta_accelerate, sparta_blas))]
fn default_library() -> Option<Library> {
    Some(linked::library())
}

/// `SPARTA_BLAS`, for callers that do not load one themselves. The Python
/// package always has by now, so this is the Rust API's and the C API's route.
#[cfg(not(any(sparta_accelerate, sparta_blas)))]
fn default_library() -> Option<Library> {
    let path = std::env::var("SPARTA_BLAS").ok().filter(|path| !path.is_empty())?;
    open(&path)
        .map_err(|e| eprintln!("SPARTA: SPARTA_BLAS={path}: {e}; the CPU solver uses its portable loops"))
        .ok()
}

#[cfg(any(sparta_accelerate, sparta_blas))]
fn open(_: &str) -> Result<Library, String> {
    unreachable!("`load` refuses before opening anything in a build that links its BLAS")
}

/// Open `path` and resolve every routine in it.
#[cfg(not(any(sparta_accelerate, sparta_blas)))]
fn open(path: &str) -> Result<Library, String> {
    // A library such as MKL's `mkl_rt` loads its own dependencies from the
    // directory it sits in, which the default Windows search order skips.
    #[cfg(windows)]
    let handle = {
        use libloading::os::windows::{
            Library, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        };
        // SAFETY: running a BLAS's initialisers is what loading one is for.
        unsafe { Library::load_with_flags(path, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }
            .map(libloading::Library::from)
    };
    // SAFETY: as above.
    #[cfg(not(windows))]
    let handle = unsafe { libloading::Library::new(path) };
    let handle = handle.map_err(|e| format!("cannot load {path}: {e}"))?;

    // The `scipy-openblas32` wheel prefixes its symbols so that it can sit
    // beside another OpenBLAS in one process. Newer builds also export some
    // under their plain names, but not all, so each routine is looked up
    // under both.
    /// The routine `name`, plain or prefixed, as the pointer type the field it
    /// is assigned to declares.
    fn find<T: Copy>(handle: &libloading::Library, name: &str) -> Result<T, String> {
        ["", "scipy_"]
            .into_iter()
            // SAFETY: each caller assigns the result to the field whose type
            // spells out the routine's standard CBLAS or LAPACK signature.
            .find_map(|prefix| unsafe { handle.get::<T>(format!("{prefix}{name}\0").as_bytes()) }.ok().map(|s| *s))
            .ok_or_else(|| name.to_string())
    }
    let missing = |name: String| format!("{path} has no {name}, which the CPU solver needs");
    macro_rules! find {
        ($name:literal) => {
            find(&handle, $name).map_err(missing)?
        };
    }
    let (per_thread, global) = match find::<unsafe extern "C" fn(i32) -> i32>(&handle, "MKL_Set_Num_Threads_Local") {
        Ok(set) => (Some(PerThread::Mkl(set)), None),
        Err(_) => (None, find::<unsafe extern "C" fn(i32)>(&handle, "openblas_set_num_threads").ok()),
    };
    let vendor = match (&per_thread, &global) {
        (Some(_), _) => "MKL",
        (_, Some(_)) => "OpenBLAS",
        _ => "BLAS",
    };
    Ok(Library {
        name: format!("{vendor} ({path})"),
        zgemm: find!("cblas_zgemm"),
        cgemm: find!("cblas_cgemm"),
        dgemm: find!("cblas_dgemm"),
        sgemm: find!("cblas_sgemm"),
        ztrsm: find!("cblas_ztrsm"),
        ctrsm: find!("cblas_ctrsm"),
        dtrsm: find!("cblas_dtrsm"),
        strsm: find!("cblas_strsm"),
        getrf: [find!("zgetrf_"), find!("cgetrf_"), find!("dgetrf_"), find!("sgetrf_")],
        sytrf: [find!("zsytrf_"), find!("csytrf_"), find!("dsytrf_"), find!("ssytrf_")],
        per_thread,
        global,
        _handle: Some(handle),
    })
}

/// Call every routine once, on this thread, before any other thread can.
///
/// MKL's `mkl_rt` binds each routine on its first call, and that binding is
/// not safe to race: Rayon workers making their first calls at once abort
/// with "Error on loading function". The library is published only after
/// this, so every worker finds its routines bound.
fn warm(library: &Library) {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const UPPER: i32 = 121;
    const UNIT: i32 = 132;
    const RIGHT: i32 = 142;
    let (mut z, mut c, mut d, mut s) = ([1.0f64, 0.0], [1.0f32, 0.0], [1.0f64], [1.0f32]);
    let (zone, cone, zzero, czero) = ([1.0f64, 0.0], [1.0f32, 0.0], [0.0f64, 0.0], [0.0f32, 0.0]);
    let (zone, cone) = (zone.as_ptr().cast::<c_void>(), cone.as_ptr().cast::<c_void>());
    let (zzero, czero) = (zzero.as_ptr().cast::<c_void>(), czero.as_ptr().cast::<c_void>());
    let (one, mut ipiv, mut info, lwork) = (1i32, [0i32], 0i32, 1i32);
    let (mut zwork, mut cwork, mut dwork, mut swork) = ([0.0f64; 2], [0.0f32; 2], [0.0f64], [0.0f32]);
    // SAFETY: every call is order one on one-element operands it may read and
    // write, with `beta = 0` so the products only overwrite their target.
    unsafe {
        let (zp, cp, dp, sp) = (
            z.as_mut_ptr().cast::<c_void>(),
            c.as_mut_ptr().cast::<c_void>(),
            d.as_mut_ptr().cast::<c_void>(),
            s.as_mut_ptr().cast::<c_void>(),
        );
        let mut product = [0.0f64; 2];
        let out = product.as_mut_ptr().cast::<c_void>();
        (library.zgemm)(ROW_MAJOR, NO_TRANS, NO_TRANS, 1, 1, 1, zone, zp, 1, zp, 1, zzero, out, 1);
        (library.cgemm)(ROW_MAJOR, NO_TRANS, NO_TRANS, 1, 1, 1, cone, cp, 1, cp, 1, czero, out, 1);
        (library.dgemm)(ROW_MAJOR, NO_TRANS, NO_TRANS, 1, 1, 1, 1.0, dp, 1, dp, 1, 0.0, out, 1);
        (library.sgemm)(ROW_MAJOR, NO_TRANS, NO_TRANS, 1, 1, 1, 1.0, sp, 1, sp, 1, 0.0, out, 1);
        (library.ztrsm)(ROW_MAJOR, RIGHT, UPPER, NO_TRANS, UNIT, 1, 1, zone, zp, 1, out, 1);
        (library.ctrsm)(ROW_MAJOR, RIGHT, UPPER, NO_TRANS, UNIT, 1, 1, cone, cp, 1, out, 1);
        (library.dtrsm)(ROW_MAJOR, RIGHT, UPPER, NO_TRANS, UNIT, 1, 1, 1.0, dp, 1, out, 1);
        (library.strsm)(ROW_MAJOR, RIGHT, UPPER, NO_TRANS, UNIT, 1, 1, 1.0, sp, 1, out, 1);
        let works = [
            zwork.as_mut_ptr().cast::<c_void>(),
            cwork.as_mut_ptr().cast(),
            dwork.as_mut_ptr().cast(),
            swork.as_mut_ptr().cast(),
        ];
        for (k, a) in [zp, cp, dp, sp].into_iter().enumerate() {
            (library.getrf[k])(&one, &one, a, &one, ipiv.as_mut_ptr(), &mut info);
            (library.sytrf[k])(b"L".as_ptr(), &one, a, &one, ipiv.as_mut_ptr(), works[k], &lwork, &mut info);
        }
    }
}

/// The library `build.rs` linked, and the symbol lookup its run-time
/// refinements need.
#[cfg(any(sparta_accelerate, sparta_blas))]
mod linked {
    use super::{ByRef, Library, PerThread};
    use std::ffi::c_void;

    // `build.rs` emits the link directive naming the library these resolve in.
    // One set per scalar type. The complex routines take `alpha` and `beta`
    // by pointer, the real ones by value.
    macro_rules! routines {
        ($gemm:ident, $trsm:ident, $sytrf:ident, $getrf:ident, $scale:ty) => {
            extern "C" {
                #[allow(clippy::too_many_arguments)]
                fn $gemm(
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
                fn $trsm(
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
                fn $getrf(m: *const i32, n: *const i32, a: *mut c_void, lda: *const i32, ipiv: *mut i32, info: *mut i32);
                #[allow(clippy::too_many_arguments)]
                fn $sytrf(
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
    routines!(cblas_zgemm, cblas_ztrsm, zsytrf_, zgetrf_, ByRef);
    routines!(cblas_cgemm, cblas_ctrsm, csytrf_, cgetrf_, ByRef);
    routines!(cblas_dgemm, cblas_dtrsm, dsytrf_, dgetrf_, f64);
    routines!(cblas_sgemm, cblas_strsm, ssytrf_, sgetrf_, f32);

    /// `$legacy`, or, where this OS exports it, Accelerate's `$NEWLAPACK`
    /// symbol of the same routine.
    ///
    /// The new interface arrived in macOS 13.3 and SPARTA supports macOS 13, so
    /// it is looked up at run time rather than linked. Both take the same
    /// 32-bit arguments; the new one is simply faster, and on M4 Pro by a wide
    /// margin: `ctrsm` on a 2680 by 996 unknown takes 6.4 ms against 58 ms, and
    /// threads where the legacy one does not; `cgetrf` at order 1286 runs at
    /// 418 GFLOP/s against 191; `zgemm` at 200 by 200 by 64 at 341 against 163.
    /// The complex128 LU of the 86k EMerge system went from 1.31 to 1.15 s and
    /// its complex64 LU from 0.63 to 0.50 s. `?sytrf` likewise: on M4 Pro,
    /// `--curl-curl 30 --force-cpu`, the `leading` phase fell from 806 to 613
    /// cpu ms and the factorization from 3.70 s to 3.46 s wall (medians).
    macro_rules! newest {
        ($legacy:expr, $kind:ty, $name:literal) => {{
            let found = if cfg!(sparta_accelerate) {
                loaded_symbol(concat!($name, "$NEWLAPACK\0").as_bytes())
            } else {
                std::ptr::null_mut()
            };
            if found.is_null() {
                $legacy as $kind
            } else {
                // SAFETY: Accelerate declares the `$NEWLAPACK` symbol with the
                // legacy routine's signature, which `$kind` spells out.
                unsafe { std::mem::transmute::<*mut (), $kind>(found) }
            }
        }};
    }

    pub(super) fn library() -> Library {
        use super::{Gemm, Getrf, Sytrf, Trsm};
        Library {
            name: env!("SPARTA_DENSE_LIBRARY").to_string(),
            zgemm: newest!(cblas_zgemm, Gemm<ByRef>, "cblas_zgemm"),
            // The legacy symbol on purpose. The `$NEWLAPACK` `cgemm` loses the
            // imaginary part when it accumulates into C: on C -= A B^T at
            // 600x600x1200, with imaginary parts a hundredth of the real ones,
            // its imaginary error is 4.5e-6 relative against the legacy 1.3e-7,
            // the real part unchanged; it tripled the residual of the 887k
            // EMerge system factored in complex64 (6.2e-4 to 1.9e-3). `zgemm`,
            // `sgemm` and `dgemm` show no such loss.
            cgemm: cblas_cgemm as Gemm<ByRef>,
            dgemm: newest!(cblas_dgemm, Gemm<f64>, "cblas_dgemm"),
            sgemm: newest!(cblas_sgemm, Gemm<f32>, "cblas_sgemm"),
            ztrsm: newest!(cblas_ztrsm, Trsm<ByRef>, "cblas_ztrsm"),
            ctrsm: newest!(cblas_ctrsm, Trsm<ByRef>, "cblas_ctrsm"),
            dtrsm: newest!(cblas_dtrsm, Trsm<f64>, "cblas_dtrsm"),
            strsm: newest!(cblas_strsm, Trsm<f32>, "cblas_strsm"),
            getrf: [
                newest!(zgetrf_, Getrf, "zgetrf"),
                newest!(cgetrf_, Getrf, "cgetrf"),
                newest!(dgetrf_, Getrf, "dgetrf"),
                newest!(sgetrf_, Getrf, "sgetrf"),
            ],
            // One lookup per type serves both the factorization and its
            // workspace query, which must go to the same symbol.
            sytrf: [
                newest!(zsytrf_, Sytrf, "zsytrf"),
                newest!(csytrf_, Sytrf, "csytrf"),
                newest!(dsytrf_, Sytrf, "dsytrf"),
                newest!(ssytrf_, Sytrf, "ssytrf"),
            ],
            per_thread: per_thread(),
            global: global(),
        }
    }

    /// A BLAS vendor whose thread control SPARTA knows how to call.
    #[derive(Clone, Copy)]
    enum Vendor {
        Accelerate,
        Mkl,
        OpenBlas,
    }

    /// Whether SPARTA's own BLAS calls can go to `vendor`, judged from the
    /// library name `build.rs` linked.
    ///
    /// The thread controls are looked up across every module in the process,
    /// and another one may carry a different vendor's: a Python process that
    /// loaded MKL for PARDISO exports `MKL_Set_Num_Threads_Local` while SPARTA
    /// calls OpenBLAS. Trusting it would spread levels across Rayon while
    /// OpenBLAS still threads every call. A library whose name names none of
    /// the three may be any of them, so every control is tried for it. A
    /// loaded library has no such problem: its controls are looked up in it.
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

    /// `name` in any loaded image, if `vendor` may be the linked library.
    fn find(vendor: Vendor, name: &[u8]) -> *mut () {
        if linked(vendor) {
            loaded_symbol(name)
        } else {
            std::ptr::null_mut()
        }
    }

    /// Accelerate's `BLASSetThreading`, which arrived in macOS 15, or MKL's
    /// `MKL_Set_Num_Threads_Local`. The lowercase `mkl_set_num_threads_local`
    /// MKL also exports is the Fortran interface, which takes its argument by
    /// reference.
    fn per_thread() -> Option<PerThread> {
        let accelerate = find(Vendor::Accelerate, b"BLASSetThreading\0");
        if !accelerate.is_null() {
            // SAFETY: the resolved symbol has Accelerate's documented signature.
            return Some(PerThread::Accelerate(unsafe {
                std::mem::transmute::<*mut (), unsafe extern "C" fn(u32) -> i32>(accelerate)
            }));
        }
        let mkl = find(Vendor::Mkl, b"MKL_Set_Num_Threads_Local\0");
        // SAFETY: the resolved symbol has MKL's documented signature.
        (!mkl.is_null())
            .then(|| PerThread::Mkl(unsafe { std::mem::transmute::<*mut (), unsafe extern "C" fn(i32) -> i32>(mkl) }))
    }

    /// OpenBLAS's `openblas_set_num_threads`, for a library without a
    /// per-thread control.
    fn global() -> Option<unsafe extern "C" fn(i32)> {
        if per_thread().is_some() {
            return None;
        }
        let set = find(Vendor::OpenBlas, b"openblas_set_num_threads\0");
        // SAFETY: the symbol has OpenBLAS's documented signature.
        (!set.is_null()).then(|| unsafe { std::mem::transmute::<*mut (), unsafe extern "C" fn(i32)>(set) })
    }

    /// Address of `name` (null-terminated) in any image already loaded into
    /// this process, or null when none exports it.
    ///
    /// `RTLD_DEFAULT` is `-2` on Apple and `0` on glibc and musl.
    #[cfg(unix)]
    fn loaded_symbol(name: &[u8]) -> *mut () {
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
    fn loaded_symbol(name: &[u8]) -> *mut () {
        extern "system" {
            fn GetCurrentProcess() -> *mut ();
            fn K32EnumProcessModules(process: *mut (), modules: *mut *mut (), bytes: u32, needed: *mut u32) -> i32;
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
}
