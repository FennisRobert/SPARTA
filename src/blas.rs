//! The dense kernels behind the CPU factorization: SpartaBLAS, SPARTA's own
//! (`crates/sparta-blas`), or Apple's Accelerate on Macs without SME.
//!
//! SpartaBLAS provides GEMM, TRSM, LU and the Bunch-Kaufman `?sytrf` for the
//! symmetric indefinite diagonal blocks, in LAPACK's conventions, compiled for
//! each instruction set (the SME matrix unit on Apple M4 and later, NEON on
//! other Arm, AVX-512 and AVX2 on x86-64) and chosen at first use.
//!
//! **Accelerate on Macs without SME.** On Apple M1 to M3 (and Intel Macs),
//! Accelerate runs its products on the AMX matrix unit, which only Apple's
//! libraries can program: its instructions are undocumented. SpartaBLAS would
//! fall back to NEON there, which is far slower, so these Macs use Accelerate,
//! as SPARTA 0.3.6 and earlier did on every Mac. M4 and later expose the same
//! unit through SME, where SpartaBLAS's `sme512` kernels match or beat
//! Accelerate. Accelerate is part of macOS, so nothing extra is installed.
//! `SPARTA_KERNEL=accelerate` forces it on any Mac, and naming a SpartaBLAS
//! set (`SPARTA_KERNEL=neon`, ...) forces SpartaBLAS.
//!
//! The table below keeps the CBLAS and LAPACK call shapes the CPU backend was
//! written against, so `cpu/dense.rs` calls either library through function
//! pointers the same way.
use crate::scalar::ScalarKind;
use sparta_blas::abi;
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
/// `?sytrf`, likewise.
pub(crate) type Sytrf =
    unsafe extern "C" fn(*const u8, *const i32, *mut c_void, *const i32, *mut i32, *mut c_void, *const i32, *mut i32);

/// A complex scalar passed by pointer.
type ByRef = *const c_void;

/// The per-thread threading control: `0` lets a call spread over several
/// threads, `1` keeps it on the calling thread. Returns zero on success.
#[derive(Clone, Copy)]
pub(crate) enum PerThread {
    /// SpartaBLAS's: a call spreads over the Rayon pool it runs in.
    Sparta(extern "C" fn(u32) -> i32),
    /// Accelerate's `BLASSetThreading` (macOS 15 and later): a call spreads
    /// over Accelerate's own threads.
    #[cfg(target_os = "macos")]
    Accelerate(unsafe extern "C" fn(u32) -> i32),
}

/// The entry points, one per routine and value type.
pub(crate) struct Library {
    /// How [`describe`] names the library.
    name: String,
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
    /// Indexed by [`slot`]; `None` would route LDLᵀ blocks to the panel factorization.
    sytrf: Option<[Sytrf; 4]>,
    pub per_thread: Option<PerThread>,
}

impl Library {
    /// The `?getrf` for `kind`.
    pub fn getrf(&self, kind: ScalarKind) -> Getrf {
        self.getrf[slot(kind)]
    }

    /// The `?sytrf` for `kind`, when there is one.
    pub fn sytrf(&self, kind: ScalarKind) -> Option<Sytrf> {
        self.sytrf.map(|table| table[slot(kind)])
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

/// The library the CPU dense kernels call, chosen on first use. Always
/// present: SpartaBLAS is compiled in, and Accelerate is part of macOS.
pub(crate) fn library() -> Option<&'static Library> {
    static LIBRARY: OnceLock<Library> = OnceLock::new();
    Some(LIBRARY.get_or_init(|| {
        #[cfg(target_os = "macos")]
        if accelerate::chosen() {
            return accelerate::library();
        }
        sparta_library()
    }))
}

/// SpartaBLAS's entry points.
fn sparta_library() -> Library {
    Library {
        name: sparta_blas::describe(),
        zgemm: abi::zgemm,
        cgemm: abi::cgemm,
        dgemm: abi::dgemm,
        sgemm: abi::sgemm,
        ztrsm: abi::ztrsm,
        ctrsm: abi::ctrsm,
        dtrsm: abi::dtrsm,
        strsm: abi::strsm,
        getrf: [abi::zgetrf, abi::cgetrf, abi::dgetrf, abi::sgetrf],
        sytrf: Some([abi::zsytrf, abi::csytrf, abi::dsytrf, abi::ssytrf]),
        per_thread: Some(PerThread::Sparta(abi::set_threading)),
    }
}

/// The CPU dense kernels in words: "SpartaBLAS (sme512)", or "Accelerate".
pub fn describe() -> String {
    library().map_or_else(String::new, |library| library.name.clone())
}

/// Apple's Accelerate, linked by `build.rs` on macOS, for Macs without SME.
#[cfg(target_os = "macos")]
mod accelerate {
    use super::{ByRef, Gemm, Getrf, Library, PerThread, Sytrf, Trsm};
    use std::ffi::c_void;

    /// Whether this process uses Accelerate: when `SPARTA_KERNEL` says
    /// `accelerate`, or when it names nothing and the CPU has no SME, which
    /// is every Mac before M4.
    pub(super) fn chosen() -> bool {
        match std::env::var("SPARTA_KERNEL") {
            Ok(name) if name == "accelerate" => true,
            Ok(name) if !name.is_empty() => false,
            _ => !sparta_blas::Isa::Sme512.supported(),
        }
    }

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

    /// `$legacy`, or, where this macOS exports it, Accelerate's `$NEWLAPACK`
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
            let found = loaded_symbol(concat!($name, "$NEWLAPACK\0").as_bytes());
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
        Library {
            name: "Accelerate".to_string(),
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
            sytrf: Some([
                newest!(zsytrf_, Sytrf, "zsytrf"),
                newest!(csytrf_, Sytrf, "csytrf"),
                newest!(dsytrf_, Sytrf, "dsytrf"),
                newest!(ssytrf_, Sytrf, "ssytrf"),
            ]),
            per_thread: per_thread(),
        }
    }

    /// `BLASSetThreading`, which arrived in macOS 15. Without it (macOS 13
    /// and 14), `level::schedule` keeps wide fronts off the spread path,
    /// which costs parallelism but never correctness.
    fn per_thread() -> Option<PerThread> {
        let set = loaded_symbol(b"BLASSetThreading\0");
        // SAFETY: the resolved symbol has Accelerate's documented signature.
        (!set.is_null()).then(|| {
            PerThread::Accelerate(unsafe { std::mem::transmute::<*mut (), unsafe extern "C" fn(u32) -> i32>(set) })
        })
    }

    /// Address of `name` (null-terminated) in any image already loaded into
    /// this process, or null when none exports it. `RTLD_DEFAULT` is `-2` on
    /// Apple.
    fn loaded_symbol(name: &[u8]) -> *mut () {
        const RTLD_DEFAULT: *mut () = -2isize as *mut ();
        extern "C" {
            fn dlsym(handle: *mut (), symbol: *const u8) -> *mut ();
        }
        // SAFETY: `name` is null-terminated and `dlsym` returns null for a
        // symbol no loaded image has.
        unsafe { dlsym(RTLD_DEFAULT, name.as_ptr()) }
    }
}
