//! The dense kernels behind the CPU factorization: SpartaBLAS, SPARTA's own
//! (`crates/sparta-blas`).
//!
//! SPARTA links no BLAS or LAPACK. GEMM, TRSM and LU come from SpartaBLAS,
//! compiled for each instruction set (the SME matrix unit on Apple M4 and
//! later, NEON on other Arm, AVX-512 and AVX2 on x86-64) and chosen at first
//! use, and Bunch-Kaufman `?sytrf` for the symmetric indefinite diagonal
//! blocks, in LAPACK's conventions.
//!
//! The table below keeps the CBLAS and LAPACK call shapes the CPU backend was
//! written against, so `cpu/dense.rs` calls SpartaBLAS through function
//! pointers exactly as it used to call a vendor library.
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

/// The per-thread threading control: `0` lets a call spread over the Rayon
/// pool, `1` keeps it on the calling thread. Returns zero.
#[derive(Clone, Copy)]
pub(crate) enum PerThread {
    Sparta(extern "C" fn(u32) -> i32),
}

/// The entry points, one per routine and value type.
pub(crate) struct Library {
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

/// The library the CPU dense kernels call. Always present: it is compiled in.
pub(crate) fn library() -> Option<&'static Library> {
    static LIBRARY: OnceLock<Library> = OnceLock::new();
    Some(LIBRARY.get_or_init(|| Library {
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
    }))
}

/// The CPU dense kernels in words, for example "SpartaBLAS (sme)".
pub fn describe() -> String {
    sparta_blas::describe()
}
