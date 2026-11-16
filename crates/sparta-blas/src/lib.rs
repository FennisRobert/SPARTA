//! SpartaBLAS: the dense kernels behind SPARTA's CPU factorization.
//!
//! Four routines in the four BLAS types (`f32`, `f64`, `Complex<f32>`,
//! `Complex<f64>`), which is everything a multifrontal factorization needs
//! from dense linear algebra:
//!
//! - [`gemm_strided`]: `C = alpha·A·B + beta·C`;
//! - [`trsm_strided`]: triangular solves with many right-hand sides;
//! - [`getrf`]: LU with partial pivoting, LAPACK's conventions;
//! - [`sytrf`]: Bunch-Kaufman `L·D·Lᵀ`, lower triangle, LAPACK's conventions.
//!
//! Every matrix is a strided view, so row-major, column-major, transposed and
//! reversed operands need no copies or flags. [`abi`] wraps the three in the
//! CBLAS and LAPACK signatures for callers written against those.
//!
//! **Kernels.** One generic microkernel per value kind (real, complex),
//! compiled for each instruction set: AVX-512 and AVX2 on x86-64, NEON on
//! aarch64, and portable Rust everywhere. Arm's matrix extension at a 512-bit
//! vector length (Apple M4 and later) has its own hand-written kernels in
//! `sme512/`. The best one the CPU runs is chosen
//! at first use; `SPARTA_KERNEL=sme512|avx512|avx2|neon|scalar` overrides it.
//!
//! **Threads.** None of its own. A call spreads over the Rayon pool it runs in,
//! unless the calling thread set [`Threading::Caller`]; see [`threading`].
mod element;
mod gemm;
mod getrf;
mod isa;
mod kernel;
mod pack;
mod simd;
#[cfg(target_arch = "aarch64")]
mod sme512;
mod sytrf;
pub mod threading;
mod trsm;

pub mod abi;

pub use element::{Element, Real};
#[doc(hidden)]
pub use gemm::{Config, Configs};
pub use gemm::gemm_strided;
pub use getrf::getrf;
pub use sytrf::sytrf;
pub use isa::{current as isa, force as force_isa, Isa};
pub use num_complex::Complex;
pub use threading::Threading;
pub use trsm::{trsm_strided, Diag, Side, Uplo};

/// "SpartaBLAS (neon)": the library and the kernel family in use.
pub fn describe() -> String {
    format!("SpartaBLAS ({})", isa::current().name())
}
