//! Multifrontal LDL^T and LU of real or complex systems: GPU single
//! precision, or CPU in the precision of the values given.
//!
//! `symbolic` orders the matrix and partitions it into supernodes; `Analysis`
//! turns that into a shared storage layout; `Solver` selects the GPU or Rust
//! numerics from `Config::force_cpu`. One build targets one GPU backend, named
//! by `Device::COMPILED`; CPU dense updates go through whichever BLAS and
//! LAPACK `build.rs` linked.
mod analysis;
mod config;
// The CPU backend is compiled once per value type from one source, so each
// copy is plain code over a concrete type: `cpu_z` complex128, `cpu_c`
// complex64, `cpu_d` float64 and `cpu_s` float32. Each reads `C`, its value
// type, and `R`, the real type of its precision, from the `cpu_types!` defined
// just before it. The complex64 copy is also the CPU half of the hybrid
// factorization, whose results go straight into the GPU's buffers.
macro_rules! cpu_types {
    () => {
        pub(crate) type R = f64;
        pub(crate) type C = num_complex::Complex64;
    };
}
#[path = "cpu/mod.rs"]
mod cpu_z;
macro_rules! cpu_types {
    () => {
        pub(crate) type R = f32;
        pub(crate) type C = num_complex::Complex32;
    };
}
#[path = "cpu/mod.rs"]
#[allow(clippy::duplicate_mod)]
mod cpu_c;
macro_rules! cpu_types {
    () => {
        pub(crate) type R = f64;
        pub(crate) type C = f64;
    };
}
#[path = "cpu/mod.rs"]
#[allow(clippy::duplicate_mod)]
mod cpu_d;
macro_rules! cpu_types {
    () => {
        pub(crate) type R = f32;
        pub(crate) type C = f32;
    };
}
#[path = "cpu/mod.rs"]
#[allow(clippy::duplicate_mod)]
mod cpu_s;
mod device;
mod gpu;
#[cfg(sparta_metal)]
mod hybrid;
pub mod optimize;
mod preset;
mod profile;
mod scalar;
pub mod profiles;
pub mod settings;
mod solver;
#[doc(hidden)]
// Public so examples/colcount_bench.rs can time the symbolic internals.
// Not part of the supported API surface.
pub mod symbolic;
mod threads;
pub use analysis::{Analysis, Kind};
pub use config::Config;
pub use device::Device;
pub use gpu::gpu_available;
pub use num_complex::{Complex32, Complex64};
pub use preset::Preset;
pub use scalar::{Scalar, ScalarKind};
/// Standalone Rust ordering API and tuning options.
pub use scalpel;
pub use solver::Solver;
pub use symbolic::OrderingMethod;
/// Why a front's pivot search gave up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PivotFailure {
    /// Every candidate in the remaining column was exactly zero: the matrix is
    /// structurally or numerically rank deficient at this column.
    ZeroColumn,
    /// The search met an infinity or NaN, so an earlier update overflowed.
    NonFinite,
    /// A 2x2 pivot block's scaled determinant fell below the arithmetic resolution.
    /// The usual cause is an ill-conditioned system rather than a singular one.
    TinyTwoByTwo,
    /// A descendant front failed, so this one was never attempted.
    ChildFailed,
}
impl std::fmt::Display for PivotFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ZeroColumn => "zero column (rank deficient)",
            Self::NonFinite => "non-finite value (earlier update overflowed)",
            Self::TinyTwoByTwo => "2x2 pivot below arithmetic resolution (ill-conditioned)",
            Self::ChildFailed => "a descendant front failed first",
        })
    }
}
#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Unsupported(String),
    /// The GPU backend failed: no device, a kernel that would not build, or a
    /// command that did not complete. Named `Metal` before SPARTA carried more
    /// than one backend.
    Device(String),
    /// A front broke down at one of its fully summed columns. `reason` says
    /// which check failed, which is what decides whether the matrix is
    /// rank deficient or the selected precision ran out of room.
    Pivot {
        node: usize,
        pivot: u32,
        reason: PivotFailure,
    },
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pivot {
                node,
                pivot,
                reason,
            } => {
                write!(f, "front {node} failed at column {pivot}: {reason}")
            }
            _ => write!(f, "{self:?}"),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

/// How a hybrid factorization divided its time; see `Config::hybrid_cpu`.
#[derive(Clone, Copy, Debug)]
pub struct HybridTiming {
    /// Fraction of the factorization work the CPU subtrees carried.
    pub share: f64,
    /// Wall seconds the CPU spent on its subtrees.
    pub cpu_seconds: f64,
    /// Wall seconds then spent waiting for the GPU's fronts below the cut.
    pub wait_seconds: f64,
    /// Wall seconds copying the CPU results into the GPU buffers.
    pub copy_seconds: f64,
    /// GPU device seconds of the fronts below the cut, which ran alongside the
    /// CPU, and of the fronts from the cut up, which ran after the hand-over.
    pub gpu_below: f64,
    pub gpu_above: f64,
}
