//! The backend for a build with no GPU kernels compiled in.
//!
//! Selected when neither `metal` nor `vulkan` is available for the target, so
//! that the symbolic analysis, the configuration surface and the complex128
//! CPU backend still build and test anywhere. Every entry point either reports
//! no device or refuses; nothing here silently degrades.
use super::{Analysis, Complex32, Error, Result};

/// No GPU kernels were compiled into this build.
pub(crate) fn available() -> bool {
    false
}

pub(crate) struct Context;

impl Context {
    pub(crate) fn new(_: &Analysis) -> Result<Self> {
        Err(Error::Unsupported(format!(
            "no GPU backend compiled in for {}-{}; build with --features metal \
             or --features vulkan, or select the CPU backend with \
             Config {{ force_cpu: true, .. }}",
            std::env::consts::OS,
            std::env::consts::ARCH,
        )))
    }
    pub(crate) fn run(
        &mut self,
        _: &[Complex32],
        _: Option<&mut [Complex32]>,
        _: u32,
        _: &mut [u32],
        _: &mut u64,
    ) -> Result<f64> {
        unreachable!("no Context is ever constructed without a backend")
    }
    pub(crate) fn release(&mut self) {}
    pub(crate) fn pivot_range(&self) -> Option<(f64, f64)> {
        None
    }
}
