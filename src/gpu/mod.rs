//! The GPU backend boundary.
//!
//! [`Solver`] is device-independent: it owns the complex64 staging buffers and
//! the pivot-status decoding, and hands everything else to `backend`. A
//! backend module is exactly this, and nothing else:
//!
//! ```text
//! pub(crate) fn available() -> bool
//! pub(crate) struct Context
//! impl Context {
//!     pub(crate) fn new(plan: &Analysis) -> Result<Self>
//!     pub(crate) fn run(&mut self, input: &[Complex32],
//!                       output: Option<&mut [Complex32]>, nrhs: u32,
//!                       status: &mut [u32], perturbed: &mut u64) -> Result<f64>
//!     pub(crate) fn release(&mut self)
//!     pub(crate) fn pivot_range(&self) -> Option<(f64, f64)>
//! }
//! ```
//!
//! `run` does both jobs: `factoring` is implied by `output.is_none()`. It
//! returns the device time in seconds, writes one status word per node, and
//! accumulates perturbed pivots. Everything it needs about the problem is in
//! `plan.dispatch_plan()`, which is device-independent by construction.
//!
//! Exactly one backend is compiled, chosen by `build.rs`:
//!
//! - `sparta_metal` -> [`metal`], the Apple Silicon path.
//! - `sparta_vulkan` -> [`vulkan`], the AMD and NVIDIA path; its design is in
//!   https://github.com/FennisRobert/SPARTA/issues/31.
//! - neither -> [`absent`], which reports no device and refuses to allocate.
//!
//! Adding a third backend means adding a module here, a `Device` variant with
//! its limits and profile, and a branch in `build.rs`. Nothing above this file
//! should need to change.
use crate::{Analysis, Complex32, Error, PivotFailure, Result};

#[cfg(sparta_metal)]
mod metal;
#[cfg(sparta_metal)]
use metal as backend;

#[cfg(sparta_vulkan)]
mod vulkan;
#[cfg(sparta_vulkan)]
use vulkan as backend;

#[cfg(not(any(sparta_metal, sparta_vulkan)))]
mod absent;
#[cfg(not(any(sparta_metal, sparta_vulkan)))]
use absent as backend;

pub fn gpu_available() -> bool {
    backend::available()
}
/// Reusable GPU allocation. Factors remain on GPU between factor and solve.
/// There is no implicit CPU fallback or precision conversion.
pub struct Solver {
    /// Dimensions, configuration and permutation only: `Analysis::runtime_only`
    /// strips the descriptor arrays, which by now live in device storage.
    plan: Analysis,
    /// Supernode count, kept because `plan.nodes` is gone.
    nodes: usize,
    context: backend::Context,
    valid: bool,
    device_seconds: f64,
    perturbed: u64,
}
impl Solver {
    /// Allocate device storage for `plan`, taking ownership of it.
    ///
    /// The symbolic arrays are dropped as soon as the backend has copied them:
    /// see [`Solver::analyzed`], which is the same thing without needing to own
    /// them in the first place.
    pub fn new(plan: Analysis) -> Result<Self> {
        Self::analyzed(&plan)
    }
    /// Allocate device storage for `plan` without taking it.
    ///
    /// `Context::new` copies every descriptor into device buffers, and nothing
    /// reads them from the host afterwards -- `factor` and `solve` need the
    /// dimensions and the permutation, and that is all. So a caller that wants
    /// to keep the analysis for a later refactorization hands it over by
    /// reference rather than cloning it: on unified memory a clone is those
    /// arrays in RAM a third time, alongside the host original and the device
    /// copy, and on a 457k EMerge system each copy is 91 MiB.
    pub fn analyzed(plan: &Analysis) -> Result<Self> {
        let context = backend::Context::new(plan)?;
        let mut runtime = plan.runtime_only();
        // This solver is on the GPU whatever the analysis was last asked to
        // report for, so its copy of the config says so.
        runtime.config.force_cpu = false;
        Ok(Self {
            nodes: plan.nodes.len(),
            plan: runtime,
            context,
            valid: false,
            device_seconds: 0.0,
            perturbed: 0,
        })
    }
    pub fn factor(&mut self, values: &[Complex32]) -> Result<()> {
        self.valid = false;
        self.device_seconds = 0.0;
        self.perturbed = 0;
        if values.len() != self.plan.nnz
            || values
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(Error::Invalid(
                "finite complex64 values must match analyzed CSC pattern".into(),
            ));
        }
        let mut status = vec![0u32; self.nodes];
        let mut perturbed = 0u64;
        self.device_seconds = self
            .context
            .run(values, None, 1, &mut status, &mut perturbed)?;
        self.perturbed = perturbed;
        if let Some((node, &code)) = status.iter().enumerate().find(|(_, v)| **v != 0) {
            // Mirrors the status encoding in kernels.metal.
            let (reason, pivot) = match code {
                0xffff_ffff => (PivotFailure::ChildFailed, 0),
                0xffff_fffe => (PivotFailure::NonFinite, 0),
                _ => {
                    let reason = match code >> 24 {
                        1 => PivotFailure::ZeroColumn,
                        2 => PivotFailure::NonFinite,
                        _ => PivotFailure::TinyTwoByTwo,
                    };
                    (reason, code & 0x00ff_ffff)
                }
            };
            return Err(Error::Pivot {
                node,
                pivot,
                reason,
            });
        }
        self.valid = true;
        Ok(())
    }
    pub fn solve(&mut self, b: &[Complex32]) -> Result<Vec<Complex32>> {
        self.solve_many(b, 1)
    }
    /// Solve for `nrhs` right-hand sides at once.
    ///
    /// `b` holds the columns end to end: column `j` occupies `b[j*n..(j+1)*n]`.
    /// The result uses the same layout. Batching is a grid dimension in the solve
    /// kernels, so the columns share one dispatch per tree level instead of one
    /// sweep each, which is where the saving is.
    pub fn solve_many(&mut self, b: &[Complex32], nrhs: usize) -> Result<Vec<Complex32>> {
        if !self.valid {
            return Err(Error::Invalid(
                "successful factorization required before solve".into(),
            ));
        }
        let n = self.plan.n;
        if nrhs == 0
            || b.len() != n * nrhs
            || b.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(Error::Invalid(
                "finite RHS columns of analyzed dimension required".into(),
            ));
        }
        // Each column is permuted into the solver's ordering independently.
        let mut rhs = Vec::with_capacity(n * nrhs);
        for j in 0..nrhs {
            rhs.extend(self.plan.perm.iter().map(|&i| b[j * n + i as usize]));
        }
        let mut x = vec![Complex32::new(0.0, 0.0); n * nrhs];
        self.context
            .run(&rhs, Some(&mut x), nrhs as u32, &mut [], &mut 0)?;
        if x.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
            return Err(Error::Device("nonfinite solve result".into()));
        }
        // The permuted RHS is spent, so it takes the solution back in the
        // caller's ordering instead of a fresh allocation.
        let mut original = rhs;
        for j in 0..nrhs {
            for (i, &old) in self.plan.perm.iter().enumerate() {
                original[j * n + old as usize] = x[j * n + i];
            }
        }
        Ok(original)
    }
    /// Release the GPU buffers and the symbolic plan this solver owns, now.
    ///
    /// Frees the Metal context, so the front workspace and the factor store go
    /// back immediately rather than when the value is dropped. The solver is
    /// spent afterwards: `factor` and `solve` return an error, and calling this
    /// again does nothing. Use it to hand back a large factorization before
    /// building the next one, when the two would otherwise overlap.
    pub fn destroy(&mut self) {
        self.context.release();
        self.plan.release();
        self.valid = false;
        self.device_seconds = 0.0;
        self.perturbed = 0;
    }
    /// Summed GPU duration of the latest factorization's commands; excludes host preparation and queue waits.
    pub fn factor_device_seconds(&self) -> f64 {
        self.device_seconds
    }
    /// Pivots the latest factorization replaced with a perturbation rather than
    /// failing on. Nonzero means the factors are of a slightly perturbed matrix:
    /// still a sound preconditioner, but a direct solve against them will not
    /// reach the accuracy the residual norm alone suggests. Drive the solution
    /// with iterative refinement or a Krylov method when this is not zero.
    pub fn perturbed_pivots(&self) -> u64 {
        self.perturbed
    }
    /// Smallest and largest pivot magnitude in the latest factors.
    ///
    /// A host scan of data already sitting in shared memory -- the block
    /// diagonal on LDL^T, U's diagonal inside the factor store on LU -- so it
    /// costs a pass over n entries and no GPU work at all. Their ratio says
    /// how near the factorization came to breaking down, which is a quality
    /// signal available without a residual and therefore without a matvec
    /// against the original matrix.
    ///
    /// None before a successful factorization and after `destroy`. 2x2 pivots contribute their block determinant relative to
    /// their largest entry rather than their own diagonal entries, which carry
    /// no meaning alone.
    pub fn pivot_range(&self) -> Option<(f64, f64)> {
        self.valid.then(|| self.context.pivot_range()).flatten()
    }
}
