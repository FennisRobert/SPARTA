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
#[cfg(sparta_metal)]
mod hybrid;

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
    /// Each node's first permuted column and fully summed column count, kept
    /// to name the unknown of a pivot failure.
    columns: Vec<(u32, u32)>,
    context: backend::Context,
    /// CPU subtrees factored alongside the GPU, when `Config::hybrid_cpu` asks.
    #[cfg(sparta_metal)]
    hybrid: Option<hybrid::Hybrid>,
    /// Real fronts factored in real arithmetic, when `Config::real_split` asks.
    #[cfg(sparta_metal)]
    split: Option<RealSplit>,
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
        #[allow(unused_mut)]
        let mut context = backend::Context::new(plan)?;
        // Only Metal has a hybrid path; other backends ignore `hybrid_cpu`.
        #[cfg(sparta_metal)]
        let hybrid = hybrid::Hybrid::new(plan, &mut context)?;
        // Not combined with the hybrid factorization, whose factor plan is the
        // GPU's share and is rebuilt as it rebalances.
        #[cfg(sparta_metal)]
        let split = (hybrid.is_none() && metal::real_split(plan)).then(|| RealSplit::new(plan));
        let mut runtime = plan.runtime_only();
        // This solver is on the GPU whatever the analysis was last asked to
        // report for, so its copy of the config says so.
        runtime.config.force_cpu = false;
        Ok(Self {
            nodes: plan.nodes.len(),
            columns: plan.nodes.iter().map(|n| (n.first, n.nc)).collect(),
            plan: runtime,
            context,
            #[cfg(sparta_metal)]
            hybrid,
            #[cfg(sparta_metal)]
            split,
            valid: false,
            device_seconds: 0.0,
            perturbed: 0,
        })
    }
    /// The matrix dimension.
    pub(crate) fn n(&self) -> usize {
        self.plan.n
    }
    pub fn factor(&mut self, values: &[Complex32]) -> Result<()> {
        self.valid = false;
        self.device_seconds = 0.0;
        self.perturbed = 0;
        if values.len() != self.plan.nnz {
            return Err(Error::Invalid(format!(
                "expected {} matrix values for the analyzed pattern, got {}",
                self.plan.nnz,
                values.len()
            )));
        }
        if let Some(at) = values.iter().position(|v| !v.re.is_finite() || !v.im.is_finite()) {
            return Err(Error::Invalid(format!(
                "matrix value {at} is not finite in complex64; values above about 3.4e38 \
                 overflow it, so rescale the matrix or factor in complex128 on the CPU"
            )));
        }
        // Static pivoting is relative to the whole matrix; see
        // `Config::static_pivot`. LU kernels never read it.
        let tiny = if self.plan.kind == crate::Kind::Ldlt {
            self.plan.config.static_pivot as f32 * values.iter().map(|v| v.norm()).fold(0.0, f32::max)
        } else {
            0.0
        };
        self.context.set_static_pivot(tiny)?;
        let mut status = vec![0u32; self.nodes];
        let mut perturbed = 0u64;
        #[cfg(sparta_metal)]
        let hybrid = self.hybrid.as_mut();
        #[cfg(not(sparta_metal))]
        let hybrid: Option<&mut ()> = None;
        #[cfg(sparta_metal)]
        let split = match self.split.as_mut() {
            Some(split) => split.prepare(&mut self.context, values)?,
            None => false,
        };
        #[cfg(not(sparta_metal))]
        let split = false;
        self.device_seconds = match hybrid {
            #[cfg(sparta_metal)]
            Some(h) => h.factor(&mut self.context, values, &mut status, &mut perturbed)?,
            #[cfg(sparta_metal)]
            _ if split => {
                self.context.factor_start(Some(values), 0, u32::MAX)?;
                self.context.factor_finish(&mut status, &mut perturbed)?
            }
            _ => self.context.run(values, None, 1, &mut status, &mut perturbed)?,
        };
        let _ = split;
        self.perturbed = perturbed;
        #[cfg(sparta_metal)]
        if let Some(split) = self.split.as_mut() {
            if status.iter().all(|&v| v == 0) {
                split.record(self.device_seconds);
            }
        }
        // A failure propagates to every ancestor as ChildFailed, so report
        // the front where it started.
        let started = status.iter().enumerate().find(|(_, v)| **v != 0 && **v != 0xffff_ffff);
        if let Some((node, &code)) = started.or_else(|| status.iter().enumerate().find(|(_, v)| **v != 0)) {
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
            return Err(Error::Pivot { node, pivot, reason, unknown: self.unknown(node, pivot) });
        }
        self.valid = true;
        Ok(())
    }
    /// The caller's index of the unknown at 1-based column `pivot` of
    /// `node`, after the interchanges the factorization recorded, or `None`
    /// when the column is outside the node. Reads the pivot order back from
    /// the device on Metal; elsewhere it names the column before interchanges.
    fn unknown(&self, node: usize, pivot: u32) -> Option<usize> {
        let &(first, nc) = self.columns.get(node)?;
        let (first, nc) = (first as usize, nc as usize);
        let k = (pivot as usize).checked_sub(1).filter(|&k| k < nc)?;
        #[allow(unused_mut)]
        let mut local = k;
        #[cfg(sparta_metal)]
        if self.plan.kind == crate::Kind::Ldlt {
            // SAFETY: the factorization has finished, so nothing writes the
            // pivot order, which holds one u32 per column.
            if let Ok(perms) = unsafe { self.context.buffer::<u32>(10) } {
                local = perms.get(first + k).map(|&l| l as usize).filter(|&l| l < nc).unwrap_or(k);
            }
        }
        self.plan.perm.get(first + local).map(|&p| p as usize)
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
        #[cfg(sparta_metal)]
        {
            self.split = None;
        }
        self.plan.release();
        self.valid = false;
        self.device_seconds = 0.0;
        self.perturbed = 0;
    }
    /// How the latest hybrid factorization divided its time, or `None` when the
    /// whole factorization ran on the GPU.
    pub fn hybrid_timing(&self) -> Option<crate::HybridTiming> {
        #[cfg(sparta_metal)]
        return self.hybrid.as_ref().map(|h| crate::HybridTiming {
            share: h.share,
            cpu_seconds: h.cpu_seconds,
            wait_seconds: h.wait_seconds,
            copy_seconds: h.copy_seconds,
            gpu_below: h.gpu_below,
            gpu_above: h.gpu_above,
        });
        #[cfg(not(sparta_metal))]
        None
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

/// `Config::real_split` on the GPU.
///
/// The fronts whose values and descendants are all real are factored by the
/// real copy of the `LDL^T` kernels, a quarter of the arithmetic and half the
/// traffic of the complex ones. Which fronts those are depends on the values,
/// so it is decided at every factorization, and the two dispatch plans are
/// rebuilt only when it changes: across a frequency sweep the complex entries
/// stay where they are.
///
/// Unlike on the CPU, the split does not always pay here. A level holding
/// both kinds of front is dispatched twice, once per copy of the kernels, and
/// its serial panel steps run one copy after the other. Whether the
/// arithmetic saved outweighs that depends on where the real fronts sit in
/// the tree and how large they are, which no flop count predicts: measured on
/// four EMerge systems, a 92% real share gained 1.26x, a 43% share 1.19x, and
/// a 90% share on a small system lost 5%. So the solver measures instead. See
/// [`Trial`].
#[cfg(sparta_metal)]
struct RealSplit {
    /// The analysis without its row lists and child maps, which the masks
    /// and the dispatch plans do not read.
    plan: Analysis,
    /// The real fronts of the installed plans; empty when none is installed.
    real: Vec<bool>,
    trial: Trial,
}

/// Where the timing comparison of the split stands.
///
/// The device time of a factorization is measured by the GPU itself, so one
/// sample of each choice is enough. The first factorization uses the split
/// and is not judged: it is the first use of the real pipelines and of the
/// storage, and was 7% slower than the later ones on the largest system. The
/// second runs without the split and the third with it, and the faster one is
/// kept for every factorization after, the split only when it is at least
/// [`MARGIN`] faster. A change of the real fronts starts the trial again.
#[cfg(sparta_metal)]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Trial {
    /// Next: with the split, not judged.
    Warm,
    /// Next: without the split.
    Without,
    /// Next: with the split, against this many seconds without.
    With(f64),
    /// Settled on using the split or not.
    Decided(bool),
}

/// Fraction the split must save to be kept. Run-to-run spread of the device
/// time on these systems is about 1%, and a tie is better without the split,
/// which then dispatches each level once.
#[cfg(sparta_metal)]
const MARGIN: f64 = 0.02;

#[cfg(sparta_metal)]
impl RealSplit {
    /// Keep what the split reads of `plan`.
    fn new(plan: &Analysis) -> Self {
        let mut plan = plan.clone();
        plan.rows = Vec::new();
        plan.maps = Vec::new();
        Self { plan, real: Vec::new(), trial: Trial::Warm }
    }
    /// Install the plans for `values` in `context` and return whether this
    /// factorization uses the split. Without it, the factorization runs on
    /// the full plan exactly as it would with `real_split` off.
    ///
    /// Arguments:
    /// - `context`: the device context; no command may be in flight.
    /// - `values`: the matrix values in the analyzed CSC order.
    fn prepare(&mut self, context: &mut backend::Context, values: &[Complex32]) -> Result<bool> {
        let real = self.plan.real_nodes(|s| values[s].im == 0.0);
        let any = real.iter().any(|&r| r);
        if real != self.real {
            if any {
                let complex: Vec<bool> = real.iter().map(|&r| !r).collect();
                context.set_factor_plan(&self.plan.dispatch_plan_for(Some(&complex)))?;
                context.set_real_plan(Some(&self.plan.dispatch_plan_for(Some(&real))))?;
            } else {
                context.set_real_plan(None)?;
            }
            self.real = real;
            self.trial = Trial::Warm;
        }
        let using = any
            && match self.trial {
                Trial::Warm | Trial::With(_) => true,
                Trial::Without => false,
                Trial::Decided(on) => on,
            };
        context.set_real_flags(using.then_some(self.real.as_slice()))?;
        Ok(using)
    }
    /// Advance the trial with the device seconds of the factorization that
    /// [`RealSplit::prepare`] last set up. Call only after it succeeded.
    fn record(&mut self, seconds: f64) {
        if !self.real.iter().any(|&r| r) {
            return;
        }
        self.trial = match self.trial {
            Trial::Warm => Trial::Without,
            Trial::Without => Trial::With(seconds),
            Trial::With(without) => {
                let on = seconds < (1.0 - MARGIN) * without;
                if crate::profile::enabled() {
                    eprintln!(
                        "SPARTA_PROFILE  real_split: {seconds:.4}s with, {without:.4}s without -> {}",
                        if on { "kept" } else { "dropped" }
                    );
                }
                Trial::Decided(on)
            }
            decided => decided,
        };
    }
}
