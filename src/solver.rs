//! Explicit backend selection without silent precision loss.
use crate::{cpu, gpu, Analysis, Complex32, Complex64, Device, Error, Result};

/// Reusable factors. `Config::force_cpu` selects complex128 Rust numerics.
/// The GPU remains the default; no automatic fallback occurs after a failure.
pub struct Solver {
    backend: Backend,
}
enum Backend {
    Cpu(cpu::Solver),
    Gpu(gpu::Solver),
}
impl Solver {
    /// Allocate numerical storage for `plan`, using its `force_cpu` setting.
    pub fn new(plan: Analysis) -> Result<Self> {
        if plan.config.force_cpu {
            return Ok(Self {
                backend: Backend::Cpu(cpu::Solver::new(plan)?),
            });
        }
        // A plan may be analyzed for any device, so that geometry can be
        // compared and a configuration written on one machine for another.
        // Running one needs the kernels, and a build carries exactly one set.
        let wanted = plan.config.device;
        if !wanted.compiled() {
            return Err(Error::Unsupported(format!(
                "this build runs {} kernels, but the plan was analyzed for {wanted}; \
                 rebuild with --features {wanted}, analyze with Config::default(), \
                 or select the CPU backend with Config {{ force_cpu: true, .. }}",
                Device::COMPILED
            )));
        }
        Ok(Self {
            backend: Backend::Gpu(gpu::Solver::new(plan)?),
        })
    }
    /// Allocate GPU numerical storage for `plan` without taking ownership.
    ///
    /// For a caller that keeps the analysis to factor against again later.
    /// The GPU backend copies every symbolic array into device storage up
    /// front and reads none of them from the host again, so it borrows what it
    /// needs and keeps only the permutation; cloning the analysis instead puts
    /// the descriptors in RAM a third time, which on a 457k EMerge system is
    /// 91 MiB wasted for as long as the factors live.
    ///
    /// GPU only, whatever `plan` says about `force_cpu` -- the CPU backend
    /// walks the symbolic arrays on every factorization and so has to own
    /// them. Use [`Solver::new`] with an owned plan for that path.
    pub fn analyzed(plan: &Analysis) -> Result<Self> {
        let wanted = plan.config.device;
        if !wanted.compiled() {
            return Err(Error::Unsupported(format!(
                "this build runs {} kernels, but the plan was analyzed for {wanted}",
                Device::COMPILED
            )));
        }
        Ok(Self {
            backend: Backend::Gpu(gpu::Solver::analyzed(plan)?),
        })
    }
    /// Factor complex64 CSC `values`; the CPU promotes them before computation.
    /// Use `factor_c128` to preserve double-precision input values.
    pub fn factor(&mut self, values: &[Complex32]) -> Result<()> {
        match &mut self.backend {
            Backend::Gpu(s) => s.factor(values),
            Backend::Cpu(s) => s.factor(
                &values
                    .iter()
                    .map(|v| Complex64::new(v.re as f64, v.im as f64))
                    .collect::<Vec<_>>(),
            ),
        }
    }
    /// Factor complex128 CSC `values` on CPU. Requires `force_cpu=true`.
    pub fn factor_c128(&mut self, values: &[Complex64]) -> Result<()> {
        match &mut self.backend {
            Backend::Cpu(s) => s.factor(values),
            Backend::Gpu(_) => Err(cpu_required()),
        }
    }
    /// Solve a complex64 right-hand side `b`, returning complex64 values.
    pub fn solve(&mut self, b: &[Complex32]) -> Result<Vec<Complex32>> {
        match &mut self.backend {
            Backend::Gpu(s) => s.solve(b),
            Backend::Cpu(_) => self.solve_many(b, 1),
        }
    }
    /// Solve `nrhs` complex64 columns packed end to end in `b`.
    pub fn solve_many(&mut self, b: &[Complex32], nrhs: usize) -> Result<Vec<Complex32>> {
        match &mut self.backend {
            Backend::Gpu(s) => s.solve_many(b, nrhs),
            Backend::Cpu(s) => {
                let out = s.solve_many(
                    &b.iter()
                        .map(|v| Complex64::new(v.re as f64, v.im as f64))
                        .collect::<Vec<_>>(),
                    nrhs,
                )?;
                let out: Vec<_> = out
                    .iter()
                    .map(|v| Complex32::new(v.re as f32, v.im as f32))
                    .collect();
                if out.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
                    return Err(Error::Invalid(
                        "CPU result exceeds complex64; use solve_c128".into(),
                    ));
                }
                Ok(out)
            }
        }
    }
    /// Solve complex128 RHS `b` on CPU, preserving double precision.
    pub fn solve_c128(&mut self, b: &[Complex64]) -> Result<Vec<Complex64>> {
        self.solve_many_c128(b, 1)
    }
    /// Solve `nrhs` complex128 columns packed end to end in `b` on CPU.
    pub fn solve_many_c128(&mut self, b: &[Complex64], nrhs: usize) -> Result<Vec<Complex64>> {
        match &mut self.backend {
            Backend::Cpu(s) => s.solve_many(b, nrhs),
            Backend::Gpu(_) => Err(cpu_required()),
        }
    }
    /// Return whether numerical operations run on CPU.
    pub fn is_cpu(&self) -> bool {
        matches!(self.backend, Backend::Cpu(_))
    }
    /// Release all numerical and symbolic storage; the solver cannot be reused.
    pub fn destroy(&mut self) {
        match &mut self.backend {
            Backend::Cpu(s) => s.destroy(),
            Backend::Gpu(s) => s.destroy(),
        }
    }
    /// Last GPU factor duration in seconds; zero on CPU.
    pub fn factor_device_seconds(&self) -> f64 {
        match &self.backend {
            Backend::Cpu(_) => 0.0,
            Backend::Gpu(s) => s.factor_device_seconds(),
        }
    }
    /// Last CPU factor wall duration in seconds; GPU command duration on Metal.
    pub fn factor_seconds(&self) -> f64 {
        match &self.backend {
            Backend::Cpu(s) => s.seconds,
            Backend::Gpu(s) => s.factor_device_seconds(),
        }
    }
    /// Number of statically perturbed pivots in the latest factorization.
    pub fn perturbed_pivots(&self) -> u64 {
        match &self.backend {
            Backend::Cpu(s) => s.perturbed,
            Backend::Gpu(s) => s.perturbed_pivots(),
        }
    }
    /// Smallest and largest pivot magnitude, or None without valid factors.
    /// LDLᵀ reports scaled block pivots; LU reports U's diagonal.
    pub fn pivot_range(&self) -> Option<(f64, f64)> {
        match &self.backend {
            Backend::Cpu(s) => s.pivot_range(),
            Backend::Gpu(s) => s.pivot_range(),
        }
    }
}
/// Explain why complex128 entry points cannot run on the selected backend.
fn cpu_required() -> Error {
    Error::Invalid("complex128 numerics require Config { force_cpu: true, .. }".into())
}
