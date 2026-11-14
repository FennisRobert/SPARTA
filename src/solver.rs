//! Explicit backend selection without silent precision loss.
//!
//! [`Solver::factor_values`] and [`Solver::solve_values`] take any of the four
//! value types and choose the numerics from it: on the CPU, the copy of the
//! backend compiled for that type; on the GPU, single precision only.
use crate::scalar::{Scalar, ScalarKind};
use crate::{cpu_c, cpu_d, cpu_s, cpu_z, gpu, Analysis, Complex32, Complex64, Device, Error, Result, SparseColumns};
use std::any::TypeId;

/// Reusable factors. `Config::force_cpu` selects the Rust CPU numerics.
/// The GPU remains the default; no automatic fallback occurs after a failure.
pub struct Solver {
    backend: Backend,
    /// Value type of the latest successful factorization.
    factored: Option<ScalarKind>,
}
// One per solver, so the size difference between the variants costs nothing.
#[allow(clippy::large_enum_variant)]
enum Backend {
    Cpu(Cpu),
    Gpu(gpu::Solver),
}
/// The CPU backend in the value type of the latest factorization. A new value
/// type replaces the storage of the old one.
enum Cpu {
    /// Validated but not yet factored: no numerical storage yet.
    Idle(Analysis),
    Z(cpu_z::Solver),
    C(cpu_c::Solver),
    D(cpu_d::Solver),
    S(cpu_s::Solver),
    /// Between two value types, or destroyed.
    Gone,
}

/// Run `$body` with `$s` bound to whichever typed CPU solver `$cpu` holds, or
/// evaluate `$idle` when it holds none.
macro_rules! each_cpu {
    ($cpu:expr, $s:ident => $body:expr, $idle:expr) => {
        match $cpu {
            Cpu::Z($s) => $body,
            Cpu::C($s) => $body,
            Cpu::D($s) => $body,
            Cpu::S($s) => $body,
            Cpu::Idle(_) | Cpu::Gone => $idle,
        }
    };
}

impl Solver {
    /// Allocate numerical storage for `plan`, using its `force_cpu` setting.
    ///
    /// On the CPU, storage is allocated by the first factorization, in the
    /// value type it is given.
    pub fn new(plan: Analysis) -> Result<Self> {
        if plan.config.force_cpu {
            plan.config.check().map_err(Error::Invalid)?;
            if plan.n == 0 {
                return Err(Error::Invalid("released analysis".into()));
            }
            return Ok(Self {
                backend: Backend::Cpu(Cpu::Idle(plan)),
                factored: None,
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
            factored: None,
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
            factored: None,
        })
    }

    /// Factor CSC `values` of any supported type: `f32`, `f64`, `Complex32`
    /// or `Complex64`.
    ///
    /// The value type decides the arithmetic. On the CPU each type is factored
    /// in its own precision, and real values use real arithmetic. The GPU
    /// factors `f32` and `Complex32` only, and
    /// refuses double precision rather than rounding it.
    ///
    /// `values` follow the entry order of the analyzed pattern.
    pub fn factor_values<T: Scalar>(&mut self, values: &[T]) -> Result<()> {
        self.factored = None;
        match &mut self.backend {
            Backend::Gpu(s) => match T::KIND {
                ScalarKind::Complex32 => s.factor(cast(values)),
                // Until the real kernels land, real single precision runs in
                // the complex kernels with a zero imaginary part.
                ScalarKind::F32 => s.factor(&convert::<T, Complex32>(values)),
                _ => Err(cpu_required(T::KIND)),
            }?,
            Backend::Cpu(cpu) => factor_cpu(cpu, values)?,
        }
        self.factored = Some(T::KIND);
        Ok(())
    }

    /// Solve `nrhs` right-hand sides packed end to end in `b`, of any
    /// supported type: column `j` is `b[j*n..(j+1)*n]`, and so is the result.
    ///
    /// `b` must have the precision the matrix was factored in. A complex
    /// right-hand side may be solved against real factors; a real one against
    /// complex factors is refused, since its solution is complex.
    pub fn solve_values<T: Scalar>(&mut self, b: &[T], nrhs: usize) -> Result<Vec<T>> {
        self.check_rhs::<T>()?;
        match &mut self.backend {
            // The GPU factors are complex64 in both cases.
            Backend::Gpu(s) => {
                let x = s.solve_many(&convert::<T, Complex32>(b), nrhs)?;
                Ok(convert(&x))
            }
            Backend::Cpu(cpu) => match cpu {
                Cpu::Z(s) => s.solve_many(&convert(b), nrhs).map(|x| convert(&x)),
                Cpu::C(s) => s.solve_many(&convert(b), nrhs).map(|x| convert(&x)),
                Cpu::D(s) => solve_real(b, nrhs, |b, k| s.solve_many(b, k)),
                Cpu::S(s) => solve_real(b, nrhs, |b, k| s.solve_many(b, k)),
                Cpu::Idle(_) | Cpu::Gone => Err(Error::Invalid("no CPU factors".into())),
            },
        }
    }

    /// Refuse a right-hand side of type `T` that does not fit the factors:
    /// none yet, another precision, or real against complex factors.
    fn check_rhs<T: Scalar>(&self) -> Result<()> {
        let Some(kind) = self.factored else {
            return Err(Error::Invalid(
                "successful factorization required before solve".into(),
            ));
        };
        if kind.is_double() != T::KIND.is_double() {
            return Err(Error::Invalid(format!(
                "the matrix was factored in {kind}; solve with a right-hand side of the same precision, not {}",
                T::KIND
            )));
        }
        if kind.is_complex() && !T::KIND.is_complex() {
            return Err(Error::Invalid(format!(
                "the matrix was factored in {kind}, so the solution is complex; pass a complex right-hand side, not {}",
                T::KIND
            )));
        }
        Ok(())
    }

    /// Solve sparse right-hand sides: `A^-1 B` for the columns of `b`, `n`
    /// values per column end to end, as [`Solver::solve_values`] returns them.
    ///
    /// The answer is the one `solve_values` gives for the same columns made
    /// dense. On CPU `LDL^T` factors it is faster: each column's forward sweep
    /// visits only the part of the factor its nonzeros connect to. Elsewhere
    /// (the GPU, the hybrid factorization, LU) the columns are solved dense.
    ///
    /// Arguments:
    /// - `b`: the right-hand sides; precision rules as for `solve_values`.
    pub fn solve_sparse<T: Scalar>(&mut self, b: &SparseColumns<'_, T>) -> Result<Vec<T>> {
        self.check_rhs::<T>()?;
        self.check_dimension(b)?;
        if let Backend::Cpu(cpu) = &self.backend {
            match sparse_cpu(cpu, b, None, false) {
                Err(Error::Unsupported(_)) => {}
                other => return other,
            }
        }
        self.solve_values(&b.dense(), b.ncols())
    }

    /// The projection `W^T A^-1 B`, `w.ncols()` rows by `b.ncols()` columns,
    /// column-major: entry `(i, j)` is column `i` of `w` applied to the
    /// solution for column `j` of `b`. The transpose is plain, never
    /// conjugated, as everywhere in SPARTA.
    ///
    /// With `w` = `None` it projects on `b` itself, `B^T A^-1 B`: the port
    /// matrix of a frequency-domain model whose columns of `b` are the port
    /// excitations, and the result is then exactly symmetric.
    ///
    /// On CPU `LDL^T` factors no solution is formed at all: two forward
    /// sweeps restricted to the columns' reach, and one small product per
    /// front they share. Elsewhere it is [`Solver::solve_sparse`] followed by
    /// the projection, the same answer.
    ///
    /// Arguments:
    /// - `b`: the right-hand sides.
    /// - `w`: the columns to project on, of the same type and dimension, or
    ///   `None` for `b`.
    pub fn solve_projected<T: Scalar>(
        &mut self,
        b: &SparseColumns<'_, T>,
        w: Option<&SparseColumns<'_, T>>,
    ) -> Result<Vec<T>> {
        self.check_rhs::<T>()?;
        self.check_dimension(b)?;
        if let Some(w) = w {
            self.check_dimension(w)?;
        }
        if let Backend::Cpu(cpu) = &self.backend {
            match sparse_cpu(cpu, b, Some(w), true) {
                Err(Error::Unsupported(_)) => {}
                other => return other,
            }
        }
        let x = self.solve_values(&b.dense(), b.ncols())?;
        Ok(w.unwrap_or(b).project(&x, b.ncols()))
    }

    /// Refuse sparse columns whose length is not the matrix dimension.
    fn check_dimension<T: Scalar>(&self, b: &SparseColumns<'_, T>) -> Result<()> {
        let n = match &self.backend {
            Backend::Gpu(s) => s.n(),
            Backend::Cpu(cpu) => each_cpu!(cpu, s => s.n(), return Err(Error::Invalid("no CPU factors".into()))),
        };
        if b.n() != n {
            return Err(Error::Invalid(format!(
                "sparse columns of length {} do not fit a matrix of dimension {n}",
                b.n()
            )));
        }
        Ok(())
    }

    /// Value type of the latest successful factorization, or `None` before one.
    pub fn factored_kind(&self) -> Option<ScalarKind> {
        self.factored
    }

    /// Factor complex64 CSC `values`; the CPU promotes them before computation.
    /// Use `factor_c128` to preserve double-precision input values, and
    /// [`Solver::factor_values`] for complex64 arithmetic on the CPU.
    pub fn factor(&mut self, values: &[Complex32]) -> Result<()> {
        match self.backend {
            Backend::Gpu(_) => self.factor_values(values),
            Backend::Cpu(_) => self.factor_values(&convert::<Complex32, Complex64>(values)),
        }
    }
    /// Factor complex128 CSC `values` on CPU. Requires `force_cpu=true`.
    pub fn factor_c128(&mut self, values: &[Complex64]) -> Result<()> {
        self.factor_values(values)
    }
    /// How the latest hybrid factorization divided its time between the CPU
    /// and the GPU, or `None` when it ran on one device only.
    pub fn hybrid_timing(&self) -> Option<crate::HybridTiming> {
        match &self.backend {
            Backend::Gpu(s) => s.hybrid_timing(),
            Backend::Cpu(_) => None,
        }
    }
    /// Solve a complex64 right-hand side `b`, returning complex64 values.
    pub fn solve(&mut self, b: &[Complex32]) -> Result<Vec<Complex32>> {
        self.solve_many(b, 1)
    }
    /// Solve `nrhs` complex64 columns packed end to end in `b`.
    ///
    /// Against complex128 CPU factors the columns are promoted, solved in
    /// complex128 and rounded back.
    pub fn solve_many(&mut self, b: &[Complex32], nrhs: usize) -> Result<Vec<Complex32>> {
        if self.factored != Some(ScalarKind::Complex64) {
            return self.solve_values(b, nrhs);
        }
        let out = self.solve_values(&convert::<Complex32, Complex64>(b), nrhs)?;
        let out: Vec<Complex32> = convert(&out);
        if out.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
            return Err(Error::Invalid(
                "CPU result exceeds complex64; use solve_c128".into(),
            ));
        }
        Ok(out)
    }
    /// Solve complex128 RHS `b` on CPU, preserving double precision.
    pub fn solve_c128(&mut self, b: &[Complex64]) -> Result<Vec<Complex64>> {
        self.solve_many_c128(b, 1)
    }
    /// Solve `nrhs` complex128 columns packed end to end in `b` on CPU.
    pub fn solve_many_c128(&mut self, b: &[Complex64], nrhs: usize) -> Result<Vec<Complex64>> {
        match self.backend {
            Backend::Cpu(_) => self.solve_values(b, nrhs),
            Backend::Gpu(_) => Err(cpu_required(ScalarKind::Complex64)),
        }
    }
    /// Return whether numerical operations run on CPU.
    pub fn is_cpu(&self) -> bool {
        matches!(self.backend, Backend::Cpu(_))
    }
    /// Release all numerical and symbolic storage; the solver cannot be reused.
    pub fn destroy(&mut self) {
        self.factored = None;
        match &mut self.backend {
            Backend::Cpu(cpu) => {
                each_cpu!(cpu, s => s.destroy(), ());
                *cpu = Cpu::Gone;
            }
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
            Backend::Cpu(cpu) => each_cpu!(cpu, s => s.seconds, 0.0),
            Backend::Gpu(s) => s.factor_device_seconds(),
        }
    }
    /// Number of statically perturbed pivots in the latest factorization.
    pub fn perturbed_pivots(&self) -> u64 {
        match &self.backend {
            Backend::Cpu(cpu) => each_cpu!(cpu, s => s.perturbed, 0),
            Backend::Gpu(s) => s.perturbed_pivots(),
        }
    }
    /// Smallest and largest pivot magnitude, or None without valid factors.
    /// LDLᵀ reports scaled block pivots; LU reports U's diagonal.
    pub fn pivot_range(&self) -> Option<(f64, f64)> {
        match &self.backend {
            Backend::Cpu(cpu) => each_cpu!(cpu, s => s.pivot_range(), None),
            Backend::Gpu(s) => s.pivot_range(),
        }
    }
}

/// Factor `values` on the CPU in their own type, first replacing the storage
/// when the previous factorization used another type.
fn factor_cpu<T: Scalar>(cpu: &mut Cpu, values: &[T]) -> Result<()> {
    let held = match cpu {
        Cpu::Z(_) => Some(ScalarKind::Complex64),
        Cpu::C(_) => Some(ScalarKind::Complex32),
        Cpu::D(_) => Some(ScalarKind::F64),
        Cpu::S(_) => Some(ScalarKind::F32),
        Cpu::Idle(_) | Cpu::Gone => None,
    };
    if held != Some(T::KIND) {
        // The old storage is dropped before the new one is allocated, so the
        // two never coexist.
        let plan = match std::mem::replace(cpu, Cpu::Gone) {
            Cpu::Idle(plan) => plan,
            Cpu::Z(s) => s.into_plan(),
            Cpu::C(s) => s.into_plan(),
            Cpu::D(s) => s.into_plan(),
            Cpu::S(s) => s.into_plan(),
            Cpu::Gone => return Err(Error::Invalid("the solver was destroyed".into())),
        };
        *cpu = match T::KIND {
            ScalarKind::Complex64 => Cpu::Z(cpu_z::Solver::new(plan)?),
            ScalarKind::Complex32 => Cpu::C(cpu_c::Solver::new(plan)?),
            ScalarKind::F64 => Cpu::D(cpu_d::Solver::new(plan)?),
            ScalarKind::F32 => Cpu::S(cpu_s::Solver::new(plan)?),
        };
    }
    match cpu {
        Cpu::Z(s) => s.factor(cast(values)),
        Cpu::C(s) => s.factor(cast(values)),
        Cpu::D(s) => s.factor(cast(values)),
        Cpu::S(s) => s.factor(cast(values)),
        Cpu::Idle(_) | Cpu::Gone => unreachable!("allocated above"),
    }
}

/// [`Solver::solve_sparse`] (`project` false) or
/// [`Solver::solve_projected`] on the CPU factors `cpu` holds.
/// `Error::Unsupported` means these factors take the dense route.
///
/// Arguments:
/// - `b`: the right-hand sides.
/// - `w`: for a projection, the columns to project on, `None` for `b`.
/// - `project`: whether to project.
fn sparse_cpu<T: Scalar>(
    cpu: &Cpu,
    b: &SparseColumns<'_, T>,
    w: Option<Option<&SparseColumns<'_, T>>>,
    project: bool,
) -> Result<Vec<T>> {
    // In the factors' own complex type, as `solve_values` converts.
    macro_rules! complex {
        ($s:expr, $c:ty, $m:ident) => {{
            let bv: Vec<$c> = convert(b.values());
            let bc = $m::Columns { ptr: b.ptr(), rows: b.rows(), values: &bv };
            let x = if project {
                let wv: Option<Vec<$c>> = w.flatten().map(|w| convert(w.values()));
                let wc = w.flatten().zip(wv.as_deref()).map(|(w, v)| $m::Columns { ptr: w.ptr(), rows: w.rows(), values: v });
                $s.project(bc, wc)?
            } else {
                $s.solve_sparse(bc)?
            };
            Ok(convert(&x))
        }};
    }
    // Real factors: a complex `b` or `w` goes as its real and imaginary parts.
    macro_rules! real {
        ($s:expr, $r:ty, $m:ident) => {{
            let split = |c: &SparseColumns<'_, T>| -> (Vec<usize>, Vec<usize>, Vec<$r>) {
                if !T::KIND.is_complex() {
                    return (c.ptr().to_vec(), c.rows().to_vec(), convert(c.values()));
                }
                let k = c.ncols();
                let mut ptr = c.ptr().to_vec();
                ptr.extend(c.ptr()[1..].iter().map(|&p| p + c.rows().len()));
                let mut rows = c.rows().to_vec();
                rows.extend_from_slice(c.rows());
                let mut values: Vec<$r> = c.values().iter().map(|v| v.re_f64() as $r).collect();
                values.extend(c.values().iter().map(|v| v.to_c64().im as $r));
                debug_assert_eq!(ptr.len(), 2 * k + 1);
                (ptr, rows, values)
            };
            let (bp, br, bv) = split(b);
            let bc = $m::Columns { ptr: &bp, rows: &br, values: &bv };
            let (p, n) = (b.ncols(), b.n());
            if !project {
                let x = $s.solve_sparse(bc)?;
                if !T::KIND.is_complex() {
                    return Ok(convert(&x));
                }
                // Columns `0..p` are the real parts' solutions, `p..2p` the
                // imaginary parts'.
                let mut out = Vec::with_capacity(n * p);
                for j in 0..p {
                    let (re, im) = (&x[j * n..(j + 1) * n], &x[(p + j) * n..(p + j + 1) * n]);
                    out.extend(re.iter().zip(im).map(|(&r, &i)| T::from_c64(Complex64::new(r as f64, i as f64))));
                }
                return Ok(out);
            }
            let wsplit = w.flatten().map(split);
            let wc = wsplit.as_ref().map(|(p, r, v)| $m::Columns { ptr: p, rows: r, values: v });
            let z = $s.project(bc, wc)?;
            if !T::KIND.is_complex() {
                return Ok(convert(&z));
            }
            // `(Wr + i Wi)^T A^-1 (Br + i Bi)` from the four real blocks of
            // the `2q` by `2p` projection.
            let q = w.flatten().map_or(p, |w| w.ncols());
            let at = |i: usize, j: usize| z[j * 2 * q + i] as f64;
            let mut out = Vec::with_capacity(q * p);
            for j in 0..p {
                for i in 0..q {
                    let re = at(i, j) - at(q + i, p + j);
                    let im = at(i, p + j) + at(q + i, j);
                    out.push(T::from_c64(Complex64::new(re, im)));
                }
            }
            Ok(out)
        }};
    }
    match cpu {
        Cpu::Z(s) => complex!(s, Complex64, cpu_z),
        Cpu::C(s) => complex!(s, Complex32, cpu_c),
        Cpu::D(s) => real!(s, f64, cpu_d),
        Cpu::S(s) => real!(s, f32, cpu_s),
        Cpu::Idle(_) | Cpu::Gone => Err(Error::Invalid("no CPU factors".into())),
    }
}

/// Solve `b` against real factors through `solve`, which takes `U` columns.
/// A complex `b` is solved as its real and imaginary parts, two real columns
/// per right-hand side.
fn solve_real<T: Scalar, U: Scalar>(
    b: &[T],
    nrhs: usize,
    solve: impl FnOnce(&[U], usize) -> Result<Vec<U>>,
) -> Result<Vec<T>> {
    if !T::KIND.is_complex() {
        return solve(&convert(b), nrhs).map(|x| convert(&x));
    }
    if nrhs == 0 || !b.len().is_multiple_of(nrhs) {
        return Err(Error::Invalid(
            "finite RHS columns of analyzed dimension required".into(),
        ));
    }
    let n = b.len() / nrhs;
    let mut parts = Vec::with_capacity(2 * b.len());
    for column in b.chunks(n) {
        parts.extend(column.iter().map(|v| U::from_c64(v.re_f64().into())));
        parts.extend(column.iter().map(|v| U::from_c64(v.to_c64().im.into())));
    }
    let x = solve(&parts, 2 * nrhs)?;
    let mut out = Vec::with_capacity(b.len());
    for pair in x.chunks(2 * n) {
        let (re, im) = pair.split_at(n);
        out.extend(
            re.iter()
                .zip(im)
                .map(|(r, i)| T::from_c64(Complex64::new(r.re_f64(), i.re_f64()))),
        );
    }
    Ok(out)
}

/// `values` as a slice of `U`, which must be the same type as `T`.
fn cast<T: Scalar, U: Scalar>(values: &[T]) -> &[U] {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>(), "value type mismatch");
    // SAFETY: `T` and `U` are the same type, checked above.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

/// `values` converted to `U`: a copy when the types match, otherwise each
/// value rounded to `U`'s precision, with the imaginary part dropped when `U`
/// is real.
fn convert<T: Scalar, U: Scalar>(values: &[T]) -> Vec<U> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        return cast(values).to_vec();
    }
    values.iter().map(|v| U::from_c64(v.to_c64())).collect()
}

/// Explain why a double precision value type cannot run on the GPU.
fn cpu_required(kind: ScalarKind) -> Error {
    Error::Invalid(format!(
        "{kind} numerics require Config {{ force_cpu: true, .. }}; the GPU factors float32 and complex64"
    ))
}
