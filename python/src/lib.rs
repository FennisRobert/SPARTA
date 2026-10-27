// PyO3 0.22 generated wrappers trigger this lint on newer Rust.
#![allow(clippy::useless_conversion)]
mod cpu;
mod residual;
mod tuning;
mod values;
use cpu::PyCpuFactor;
use numpy::PyReadonlyArray1;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use sparta::{scalpel, Analysis, Config, OrderingMethod, Preset, Solver};
fn error(e: sparta::Error) -> PyErr {
    match e {
        sparta::Error::Invalid(_) => PyValueError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}
#[pyclass(name = "_Analysis")]
struct PyAnalysis {
    analysis: Analysis,
    ordering: OrderingMethod,
}
#[pymethods]
impl PyAnalysis {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (n, ptr, rows, max_front, lu, ordering="scalpel", nemin=None,
                        small_front_rows=None, imbalance=None, wide_cycles=None,
                        fm_passes=None, separator_trials=None, schur_batch=None,
                        panel_threads=None, solve_threads=None, lu_diag_threshold=None,
                        panel=None, schur_block=None, strip_block=None,
                        narrow_cols=None, pivot_threads=None, schur_chunk=None,
                        force_cpu=false, cpu_panel=None, cpu_block_rows=None, cpu_blas_min_rows=None,
                        threads=None))]
    fn new(
        n: usize,
        ptr: PyReadonlyArray1<'_, usize>,
        rows: PyReadonlyArray1<'_, usize>,
        max_front: usize,
        lu: bool,
        ordering: &str,
        nemin: Option<usize>,
        small_front_rows: Option<u32>,
        imbalance: Option<f64>,
        wide_cycles: Option<usize>,
        fm_passes: Option<usize>,
        separator_trials: Option<usize>,
        schur_batch: Option<u32>,
        panel_threads: Option<u32>,
        solve_threads: Option<u32>,
        lu_diag_threshold: Option<f64>,
        panel: Option<u32>,
        schur_block: Option<u32>,
        strip_block: Option<u32>,
        narrow_cols: Option<u32>,
        pivot_threads: Option<u32>,
        schur_chunk: Option<u32>,
        force_cpu: bool,
        cpu_panel: Option<usize>,
        cpu_block_rows: Option<usize>,
        cpu_blas_min_rows: Option<usize>,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let ordering = match ordering {
            "scalpel" => OrderingMethod::Scalpel,
            _ => return Err(PyValueError::new_err("ordering must be scalpel")),
        };
        // None keeps whatever the build profile compiled in.
        let base = Config::default();
        let config = Config {
            ordering,
            // One extension module carries one backend's kernels, so there is
            // nothing here for a caller to choose.
            device: base.device,
            // Python's None for an unlimited front arrives as usize::MAX.
            max_front: (max_front != usize::MAX).then_some(max_front),
            nemin: nemin.unwrap_or(base.nemin),
            small_front_rows: small_front_rows.unwrap_or(base.small_front_rows),
            scalpel: scalpel::Options {
                imbalance: imbalance.unwrap_or(base.scalpel.imbalance),
                wide_cycles: wide_cycles.unwrap_or(base.scalpel.wide_cycles),
                fm_passes: fm_passes.unwrap_or(base.scalpel.fm_passes),
                separator_trials: separator_trials.unwrap_or(base.scalpel.separator_trials),
                ..base.scalpel.clone()
            },
            schur_batch: schur_batch.unwrap_or(base.schur_batch),
            panel_threads: panel_threads.unwrap_or(base.panel_threads),
            solve_threads: solve_threads.unwrap_or(base.solve_threads),
            lu_diag_threshold: lu_diag_threshold.unwrap_or(base.lu_diag_threshold),
            panel: panel.unwrap_or(base.panel),
            schur_block: schur_block.unwrap_or(base.schur_block),
            strip_block: strip_block.unwrap_or(base.strip_block),
            narrow_cols: narrow_cols.unwrap_or(base.narrow_cols),
            pivot_threads: pivot_threads.unwrap_or(base.pivot_threads),
            schur_chunk: schur_chunk.unwrap_or(base.schur_chunk),
            force_cpu,
            cpu_panel: cpu_panel.unwrap_or(base.cpu_panel),
            cpu_block_rows: cpu_block_rows.unwrap_or(base.cpu_block_rows),
            cpu_blas_min_rows: cpu_blas_min_rows.unwrap_or(base.cpu_blas_min_rows),
            // Not a Python argument yet: the default, or SPARTA_HYBRID_CPU.
            hybrid_cpu: base.hybrid_cpu,
            threads: threads.unwrap_or(base.threads),
        };
        config.check().map_err(PyValueError::new_err)?;
        let (ptr, rows) = (ptr.as_slice()?, rows.as_slice()?);
        let analysis = if lu {
            Analysis::lu(n, ptr, rows, &config)
        } else {
            Analysis::ldlt(n, ptr, rows, &config)
        };
        Ok(Self {
            analysis: analysis.map_err(error)?,
            ordering,
        })
    }
    fn info(&self) -> (usize, usize, usize, usize, usize, usize, usize, usize, usize) {
        (
            self.analysis.dimension(),
            self.analysis.supernodes(),
            self.analysis.levels(),
            self.analysis.front_bytes(),
            self.analysis.max_front_size(),
            self.analysis.large_fronts(),
            self.analysis.factor_bytes(),
            self.analysis.upper_bytes(),
            self.analysis.device_bytes(),
        )
    }
    /// Print four bounded symbolic-statistics lines when Python verbosity requests them.
    fn _print_stats(&self) {
        eprintln!(
            "SPARTA | path={:?} | ordering={} nested dissection",
            self.analysis.kind(),
            match self.ordering {
                OrderingMethod::Scalpel => "SCALPEL",
            }
        );
        eprintln!(
            "  matrix: n={} | supernodes={} | levels={}",
            self.analysis.dimension(),
            self.analysis.supernodes(),
            self.analysis.levels()
        );
        eprintln!(
            "  symbolic fill ratio={:.3}x | largest front={} rows",
            self.analysis.fill_ratio(),
            self.analysis.max_front_size()
        );
        eprintln!(
            "  factor storage={:.2} MiB | front workspace={:.2} MiB",
            self.analysis.factor_bytes() as f64 / 1048576.0,
            self.analysis.front_bytes() as f64 / 1048576.0
        );
    }
    // Private diagnostic for tests: prove fixtures actually reach narrow and
    // wide cooperative routes after ordering/amalgamation.
    fn cooperative_front_shapes(&self) -> Vec<(usize, usize)> {
        self.analysis.large_nodes()
    }
    /// Release the symbolic arrays this analysis owns.
    fn destroy(&mut self) {
        self.analysis.release();
    }
    /// Select backend `force_cpu` for memory reporting without repeating symbolic analysis.
    fn set_force_cpu(&mut self, force_cpu: bool) {
        self.analysis.set_force_cpu(force_cpu);
    }
    /// Factor CSC `values` with CPU numerics and this symbolic layout, in the
    /// values' own dtype.
    fn factor_cpu(&self, values: &Bound<'_, PyAny>) -> PyResult<PyCpuFactor> {
        let mut plan = self.analysis.clone();
        plan.set_force_cpu(true);
        let mut solver = Solver::new(plan).map_err(error)?;
        values::factor(&mut solver, values)?;
        Ok(PyCpuFactor { solver })
    }
    /// Factor float32 or complex64 CSC `values` on the GPU, with `hybrid_cpu`
    /// of the work on the CPU alongside it (0 for none; ignored where the
    /// backend has no hybrid path).
    #[pyo3(signature = (values, hybrid_cpu=0.0))]
    fn factor(&mut self, values: &Bound<'_, PyAny>, hybrid_cpu: f64) -> PyResult<PyFactor> {
        if !(0.0..1.0).contains(&hybrid_cpu) {
            return Err(PyValueError::new_err("hybrid_cpu must be at least 0 and below 1"));
        }
        self.analysis.set_hybrid_cpu(hybrid_cpu);
        // Borrowed, not cloned: this analysis stays alive for the next
        // factorization, and the GPU backend keeps its own device copy of every
        // symbolic array. Cloning put them in RAM a third time -- 91 MiB of
        // them on a 457k EMerge system -- for the life of the factors.
        let mut solver = Solver::analyzed(&self.analysis).map_err(error)?;
        values::factor(&mut solver, values)?;
        Ok(PyFactor { solver })
    }
}
#[pyclass(name = "_Factor", unsendable)]
struct PyFactor {
    solver: Solver,
}
#[pymethods]
impl PyFactor {
    /// Replace the GPU factors with float32 or complex64 CSC `values`.
    fn refactor(&mut self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        values::factor(&mut self.solver, values)
    }
    /// Release the Metal buffers and the analysis copy this factor owns.
    fn destroy(&mut self) {
        self.solver.destroy();
    }
    /// Solve vector `rhs` and return a NumPy vector of its dtype.
    fn solve<'py>(&mut self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        values::solve(py, &mut self.solver, rhs)
    }
    /// Solve every column of a (nrhs, n) C-contiguous block in one pass.
    ///
    /// The block is taken as rows rather than columns so the caller's array is
    /// contiguous per right-hand side, which is the layout the bridge copies.
    fn solve_many<'py>(&mut self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        values::solve_many(py, &mut self.solver, rhs)
    }
    /// Numerical factorization duration in seconds (CPU wall time or GPU commands).
    #[getter]
    fn factor_seconds(&self) -> f64 {
        self.solver.factor_seconds()
    }
    #[getter]
    fn factor_device_seconds(&self) -> f64 {
        self.solver.factor_device_seconds()
    }
    #[getter]
    fn perturbed_pivots(&self) -> u64 {
        self.solver.perturbed_pivots()
    }
    /// (smallest, largest) pivot magnitude, or None when unavailable.
    #[getter]
    fn pivot_range(&self) -> Option<(f64, f64)> {
        self.solver.pivot_range()
    }
    /// How the latest hybrid factorization divided its time: (CPU share of
    /// the work, CPU seconds, wait seconds, hand-over seconds, GPU seconds
    /// below the cut, GPU seconds above it), or None when it ran on the GPU
    /// alone.
    #[getter]
    fn hybrid_timing(&self) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.solver.hybrid_timing().map(|h| {
            (h.share, h.cpu_seconds, h.wait_seconds, h.copy_seconds, h.gpu_below, h.gpu_above)
        })
    }
}
#[pyfunction]
fn gpu_available() -> bool {
    sparta::gpu_available()
}
/// Returns the GPU backend this extension was built for: "metal" or "vulkan".
///
/// One build carries one backend's kernels, so this describes the module and
/// not the machine; `gpu_available()` reports whether it can run here.
#[pyfunction]
fn device() -> &'static str {
    match sparta::Device::COMPILED {
        sparta::Device::Metal => "metal",
        sparta::Device::Vulkan => "vulkan",
    }
}
/// Loads the BLAS and LAPACK at `path` for the CPU solver and returns how
/// `blas_library()` names it. Raises when the library lacks a routine, or
/// once a library is already chosen.
#[pyfunction]
fn _load_blas(path: &str) -> PyResult<String> {
    sparta::load_blas(path).map_err(PyRuntimeError::new_err)
}
/// Names the CPU solver's BLAS and LAPACK, or "portable loops" without one.
#[pyfunction]
fn blas_library() -> String {
    sparta::blas_library()
}
/// Returns one setting map for `preset`. Keys are the analyze() keyword names.
#[pyfunction]
#[pyo3(signature = (preset))]
fn _settings(py: Python<'_>, preset: &str) -> PyResult<Py<pyo3::types::PyDict>> {
    let config = Preset::parse(preset)
        .ok_or_else(|| {
            let names: Vec<_> = Preset::ALL.iter().map(|p| p.name()).collect();
            PyValueError::new_err(format!(
                "unknown preset '{preset}'. Available: {}",
                names.join(", ")
            ))
        })?
        .config();
    tuning::settings(py, &config)
}

/// Returns the names and one-line summaries of every built-in preset.
#[pyfunction]
fn _presets() -> Vec<(String, String)> {
    Preset::ALL
        .iter()
        .map(|p| (p.name().to_string(), p.summary().to_string()))
        .collect()
}

#[pymodule]
fn _sparta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAnalysis>()?;
    m.add_class::<PyFactor>()?;
    m.add_class::<PyCpuFactor>()?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(device, m)?)?;
    m.add_function(wrap_pyfunction!(_load_blas, m)?)?;
    m.add_function(wrap_pyfunction!(blas_library, m)?)?;
    m.add_function(wrap_pyfunction!(_settings, m)?)?;
    m.add_function(wrap_pyfunction!(_presets, m)?)?;
    m.add_function(wrap_pyfunction!(residual::_residual_product, m)?)?;
    m.add_class::<tuning::PyOutcome>()?;
    m.add_function(wrap_pyfunction!(tuning::_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(tuning::_profile_save, m)?)?;
    m.add_function(wrap_pyfunction!(tuning::_profile_load, m)?)?;
    m.add_function(wrap_pyfunction!(tuning::_profile_delete, m)?)?;
    m.add_function(wrap_pyfunction!(tuning::_profile_list, m)?)?;
    m.add_function(wrap_pyfunction!(tuning::_profile_path, m)?)?;
    m.add_function(wrap_pyfunction!(tuning::_profile_check, m)?)?;
    // Baked into the compiled extension rather than read from the installed
    // distribution metadata, so it reports the version of the binary actually
    // loaded. A stale extension left behind by a partial reinstall keeps its
    // own old version here while the metadata next to it claims the new one,
    // which is the only way to tell the two apart from inside Python.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
