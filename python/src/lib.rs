// PyO3 0.22 generated wrappers trigger this lint on newer Rust.
#![allow(clippy::useless_conversion)]
mod cpu;
mod tuning;
use cpu::PyCpuFactor;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use rayon::prelude::*;
use sparta::{scalpel, Analysis, Complex32, Config, OrderingMethod, Preset, Solver};
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
                        force_cpu=false, cpu_panel=None, cpu_block_rows=None, cpu_blas_min_rows=None))]
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
    /// Factor complex128 CSC values using CPU numerics and this symbolic layout.
    fn factor_cpu(&self, values: PyReadonlyArray1<'_, Complex64>) -> PyResult<PyCpuFactor> {
        let mut plan = self.analysis.clone();
        plan.set_force_cpu(true);
        let mut solver = Solver::new(plan).map_err(error)?;
        solver.factor_c128(values.as_slice()?).map_err(error)?;
        Ok(PyCpuFactor { solver })
    }
    fn factor(&self, values: PyReadonlyArray1<'_, Complex32>) -> PyResult<PyFactor> {
        // Borrowed, not cloned: this analysis stays alive for the next
        // factorization, and the GPU backend keeps its own device copy of every
        // symbolic array. Cloning put them in RAM a third time -- 91 MiB of
        // them on a 457k EMerge system -- for the life of the factors.
        let mut solver = Solver::analyzed(&self.analysis).map_err(error)?;
        solver.factor(values.as_slice()?).map_err(error)?;
        Ok(PyFactor { solver })
    }
}
#[pyclass(name = "_Factor", unsendable)]
struct PyFactor {
    solver: Solver,
}
#[pymethods]
impl PyFactor {
    fn refactor(&mut self, values: PyReadonlyArray1<'_, Complex32>) -> PyResult<()> {
        self.solver.factor(values.as_slice()?).map_err(error)
    }
    /// Release the Metal buffers and the analysis copy this factor owns.
    fn destroy(&mut self) {
        self.solver.destroy();
    }
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        rhs: PyReadonlyArray1<'_, Complex32>,
    ) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
        Ok(self
            .solver
            .solve(rhs.as_slice()?)
            .map_err(error)?
            .into_pyarray_bound(py))
    }
    /// Solve every column of a (nrhs, n) C-contiguous block in one pass.
    ///
    /// The block is taken as rows rather than columns so the caller's array is
    /// contiguous per right-hand side, which is the layout the bridge copies.
    fn solve_many<'py>(
        &mut self,
        py: Python<'py>,
        rhs: PyReadonlyArray2<'_, Complex32>,
    ) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
        let dims = rhs.as_array().raw_dim();
        let (nrhs, n) = (dims[0], dims[1]);
        let out = self
            .solver
            .solve_many(rhs.as_slice()?, nrhs)
            .map_err(error)?;
        let block = numpy::ndarray::Array2::from_shape_vec((nrhs, n), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(block.into_pyarray_bound(py))
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

/// Rows one worker takes at a time in the residual product. Large enough that
/// the per-task overhead disappears against a row's own work, small enough that
/// the last worker to finish is not holding up the rest.
const RESIDUAL_ROW_BLOCK: usize = 512;

/// Multiplies a complex128 CSR matrix by a dense block of right-hand sides.
///
/// Iterative refinement measures its residual against the original matrix in
/// complex128, and that product is the only part of a solve that is not on the
/// GPU. It stops being negligible as soon as there are several right-hand
/// sides: the GPU batches them into one triangular solve, while SciPy's product
/// is one thread and scales linearly with their number, so past about eight
/// ports it is the larger half of the solve.
///
/// Rows are split into blocks and each worker owns its block's output rows
/// outright, so nothing is shared and no accumulation is atomic. Within a row
/// every right-hand side is advanced together, so the row's values and column
/// indices are read once for all of them rather than once each.
///
/// `indptr` and `indices` describe the CSR pattern, `values` its entries, and
/// `block` is (rows, columns) in C order. The accumulation order matches a
/// row-wise serial product, so the result does not depend on the worker count.
#[pyfunction]
fn _residual_product<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, Complex64>,
    block: PyReadonlyArray2<'py, Complex64>,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let values = values.as_slice()?;
    let block = block.as_array();
    let (n, k) = (block.shape()[0], block.shape()[1]);
    if indptr.len() != n + 1 {
        return Err(PyValueError::new_err(
            "row pointer length must be one more than the block's row count",
        ));
    }
    let x = block
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("the right-hand side block must be C-contiguous"))?;
    let nnz = *indptr.last().unwrap_or(&0);
    if nnz < 0 || nnz as usize > values.len() || nnz as usize > indices.len() {
        return Err(PyValueError::new_err(
            "row pointers overrun the stored entries",
        ));
    }
    let mut out = vec![Complex64::new(0.0, 0.0); n * k];
    // The GIL is not needed by any of this, and holding it would serialise the
    // workers against anything else the interpreter wants to do.
    py.allow_threads(|| {
        out.par_chunks_mut(RESIDUAL_ROW_BLOCK * k)
            .enumerate()
            .for_each(|(chunk, dst)| {
                let first = chunk * RESIDUAL_ROW_BLOCK;
                for (offset, row) in dst.chunks_mut(k).enumerate() {
                    let i = first + offset;
                    for slot in indptr[i] as usize..indptr[i + 1] as usize {
                        let value = values[slot];
                        let column = indices[slot] as usize * k;
                        for c in 0..k {
                            row[c] += value * x[column + c];
                        }
                    }
                }
            });
    });
    let shaped = numpy::ndarray::Array2::from_shape_vec((n, k), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(shaped.into_pyarray_bound(py))
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
    m.add_function(wrap_pyfunction!(_settings, m)?)?;
    m.add_function(wrap_pyfunction!(_presets, m)?)?;
    m.add_function(wrap_pyfunction!(_residual_product, m)?)?;
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
