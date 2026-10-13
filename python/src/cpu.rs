//! Complex128 Python arrays at the CPU solver boundary.
use crate::error;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use sparta::Solver;

#[pyclass(name = "_CpuFactor", unsendable)]
pub(crate) struct PyCpuFactor {
    pub(crate) solver: Solver,
}
#[pymethods]
impl PyCpuFactor {
    /// Replace the CPU factors with complex128 CSC `values`.
    fn refactor(&mut self, values: PyReadonlyArray1<'_, Complex64>) -> PyResult<()> {
        self.solver.factor_c128(values.as_slice()?).map_err(error)
    }
    /// Release the CPU buffers and the analysis copy this factor owns.
    fn destroy(&mut self) {
        self.solver.destroy();
    }
    /// Solve complex128 vector `rhs` and return a NumPy vector owned by `py`.
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        rhs: PyReadonlyArray1<'_, Complex64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        Ok(self
            .solver
            .solve_c128(rhs.as_slice()?)
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
        rhs: PyReadonlyArray2<'_, Complex64>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let dims = rhs.as_array().raw_dim();
        let (nrhs, n) = (dims[0], dims[1]);
        let out = self
            .solver
            .solve_many_c128(rhs.as_slice()?, nrhs)
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
    /// GPU command time in seconds; always zero for CPU numerics.
    #[getter]
    fn factor_device_seconds(&self) -> f64 {
        self.solver.factor_device_seconds()
    }
    /// Number of dead pivots rescued by the latest CPU factorization.
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
