//! The CPU factors behind the Python solver.
use crate::values;
use pyo3::prelude::*;
use sparta::Solver;

#[pyclass(name = "_CpuFactor", unsendable)]
pub(crate) struct PyCpuFactor {
    pub(crate) solver: Solver,
}
#[pymethods]
impl PyCpuFactor {
    /// Replace the CPU factors with CSC `values`, factored in their own dtype.
    fn refactor(&mut self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        values::factor(&mut self.solver, values)
    }
    /// Release the CPU buffers and the analysis copy this factor owns.
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
