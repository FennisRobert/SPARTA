//! NumPy arrays of any of the four value types at the solver boundary.
//!
//! The array's dtype picks the numerics: `float32`, `float64`, `complex64` or
//! `complex128`, each passed to the solver without conversion.
use crate::error;
use num_complex::{Complex32, Complex64};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
use sparta::{Scalar, Solver, SparseRhs};

const TYPES: &str = "a contiguous float32, float64, complex64 or complex128 array";

/// Factor `solver` with the CSC values in the 1-D array `values`.
pub(crate) fn factor(solver: &mut Solver, values: &Bound<'_, PyAny>) -> PyResult<()> {
    macro_rules! typed {
        ($t:ty) => {
            if let Ok(array) = values.downcast::<PyArray1<$t>>() {
                return solver.factor_values(array.readonly().as_slice()?).map_err(error);
            }
        };
    }
    typed!(f32);
    typed!(f64);
    typed!(Complex32);
    typed!(Complex64);
    Err(PyTypeError::new_err(format!("matrix values must be {TYPES}")))
}

/// Solve the 1-D right-hand side `rhs`, returning an array of its dtype.
pub(crate) fn solve<'py>(
    py: Python<'py>,
    solver: &mut Solver,
    rhs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! typed {
        ($t:ty) => {
            if let Ok(array) = rhs.downcast::<PyArray1<$t>>() {
                let x = solver.solve_values(array.readonly().as_slice()?, 1).map_err(error)?;
                return Ok(x.into_pyarray_bound(py).into_any());
            }
        };
    }
    typed!(f32);
    typed!(f64);
    typed!(Complex32);
    typed!(Complex64);
    Err(PyTypeError::new_err(format!("the right-hand side must be {TYPES}")))
}

/// Solve every row of the C-contiguous (nrhs, n) block `rhs` in one pass,
/// returning a block of the same shape and dtype.
pub(crate) fn solve_many<'py>(
    py: Python<'py>,
    solver: &mut Solver,
    rhs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! typed {
        ($t:ty) => {
            if let Ok(array) = rhs.downcast::<PyArray2<$t>>() {
                let (nrhs, n) = (array.shape()[0], array.shape()[1]);
                let x = solver
                    .solve_values(array.readonly().as_slice()?, nrhs)
                    .map_err(error)?;
                let block = numpy::ndarray::Array2::from_shape_vec((nrhs, n), x)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                return Ok(block.into_pyarray_bound(py).into_any());
            }
        };
    }
    typed!(f32);
    typed!(f64);
    typed!(Complex32);
    typed!(Complex64);
    Err(PyTypeError::new_err(format!("the right-hand side block must be {TYPES}")))
}

/// The sparse block `n` by `ptr.len() - 1` with CSC `ptr`, `rows` and the
/// values of the 1-D array `values`, in that array's dtype.
fn block<T: Scalar + numpy::Element>(
    n: usize,
    ptr: &PyReadonlyArray1<'_, usize>,
    rows: &PyReadonlyArray1<'_, usize>,
    values: &Bound<'_, PyArray1<T>>,
) -> PyResult<SparseRhs<T>> {
    SparseRhs::new(
        n,
        ptr.as_slice()?.to_vec(),
        rows.as_slice()?.to_vec(),
        values.readonly().as_slice()?.to_vec(),
    )
    .map_err(error)
}

/// The port matrix `B^T A^-1 B` of the CSC block `B` (`n` rows, columns
/// `ptr`, `rows`, `values`), as a (ncols, ncols) array of the values' dtype.
/// It is symmetric, so its storage order does not matter.
pub(crate) fn port_matrix<'py>(
    py: Python<'py>,
    solver: &Solver,
    n: usize,
    ptr: PyReadonlyArray1<'py, usize>,
    rows: PyReadonlyArray1<'py, usize>,
    values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! typed {
        ($t:ty) => {
            if let Ok(array) = values.downcast::<PyArray1<$t>>() {
                let b = block(n, &ptr, &rows, array)?;
                let p = b.ncols();
                let z = solver.port_matrix(&b).map_err(error)?;
                let z = numpy::ndarray::Array2::from_shape_vec((p, p), z)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                return Ok(z.into_pyarray_bound(py).into_any());
            }
        };
    }
    typed!(f32);
    typed!(f64);
    typed!(Complex32);
    typed!(Complex64);
    Err(PyTypeError::new_err(format!("the sparse block's values must be {TYPES}")))
}

/// Solve `A X = B` for the CSC block `B`, returning the solutions as a
/// C-contiguous (ncols, n) block of the values' dtype, the layout of
/// [`solve_many`].
pub(crate) fn solve_sparse<'py>(
    py: Python<'py>,
    solver: &Solver,
    n: usize,
    ptr: PyReadonlyArray1<'py, usize>,
    rows: PyReadonlyArray1<'py, usize>,
    values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! typed {
        ($t:ty) => {
            if let Ok(array) = values.downcast::<PyArray1<$t>>() {
                let b = block(n, &ptr, &rows, array)?;
                let p = b.ncols();
                let x = solver.solve_sparse(&b).map_err(error)?;
                let x = numpy::ndarray::Array2::from_shape_vec((p, n), x)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                return Ok(x.into_pyarray_bound(py).into_any());
            }
        };
    }
    typed!(f32);
    typed!(f64);
    typed!(Complex32);
    typed!(Complex64);
    Err(PyTypeError::new_err(format!("the sparse block's values must be {TYPES}")))
}

/// [`Solver::sparse_reach`] as (fronts, pivot columns, factor entries, total
/// fronts, total factor entries).
pub(crate) fn sparse_reach(
    solver: &Solver,
    rows: PyReadonlyArray1<'_, usize>,
) -> PyResult<(usize, usize, usize, usize, usize)> {
    let r = solver.sparse_reach(rows.as_slice()?).map_err(error)?;
    Ok((r.fronts, r.columns, r.factor_entries, r.total_fronts, r.total_factor_entries))
}
