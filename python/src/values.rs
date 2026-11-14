//! NumPy arrays of any of the four value types at the solver boundary.
//!
//! The array's dtype picks the numerics: `float32`, `float64`, `complex64` or
//! `complex128`, each passed to the solver without conversion.
use crate::error;
use num_complex::{Complex32, Complex64};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
use sparta::Solver;

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

/// Sparse columns from CSC arrays: `ptr` and `rows` as `uintp`, `values` of
/// one of the four value types, which picks the numerics as in `solve`.
macro_rules! with_columns {
    ($n:expr, $ptr:expr, $rows:expr, $values:expr, |$cols:ident : $t:ty| $body:expr) => {{
        let ptr = $ptr.readonly();
        let rows = $rows.readonly();
        if let Ok(array) = $values.downcast::<PyArray1<$t>>() {
            let values = array.readonly();
            let $cols = sparta::SparseColumns::new($n, ptr.as_slice()?, rows.as_slice()?, values.as_slice()?)
                .map_err(error)?;
            Some($body)
        } else {
            None
        }
    }};
}

/// Solve the sparse columns `(ptr, rows, values)` of length `n`, returning
/// a C-contiguous (ncols, n) block of their dtype, one solution per row.
pub(crate) fn solve_sparse<'py>(
    py: Python<'py>,
    solver: &mut Solver,
    n: usize,
    ptr: &Bound<'py, PyArray1<usize>>,
    rows: &Bound<'py, PyArray1<usize>>,
    values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! typed {
        ($t:ty) => {
            if let Some(out) = with_columns!(n, ptr, rows, values, |b: $t| {
                let x = solver.solve_sparse(&b).map_err(error)?;
                let block = numpy::ndarray::Array2::from_shape_vec((b.ncols(), n), x)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                block.into_pyarray_bound(py).into_any()
            }) {
                return Ok(out);
            }
        };
    }
    typed!(f32);
    typed!(f64);
    typed!(Complex32);
    typed!(Complex64);
    Err(PyTypeError::new_err(format!("sparse right-hand side values must be {TYPES}")))
}

/// The projection `W^T A^-1 B` for sparse columns `b` and `w` (or `b` itself
/// when `w` is None), as a C-contiguous (b columns, w columns) block: row `j`
/// is the projection of the solution for column `j` of `b`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project<'py>(
    py: Python<'py>,
    solver: &mut Solver,
    n: usize,
    ptr: &Bound<'py, PyArray1<usize>>,
    rows: &Bound<'py, PyArray1<usize>>,
    values: &Bound<'py, PyAny>,
    w: Option<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>, Bound<'py, PyAny>)>,
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! typed {
        ($t:ty) => {
            if let Some(out) = with_columns!(n, ptr, rows, values, |b: $t| {
                let z = match &w {
                    None => solver.solve_projected(&b, None).map_err(error)?,
                    Some((wp, wr, wv)) => match with_columns!(n, wp, wr, wv, |w: $t| {
                        solver.solve_projected(&b, Some(&w)).map_err(error)?
                    }) {
                        Some(z) => z,
                        None => return Err(PyTypeError::new_err("the projection must have the dtype of the right-hand side")),
                    },
                };
                let q = z.len() / b.ncols().max(1);
                let block = numpy::ndarray::Array2::from_shape_vec((b.ncols(), q), z)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                block.into_pyarray_bound(py).into_any()
            }) {
                return Ok(out);
            }
        };
    }
    typed!(f32);
    typed!(f64);
    typed!(Complex32);
    typed!(Complex64);
    Err(PyTypeError::new_err(format!("sparse right-hand side values must be {TYPES}")))
}
