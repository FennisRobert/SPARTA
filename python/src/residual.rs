//! The residual product of iterative refinement, read straight from the
//! matrix's canonical CSC storage.
//!
//! Iterative refinement measures its residual against the original matrix in
//! double precision, and that product is the only part of a solve that is not
//! on the GPU. It stops being negligible as soon as there are several
//! right-hand sides: the GPU batches them into one triangular solve, while
//! SciPy's product is one thread and scales linearly with their number, so
//! past about eight ports it is the larger half of the solve.
//!
//! The product reads the CSC arrays the solver already keeps, not a CSR copy
//! of its own, so the only extra storage is a transposed index: for each row,
//! the columns of its entries and their slots in the CSC values. For LDL^T the
//! CSC holds the lower triangle; column `i` of it is row `i` of the upper
//! triangle, so the transposed index needs only the strict lower triangle. For
//! LU the transposed index covers every entry.
//!
//! Rows are split into blocks and each worker owns its block's output rows
//! outright, so nothing is shared and no accumulation is atomic. Within a row
//! every right-hand side is advanced together, so each entry is read once for
//! all of them. The accumulation order is fixed per row, so the result does
//! not depend on the worker count.
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

/// Rows one worker takes at a time. Large enough that the per-task overhead
/// disappears against a row's own work, small enough that the last worker to
/// finish is not holding up the rest.
const ROW_BLOCK: usize = 512;

/// The matrix times the dense (n, k) C-order block `block`.
///
/// - `indptr`, `indices`, `values`: the canonical CSC matrix, the lower
///   triangle when `symmetric` and every entry otherwise.
/// - `t_indptr`, `t_indices`, `t_source`: the transposed index, per row the
///   columns of its entries and their slots in `values`; strict lower
///   triangle only when `symmetric`.
///
/// The index arrays are all int32 or all int64. `values` and `block` are both
/// float64 or both complex128; a complex block against a real matrix is passed
/// as its float64 view, a real block with the real and imaginary parts as
/// separate columns.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _residual_product<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    t_indptr: &Bound<'py, PyAny>,
    t_indices: &Bound<'py, PyAny>,
    t_source: &Bound<'py, PyAny>,
    symmetric: bool,
    block: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! typed {
        ($i:ty, $t:ty) => {
            if let (Ok(p), Ok(r), Ok(tp), Ok(tr), Ok(ts), Ok(v), Ok(b)) = (
                indptr.downcast::<PyArray1<$i>>(),
                indices.downcast::<PyArray1<$i>>(),
                t_indptr.downcast::<PyArray1<$i>>(),
                t_indices.downcast::<PyArray1<$i>>(),
                t_source.downcast::<PyArray1<$i>>(),
                values.downcast::<PyArray1<$t>>(),
                block.downcast::<PyArray2<$t>>(),
            ) {
                let read = |a: &Bound<'py, PyArray1<$i>>| -> PyReadonlyArray1<'py, $i> { a.readonly() };
                let (p, r, tp, tr, ts) = (read(p), read(r), read(tp), read(tr), read(ts));
                let (v, b) = (v.readonly(), b.readonly());
                let csc = Pattern { indptr: p.as_slice()?, indices: r.as_slice()? };
                let transposed = Transposed {
                    indptr: tp.as_slice()?,
                    indices: tr.as_slice()?,
                    source: ts.as_slice()?,
                };
                let out = product(py, csc, transposed, v.as_slice()?, symmetric, b.as_array())?;
                return Ok(out.into_pyarray_bound(py).into_any());
            }
        };
    }
    typed!(i32, f64);
    typed!(i32, Complex64);
    typed!(i64, f64);
    typed!(i64, Complex64);
    Err(PyValueError::new_err(
        "index arrays must all be int32 or all int64, and values and block both float64 or both complex128",
    ))
}

/// One CSC pattern: column pointers and row indices.
#[derive(Clone, Copy)]
struct Pattern<'a, I> {
    indptr: &'a [I],
    indices: &'a [I],
}

/// The transposed index: per row, the columns of its entries and their slots
/// in the CSC values.
#[derive(Clone, Copy)]
struct Transposed<'a, I> {
    indptr: &'a [I],
    indices: &'a [I],
    source: &'a [I],
}

/// An index type the product reads.
trait Index: Copy + Send + Sync {
    fn at(self) -> usize;
    /// Whether the value is negative, which no valid index is.
    fn negative(self) -> bool;
}
impl Index for i32 {
    #[inline(always)]
    fn at(self) -> usize {
        self as usize
    }
    fn negative(self) -> bool {
        self < 0
    }
}
impl Index for i64 {
    #[inline(always)]
    fn at(self) -> usize {
        self as usize
    }
    fn negative(self) -> bool {
        self < 0
    }
}

/// Reject index arrays that would read outside the values, the block or each
/// other, so the product itself can index without checks failing midway.
fn check<I: Index>(csc: Pattern<'_, I>, transposed: Transposed<'_, I>, values: usize, n: usize, symmetric: bool) -> PyResult<()> {
    let bad = |what: &str| Err(PyValueError::new_err(format!("{what} does not describe the block")));
    if transposed.indptr.len() != n + 1 || (symmetric && csc.indptr.len() != n + 1) {
        return bad("a row or column pointer");
    }
    let ends = |p: &[I], len: usize| p.first().is_some_and(|f| f.at() == 0 && !f.negative())
        && p.windows(2).all(|w| !w[1].negative() && w[0].at() <= w[1].at())
        && p.last().is_some_and(|l| l.at() <= len);
    if !ends(transposed.indptr, transposed.indices.len().min(transposed.source.len())) {
        return bad("the transposed row pointer");
    }
    let within = |a: &[I], limit: usize| a.iter().all(|v| !v.negative() && v.at() < limit);
    if !within(transposed.indices, n) || !within(transposed.source, values) {
        return bad("the transposed index");
    }
    if symmetric {
        if !ends(csc.indptr, csc.indices.len().min(values)) {
            return bad("the column pointer");
        }
        if !within(&csc.indices[..csc.indptr[n].at()], n) {
            return bad("a row index");
        }
    }
    Ok(())
}

/// The product itself; see [`_residual_product`].
fn product<I: Index, T>(
    py: Python<'_>,
    csc: Pattern<'_, I>,
    transposed: Transposed<'_, I>,
    values: &[T],
    symmetric: bool,
    block: numpy::ndarray::ArrayView2<'_, T>,
) -> PyResult<numpy::ndarray::Array2<T>>
where
    T: Copy + Default + Send + Sync + std::ops::Mul<Output = T> + std::ops::AddAssign,
{
    let (n, k) = (block.shape()[0], block.shape()[1]);
    let x = block
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("the right-hand side block must be C-contiguous"))?;
    check(csc, transposed, values.len(), n, symmetric)?;
    let mut out = vec![T::default(); n * k];
    if k == 0 {
        return numpy::ndarray::Array2::from_shape_vec((n, 0), out)
            .map_err(|e| PyValueError::new_err(e.to_string()));
    }
    // The GIL is not needed by any of this, and holding it would serialise the
    // workers against anything else the interpreter wants to do.
    py.allow_threads(|| {
        out.par_chunks_mut(ROW_BLOCK * k)
            .enumerate()
            .for_each(|(chunk, dst)| {
                let first = chunk * ROW_BLOCK;
                for (offset, row) in dst.chunks_mut(k).enumerate() {
                    let i = first + offset;
                    // Row i of the upper triangle is column i of the lower.
                    if symmetric {
                        let span = csc.indptr[i].at()..csc.indptr[i + 1].at();
                        for (&value, r) in values[span.clone()].iter().zip(&csc.indices[span]) {
                            let column = r.at() * k;
                            for c in 0..k {
                                row[c] += value * x[column + c];
                            }
                        }
                    }
                    for t in transposed.indptr[i].at()..transposed.indptr[i + 1].at() {
                        let value = values[transposed.source[t].at()];
                        let column = transposed.indices[t].at() * k;
                        for c in 0..k {
                            row[c] += value * x[column + c];
                        }
                    }
                }
            });
    });
    numpy::ndarray::Array2::from_shape_vec((n, k), out).map_err(|e| PyValueError::new_err(e.to_string()))
}
