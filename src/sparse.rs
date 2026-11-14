//! Sparse right-hand sides and projections, at the public boundary.
use crate::scalar::Scalar;
use crate::{Error, Result};

/// A block of sparse columns of length `n`, in compressed sparse column form.
///
/// Column `j` holds the values `values[ptr[j]..ptr[j + 1]]` at the rows
/// `rows[ptr[j]..ptr[j + 1]]`, numbered as the analyzed matrix. Rows need not
/// be sorted, and a row listed twice in one column counts as the sum. A column
/// with no entries is zero.
///
/// Used for right-hand sides that touch few unknowns, such as port
/// excitations, by [`crate::Solver::solve_sparse`], and for both sides of
/// [`crate::Solver::solve_projected`].
#[derive(Clone, Copy, Debug)]
pub struct SparseColumns<'a, T> {
    n: usize,
    ptr: &'a [usize],
    rows: &'a [usize],
    values: &'a [T],
}

impl<'a, T: Scalar> SparseColumns<'a, T> {
    /// Check and wrap a block of sparse columns.
    ///
    /// Arguments:
    /// - `n`: the length of every column, the matrix dimension.
    /// - `ptr`: column starts, one more than the number of columns, from 0 up
    ///   to `rows.len()` and never decreasing.
    /// - `rows`: the row of each entry, each below `n`.
    /// - `values`: the value of each entry, finite, as many as `rows`.
    pub fn new(n: usize, ptr: &'a [usize], rows: &'a [usize], values: &'a [T]) -> Result<Self> {
        if ptr.is_empty() || ptr[0] != 0 || ptr.windows(2).any(|w| w[0] > w[1]) {
            return Err(Error::Invalid("sparse column starts must begin at 0 and never decrease".into()));
        }
        if ptr[ptr.len() - 1] != rows.len() || rows.len() != values.len() {
            return Err(Error::Invalid(format!(
                "sparse columns need as many rows and values as the last column start ({}), got {} rows and {} values",
                ptr[ptr.len() - 1],
                rows.len(),
                values.len()
            )));
        }
        if let Some(&row) = rows.iter().find(|&&r| r >= n) {
            return Err(Error::Invalid(format!("sparse column row {row} is outside the matrix of dimension {n}")));
        }
        if let Some(at) = values.iter().position(|v| !v.finite()) {
            return Err(Error::Invalid(format!("sparse column value {at} is not finite")));
        }
        Ok(Self { n, ptr, rows, values })
    }

    /// The length of every column.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The number of columns.
    pub fn ncols(&self) -> usize {
        self.ptr.len() - 1
    }

    pub(crate) fn ptr(&self) -> &'a [usize] {
        self.ptr
    }

    pub(crate) fn rows(&self) -> &'a [usize] {
        self.rows
    }

    pub(crate) fn values(&self) -> &'a [T] {
        self.values
    }

    /// The columns as a dense block, column `j` at `[j * n..(j + 1) * n]`.
    pub(crate) fn dense(&self) -> Vec<T> {
        let mut out = vec![T::from_c64(Default::default()); self.n * self.ncols()];
        for j in 0..self.ncols() {
            for e in self.ptr[j]..self.ptr[j + 1] {
                let at = j * self.n + self.rows[e];
                out[at] = T::from_c64(out[at].to_c64() + self.values[e].to_c64());
            }
        }
        out
    }

    /// `self^T x` for the dense block `x` of `nrhs` columns of length `n`:
    /// `ncols()` rows by `nrhs` columns, column-major.
    pub(crate) fn project<U: Scalar>(&self, x: &[U], nrhs: usize) -> Vec<U> {
        let q = self.ncols();
        let mut z = vec![U::from_c64(Default::default()); q * nrhs];
        for j in 0..nrhs {
            let column = &x[j * self.n..(j + 1) * self.n];
            for i in 0..q {
                let mut sum = num_complex::Complex64::new(0.0, 0.0);
                for e in self.ptr[i]..self.ptr[i + 1] {
                    sum += self.values[e].to_c64() * column[self.rows[e]].to_c64();
                }
                z[j * q + i] = U::from_c64(sum);
            }
        }
        z
    }
}
