//! Sparse right-hand sides, for [`Solver::port_matrix`](crate::Solver::port_matrix)
//! and [`Solver::solve_sparse`](crate::Solver::solve_sparse).
use crate::scalar::Scalar;
use crate::{Analysis, Error, Result};

/// An `n` by `ncols` block of right-hand sides stored by columns: column `c`
/// holds rows `rows[col_ptr[c]..col_ptr[c + 1]]` with the matching `values`.
///
/// Rows are in the matrix's own (unpermuted) numbering. Within a column they
/// need not be sorted, and a row listed more than once counts with the sum of
/// its values. A column with no entries is a zero right-hand side.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseRhs<T> {
    n: usize,
    col_ptr: Vec<usize>,
    rows: Vec<usize>,
    values: Vec<T>,
}

impl<T: Scalar> SparseRhs<T> {
    /// Check and take a column-compressed block: `col_ptr` holds `ncols + 1`
    /// nondecreasing offsets from zero to `rows.len()`, every row is below
    /// `n`, and every value is finite.
    pub fn new(n: usize, col_ptr: Vec<usize>, rows: Vec<usize>, values: Vec<T>) -> Result<Self> {
        if n == 0 {
            return Err(Error::Invalid(
                "sparse right-hand side of dimension zero".into(),
            ));
        }
        if col_ptr.first() != Some(&0)
            || col_ptr.last() != Some(&rows.len())
            || col_ptr.windows(2).any(|w| w[0] > w[1])
        {
            return Err(Error::Invalid(
                "sparse right-hand side: col_ptr must rise from 0 to the number of entries".into(),
            ));
        }
        if rows.len() != values.len() {
            return Err(Error::Invalid(
                "sparse right-hand side: one value per row index required".into(),
            ));
        }
        if let Some(&row) = rows.iter().find(|&&row| row >= n) {
            return Err(Error::Invalid(format!(
                "sparse right-hand side: row {row} outside dimension {n}"
            )));
        }
        if values.iter().any(|v| !v.finite()) {
            return Err(Error::Invalid(
                "sparse right-hand side: finite values required".into(),
            ));
        }
        Ok(Self {
            n,
            col_ptr,
            rows,
            values,
        })
    }

    /// Build a block from `(row, value)` lists, one per column.
    pub fn from_columns(n: usize, columns: &[Vec<(usize, T)>]) -> Result<Self> {
        let mut col_ptr = Vec::with_capacity(columns.len() + 1);
        col_ptr.push(0);
        let (mut rows, mut values) = (Vec::new(), Vec::new());
        for column in columns {
            for &(row, value) in column {
                rows.push(row);
                values.push(value);
            }
            col_ptr.push(rows.len());
        }
        Self::new(n, col_ptr, rows, values)
    }

    /// The dimension of the system the columns belong to.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The number of right-hand sides.
    pub fn ncols(&self) -> usize {
        self.col_ptr.len() - 1
    }

    /// The number of stored entries, duplicates included.
    pub fn nnz(&self) -> usize {
        self.rows.len()
    }

    /// Column `c`'s rows and values.
    pub fn column(&self, c: usize) -> (&[usize], &[T]) {
        let range = self.col_ptr[c]..self.col_ptr[c + 1];
        (&self.rows[range.clone()], &self.values[range])
    }

    /// The column offsets, row indices and values, as given.
    pub fn parts(&self) -> (&[usize], &[usize], &[T]) {
        (&self.col_ptr, &self.rows, &self.values)
    }

    /// The block as dense columns packed end to end, the layout
    /// [`Solver::solve_values`](crate::Solver::solve_values) takes.
    pub fn to_dense(&self) -> Vec<T> {
        let mut out = vec![T::default(); self.n * self.ncols()];
        for c in 0..self.ncols() {
            let (rows, values) = self.column(c);
            for (&row, &value) in rows.iter().zip(values) {
                out[c * self.n + row] += value;
            }
        }
        out
    }

    /// `B^T X` for `k` dense columns `x` packed end to end, as an `ncols` by
    /// `k` column-major block. Plain transpose: nothing is conjugated.
    pub fn transpose_times(&self, x: &[T], k: usize) -> Result<Vec<T>> {
        if self.n.checked_mul(k) != Some(x.len()) {
            return Err(Error::Invalid(
                "transpose_times: k columns of the block's dimension required".into(),
            ));
        }
        let p = self.ncols();
        let mut out = vec![T::default(); p * k];
        for j in 0..k {
            let column = &x[j * self.n..(j + 1) * self.n];
            for c in 0..p {
                let (rows, values) = self.column(c);
                let mut sum = T::default();
                for (&row, &value) in rows.iter().zip(values) {
                    sum += value * column[row];
                }
                out[j * p + c] = sum;
            }
        }
        Ok(out)
    }
}

/// How much of a CPU `LDL^T` factor a right-hand side reaches: the fronts on
/// the paths from its nonzeros' fronts to the root of the assembly tree, which
/// are the only fronts its forward sweep touches. See
/// [`Solver::sparse_reach`](crate::Solver::sparse_reach).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseReach {
    /// Fronts reached.
    pub fronts: usize,
    /// Pivot columns of those fronts.
    pub columns: usize,
    /// Stored factor entries of those fronts, `nr * nc` each.
    pub factor_entries: usize,
    /// Fronts in the whole factor.
    pub total_fronts: usize,
    /// Stored factor entries of the whole factor.
    pub total_factor_entries: usize,
}

/// The assembly tree of a plan, as the sparse solves walk it.
///
/// The CPU `LDL^T` factor pivots inside each front's fully summed block and
/// never delays a pivot to the parent, so the factored fronts are exactly the
/// analyzed ones and the reach can be taken from the plan.
pub(crate) struct Tree {
    /// Each front's parent, `u32::MAX` at a root.
    parent: Vec<u32>,
    /// The front owning each permuted unknown.
    pub owner: Vec<u32>,
    /// `iperm[old] = new`, the inverse of the plan's fill-reducing order.
    pub iperm: Vec<u32>,
}

impl Tree {
    pub(crate) fn new(plan: &Analysis) -> Self {
        let mut parent = vec![u32::MAX; plan.nodes.len()];
        let mut owner = vec![0; plan.n];
        for (s, node) in plan.nodes.iter().enumerate() {
            let at = node.children as usize;
            for child in &plan.children[at..at + node.count as usize] {
                parent[child.node as usize] = s as u32;
            }
            owner[node.first as usize..(node.first + node.nc) as usize].fill(s as u32);
        }
        let mut iperm = vec![0; plan.n];
        for (new, &old) in plan.perm.iter().enumerate() {
            iperm[old as usize] = new as u32;
        }
        Self {
            parent,
            owner,
            iperm,
        }
    }

    /// The fronts on the paths from `starts` to their roots, ascending. Fronts
    /// are numbered children first, so this is also an order in which every
    /// front comes after its children. `mark` holds one `false` per front and
    /// is left that way.
    pub(crate) fn reach(
        &self,
        starts: impl IntoIterator<Item = u32>,
        mark: &mut [bool],
    ) -> Vec<u32> {
        let mut out = Vec::new();
        for start in starts {
            let mut s = start;
            while s != u32::MAX && !mark[s as usize] {
                mark[s as usize] = true;
                out.push(s);
                s = self.parent[s as usize];
            }
        }
        for &s in &out {
            mark[s as usize] = false;
        }
        out.sort_unstable();
        out
    }

    /// The size of the reach `fronts` against the whole factor of `plan`.
    pub(crate) fn summary(plan: &Analysis, fronts: &[u32]) -> SparseReach {
        let entries = |s: &u32| {
            let node = plan.nodes[*s as usize];
            node.nr as usize * node.nc as usize
        };
        SparseReach {
            fronts: fronts.len(),
            columns: fronts
                .iter()
                .map(|&s| plan.nodes[s as usize].nc as usize)
                .sum(),
            factor_entries: fronts.iter().map(entries).sum(),
            total_fronts: plan.nodes.len(),
            total_factor_entries: plan
                .nodes
                .iter()
                .map(|n| n.nr as usize * n.nc as usize)
                .sum(),
        }
    }
}

/// [`Solver::sparse_reach`](crate::Solver::sparse_reach) on a plan: the reach
/// of one right-hand side with nonzeros at `rows`.
pub(crate) fn reach_of(plan: &Analysis, rows: &[usize]) -> Result<SparseReach> {
    if plan.kind != crate::Kind::Ldlt {
        return Err(Error::Unsupported(
            "sparse right-hand sides need an LDL^T plan".into(),
        ));
    }
    if let Some(&row) = rows.iter().find(|&&row| row >= plan.n) {
        return Err(Error::Invalid(format!(
            "sparse_reach: row {row} outside dimension {}",
            plan.n
        )));
    }
    let tree = Tree::new(plan);
    let mut mark = vec![false; plan.nodes.len()];
    let starts = rows.iter().map(|&row| tree.owner[tree.iperm[row] as usize]);
    Ok(Tree::summary(plan, &tree.reach(starts, &mut mark)))
}
