//! Test-matrix generators for SPARTA's examples and benchmarks.
//!
//! A dev-only crate: a dev-dependency of the solver, so it is compiled for
//! tests, examples and benchmarks and never linked into the shipped library.

use num_complex::Complex64;

/// Element stride of each grid axis, for a column-major (`x` fastest) layout.
fn strides(dims: &[usize]) -> Vec<usize> {
    let mut stride = vec![1usize; dims.len()];
    for axis in 1..dims.len() {
        stride[axis] = stride[axis - 1] * dims[axis - 1];
    }
    stride
}

/// Position of node `node` along `axis`.
#[inline]
fn coord(node: usize, axis: usize, dims: &[usize], stride: &[usize]) -> usize {
    (node / stride[axis]) % dims[axis]
}

/// Time-harmonic curl-curl Maxwell operator on a structured grid.
///
/// ```text
/// A = curl curl - omega^2 M + i omega sigma M
/// ```
///
/// discretised through the vector identity `curl curl = L - grad div`, where
/// `L` is the component-wise vector Laplacian and `grad div = D^T D` for the
/// discrete divergence `D`. `M` is the lumped mass, so the identity.
///
/// The three terms are what make this a realistic stress test rather than a
/// Laplacian:
///
/// - `-omega^2 M` shifts the spectrum and makes the operator **indefinite**, so
///   the factorization has to pivot rather than run a plain Cholesky;
/// - `i omega sigma M` adds conductivity and makes it **complex symmetric** —
///   `A == A^T`, not Hermitian;
/// - `curl curl` is singular on discrete gradients, giving the gradient
///   near-null-space that characterises edge-element electromagnetics and
///   produces the dense fill SPARTA is built for.
///
/// Degrees of freedom are component-major: component `a` of node `i` is
/// `a * nodes + i`. `dims` is `[nx, ny]` or `[nx, ny, nz]`.
///
/// Returns the **lower triangle** as sorted, duplicate-free CSC:
/// `(ndof, col_ptr, row_idx, values)`.
pub fn curl_curl(
    dims: &[usize],
    omega: f64,
    sigma: f64,
) -> (usize, Vec<usize>, Vec<usize>, Vec<Complex64>) {
    let ndim = dims.len();
    let nodes: usize = dims.iter().product();
    let ndof = ndim * nodes;
    let stride = strides(dims);

    // Lower-triangle contributions as `(col, row, value)` with `row >= col`.
    // A plain vector sorted once beats a hash map here: the count is known
    // within a small factor, and the order is deterministic by construction.
    let mut entries: Vec<(usize, usize, f64)> =
        Vec::with_capacity(ndof * (2 * ndim + 1) + nodes * ndim * (2 * ndim - 1));

    // --- Vector Laplacian -------------------------------------------------
    //
    // Component-decoupled: each component carries the scalar Dirichlet
    // Laplacian over the grid. Every axis contributes 1 to a node's diagonal
    // per side — from a neighbour if there is one, from the Dirichlet ghost if
    // the node is on the boundary — so the diagonal is 2 * ndim throughout,
    // and each interior edge contributes -1 off the diagonal.
    for component in 0..ndim {
        let base = component * nodes;
        for node in 0..nodes {
            let row = base + node;
            for axis in 0..ndim {
                let at = coord(node, axis, dims, &stride);
                if at + 1 < dims[axis] {
                    let neighbour = row + stride[axis];
                    entries.push((row, row, 1.0));
                    entries.push((neighbour, neighbour, 1.0));
                    entries.push((row, neighbour, -1.0));
                } else {
                    entries.push((row, row, 1.0)); // ghost above
                }
                if at == 0 {
                    entries.push((row, row, 1.0)); // ghost below
                }
            }
        }
    }

    // --- minus grad div, as -D^T D ---------------------------------------
    //
    // Row `node` of the discrete divergence holds a central difference along
    // each axis: +1/2 at the neighbour above, -1/2 at the one below, with
    // boundary neighbours dropped. Subtracting `D^T D` means subtracting each
    // row's outer product with itself.
    let mut row_entries: Vec<(usize, f64)> = Vec::with_capacity(2 * ndim);
    for node in 0..nodes {
        row_entries.clear();
        for axis in 0..ndim {
            let at = coord(node, axis, dims, &stride);
            if at + 1 < dims[axis] {
                row_entries.push((axis * nodes + node + stride[axis], 0.5));
            }
            if at > 0 {
                row_entries.push((axis * nodes + node - stride[axis], -0.5));
            }
        }
        for (a, &(p, cp)) in row_entries.iter().enumerate() {
            entries.push((p, p, -cp * cp));
            for &(q, cq) in &row_entries[a + 1..] {
                let (row, col) = if p > q { (p, q) } else { (q, p) };
                entries.push((col, row, -cp * cq));
            }
        }
    }

    // --- assemble ---------------------------------------------------------
    entries.sort_unstable_by_key(|&(col, row, _)| (col, row));

    let shift = Complex64::new(-omega * omega, omega * sigma);
    let mut col_ptr = Vec::with_capacity(ndof + 1);
    let mut row_idx: Vec<usize> = Vec::new();
    let mut values: Vec<Complex64> = Vec::new();

    let mut at = 0usize;
    for col in 0..ndof {
        col_ptr.push(row_idx.len());

        // The diagonal is always present: every node carries the Laplacian's
        // 2 * ndim plus the shift, even where the real part cancels.
        let mut diagonal = shift;
        while at < entries.len() && entries[at].0 == col {
            let row = entries[at].1;
            let mut sum = 0.0;
            while at < entries.len() && entries[at].0 == col && entries[at].1 == row {
                sum += entries[at].2;
                at += 1;
            }
            if row == col {
                diagonal += sum;
            } else if sum != 0.0 {
                row_idx.push(row);
                values.push(Complex64::new(sum, 0.0));
            }
        }

        // Splice the diagonal in at the front: rows within a column ascend,
        // and the diagonal is the lowest row index in its own column.
        row_idx.insert(col_ptr[col], col);
        values.insert(col_ptr[col], diagonal);
    }
    col_ptr.push(row_idx.len());

    (ndof, col_ptr, row_idx, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_triangle_is_canonical_csc() {
        let (n, ptr, rows, vals) = curl_curl(&[5, 4, 3], 1.0, 0.1);
        assert_eq!(ptr.len(), n + 1);
        assert_eq!(ptr[0], 0);
        assert_eq!(*ptr.last().unwrap(), rows.len());
        assert_eq!(rows.len(), vals.len());
        for col in 0..n {
            let block = &rows[ptr[col]..ptr[col + 1]];
            assert_eq!(block.first(), Some(&col), "column {col} must open on its diagonal");
            assert!(block.windows(2).all(|w| w[0] < w[1]), "column {col} must ascend");
            assert!(block.iter().all(|&r| r >= col), "column {col} must be lower triangular");
            assert!(block.iter().all(|&r| r < n));
        }
    }

    #[test]
    fn is_complex_symmetric_and_indefinite() {
        // Non-zero conductivity gives a genuinely complex diagonal, and the
        // -omega^2 shift drives some real diagonal entries negative.
        let (_, ptr, _, vals) = curl_curl(&[4, 4, 4], 8.0, 0.5);
        let diagonal: Vec<_> = (0..ptr.len() - 1).map(|c| vals[ptr[c]]).collect();
        assert!(diagonal.iter().all(|d| d.im > 0.0), "conductivity term missing");
        assert!(diagonal.iter().any(|d| d.re < 0.0), "shift should make it indefinite");
    }
}
