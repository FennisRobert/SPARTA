"""Diagonal equilibration without changing the sparse storage pattern."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csc_matrix


def _equilibrate(pattern: csc_matrix, values: np.ndarray, symmetric: bool
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance real or complex CSC values in five infinity-norm passes.

    pattern supplies row/column indices; symmetric selects D A D scaling for
    lower-triangle storage. Return scaled values, float64 for real input and
    complex128 for complex input, and positive real
    row/column multipliers. Empty rows/columns retain unit multipliers.
    """
    rows = pattern.indices
    cols = np.repeat(np.arange(pattern.shape[1]), np.diff(pattern.indptr))
    # Group row entries once; each pass can then use contiguous reductions.
    ordering = csc_matrix((np.arange(len(values)), rows, pattern.indptr), shape=pattern.shape).tocsr()
    row_nonempty = np.flatnonzero(np.diff(ordering.indptr))
    col_nonempty = np.flatnonzero(np.diff(pattern.indptr))
    data = np.array(values, dtype=np.result_type(values.dtype, np.float64), copy=True)
    left = np.ones(pattern.shape[0])
    right = np.ones(pattern.shape[1])
    for _ in range(5):
        magnitude = np.abs(data)
        rowmax, colmax = np.zeros_like(left), np.zeros_like(right)
        rowmax[row_nonempty] = np.maximum.reduceat(magnitude[ordering.data], ordering.indptr[row_nonempty])
        colmax[col_nonempty] = np.maximum.reduceat(magnitude, pattern.indptr[col_nonempty])
        if symmetric:
            rowmax = np.maximum(rowmax, colmax)
            colmax = rowmax
        r = 1 / np.sqrt(np.where(rowmax > 0, rowmax, 1))
        c = 1 / np.sqrt(np.where(colmax > 0, colmax, 1))
        data *= r[rows]
        data *= c[cols]
        left *= r
        right *= c
    if not all(np.isfinite(x).all() for x in (data, left, right)):
        raise ValueError("equilibration produced nonfinite values")
    return data, left, right
