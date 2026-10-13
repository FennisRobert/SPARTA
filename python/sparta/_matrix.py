"""Input pipeline: sparse matrix → validated CSC storage → complex64 GPU values.

Solver calls _prepare_matrix during analysis and on replacement matrices.
Solver.factorize calls _gpu_values only after optional scaling; original precision
is retained in the CSC matrix for residuals and future factorization.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from ._symmetry import _is_symmetric
from ._types import MatrixType, Path

def _gpu_values(values: np.ndarray, name: str, safe: bool = True) -> np.ndarray:
    """Convert array values to complex64 for the GPU; name labels validation errors.

    safe=True rejects non-complex input, non-finite values and values that
    overflow complex64. safe=False performs only the conversion: an infinity or
    a NaN then reaches the factorization, where it propagates into the factors
    rather than raising here.
    """
    values = np.asarray(values)
    if not safe:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.ascontiguousarray(values, dtype=np.complex64)
    if values.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError(f"{name} must be complex64 or complex128")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.ascontiguousarray(values, dtype=np.complex64)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} overflows complex64 GPU precision")
    return converted


def _standardize(a: sparse.spmatrix, triangle: str, path: Path) -> sparse.csc_matrix:
    """Reshape sparse a into the CSC layout Rust consumes, checking nothing.

    Sums duplicates, sorts row indices within each column, and drops the upper
    triangle on the LDLT path. This is the structural half of
    :func:`_canonical_matrix`; the validating half lives there.
    """
    a = a.tocsc(copy=True)
    a.sum_duplicates()
    a.sort_indices()
    return a if path == Path.LU else sparse.tril(a, format="csc")


def _value_gather(original: sparse.spmatrix, canonical: sparse.csc_matrix,
                  triangle: str, path: Path) -> np.ndarray | None:
    """Index array `g` with `canonical.data == original.tocsc().data[g]`.

    Lets a later factorize reproduce the canonical values from a replacement
    matrix of the same layout with one gather, instead of repeating the whole
    input pipeline. Returns None when no such index array exists, which happens
    when the pattern carries duplicate entries for one position, so the caller
    must keep using the checked path.
    """
    probe = original.tocsc(copy=True)
    stored = probe.nnz
    # Mark each input entry with its own 1-based position, then push the marks
    # through the same reshaping. Summed duplicates and fabricated entries both
    # land outside 1..stored and are rejected below.
    marked = sparse.csc_matrix((np.arange(1, stored + 1, dtype=np.float64),
                                probe.indices, probe.indptr), shape=probe.shape)
    produced = _standardize(marked, triangle, path)
    g = produced.data
    if (produced.nnz != canonical.nnz
            or not np.array_equal(produced.indptr, canonical.indptr)
            or not np.array_equal(produced.indices, canonical.indices)
            or not np.array_equal(g, np.round(g))
            or g.min(initial=1.0) < 1.0
            or g.max(initial=1.0) > stored):
        return None
    return g.astype(np.intp) - 1

def _canonical_matrix(a: sparse.spmatrix, triangle: str, path: Path) -> sparse.csc_matrix:
    """Copy sparse a into the exact CSC layout consumed by Rust's symbolic plan.

    a contains complex64/complex128 values, promoted by _prepare_matrix if real.
    triangle specifies full or lower input; path is a resolved LU or LDLT enum.
    Returns sorted, duplicate-free CSC: all entries for LU, lower triangle for
    LDLT. Keeps explicit zeros because their positions belong to the pattern.
    Never modifies the caller's matrix and never casts its values to float32.
    """
    if not sparse.issparse(a):
        raise TypeError("A must be a SciPy sparse matrix")
    # Copy before sorting or summing: the caller retains its original storage.
    a = a.tocsc(copy=True)
    a.check_format(full_check=True)
    if a.shape[0] != a.shape[1] or not a.shape[0]:
        raise ValueError("A must be nonempty and square")
    if a.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError("A must be complex64 or complex128")
    if not np.isfinite(a.data).all():
        raise ValueError("A must contain finite values")
    # Rust requires sorted, unique row indices within each column.
    a.sum_duplicates()
    a.sort_indices()
    if not np.isfinite(a.data).all():
        raise ValueError("A contains nonfinite values after summing duplicates")
    # LU stores both sides; LDLT stores lower values and reconstructs the upper.
    if path == Path.LU:
        if triangle != "full":
            raise ValueError("path='lu' needs the whole matrix: pass triangle='full'")
        return a
    if triangle == "full":
        if not _is_symmetric(a):
            raise ValueError("A must be complex-symmetric within sparta.EPS, not Hermitian; supply both triangles")
    return sparse.tril(a, format="csc")


def _resolve_path(a: sparse.spmatrix, path: Path, triangle: str) -> Path:
    """Turn path='auto' into a concrete factorization.

    Only a full matrix can be unsymmetric as far as this layer can tell -- a
    lower triangle carries no evidence either way, and reading it as anything
    but complex-symmetric would be a guess.
    """
    if path != Path.AUTO:
        return path
    if triangle != "full":
        return Path.LDLT
    return Path.LDLT if _is_symmetric(a) else Path.LU


def _prepare_matrix(a: sparse.spmatrix, mtype: MatrixType, path: Path, triangle: str) -> tuple[sparse.csc_matrix, Path]:
    """Validate sparse matrix a against its declared type and select a GPU path.

    triangle specifies full or lower storage; path may explicitly override AUTO.
    Real float inputs are promoted to matching-width complex storage. Returns
    canonical CSC values and the selected Path, ready for symbolic analysis.
    """
    _validate_choices(mtype, path)
    if not sparse.issparse(a):
        raise TypeError('A must be a SciPy sparse matrix')
    if mtype.value.startswith('real_') and np.iscomplexobj(a.data) and np.any(a.data.imag):
        raise ValueError('Declared real matrix contains imaginary values')
    # Promotion preserves the input precision; GPU conversion happens later.
    if a.dtype in (np.dtype('float32'), np.dtype('float64')):
        a = a.astype(np.complex64 if a.dtype == np.float32 else np.complex128)
    if mtype == MatrixType.AUTO:
        selected = _resolve_path(a, path, triangle)
        return _canonical_matrix(a, triangle, selected), selected
    symmetric = mtype in (MatrixType.REAL_SYMMETRIC, MatrixType.COMPLEX_SYMMETRIC)
    if not symmetric and triangle != 'full':
        raise ValueError('This matrix type requires triangle="full"')
    # Validate the declared structure before discarding any upper-triangle values.
    if triangle == 'full':
        c = a.tocsc(copy=True)
        c.sum_duplicates()
        if symmetric and not _is_symmetric(c):
            raise ValueError('Declared symmetric matrix differs from its transpose beyond sparta.EPS')
        if mtype == MatrixType.HERMITIAN and not _is_symmetric(c, hermitian=True):
            raise ValueError('Declared Hermitian matrix differs from its conjugate transpose beyond sparta.EPS')
        if 'structurally' in mtype.value:
            c.data = np.ones(c.nnz)
            if np.any((c-c.T).data != 0):
                raise ValueError('Declared structural symmetry does not match stored pattern')
    selected = path
    if selected == Path.AUTO:
        selected = Path.LDLT if symmetric else Path.LU
    return _canonical_matrix(a, triangle, selected), selected


def _validate_choices(mtype: MatrixType, path: Path) -> None:
    """Reject reserved enum choices before allocating or inspecting matrix storage."""
    if path not in (Path.AUTO, Path.LU, Path.LDLT):
        raise NotImplementedError(f"Factorization {path.value} is not implemented")
    supported = (MatrixType.AUTO, MatrixType.REAL_SYMMETRIC,
                 MatrixType.REAL_UNSYMMETRIC, MatrixType.REAL_STRUCTURALLY_SYMMETRIC,
                 MatrixType.COMPLEX_SYMMETRIC, MatrixType.COMPLEX_UNSYMMETRIC,
                 MatrixType.COMPLEX_STRUCTURALLY_SYMMETRIC, MatrixType.HERMITIAN)
    if mtype not in supported:
        raise NotImplementedError(f"Matrix type {mtype.value} is not implemented")
