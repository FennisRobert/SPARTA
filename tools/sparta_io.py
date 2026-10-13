"""Reader and writer for the SPARTA benchmark container.

`npz_to_sparta.py` documents the byte layout. The reader mirrors
`examples/common/mod.rs`, so a Python script can look at the same `A`, `b` and
`x` the Rust benchmarks read. `write` is the one place the layout is produced,
also for new systems.
"""
import numpy as np
import scipy.sparse as sp

MAGIC = b"SPARTA02"
FLAG_SOLUTION = 1
FLAG_FULL = 2
#: Largest |A - A^T| relative to max |A| still stored as symmetric. It matches
#: the sparta.EPS the Python API uses to choose between LDL^T and LU.
SYMMETRY_TOLERANCE = 1e-10


class System:
    """One assembled system: `A`, its right-hand sides, and its reference `x`.

    `full` is False when `A` holds only the lower triangle of a complex-
    symmetric matrix, in which case `matrix()` mirrors it back out.
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic not in (b"SPARTA01", b"SPARTA02"):
                raise ValueError(f"{path}: not a SPARTA benchmark file")
            version = int(magic[-1:])
            head = np.fromfile(f, dtype="<u8", count=4 if version >= 2 else 3)
            n, nnz = int(head[0]), int(head[1])
            # v1 had no port count: one right-hand side, lower triangle only.
            nrhs, flags = (int(head[2]), int(head[3])) if version >= 2 else (1, int(head[2]))
            k0 = float(np.fromfile(f, dtype="<f8", count=1)[0])
            indptr = np.fromfile(f, dtype="<u8", count=n + 1).astype(np.int64)
            indices = np.fromfile(f, dtype="<u8", count=nnz).astype(np.int64)
            data = np.fromfile(f, dtype="<c16", count=nnz)
            rhs = np.fromfile(f, dtype="<c16", count=n * nrhs).reshape(nrhs, n).T
            solution = (np.fromfile(f, dtype="<c16", count=n * nrhs).reshape(nrhs, n).T
                        if flags & FLAG_SOLUTION else None)
        self.path, self.n, self.nrhs, self.k0 = path, n, nrhs, k0
        self.full = bool(flags & FLAG_FULL)
        self.stored = sp.csc_matrix((data, indices, indptr), shape=(n, n))
        self.rhs, self.solution = rhs, solution

    def matrix(self):
        """`A` itself, mirrored out of the lower triangle when symmetric."""
        if self.full:
            return self.stored
        return self.stored + sp.tril(self.stored, -1).T


def canonical(a):
    """Return `a` as sorted, duplicate-free CSC, which `Plan::analyze` validates.

    a: any SciPy sparse matrix. Explicit zeros are kept, so a pattern shared
    across a sweep survives even where one snapshot's value is zero.
    """
    a = sp.csc_matrix(a)
    a.sum_duplicates()
    a.sort_indices()
    return a


def is_symmetric(a):
    """Return True when `a` equals its plain transpose to SYMMETRY_TOLERANCE.

    a: square sparse matrix. The transpose is not conjugated: complex-symmetric
    means A == A^T, and a Hermitian matrix with a complex diagonal or complex
    off-diagonal entries is not symmetric in this sense.

    The test is relative, not exact. Assembly leaves noise in the last bits --
    one measured export differs from its own transpose by 8.7e-19 against
    entries reaching 6.8e4, far below even complex128 resolution -- and an exact
    test would call that unsymmetric, double the stored size and send the
    benchmark down the LU path instead of the LDL^T path the model solves on.
    """
    difference = a - a.T
    scale = abs(a).max() if a.nnz else 0.0
    return difference.nnz == 0 or abs(difference).max() <= SYMMETRY_TOLERANCE*max(scale, 1.0)


def write(path, matrix, rhs, solution=None, k0=0.0, symmetric=None):
    """Write one system as a SPARTA02 container and return how it was stored.

    path: output file, conventionally ending in `.sparta`.
    matrix: square SciPy sparse matrix, real or complex, with both triangles
        present. Stored as complex128.
    rhs: right-hand sides, shape (n,) or (n, nrhs); one column per port.
    solution: reference solutions with the shape of `rhs`, or None. Should be
        accurate in complex128, since tools measure SPARTA's error against it.
    k0: free-space wavenumber in rad/m, or 0 when the system has none.
    symmetric: True stores the lower triangle only, False the full matrix, and
        None decides with `is_symmetric`. True on a matrix that is not
        symmetric raises ValueError instead of discarding its upper triangle.

    Returns "lower" or "full". Raises ValueError for a non-square matrix, a
    right-hand side or solution of the wrong shape, or a non-finite value.
    """
    a = canonical(matrix).astype(np.complex128)
    n = a.shape[0]
    if a.shape != (n, n) or n == 0:
        raise ValueError(f"matrix must be square and nonempty, got {a.shape}")
    rhs = _columns(rhs, n, "rhs")
    if solution is not None:
        solution = _columns(solution, n, "solution")
        if solution.shape != rhs.shape:
            raise ValueError(f"solution shape {solution.shape} differs from rhs {rhs.shape}")
    for name, values in (("matrix", a.data), ("rhs", rhs), ("solution", solution)):
        if values is not None and not np.isfinite(values).all():
            raise ValueError(f"{name} holds NaN or infinite values")
    detected = is_symmetric(a)
    if symmetric and not detected:
        raise ValueError("symmetric=True, but the matrix differs from its transpose")
    full = not detected if symmetric is None else not symmetric
    # The lower-triangle form is mirrored back by the reader, so a symmetric
    # matrix has to be cut down here; writing it whole under that flag would
    # read back as something that is not symmetric at all.
    stored = a if full else canonical(sp.tril(a))
    flags = (FLAG_SOLUTION if solution is not None else 0) | (FLAG_FULL if full else 0)
    with open(path, "wb") as out:
        out.write(MAGIC)
        np.array([n, stored.nnz, rhs.shape[1], flags], dtype="<u8").tofile(out)
        np.array([k0], dtype="<f8").tofile(out)
        stored.indptr.astype("<u8").tofile(out)
        stored.indices.astype("<u8").tofile(out)
        stored.data.astype("<c16").tofile(out)
        # Port-major, so a reader can take one port without a stride.
        np.ascontiguousarray(rhs.T).astype("<c16").tofile(out)
        if solution is not None:
            np.ascontiguousarray(solution.T).astype("<c16").tofile(out)
    return "full" if full else "lower"


def _columns(values, n, name):
    """Return `values` as a complex128 (n, k) array, or raise ValueError.

    values: array of shape (n,) or (n, k). name: label used in the error.
    """
    values = np.asarray(values, dtype=np.complex128)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != n or values.shape[1] == 0:
        raise ValueError(f"{name} must have shape ({n},) or ({n}, k), got {values.shape}")
    return values
