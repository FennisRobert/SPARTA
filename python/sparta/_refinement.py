"""Complex128 residual correction using an existing complex64 factor."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, tril

try:
    from ._sparta import _residual_product
except ImportError:  # pragma: no cover - only when the extension predates it
    _residual_product = None


class RefinementStatus(str, Enum):
    """Reason residual correction stopped; convergence concerns residual, not x error."""
    CONVERGED = "converged"
    STAGNATED = "stagnated"
    ITERATION_LIMIT = "iteration_limit"
    NONFINITE = "nonfinite"
    #: Refinement was switched off, so the residual was measured once and the
    #: solve returned as it came off the factors.
    UNREFINED = "unrefined"


# A bare complex64 solve of a healthy system lands near 1e-5 relative residual,
# and the worst well-behaved fixture measured here reaches 1.6e-3 on a matrix
# whose condition number is about 2.4e9. Past this the factors are not merely
# inaccurate, they have stopped being a usable approximation of the matrix: the
# badly conditioned EMerge export that motivated this reads 5.4e-1, and no
# number of corrections moves it because the iteration does not contract.
SUSPECT_RESIDUAL = 1e-2


@dataclass(frozen=True)
class RefinementReport:
    """Per-RHS residual history and correction count, with dimensionless residuals."""
    status: RefinementStatus
    corrections: int
    initial_relative_residual: float
    relative_residual: float
    history: tuple[float, ...]
    error_estimate: float

    @property
    def estimated_relative_error(self) -> float:
        """Estimated relative error of the returned solution, or nan if unknown.

        The residual is not the error. Refinement stops on the residual, and on
        an ill-conditioned system the solution can be far less accurate than the
        residual suggests: on the measured em_m2_sym1, whose condition number is
        about 2.4e9, stopping at a relative residual of 4e-7 left a solution
        error near 8e-5, a factor of 1200.

        The last correction refinement applied is a direct measure of how wrong
        the iterate before it was, and so an estimate -- a generous one, by
        roughly the factor each step contracts by -- of what is left in the
        answer returned. Measured against a complex128 reference it tracked the
        true error to within about 15% while refinement was converging.

        It is nan when no correction was applied, and it understates once the
        history shows refinement has stagnated, because the corrections are then
        noise rather than error.
        """
        return self.error_estimate

    @property
    def converged(self) -> bool:
        """Return whether the requested relative-residual tolerance was reached."""
        return self.status == RefinementStatus.CONVERGED


# Corrections allowed when the caller asks for as many as it takes. Stagnation
# normally stops the loop long before this; the ceiling only bounds a pathological
# case that keeps improving by a hair every step.
UNBOUNDED_CEILING = 50


def _column_scale(x: np.ndarray) -> np.ndarray:
    """Return each column's largest real/imaginary component magnitude."""
    return np.maximum(np.abs(x.real).max(axis=0), np.abs(x.imag).max(axis=0))


class _Transposed:
    """Per row of a CSC matrix, the columns of its entries and their slots.

    Built once per pattern and shared by every _Operator on it. With
    symmetric=True the CSC is a lower triangle and only its strict lower part
    is indexed, because the triangle's own columns already give the upper
    part's rows. Indices are int32 when they fit, otherwise int64.
    """

    def __init__(self, a: csc_matrix, symmetric: bool) -> None:
        """Index the transposed pattern of canonical CSC a."""
        n, nnz = a.shape[0], a.nnz
        self.index = np.dtype(np.int32 if max(n, nnz) < 2**31 else np.int64)
        rows = np.ascontiguousarray(a.indices, dtype=self.index)
        columns = np.repeat(np.arange(n, dtype=self.index), np.diff(a.indptr))
        source = np.arange(nnz, dtype=self.index)
        if symmetric:
            off = rows != columns
            rows, columns, source = rows[off], columns[off], source[off]
        # Stable, so each row keeps its columns in ascending order.
        order = np.argsort(rows, kind='stable')
        self.indices = columns[order]
        self.source = source[order]
        self.indptr = np.zeros(n + 1, dtype=self.index)
        np.cumsum(np.bincount(rows, minlength=n), out=self.indptr[1:])


class _Operator:
    """The original matrix in double precision, applied to dense blocks of columns.

    Refinement measures its residual against this, and it is the only part of a
    solve that does not run on the GPU. That matters as soon as there are
    several right-hand sides: the GPU batches them into a single triangular
    solve, so its cost barely grows, while a single-threaded product grows with
    their number and overtakes it at about eight.

    So the product is handed to the compiled extension, which splits the rows
    across workers and advances every right-hand side together within a row.
    It reads the solver's own canonical CSC arrays plus a _Transposed index,
    rather than a CSR copy of the matrix. The accumulation order is fixed per
    row, so the result does not depend on how many workers ran. SciPy remains
    the fallback when the extension is unavailable.
    """

    def __init__(self, a: csc_matrix, symmetric: bool, transposed: _Transposed | None = None) -> None:
        """Wrap canonical CSC a: its lower triangle when symmetric, else all of it.

        transposed is a's _Transposed index, shared across factorizations of
        one pattern; None builds it. The arrays of a are used in place when
        they already have the precision and index type the product needs.
        """
        self.shape = a.shape
        self.symmetric = symmetric
        self.real = not np.iscomplexobj(a.data)
        self._transposed = _Transposed(a, symmetric) if transposed is None else transposed
        index = self._transposed.index
        self._indptr = np.ascontiguousarray(a.indptr, dtype=index)
        self._indices = np.ascontiguousarray(a.indices, dtype=index)
        self._values = np.ascontiguousarray(a.data, dtype=np.float64 if self.real else np.complex128)

    def _scipy(self) -> csc_matrix:
        """The represented matrix as SciPy CSC, for the fallback product."""
        a = csc_matrix((self._values, self._indices, self._indptr), shape=self.shape)
        return a + tril(a, -1, format='csc').T if self.symmetric else a

    def _multiply(self, x: np.ndarray) -> np.ndarray:
        """Return the matrix times a dense (n, k) block, in float64 when both
        are real and complex128 otherwise."""
        complex_x = np.iscomplexobj(x)
        block = np.ascontiguousarray(x, dtype=np.complex128 if complex_x or not self.real else np.float64)
        if _residual_product is None:
            matrix = self._scipy()
            return np.column_stack([matrix @ block[:, j] for j in range(block.shape[1])])
        t = self._transposed
        arrays = (self._indptr, self._indices, self._values, t.indptr, t.indices, t.source, self.symmetric)
        if self.real and complex_x:
            # A real matrix applies to the real and imaginary parts alike, so
            # the complex block goes in as its float64 view, twice as wide.
            return _residual_product(*arrays, block.view(np.float64)).view(np.complex128)
        return _residual_product(*arrays, block)

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Support `operator @ vector` for the one-off residuals verbose prints."""
        if np.ndim(x) == 1:
            return self._multiply(np.asarray(x)[:, None])[:, 0]
        return self._multiply(x)


def _apply(a: _Operator | csc_matrix | csr_matrix, x: np.ndarray) -> np.ndarray:
    """Multiply the original matrix by a dense (n, k) block.

    Threaded through the extension when `a` carries it. The SciPy path is kept
    for callers that still pass a bare matrix: its sparse-times-dense-matrix
    routine is slower than repeated matrix-vector products at the widths
    refinement uses, so it does those.
    """
    if isinstance(a, _Operator):
        return a._multiply(x)
    if x.shape[1] == 1:
        return (a @ x[:, 0])[:, None]
    return np.column_stack([a @ x[:, j] for j in range(x.shape[1])])


def _refine_block(a: csc_matrix | csr_matrix, b: np.ndarray,
                  solve: Callable[[np.ndarray], np.ndarray], steps: int, rtol: float,
                  minimize: bool = False, single: bool = True,
                  ) -> tuple[np.ndarray, tuple[RefinementReport, ...]]:
    """Refine every column of b together, one batched solve per correction.

    a is the original matrix, b an (n, k) right-hand-side block, and solve
    applies the factors to a whole block. single says the factors are single
    precision (the GPU's), so each right-hand side is narrowed to float32 or
    complex64 before solve; double precision factors (the CPU's) get it
    unrounded. The work is float64
    when a and b are both real, complex128 otherwise. steps is the
    maximum correction count or -1 for as many as rtol needs; minimize selects an
    optimal complex step length per column. Columns stop independently: once one
    converges or stalls it leaves the batch, so later corrections only carry the
    columns that still need them. Returns the block of best iterates and one
    report per column, in the caller's column order.
    """
    real = not np.iscomplexobj(b) and getattr(a, 'real', False) is True
    b = np.asarray(b, dtype=np.float64 if real else np.complex128)
    narrow = (np.float32 if real else np.complex64) if single else b.dtype
    n, k = b.shape
    scale = _column_scale(b)
    # A zero column has no scale to divide by and is measured absolutely.
    unit = np.where(scale > 0, scale, 1.0)
    divisor = np.linalg.norm(b/unit, axis=0)
    divisor = np.where(scale > 0, np.where(divisor > 0, divisor, 1.0), 1.0)

    def _correction(rhs: np.ndarray) -> np.ndarray:
        """Scale a block column by column, narrow it to the factors' precision and solve it."""
        size = _column_scale(rhs)
        by = np.where(size > 0, size, 1.0)
        out = np.asarray(solve(np.ascontiguousarray(rhs/by, dtype=narrow)), dtype=b.dtype)
        return out*by

    def _residual(x: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the residual block and its relative norm for the named columns."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            r = b[:, columns] - _apply(a, x)
            value = np.linalg.norm(r/unit[columns], axis=0)/divisor[columns]
        return r, value

    every = np.arange(k)
    x = _correction(b)
    r, best = _residual(x, every)
    if not np.isfinite(x).all() or not np.isfinite(best).all():
        raise FloatingPointError("Initial refined solve produced nonfinite output or residual")

    history = [[float(v)] for v in best]
    status = [RefinementStatus.CONVERGED if v <= rtol else RefinementStatus.ITERATION_LIMIT
              for v in best]
    attempted = [0]*k
    # The size of the last correction accepted, relative to the solution it was
    # applied to. See RefinementReport.estimated_relative_error.
    correction = [float('nan')]*k
    # Only the columns that have not settled stay in the batch.
    active = every[best > rtol]
    limit = UNBOUNDED_CEILING if steps < 0 else steps
    for step in range(1, limit + 1):
        if not active.size:
            break
        delta = _correction(r)
        if minimize:
            # One matvec for the whole block, then a step length per column.
            ad = _apply(a, delta)
            size = np.maximum(_column_scale(ad), _column_scale(r))
            safe = (size > 0) & np.isfinite(size)
            by = np.where(safe, size, 1.0)
            v, target = ad/by, r/by
            denominator = np.einsum('ij,ij->j', v.conj(), v).real
            usable = safe & (denominator > 0)
            alpha = np.where(usable, np.einsum('ij,ij->j', v.conj(), target)/np.where(denominator > 0, denominator, 1.0), 1.0)
            delta = delta*alpha
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = x[:, active] + delta
        new_r, value = _residual(candidate, active)
        keep = []
        for slot, column in enumerate(active):
            attempted[column] = step
            history[column].append(float(value[slot]))
            if not np.isfinite(candidate[:, slot]).all() or not np.isfinite(value[slot]):
                status[column] = RefinementStatus.NONFINITE
                continue
            if value[slot] >= best[column]:
                status[column] = RefinementStatus.STAGNATED
                continue
            x[:, column] = candidate[:, slot]
            size = float(np.linalg.norm(candidate[:, slot]))
            correction[column] = (float(np.linalg.norm(delta[:, slot]))/size
                                  if size > 0 else float('nan'))
            best[column] = value[slot]
            if value[slot] <= rtol:
                status[column] = RefinementStatus.CONVERGED
                continue
            status[column] = RefinementStatus.ITERATION_LIMIT
            keep.append(slot)
        # Carry only the surviving columns, with their matching residuals.
        surviving = np.array(keep, dtype=np.intp)
        r = new_r[:, surviving] if surviving.size else new_r[:, :0]
        active = active[surviving]
    reports = tuple(RefinementReport(status[j], attempted[j], history[j][0], float(best[j]),
                                     tuple(history[j]), correction[j]) for j in range(k))
    return x, reports


def _refine(a: csc_matrix | csr_matrix, b: np.ndarray, solve: Callable[[np.ndarray], np.ndarray],
            steps: int, rtol: float, minimize: bool = False) -> tuple[np.ndarray, RefinementReport]:
    """Refine one right-hand side, where solve applies the factors to one vector.

    A single-column convenience wrapper over _refine_block, which carries the
    algorithm. b has shape (n,); the result is the best iterate and its report.
    """
    def _block(rhs: np.ndarray) -> np.ndarray:
        return np.asarray(solve(rhs[:, 0]))[:, None]

    x, reports = _refine_block(a, np.asarray(b)[:, None], _block, steps, rtol, minimize)
    return x[:, 0], reports[0]
