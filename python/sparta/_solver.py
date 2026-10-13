"""The analyze, factorize, solve lifecycle."""
from __future__ import annotations
from dataclasses import asdict, replace
import math
import sys
import time
import numpy as np
from scipy import sparse
from ._sparta import _Analysis, _Factor, _CpuFactor, device as _device
from ._types import Ordering, MatrixType, Path
from ._options import _Options
from ._matrix import _prepare_matrix, _validate_choices, _gpu_values, _value_gather
from ._scaling import _equilibrate
from ._refinement import RefinementReport, _Operator, _refine_block
from ._quality import FactorQuality


def _symbolic(a: sparse.csc_matrix, path: Path, options: _Options) -> _Analysis:
    """Run the Rust symbolic analysis for canonical CSC a.

    a is the validated CSC storage from _prepare_matrix (full for LU, lower
    triangle for LDLT); path is the resolved factorization and options carries
    the ordering and tuning fields. Rust owns the ordering, the elimination
    tree and the GPU storage layout.
    """
    return _Analysis(a.shape[0], np.ascontiguousarray(a.indptr, dtype=np.uintp),
                     np.ascontiguousarray(a.indices, dtype=np.uintp),
                     2**64-1 if options.max_front is None else options.max_front,
                     path == Path.LU, options.ordering.value,
                     nemin=options.nemin, small_front_rows=options.small_front_rows,
                     imbalance=options.imbalance, wide_cycles=options.wide_cycles,
                     fm_passes=options.fm_passes,
                     separator_trials=options.separator_trials,
                     schur_batch=options.schur_batch,
                     panel_threads=options.panel_threads,
                     solve_threads=options.solve_threads,
                     lu_diag_threshold=options.lu_diag_threshold,
                     panel=options.panel, schur_block=options.schur_block,
                     strip_block=options.strip_block,
                     narrow_cols=options.narrow_cols,
                     pivot_threads=options.pivot_threads,
                     schur_chunk=options.schur_chunk, force_cpu=options.force_cpu,
                     cpu_panel=options.cpu_panel, cpu_block_rows=options.cpu_block_rows,
                     cpu_blas_min_rows=options.cpu_blas_min_rows)


def _reader(matrix: sparse.spmatrix, a: sparse.csc_matrix, triangle: str,
            path: Path) -> tuple[np.ndarray | None, tuple | None]:
    """Return the fast value gather for the caller's storage of a pattern.

    matrix is the caller's sparse matrix and a its canonical CSC copy. The
    result is (gather, layout): gather rebuilds a.data from a replacement
    matrix's CSC data in one indexing step, and layout is the caller's CSC
    shape, indptr and indices, which a replacement must match exactly before
    the gather may be used. Both are None when the pattern carries duplicate
    entries, which leaves only the checked path.
    """
    gather = _value_gather(matrix, a, triangle, path)
    if gather is None:
        return None, None
    c = matrix.tocsc()
    # Copies: tocsc() returns the caller's own arrays for CSC input.
    return gather, (c.shape, c.indptr.copy(), c.indices.copy())


def _csr_layout(a: sparse.csc_matrix, ldlt: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the CSR layout of the matrix a represents, and its value map.

    a is canonical CSC (lower triangle when ldlt is True). The result is
    (indptr, indices, source): the first two describe the CSR pattern of the
    represented matrix, and source[k] is the index in a.data that belongs to
    CSR slot k. LDLT mirrors each off-diagonal entry, so one CSC value can feed
    two CSR slots; nothing is conjugated.
    """
    columns = np.repeat(np.arange(a.shape[0], dtype=np.intp), np.diff(a.indptr))
    rows = a.indices
    source = np.arange(a.nnz, dtype=np.int64)
    if ldlt:
        # The stored lower triangle also stands for its transpose.
        off = rows != columns
        rows, columns = np.concatenate([rows, columns[off]]), np.concatenate([columns, rows[off]])
        source = np.concatenate([source, source[off]])
    # Carrying source+1 as the values makes tocsr permute the map for us.
    layout = sparse.coo_matrix((source + 1, (rows, columns)), shape=a.shape).tocsr()
    if layout.nnz != source.size:
        raise ValueError('The represented matrix has duplicate stored entries')
    return layout.indptr, layout.indices, layout.data - 1


class Solver:
    """Reusable analysis and GPU factorization, created by :func:`sparta.analyze`.

    Ownership and data flow:
    - _a: latest successful CSC values on CPU, full for LU or lower for LDLT.
    - _analysis: Rust symbolic analysis; fixed ordering, pattern and buffer layout.
    - _native: Rust/Metal factor allocation; created by factorize, reused on updates.
    - _row_scale/_column_scale: optional transformations around native solves.
    - _residual_matrix: original values in complex128 CSR behind the threaded
      product, only for refinement.

    analyze prepares _a and _analysis. factorize sends scaled complex64 values to
    _native. solve either calls that native factor once or uses it repeatedly
    inside _refine, which measures residuals using _residual_matrix.

    Use this object only on its creating thread. factorize replaces values on
    the analyzed pattern; given a different pattern it repeats the symbolic
    analysis first. solve and solve_many reuse the factors.
    Numerical factors use complex64 on Metal or complex128 with force_cpu=True.
    Refinement retains original complex128 values and returns complex128 solutions.
    """

    def __init__(self, matrix: sparse.spmatrix, path: Path, options: _Options) -> None:
        """Create symbolic storage from sparse matrix, requested path and validated options."""
        self._options = options
        # Python validates the values and chooses the storage Rust expects.
        self._a, self.path = _prepare_matrix(matrix, options.mtype, Path(path), options.triangle)
        # Kept so n and path still answer after destroy() drops the matrix copy.
        self._n = self._a.shape[0]
        self._analysis = _symbolic(self._a, self.path, options)
        self._gather, self._input_pattern = _reader(matrix, self._a, options.triangle, self.path)
        # How many times factorize() met a new pattern and analyzed again.
        self._reanalyses = 0
        # No GPU memory is allocated until factorize() succeeds.
        self._native: _Factor | _CpuFactor | None = None
        self._row_scale: np.ndarray | None = None
        self._column_scale: np.ndarray | None = None
        self._residual_matrix: _Operator | None = None
        # Fixed CSR layout for CPU residuals, built on first use.
        self._residual_map: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._refinement_reports: tuple[RefinementReport, ...] = ()

    def factorize(self, matrix: sparse.spmatrix | None = None, *,
                  safe_mode: bool | None = None, force_cpu: bool | None = None,
                  ordering: Ordering | None = None, mtype: MatrixType | None = None,
                  path: Path | None = None, triangle: str | None = None,
                  max_front: int | None = None, equilibrate: bool | None = None,
                  refinement_steps: int | None = None, refinement_rtol: float | None = None,
                  refinement_minimize: bool | None = None, verbose: bool | None = None) -> Solver:
        """Factor a matrix and return this solver; reuse storage while the backend is unchanged.

        Parameters
        ----------
        matrix : SciPy sparse matrix or None
            New full matrix (or lower triangle if selected during analysis).
            With the analyzed stored pattern, the analysis and its memory are
            reused. With a different pattern, including a different size, the
            symbolic analysis runs again first: the ordering, the plan and the
            factor memory are rebuilt, the ordering and tuning settings are kept,
            and reanalyses counts it. None reuses the latest successful values,
            initially the analyzed matrix. Raw data arrays are not accepted.
        force_cpu : bool or None
            True selects CPU complex128; False selects Metal complex64. None keeps
            the current backend, initially chosen by analyze(). Switching reuses
            symbolic analysis and releases old numerical storage before allocating
            the new factors. A failed switch disables solves until factorize succeeds.
        safe_mode : bool or None
            Run every new matrix through the full checked input pipeline. Off, a
            matrix stored exactly like the analyzed one is read straight into the
            analyzed layout, and non-finite values reach the factorization rather
            than raising; a different pattern or storage order is still detected
            and takes the checked path. None keeps the current setting.
        ordering : Ordering or None
            Optional repetition of the analyzed ordering; changing it requires analyze.
        mtype : MatrixType or None
            Optional repetition of the declared matrix type; cannot change here.
        path : Path or None
            AUTO or the already selected path. A different path requires analyze.
        triangle : {"full", "lower"} or None
            Optional repetition of the analyzed storage convention.
        max_front : int or None
            Optional repetition of the analyzed front-size limit, in rows.
        equilibrate : bool or None
            Balance matrix magnitudes before float32 conversion. Extra CPU work
            and memory; recomputed for each factorization. None keeps current setting.
        refinement_steps : int or None
            Correction solves allowed per RHS after the initial GPU solve: -1 for
            as many as refinement_rtol needs, 0 to disable refinement, a positive
            value for a hard cap. None keeps the current setting.
        refinement_rtol : float or None
            Relative residual tolerance, between 0 and 1, measured against the
            original matrix. Does not bound solution error. None keeps current value.
        refinement_minimize : bool or None
            Optimize each correction's complex step length. Adds a CPU matrix-vector
            product per correction; may reduce iterations. None keeps current setting.

        verbose : bool or None
            Print bounded per-solve summaries to stderr; None keeps current setting.
            Extra CPU residual evaluation is performed when enabled.

        Raises
        ------
        ValueError
            Analysis options changed, the new matrix does not fit the analyzed
            path or matrix type, or numerical settings are invalid.
        NotImplementedError
            A reserved ordering, matrix type or factorization is selected.
        RuntimeError
            Metal allocation or numerical factorization failed, and solve is then
            disabled until a successful factorize; or the analysis of a new pattern
            failed, for example on max_front, which keeps the old factors usable.
            Input validation failures preserve old factors too.
        """
        self._live()
        updates = dict(force_cpu=force_cpu, verbose=verbose, safe_mode=safe_mode, ordering=ordering, mtype=mtype, triangle=triangle, max_front=max_front,
                       equilibrate=equilibrate, refinement_steps=refinement_steps,
                       refinement_rtol=refinement_rtol, refinement_minimize=refinement_minimize)
        # None means keep the current value; _Options has no separate defaults.
        overrides = {name: value for name, value in updates.items() if value is not None}
        options = replace(self._options, **overrides)
        requested = self.path if path is None else Path(path)
        # Validate future choices before comparing immutable analysis options.
        _validate_choices(options.mtype, requested)
        for key in ('ordering', 'mtype', 'triangle', 'max_front'):
            if getattr(options, key) != getattr(self._options, key):
                raise ValueError(f'Changing {key} requires new analysis')
        if requested not in (Path.AUTO, self.path):
            raise ValueError('Changing path requires new analysis')
        a, changed = self._validated(matrix, options.safe_mode)
        # A new pattern needs a new plan. Built before anything is released, so
        # an analysis failure leaves the old factors usable.
        analysis = _symbolic(a, self.path, options) if changed else self._analysis
        layout = (_csr_layout(a, self.path == Path.LDLT)
                  if changed and options.refinement_steps != 0 else None)
        try:
            scaled, left, right = self._scaled(a, options)
            converted = self._native_values(scaled, 'A', options.safe_mode, force_cpu=options.force_cpu)
            # Prepare all CPU data before touching an existing valid factorization.
            residual_matrix = self._build_residual_matrix(a, options.refinement_steps != 0, layout)
        except Exception:
            if changed:
                analysis.destroy()
            raise
        switching = options.force_cpu != self._options.force_cpu
        try:
            if (switching or changed) and self._native is not None:
                self._native.destroy()
                self._native = None
            if self._native is None:
                # The selected backend allocates buffers and factors the CSC values.
                self._native = (analysis.factor_cpu(converted) if options.force_cpu
                                else analysis.factor(converted))
            else:
                # Same pattern: retain the allocation and replace numerical factors.
                self._native.refactor(converted)
        except Exception:
            # A failed native factorization must never be used by a later solve.
            # The old analysis stays, matching the values in self._a.
            self._native = None
            self._residual_matrix = None
            self._refinement_reports = ()
            if changed:
                analysis.destroy()
            raise

        # Publish matching values, scaling and settings only after success.
        if changed:
            self._adopt(analysis, matrix, a, layout, options.verbose)
        if switching:
            self._analysis.set_force_cpu(options.force_cpu)
        self._a = a
        self._options = options
        self._row_scale, self._column_scale = left, right
        self._residual_matrix = residual_matrix
        self._refinement_reports = ()
        return self

    def _validated(self, matrix: sparse.spmatrix | None, safe: bool = True) -> tuple[sparse.csc_matrix, bool]:
        """Return replacement values in canonical CSC, and whether the pattern changed.

        matrix=None returns the latest successful CPU values. With safe=False
        and a matrix stored exactly like the analyzed one (same shape, indptr
        and indices), the values are gathered straight into the stored layout.
        Anything else goes through the whole checked input pipeline. The bool
        is True when the canonical pattern differs from the analyzed one, in
        which case the caller must analyze again before factoring.
        """
        if matrix is None:
            return self._a, False
        # Without this a raw ndarray reaches .tocsc() and raises AttributeError
        # instead of saying what is wrong.
        if not sparse.issparse(matrix):
            raise TypeError('A must be a SciPy sparse matrix')
        if not safe and self._gather is not None:
            c = matrix.tocsc()
            shape, indptr, indices = self._input_pattern
            if (c.shape == shape and np.array_equal(c.indptr, indptr)
                    and np.array_equal(c.indices, indices)):
                data = c.data[self._gather]
                if data.dtype.kind != 'c':
                    data = data.astype(np.complex128)
                return sparse.csc_matrix((data, self._a.indices, self._a.indptr),
                                         shape=self._a.shape, copy=False), False
        a, _ = _prepare_matrix(matrix, self._options.mtype, self.path, self._options.triangle)
        same_pattern = (
            a.shape == self._a.shape
            and np.array_equal(a.indptr, self._a.indptr)
            and np.array_equal(a.indices, self._a.indices)
        )
        return a, not same_pattern

    def _adopt(self, analysis: _Analysis, matrix: sparse.spmatrix, a: sparse.csc_matrix,
               layout: tuple[np.ndarray, np.ndarray, np.ndarray] | None, verbose: bool) -> None:
        """Replace the analyzed pattern after a factorization on a new one succeeded.

        analysis is the new Rust plan, matrix the caller's matrix and a its
        canonical CSC copy. layout is a's CSR residual layout, or None to build
        it on first use. verbose prints one line to stderr.
        """
        self._analysis.destroy()
        self._analysis = analysis
        self._n = a.shape[0]
        self._gather, self._input_pattern = _reader(matrix, a, self._options.triangle, self.path)
        self._residual_map = layout
        self._reanalyses += 1
        if verbose:
            print(f"  new sparsity pattern: repeated the symbolic analysis (n={a.shape[0]}, "
                  f"stored entries={a.nnz})", file=sys.stderr)

    def _scaled(self, a: sparse.csc_matrix, options: _Options) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Prepare complex CSC values for factorization using numerical options.

        a is the canonical CSC matrix being factored. Returns values to send to
        the GPU and optional real row/column multipliers for RHS/solution
        conversion. None multipliers mean no scaling, avoiding unnecessary
        vector allocation.
        """
        if options.equilibrate:
            return _equilibrate(a, a.data, self.path == Path.LDLT)
        return a.data, None, None

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve A x = b using existing factors.

        b : ndarray, shape (n,)
            Finite float32, float64, complex64 or complex128 RHS vector.
        Returns a complex vector of shape (n,); refinement returns complex128,
        otherwise precision follows the wider matrix/RHS input dtype.
        Raises ValueError if factorize has not succeeded or b has invalid shape/type.
        Inspect refinement_reports: an unconverged best iterate can be returned.
        """
        started = time.perf_counter() if self._options.verbose else 0.0
        result = self._solve_vector(b)
        if self._options.verbose:
            self._print_summary(np.asarray(b)[:, None], result[:, None], time.perf_counter() - started)
        return result

    def _solve_vector(self, b: np.ndarray) -> np.ndarray:
        """Solve one RHS quietly; solve_many aggregates logs outside this operation."""
        self._require_factor()
        b = self._validate_rhs(b, dimensions=1)
        self._refinement_reports = ()
        if self._residual_matrix is not None:
            # Residuals use original values; correction solves reuse GPU factors.
            result, reports = _refine_block(
                self._residual_matrix, b[:, None], self._solve_scaled_block,
                self._options.refinement_steps, self._options.refinement_rtol,
                self._options.refinement_minimize,
            )
            self._refinement_reports = reports
            return result[:, 0]
        result = self._solve_scaled(b)
        return result.astype((np.complex128 if self._options.force_cpu else np.result_type(self._a.dtype, b.dtype)), copy=False)

    def solve_many(self, b: np.ndarray) -> np.ndarray:
        """Solve A X = B using existing factors.

        b : ndarray, shape (n, nrhs)
            Finite real or complex RHS columns. Rows are matrix unknowns, not ports.
        Returns a complex array with the same shape. Columns currently execute
        sequentially; refinement_reports contains one report per column.
        Zero-column arrays are accepted only after successful factorization.
        """
        started = time.perf_counter() if self._options.verbose else 0.0
        self._require_factor()
        b = self._validate_rhs(b, dimensions=2)
        self._refinement_reports = ()
        if b.shape[1] == 0:
            dtype = np.complex128 if self._options.refinement_steps else (np.complex128 if self._options.force_cpu else np.result_type(self._a.dtype, b.dtype))
            result = np.empty(b.shape, dtype=dtype)
            if self._options.verbose:
                self._print_summary(b, result, time.perf_counter() - started)
            return result

        if self._residual_matrix is None:
            # No refinement: every column goes through the factors together.
            result = self._solve_scaled_block(b)
            result = result.astype((np.complex128 if self._options.force_cpu else np.result_type(self._a.dtype, b.dtype)), copy=False)
        else:
            # Refinement batches as well: one dispatch per correction, not per column.
            result, reports = _refine_block(
                self._residual_matrix, b, self._solve_scaled_block,
                self._options.refinement_steps, self._options.refinement_rtol,
                self._options.refinement_minimize,
            )
            self._refinement_reports = reports
        if self._options.verbose:
            self._print_summary(b, result, time.perf_counter() - started)
        return result

    def _require_factor(self) -> _Factor | _CpuFactor:
        """Return valid numerical storage or reject a premature solve."""
        self._live()
        if self._native is None:
            raise ValueError('Call factorize successfully before solving')
        return self._native

    @property
    def n(self) -> int:
        """Number of matrix rows and columns."""
        return self._n

    def destroy(self) -> None:
        """Release the GPU factors, the plan and every matrix copy, right now.

        Frees the Metal buffers, the symbolic plan and the CPU-side matrix copies
        without waiting for the garbage collector. Use it before analyzing the
        next matrix when the factors are large: a solver still referenced
        anywhere -- a list, a traceback, a notebook's output history -- holds its
        GPU memory until that reference goes.

        Calling it more than once is harmless. Afterwards the solver still
        reports n and path, and factorize, solve and info raise ValueError.
        """
        # Hand the Metal buffers and the symbolic arrays back through the
        # extension rather than waiting for the reference count to reach zero:
        # a solver still referenced anywhere would otherwise keep both.
        if self._native is not None:
            self._native.destroy()
        if self._analysis is not None:
            self._analysis.destroy()
        self._native = None
        self._analysis = None
        self._a = None
        self._residual_matrix = None
        self._residual_map = None
        self._row_scale = None
        self._column_scale = None
        self._refinement_reports = ()

    def _live(self) -> None:
        """Reject any operation that needs storage destroy() has already released."""
        if self._analysis is None:
            raise ValueError('This solver was destroyed; analyze again to get a new one')

    def __enter__(self) -> Solver:
        """Return this solver, so `with sparta.analyze(a) as f:` frees it on exit."""
        return self

    def __exit__(self, *exception: object) -> None:
        """Release everything the solver holds, whether or not the block raised."""
        self.destroy()

    @property
    def info(self) -> dict[str, int | Path]:
        """Symbolic sizes; byte fields are allocation estimates, not device timings."""
        self._live()
        (n, nodes, levels, front_bytes, largest, large, factor_bytes, upper,
         device_bytes) = self._analysis.info()
        return {
            'n': n, 'supernodes': nodes, 'levels': levels,
            'front_bytes': front_bytes, 'max_front': largest,
            'large_fronts': large, 'factor_bytes': factor_bytes,
            'upper_bytes': upper, 'path': self.path,
            'gpu_bytes': device_bytes,
            'cpu_bytes': front_bytes + factor_bytes if self._options.force_cpu else 0,
        }

    @property
    def reanalyses(self) -> int:
        """How many times factorize met a new sparsity pattern and analyzed it again.

        Each one costs a full symbolic analysis and new factor memory. A count
        that keeps rising in a loop means the pattern is not stable between
        calls, for example because explicit zeros come and go.
        """
        return self._reanalyses

    @property
    def refinement_reports(self) -> tuple[RefinementReport, ...]:
        """Last solve's per-RHS convergence reports; empty when refinement is off."""
        return self._refinement_reports

    @property
    def perturbed_pivots(self) -> int:
        """Count of rescued pivots; zero does not guarantee numerical accuracy."""
        return self._require_factor().perturbed_pivots

    @property
    def quality(self) -> FactorQuality:
        """How much to trust this factorization, at no extra cost.

        Available the moment `factorize` returns, before any solve. The pivot
        range behind it is a host scan of the block diagonal already in shared
        memory, so nothing here requires a residual and therefore nothing here
        costs a matvec against the original matrix.

        `relative_residual` is filled in only when iterative refinement has run
        and already measured it; otherwise it is nan and the verdict rests on
        the pivots alone, which on the measured fixtures is enough to separate
        a healthy factorization from one refinement cannot rescue.
        """
        native = self._require_factor()
        # None only when the factors report no finite nonzero pivot.
        span = native.pivot_range
        low, high = span if span else (math.nan, math.nan)
        residual = (self._refinement_reports[0].initial_relative_residual
                    if self._refinement_reports else math.nan)
        return FactorQuality(native.perturbed_pivots, low, high, residual,
                             single_precision=not self._options.force_cpu,
                             lu=self.path == Path.LU)

    @property
    def factor_device_seconds(self) -> float:
        """GPU factorization time in seconds; excludes analysis and Python preparation."""
        return self._require_factor().factor_device_seconds

    @property
    def factor_seconds(self) -> float:
        """CPU factor wall time or GPU command duration, in seconds."""
        return self._require_factor().factor_seconds

    def diagnostics(self) -> dict[str, object]:
        """Return symbolic sizes, current options and latest numerical diagnostics."""
        return {
            **self.info,
            'backend': 'cpu' if self._options.force_cpu else _device(),
            'compute_dtype': 'complex128' if self._options.force_cpu else 'complex64',
            'options': asdict(self._options),
            'factorized': self._native is not None,
            'reanalyses': self._reanalyses,
            'refinement_reports': [asdict(report) for report in self.refinement_reports],
        }


    def _residual_layout(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the CSR layout and value map of the analyzed pattern, see _csr_layout.

        Built once per analysis; factorize replaces it when the pattern changes.
        """
        if self._residual_map is None:
            self._residual_map = _csr_layout(self._a, self.path == Path.LDLT)
        return self._residual_map

    def _build_residual_matrix(self, matrix: sparse.csc_matrix, enabled: bool,
                               layout: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> _Operator | None:
        """Copy the represented matrix into complex128 CSR for CPU residuals.

        matrix is canonical CSC (lower for LDLT, full for LU); enabled determines
        whether refinement needs this extra storage. Returns None when disabled.
        LDLT reconstructs the upper triangle by transpose, never conjugation.
        layout is matrix's CSR layout from _csr_layout; None uses the cached
        layout of the analyzed pattern, so each factorization only gathers values.
        """
        if not enabled:
            return None
        indptr, indices, source = self._residual_layout() if layout is None else layout
        values = matrix.data.astype(np.complex128, copy=False)[source]
        # CSR makes repeated row-wise residual matrix-vector products cheaper,
        # and is the layout the threaded product in the extension expects.
        return _Operator(sparse.csr_matrix((values, indices, indptr), shape=matrix.shape))

    def _validate_rhs(self, values: np.ndarray, dimensions: int) -> np.ndarray:
        """Return a finite complex RHS array with one or two dimensions.

        values holds n rows of real/complex floating-point data. dimensions is
        1 for solve and 2 for solve_many. Real inputs retain their precision when
        promoted to complex. The caller's array is never modified.
        """
        rhs = np.asarray(values)
        if rhs.dtype in (np.dtype('float32'), np.dtype('float64')):
            rhs = rhs.astype(np.complex64 if rhs.dtype == np.float32 else np.complex128)
        if rhs.ndim != dimensions or rhs.shape[0] != self.n:
            raise ValueError('RHS must have n rows and match solve/solve_many dimensions')
        if rhs.dtype not in (np.dtype('complex64'), np.dtype('complex128')) or not np.isfinite(rhs).all():
            raise ValueError('RHS must contain finite real or complex floating-point values')
        return rhs

    def _native_values(self, values: np.ndarray, name: str, safe: bool = True,
                       force_cpu: bool | None = None) -> np.ndarray:
        """Convert values for force_cpu (None keeps current backend); name labels safe validation errors."""
        cpu = self._options.force_cpu if force_cpu is None else force_cpu
        if cpu:
            out = np.ascontiguousarray(values, dtype=np.complex128)
            if not np.isfinite(out).all():
                raise ValueError(f'{name} must contain finite complex128 values')
            return out
        return _gpu_values(values, name, safe)

    def _solve_scaled(self, rhs: np.ndarray) -> np.ndarray:
        """Apply existing GPU factors to an original-system complex RHS vector.

        rhs has shape (n,). For factors of R A C, solve (R A C)y = R rhs, then
        return x = C y. Normalization keeps the scaled RHS representable in float32.
        This same function serves both the initial solve and refinement corrections.
        """
        native = self._require_factor()
        if self._row_scale is None:
            return native.solve(self._native_values(rhs, 'b'))

        scaled_rhs = np.asarray(rhs, dtype=np.complex128) * self._row_scale
        magnitude = float(max(
            np.max(np.abs(scaled_rhs.real), initial=0),
            np.max(np.abs(scaled_rhs.imag), initial=0),
        ))
        normalized = scaled_rhs / magnitude if magnitude else scaled_rhs
        solution = native.solve(self._native_values(normalized, 'scaled b')).astype(np.complex128)
        if magnitude:
            solution *= magnitude
        return solution * self._column_scale


    def _solve_scaled_block(self, rhs: np.ndarray) -> np.ndarray:
        """Apply existing GPU factors to every column of a complex RHS block.

        rhs has shape (n, nrhs). Returns the solutions in the same shape. This is
        the same transformation _solve_scaled performs, applied per column, but
        the columns share one dispatch per tree level inside the native solve
        instead of one sweep each. Each column is normalised on its own so a
        quiet column stays representable in float32 next to a loud one.
        """
        native = self._require_factor()
        if self._row_scale is None:
            block = self._native_values(np.ascontiguousarray(rhs.T), 'b')
            return np.asarray(native.solve_many(block)).T

        scaled = np.asarray(rhs, dtype=np.complex128) * self._row_scale[:, None]
        magnitude = np.maximum(np.abs(scaled.real).max(axis=0), np.abs(scaled.imag).max(axis=0))
        divisor = np.where(magnitude > 0, magnitude, 1.0)
        block = self._native_values(np.ascontiguousarray((scaled/divisor).T), 'scaled b')
        solution = np.asarray(native.solve_many(block)).T.astype(np.complex128)
        return solution * divisor * self._column_scale[:, None]

    def _print_summary(self, rhs: np.ndarray, solution: np.ndarray, seconds: float) -> None:
        """Print one bounded summary for RHS columns and solution, with solve wall seconds.

        Rust prints symbolic layout statistics; Python measures residuals against
        the original represented matrix. At most eight RHS rows are printed.
        Logging work is excluded from the displayed solve duration.
        """
        self._analysis._print_stats()
        backend = 'CPU complex128' if self._options.force_cpu else 'Metal complex64'
        print(f"  {backend} | input={self._a.dtype} | RHS={rhs.shape[1]}", file=sys.stderr)
        print(f"  factor={self.factor_seconds:.6f}s | solve={seconds:.6f}s | rescued pivots={self.perturbed_pivots}", file=sys.stderr)
        print(f"  scaling={self._options.equilibrate} | refinement cap={self._options.refinement_steps} | minimize={self._options.refinement_minimize}", file=sys.stderr)
        matrix = self._residual_matrix
        if matrix is None:
            matrix = self._build_residual_matrix(self._a, True)
        worst = 0.0
        for j in range(rhs.shape[1]):
            b = np.asarray(rhs[:, j], dtype=np.complex128)
            r = b - matrix @ solution[:, j].astype(np.complex128)
            scale = float(max(np.max(np.abs(b.real), initial=0), np.max(np.abs(b.imag), initial=0)))
            residual = float(np.linalg.norm(r / scale) / np.linalg.norm(b / scale)) if scale else float(np.linalg.norm(r))
            worst = max(worst, residual)
            if j < 8:
                detail = "unrefined"
                if self._refinement_reports:
                    report = self._refinement_reports[j]
                    detail = f"{report.status.value}, corrections={report.corrections}"
                    # The residual is not the error; on an ill-conditioned
                    # system it can be a thousand times smaller.
                    if report.estimated_relative_error == report.estimated_relative_error:
                        detail += f", est. solution error<={report.estimated_relative_error:.1e}"
                print(f"  RHS {j}: relative residual={residual:.3e} | {detail}", file=sys.stderr)
        if rhs.shape[1] > 8:
            print(f"  {rhs.shape[1]-8} more RHS omitted | worst residual across all RHS={worst:.3e}", file=sys.stderr)
