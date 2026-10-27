"""Sparse GPU solves through analyze → factorize → solve.

See python/API.md for argument reference, examples and supported enum choices.
"""
from __future__ import annotations
from scipy import sparse

# Absolute entrywise tolerance for numerical symmetry, in matrix-value units.
EPS: float = 1e-10

from ._sparta import gpu_available, __version__
from ._types import Ordering, MatrixType, Path, Backend
from ._options import _Options, _hybrid_share
from ._config import Config, presets, profiles
from ._optimize import optimize, Optimization, Progress
from ._solver import Solver
from ._refinement import RefinementReport, RefinementStatus
from ._quality import FactorQuality

__all__ = ['EPS', '__version__', 'analyze', 'Solver', 'Config', 'presets',
           'profiles', 'optimize', 'Optimization', 'Progress',
           'Ordering', 'MatrixType', 'Path', 'Backend',
           'gpu_available', 'RefinementReport', 'RefinementStatus', 'FactorQuality']


def analyze(matrix: sparse.spmatrix, *, config: Config | str | None = None,
            ordering: Ordering | None = None,
            mtype: MatrixType = MatrixType.AUTO, path: Path = Path.AUTO,
            triangle: str = 'full', max_front: int | None = None,
            equilibrate: bool = False, refinement_steps: int = -1,
            refinement_rtol: float = 1e-6, refinement_minimize: bool = True,
            safe_mode: bool = False, verbose: bool = False,
            nemin: int | None = None, small_front_rows: int | None = None,
            imbalance: float | None = None, wide_cycles: int | None = None,
            fm_passes: int | None = None, separator_trials: int | None = None,
            schur_batch: int | None = None, panel_threads: int | None = None,
            solve_threads: int | None = None,
            lu_diag_threshold: float | None = None,
            panel: int | None = None, schur_block: int | None = None,
            strip_block: int | None = None, narrow_cols: int | None = None,
            pivot_threads: int | None = None,
            schur_chunk: int | None = None,
            force_cpu: bool | None = None, hybrid: bool | float | None = None,
            cpu_panel: int | None = None,
            cpu_block_rows: int | None = None, cpu_blas_min_rows: int | None = None,
            threads: int | None = None) -> Solver:
    """Analyze a matrix pattern on CPU and return an unfactorized Solver.

    Parameters
    ----------
    force_cpu : bool or None, default None
        True uses Rust complex128 factorization and solves on CPU, with Apple
        Accelerate for large updates. False uses Metal complex64. None uses the
        config setting, or False without a config. factorize(force_cpu=...) can
        override this later without repeating symbolic analysis.
    hybrid : bool, float or None, default None
        Factor with the CPU and the GPU together. The CPU takes whole branches
        of the elimination tree while the GPU factors the rest, then the GPU
        finishes from where they meet. True starts the CPU at 30% of the work;
        a number from 0 to below 1 starts it there instead; False is the GPU
        alone. The share then adapts to the measured speed of each device from
        the third factorization on. Works for LDLT and LU. Only Metal has a
        hybrid path: with force_cpu, on Vulkan, or without a GPU the setting is
        ignored. None uses the config's hybrid_cpu, or False without a config.
        factorize(hybrid=...) can change it later.
    cpu_panel : int or None, default None
        CPU panel width in pivots; 2x2 pivots may extend it by one column.
    cpu_block_rows : int or None, default None
        Rows per tile for small CPU updates below the Accelerate threshold.
    cpu_blas_min_rows : int or None, default None
        Minimum trailing rows before CPU updates use Apple Accelerate.
    threads : int or None, default None
        CPU threads for SPARTA's own work: the analysis, CPU factorization and
        solves, and the hybrid's CPU share. 0 uses the process-wide pool, which
        RAYON_NUM_THREADS sizes; a positive count, up to 4096, gives the solver
        a pool of its own and keeps Accelerate from threading past it. The
        SPARTA_NUM_THREADS environment variable, when set to a positive count,
        overrides this. None uses the config's threads, or 0 without a config.
    config : Config, str or None, default None
        Every tuning setting at once, from Config.defaults(), Config.preset()
        or Config.load(). A string names a profile saved by sparta.optimize()
        or Config.save(), else a preset. Individual keyword arguments below
        still take precedence, so a preset can be adjusted at the call site.
    matrix : SciPy sparse matrix
        Nonempty square float32/float64/complex64/complex128 matrix. Converted
        to sorted CSC; duplicate entries are summed, stored zeros retained.
        Keep original complex128 values when requesting accurate refinement.
    ordering : Ordering or None, default None
        Fill-reducing ordering. SCALPEL, SPARTA's parallel nested dissection,
        is the only one; None takes it from `config`. Other named choices
        raise NotImplementedError. Does not change between factorize calls.
    mtype : MatrixType, default AUTO
        Declared mathematical structure. AUTO detects transpose symmetry within EPS
        in full storage. Symmetric types select LDLT; unsymmetric, structurally
        symmetric and Hermitian types select LU. Declarations are validated.
        Reserved matrix types raise NotImplementedError (see MatrixType).
    path : Path, default AUTO
        Automatic selection or explicit LU/LDLT override. LDLT requires A == A.T within EPS,
        not A == A.conjugate().T. Other paths raise NotImplementedError.
    triangle : {"full", "lower"}, default "full"
        Input storage. Full reads both triangles and checks declared symmetry.
        Lower reads the lower triangle as a transpose-symmetric matrix; LU and
        Hermitian declarations require full storage. Fixed during analysis.
    max_front : int or None, default None
        Optional maximum frontal matrix size in rows; larger fronts fail analysis.
        None imposes no user limit (GPU memory limits still apply).
    equilibrate : bool, default False
        Balance row/column magnitudes before float32 conversion at factorization.
        LDLT preserves symmetry using the same scaling on both sides. Costs extra
        CPU work and storage; not always beneficial.
    refinement_steps : int, default -1
        How many GPU correction solves to allow per RHS after its initial solve.
        -1 takes as many as refinement_rtol needs, 0 switches refinement off, and
        a positive value is a hard cap that also stops early once rtol is reached.
        Refinement retains the original complex128 matrix for CPU residuals and
        accumulates the solution in complex128. It is on by default because the
        GPU factors are single precision: a bare solve lands near 1e-5 relative
        residual, and the first correction is worth far more than the last.
    refinement_rtol : float, default 1e-6
        Stop refinement when ||b - A x||2 / ||b||2 reaches this tolerance (0 < rtol < 1).
        A zero RHS uses absolute residual.

        This is a residual, not a solution error, and on an ill-conditioned
        system the two are far apart. Measured on em_m2_sym1, whose condition
        number is about 2.4e9: stopping at the default 1e-6 leaves a solution
        error near 8e-5, and reaching a solution error of 2e-8 needs a
        refinement_rtol of 1e-10. A well-conditioned system amplifies by only a
        few hundred. So ask for a tighter residual than the solution accuracy
        you need, and read estimated_relative_error off the reports afterwards
        to see what you actually got -- it costs nothing and it is the number
        that answers "how accurate is this answer".
    refinement_minimize : bool, default True
        Minimize residual along each correction direction. Adds a CPU sparse
        matrix-vector product per step, and roughly halves the correction count on
        badly conditioned matrices. Not full GMRES.

    safe_mode : bool, default False
        Re-check every replacement matrix passed to factorize(). Analysis always
        validates its input; this controls the repeat checks afterwards. Off, a
        replacement matrix stored exactly like the analyzed one is read straight
        into the analyzed layout, which is much faster, and its values are not
        checked for infinities or NaNs. Either way, a matrix with a different
        pattern is detected, and factorize() analyzes it again before factoring.
    verbose : bool, default False
        Print a compact stderr summary per solve call: native symbolic stats,
        factor timing and original-system residuals. Residual evaluation adds
        CPU work even when refinement is off. At most 16 lines per call.

    nemin : int or None, default None
        Fewest eliminated columns before two supernodes may merge. Larger
        values give fewer, wider GPU fronts for more factor storage. 16 suits
        unstructured meshes; 32 often suits structured grids.
    small_front_rows : int or None, default None
        Front height at which the GPU switches from its serial kernel to the
        cooperative one. Measured best near 32; higher is clearly worse.
    imbalance : float or None, default None
        Largest fraction of remaining weight allowed in either half of a
        nested-dissection split, from 0.5 to 0.9. Allowing imbalance produces
        smaller separators and less fill.
    wide_cycles : int or None, default None
        Separator refinement cycles per split. 0 orders faster and fills more;
        the default spends about 70 ms for roughly 4% fewer factor entries on
        a 109k system. Raise it if you factor one pattern many times.
    fm_passes : int or None, default None
        Fiduccia-Mattheyses refinement passes. 0 is much faster and much
        worse; rarely the right choice.
    separator_trials : int or None, default None
        Independent starting separators tried per split, best kept. Costs
        analysis time about linearly.
    schur_batch : int or None, default None
        Pivot panels between two Schur drains. 0 drains once per level.
    panel_threads : int or None, default None
        Threads per threadgroup in the panel factorization.
    solve_threads : int or None, default None
        Threads per front in the triangular solves. 0 picks per front width.
    lu_diag_threshold : float or None, default None
        LU only. 0 keeps strict partial pivoting. Above 0 a diagonal pivot is
        accepted when within this fraction of the column's best candidate:
        faster, less stable. This one changes your answers.

    panel : int or None, default None
        Pivots accumulated per panel before the trailing update runs. Only 16
        is supported; other widths need matching kernel changes.
    schur_block : int or None, default None
        Per-thread register block edge in the Schur update, 1 to 8. The tile a
        threadgroup owns is 16 * schur_block square. Larger tiles mean fewer,
        bigger dispatches and more threadgroup memory.
    strip_block : int or None, default None
        Per-thread register block height in the strip update, 1 to 8. A sweep
        over 2/4/8 picked 4. 16 * strip_block must be a multiple of panel.
    narrow_cols : int or None, default None
        Fully summed width above which a front takes the wide kernel path.
    pivot_threads : int or None, default None
        Threads cooperating on one pivot search; a multiple of 32.
    schur_chunk : int or None, default None
        Panel columns staged per Schur pass. Two arrays of
        16 * schur_block * schur_chunk complex floats must fit 32 KiB of
        threadgroup memory.

    Tuning arguments from nemin onward are fixed for the life of the analysis,
    except force_cpu and hybrid, which can also be selected by factorize().
    The six from panel onward are compiled into the Metal kernels when the
    first factorization creates them, so a combination this process has not
    used before costs roughly 300 ms of shader compilation once. Leave them
    alone unless you are tuning for hardware other than the one they were
    measured on.

    Returns
    -------
    Solver
        Call factorize() before solve(b) or solve_many(B). Factorization and solves
        use the selected CPU or Metal backend; analysis requires no GPU. Use on the creating thread only.
    """
    # A Config supplies every tuning field at once. An explicit keyword still
    # wins, so a preset can be adjusted at the call site without rebuilding it.
    tuning = dict(nemin=nemin, small_front_rows=small_front_rows, imbalance=imbalance,
                  wide_cycles=wide_cycles, fm_passes=fm_passes,
                  separator_trials=separator_trials, schur_batch=schur_batch,
                  panel_threads=panel_threads, solve_threads=solve_threads,
                  lu_diag_threshold=lu_diag_threshold, panel=panel,
                  schur_block=schur_block, strip_block=strip_block,
                  narrow_cols=narrow_cols, pivot_threads=pivot_threads,
                  schur_chunk=schur_chunk, force_cpu=force_cpu,
                  hybrid_cpu=None if hybrid is None else _hybrid_share(hybrid),
                  cpu_panel=cpu_panel, cpu_block_rows=cpu_block_rows,
                  cpu_blas_min_rows=cpu_blas_min_rows, threads=threads)
    if config is not None:
        base = Config.resolve(config).to_dict()
        tuning = {name: base[name] if value is None else value
                  for name, value in tuning.items()}
        if max_front is None:
            max_front = base['max_front']
        if ordering is None:
            ordering = Ordering(base['ordering'])
    if ordering is None:
        ordering = Ordering(Config.defaults().ordering)
    if tuning["force_cpu"] is None:
        tuning["force_cpu"] = False
    if tuning["hybrid_cpu"] is None:
        tuning["hybrid_cpu"] = 0.0
    options = _Options(**tuning, verbose=verbose, safe_mode=safe_mode, ordering=ordering, mtype=mtype, triangle=triangle,
                      max_front=max_front, equilibrate=equilibrate,
                      refinement_steps=refinement_steps, refinement_rtol=refinement_rtol,
                      refinement_minimize=refinement_minimize)
    return Solver(matrix, path, options)
