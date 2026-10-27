# SPARTA Python API

The public workflow is **analyze → factorize → solve**. One Solver owns the
symbolic analysis, current matrix values and GPU factors. There is no separate
Settings, Analysis or Factor API and no one-shot lu/ldlt/solve helper.

## Example

```python
import sparta

settings = {
    "ordering": sparta.Ordering.SCALPEL,
    "mtype": sparta.MatrixType.COMPLEX_SYMMETRIC,
    "path": sparta.Path.AUTO,
    "refinement_steps": 30,
    "refinement_rtol": 1e-7,
    "refinement_minimize": True,
}
F = sparta.analyze(A1, **settings)
F.factorize()                    # Optional: F.factorize(**settings)
x1 = F.solve(b1)                  # b1.shape == (n,)
X1 = F.solve_many(B1)             # B1.shape == (n, nrhs)
F.factorize(matrix=A2)            # New values; a new pattern is analyzed again
X2 = F.solve_many(B2)
F.factorize(refinement_rtol=1e-9) # Reuses A2; keeps other numerical options
```

Analysis runs on CPU. Factorization and triangular solves use Metal complex64
by default, or CPU complex128 with `force_cpu=True`. Optional refinement
evaluates residuals and accumulates solutions in complex128. Backend selection
is explicit. Use F only on its creating thread.

## analyze(matrix, *, options...) → Solver

The signature lists every accepted keyword; unknown arguments raise TypeError.
All numeric tolerances below are dimensionless.

| Argument | Default | Meaning |
|---|---|---|
| `matrix` | required | Nonempty square SciPy sparse matrix with float32, float64, complex64 or complex128 values. Copied into sorted CSC, duplicates summed, explicit zeros retained. |
| `ordering` | `None` | Fill-reducing ordering. `Ordering.SCALPEL` is the only one; `None` takes it from `config`. Other enum choices are reserved. |
| `mtype` | `MatrixType.AUTO` | Declared mathematical structure; see routing table. Declarations are checked. |
| `path` | `Path.AUTO` | Automatic routing or explicit LU/LDLT. An override must be mathematically compatible with the matrix. |
| `triangle` | `"full"` | Full storage reads both triangles. `"lower"` uses only the lower triangle and interprets it as transpose-symmetric; only LDLT supports it. |
| `max_front` | `None` | Optional positive maximum front size in rows. Analysis fails if exceeded. None means no user cap, not unlimited GPU memory. |
| `equilibrate` | `False` | Balance row/column magnitudes before converting factors to complex64. Adds CPU work and memory. Recomputed on each factorization. Not always faster or more accurate. |
| `refinement_steps` | `-1` | Correction solves per RHS after the initial solve: `-1` for as many as `refinement_rtol` needs (ceiling 50), `0` to disable refinement, positive for a hard cap that also stops early at the target. Refinement retains the original matrix in complex128. |
| `refinement_rtol` | `1e-6` | Refinement stops when `norm(b-A@x)/norm(b)` reaches this value; must lie strictly between 0 and 1. Zero RHS uses absolute residual. This is a residual, not a solution error, and on an ill-conditioned system the two are far apart: on em_m2_sym1 (condition ~2.4e9) the default leaves a solution error near 8e-5, and reaching 2e-8 needs `1e-10`. Read `estimated_relative_error` off the reports to see what you actually got. |
| `refinement_minimize` | `True` | Optimize the complex step length of each correction; adds a CPU sparse matrix-vector product and roughly halves the correction count on badly conditioned matrices. No effect when refinement is disabled. Not full GMRES. |

Real inputs are promoted to matching-width complex storage. Results are complex,
even for real input. Without refinement, output precision follows the wider
matrix/RHS dtype; refinement always returns complex128. Widening an unrefined
result does not restore lost precision. Preserve the original complex128 matrix
when accuracy matters.

## F.factorize(matrix=None, *, options...) → F

Creates factors on the first call and reuses GPU storage on later calls. `matrix`
is a SciPy sparse matrix. With the analyzed stored pattern, including explicit
zeros, the analysis and its storage are reused. With a different pattern or size,
the symbolic analysis runs again first, with the same analysis settings, and new
factor storage is allocated; `F.reanalyses` counts these. If that analysis fails
(for example on `max_front`), the previous factors stay usable. Raw CSC data
arrays are not accepted. `None` reuses the latest successful matrix values,
initially those supplied to analyze. The caller's matrix is copied.

Every option in the analyze table is accepted here with default `None`, meaning
keep the current value. Numerical options (`equilibrate`, `refinement_steps`,
`refinement_rtol`, `refinement_minimize`) can change and persist. Analysis options
(`ordering`, `mtype`, `path`, `triangle`, `max_front`) can be repeated unchanged,
so `factorize(**settings)` works. Changing them requires a new analyze call.
`Path.AUTO` also accepts the already selected path; it does not reselect it.

Invalid input/options preserve an existing factorization. A failure during GPU
factorization disables solves until factorize succeeds again. Successful
factorization clears old refinement reports. Refactorization is performed even
when only refinement options change; configure those options before factorizing
when avoiding unnecessary work matters.

## F.solve(b) → x and F.solve_many(B) → X

`solve` takes one finite vector of length n. `solve_many` takes an `(n, nrhs)` array
with RHS vectors in **columns**, not rows. Both accept float32/float64/complex64/
complex128 arrays and return arrays with matching shapes. A successful factorize
is required first. Empty RHS collections `(n, 0)` are supported after factorization.
`solve_many` solves the columns as one block: they share a dispatch per tree level
rather than one sweep each, and refinement corrects them together, with columns
leaving the batch as they converge. A bare eight-column block measured 4x-6x
faster than eight separate solves. Results are unchanged beyond the usual
float-atomic nondeterminism.

An unconverged refinement returns its best accepted finite iterate. Inspect:

```python
for report in F.refinement_reports:
    print(report.status, report.corrections, report.relative_residual)
```

`RefinementStatus` values are CONVERGED, STAGNATED, ITERATION_LIMIT and NONFINITE.
With `refinement_steps=-1`, ITERATION_LIMIT means the safety ceiling of 50 was hit.
Reports contain the initial and final relative residual, attempted correction
count, residual history, and a `converged` property. History can include a rejected
final correction. An invalid initial numerical result raises FloatingPointError.
A small residual does not guarantee a small field-vector or S-parameter error.

## Matrix types and factorization paths

| MatrixType | Interpretation | AUTO path |
|---|---|---|
| AUTO | Inspect transpose symmetry within EPS of full input | LDLT if symmetric; otherwise LU |
| REAL_SYMMETRIC | Real entries, `A == A.T` | LDLT |
| COMPLEX_SYMMETRIC | Complex entries, `A == A.T` | LDLT |
| REAL_UNSYMMETRIC | General real square matrix; symmetry not required | LU |
| COMPLEX_UNSYMMETRIC | General complex square matrix; symmetry not required | LU |
| REAL_STRUCTURALLY_SYMMETRIC | Real matrix with symmetric stored pattern | LU |
| COMPLEX_STRUCTURALLY_SYMMETRIC | Complex matrix with symmetric stored pattern | LU |
| HERMITIAN | `A == A.conjugate().T` | LU |

Hermitian is not complex transpose symmetry: the implemented LDLT cannot
substitute for LDLH. Structural symmetry includes stored zeros and does not
establish numerical symmetry. Lower storage is accepted only for AUTO or the
numerically symmetric declarations, and implies LDLT under AUTO.

`Path.AUTO`, `Path.LU` and `Path.LDLT` are implemented. LDLH, CHOLESKY, QR, LQ and
SVD are reserved and raise NotImplementedError.

Only `Ordering.SCALPEL` is implemented. NATURAL, AMD, COLAMD, SYMAMD,
MMD_AT_PLUS_A, MMD_ATA, SCOTCH, NESDIS and USER raise NotImplementedError.

Reserved matrix types also raise NotImplementedError: REAL_SYMMETRIC_POSITIVE_DEFINITE,
REAL_SYMMETRIC_POSITIVE_SEMIDEFINITE, REAL_SYMMETRIC_INDEFINITE,
COMPLEX_SYMMETRIC_INDEFINITE, HERMITIAN_POSITIVE_DEFINITE,
HERMITIAN_POSITIVE_SEMIDEFINITE, HERMITIAN_INDEFINITE, REAL_SKEW_SYMMETRIC,
COMPLEX_SKEW_SYMMETRIC, SKEW_HERMITIAN, REAL_RECTANGULAR and COMPLEX_RECTANGULAR.
No automatic fallback or definiteness verification is implied by these names.

## F.destroy() and the context manager

`destroy()` releases the Metal buffers, the symbolic analysis and the CPU matrix
copies immediately, through the extension, rather than waiting for the solver's
reference count to reach zero. Use it when a large factorization is still
referenced -- from a list, a traceback, or a notebook's output history -- and the
next one is about to be built.

It is idempotent. After it, `n` and `path` still answer; `factorize`, `solve`,
`solve_many`, `info`, `diagnostics` and the numerical properties raise
`ValueError`. `Solver` implements `__enter__`/`__exit__`, so a `with` block frees
the solver on exit whether or not the body raised.

Underneath, `sparta::Solver::destroy` releases the Metal context and clears the
plan's arrays, and `Plan::release` does the same for a plan alone. Both are
idempotent and leave `Drop` with nothing to do.

## Inspection and availability

- `sparta.gpu_available()` reports whether Metal is visible to the process.
- `F.path` is the selected Path enum; `F.n` is the matrix dimension.
- `F.info` reports symbolic counts and byte estimates.
- `F.factor_device_seconds` is GPU factor time, excluding CPU analysis/preparation.
- `F.perturbed_pivots` counts rescued pivots; zero is not an accuracy certificate.
- `F.reanalyses` counts the factorize calls that met a new pattern and analyzed again.
- `F.diagnostics()` reports sizes, current options, factorization state and reports.
- `F.quality` returns a `FactorQuality` describing how far to trust the factors.
  It is ready as soon as `factorize` returns, needs no solve, and costs no GPU
  work and no matvec: the pivot range behind it is a host scan of the block
  diagonal already resident in shared memory.
  - `.verdict` is `'good'`, `'degraded'`, `'unusable'` or `'unknown'`, and
    `.usable` is False only for `'unusable'`. `'degraded'` means the answer is
    worth having but less accurate than its residual suggests, so refine it and
    read `estimated_relative_error`. `'unusable'` means corrections cannot help,
    because the iteration will not contract.
  - `.pivot_ratio` is the smallest pivot magnitude over the largest, bounded
    above by the reciprocal of the factorization's condition number. 2x2 pivots
    contribute their block determinant, not their own diagonal entries, which
    alone mean nothing. Measured: 4.7e-5 to 2.1e-6 on healthy fixtures, 3.4e-8
    on one that still solves to 1.6e-3, and 9.2e-19 on one refinement cannot
    move at all.
  - `.relative_residual` is nan unless refinement ran and already measured it.
    Where both are known, the residual decides the verdict.
  - The LU path reports `'unknown'`: its pivots sit on U's diagonal inside the
    factor store, out of the scan's reach.

The experimental `SPARTA_LU_DIAG_THRESHOLD` environment variable is separate from
this API. It is captured at factor allocation: zero uses strict partial pivoting;
positive values prefer a diagonal at least that fraction of the eligible column
maximum. The tested 0.1 can help some matrices; 0.001 failed controlled tests.
It does not affect LDLT.


## optimize(problems=None, *, name=None, ...) → Optimization

Tunes a `Config` to this computer by measuring: every candidate setting is
timed on the given problems (median factorize + solve, refinement off) and
kept only when it beats the current value by more than `min_gain` and by more
than the timing noise (interquartile range over median of its samples), holds
up in a head-to-head recheck, and has a residual no worse than 10x the
defaults'. The search runs in Rust
(`sparta::optimize`); Python prepares the matrices exactly as `analyze`
would and prints the progress lines Rust formats.

```python
result = sparta.optimize([A1, (A2, b2)], name="antenna", time_limit=300)
print(result.speedup, result.changes)
solver = sparta.analyze(A, config="antenna")
```

| Argument | Default | Meaning |
|---|---|---|
| `problems` | `None` | A sparse matrix, a `(matrix, rhs)` pair, or a list of either. `rhs` is a vector or an `(n, k)` block. `None` uses two standard Q1 FEM Helmholtz cubes (14k and 47k unknowns). |
| `name` | `None` | Save the result as this profile. Preset names raise ValueError before anything runs. |
| `gpu`, `cpu` | `True` | Backends to tune. The GPU is left out when unavailable. The result's `force_cpu` selects the faster; both keep their tuned fields. |
| `start` | defaults | `Config`, profile name or preset name: the starting point and the comparison baseline. |
| `analysis` | `False` | Add analysis wall time to each measurement. |
| `time_limit` | `None` | Seconds of search before keeping the best so far; the closing comparison still runs. |
| `iterations` | `1` | Full passes over all settings, each starting from the previous pass's result. Stops early after a pass that changes nothing. |
| `min_gain` | `0.03` | Smallest improvement kept, as a fraction; raised to the timing noise when that is larger. `0 <= min_gain < 1`. |
| `verbose` | `True` | Print each step to stderr as `[done/total elapsed \| ~left] ...`, then the summary. |
| `progress` | `None` | Called with a `sparta.Progress(line, done, total, elapsed, remaining)` per step. Raising from it stops the search and propagates; so does Ctrl-C. |
| `mtype`, `path`, `triangle` | as `analyze` | Applied to every problem. |
| `note` | `""` | Stored with the profile. |

`Optimization` has `config` (Config), `backend` (`sparta.Backend`),
`speedup` (default seconds on the default backend over tuned seconds on the
chosen one), `iterations` (`(run, asked)`), `seconds` (`{Backend: (default, tuned)}`, measured back to back
at the end), `changes` (`{name: (default, tuned)}`), `trials` and `elapsed`.
`str(result)` is the summary; `result.save(name)` saves it later.

Profiles are stored by the Rust `sparta::profiles` module in `configs.json`
in `~/Library/SPARTA` (macOS) or `~/.sparta`, or in the file `SPARTA_CONFIG`
names. `Config.save(name, note=)`, `Config.load(name)`, `Config.delete(name)`,
`Config.store()` and `sparta.profiles()` manage it. `load` and `delete` raise
KeyError for an unknown name. `load` warns when the profile was tuned on
another chip, and fills settings an older profile lacks with defaults.
`analyze(config=...)` accepts a profile name, then a preset name.

## Numerical symmetry tolerance

`sparta.EPS` (default `1e-10`) is the global **absolute**, entrywise tolerance in matrix-value
units: `abs(A[i,j] - A[j,i]) <= EPS`. Hermitian checks conjugate the mirrored
entry. AUTO detection, declared numerical symmetry and explicit LDLT validation
all use this value, including replacement matrices during factorize.

```python
sparta.EPS = 1e-14  # Set before analyze/factorize; zero requires exact symmetry.
```

Structural symmetry still checks the stored pattern exactly. Increasing EPS
can cause AUTO to select LDLT for slightly asymmetric matrices. LDLT then uses
the lower triangle and its transpose (also for refinement residuals), not an
average of the two triangles. Use LU if those differences must be retained.
Changing EPS does not change the path of an existing solver. The value must be
finite and nonnegative; it is not scaled by the matrix norm.

## Reading the implementation

Start with `Solver` in `_solver.py`. It is the only Python class that owns matrix
or factor data. The former NumericFactor wrapper is gone.

```text
analyze(matrix, arguments with defaults)
    ├─ _Options: validate and record every argument (no defaults here)
    ├─ _prepare_matrix: select path and copy into full/lower CSC
    └─ Rust _Analysis: build ordering, elimination tree and storage layout

F.factorize(matrix=None, overrides=None)
    ├─ copy current _Options and apply only supplied overrides
    ├─ validate replacement matrix; new pattern → Rust _Analysis again
    ├─ optional _equilibrate → _gpu_values (complex64)
    ├─ Rust _Factor: allocate or reuse Metal factor buffers
    └─ retain matching original matrix, scales and options after success

F.solve(b)
    ├─ validate RHS
    └─ _solve_scaled(b)
         └─ Rust _Factor.solve: GPU triangular solve
    or _refine(original matrix, b, callback=_solve_scaled)
         ├─ CPU complex128 residual and solution updates
         └─ same GPU triangular solve for each correction
```

`_Options` is only an immutable record of the solver's current configuration.
Initial defaults live in `analyze`'s signature. `factorize`'s None defaults mean
“keep the existing value”, not “reset to analyze's defaults”. Numerical updates
persist only when factorization succeeds.

Internal helpers begin with `_`. `_types.py` contains only public enum definitions;
`_matrix.py` contains matrix validation/storage preparation; `_scaling.py` and
`_refinement.py` contain the optional numerical algorithms. The native `_Analysis`
and `_Factor` are Rust extension objects, not additional public Python APIs.

## Verbose solve summary

Pass `verbose=True` to analyze, or toggle it with `F.factorize(verbose=True)`.
The default is False. Each solve or solve_many call prints one stderr summary
of at most 16 lines; individual refinement corrections stay quiet.

```python
F = sparta.analyze(A, verbose=True, refinement_steps=12).factorize()
x = F.solve(b)
```

Rust reports the selected LU/LDLT path, the nested-dissection ordering, matrix
size, supernodes, tree levels, largest front, factor/workspace storage and
symbolic fill ratio. This ratio counts the supernodal factor envelope divided by
stored input entries: one triangle including diagonal for LDLT, L+U with the
diagonal counted once for LU. Padding/numerical cancellation mean this is not
an actual count of numerically nonzero factors or allocated factor elements.

Python reports GPU factor time, solve wall time, scaling/refinement settings,
rescued pivots, and residual/status for up to eight RHS. Additional RHS are
summarized by the worst residual across all columns. Residual calculation uses
complex128 original represented values; LDLT mirrors the lower triangle.
Without refinement, verbose mode builds a temporary residual matrix and performs
extra CPU matvecs. Logging costs are excluded from the printed solve time.

## CPU complex128

Select `force_cpu=True` in Python to keep factorization and solves in double
precision on the CPU. Large dense updates use Apple Accelerate on Apple Silicon.
Both LDLT and LU use the same symbolic analysis and multifrontal architecture as
Metal. This is explicit selection; GPU failures never trigger an automatic fallback.

```python
solver = sparta.analyze(A, force_cpu=True, refinement_steps=0).factorize()
x = solver.solve(b)
```

Keep original complex128 matrix values and right-hand sides. CPU outputs are
complex128 even when the inputs are single precision. `refinement_steps=0`
measures the direct solve; the existing refinement and scaling options also work.
CPU tuning uses `cpu_panel`, `cpu_block_rows`, and `cpu_blas_min_rows`.
`threads=n` runs the solver's analysis, factorization and solves on a pool of
its own `n` threads instead of the process-wide pool that `RAYON_NUM_THREADS`
sizes; `SPARTA_NUM_THREADS`, when set, overrides it. The
`factor_seconds` property reports CPU factorization wall time;
`factor_device_seconds` is zero on CPU. `diagnostics()` identifies the backend.

Rust uses `Config { force_cpu: true, ..Config::default() }` with
`factor_c128`, `solve_c128`, or `solve_many_c128`.

The backend can also be selected at factorization time without repeating symbolic
analysis:

```python
solver = sparta.analyze(A)
solver.factorize(force_cpu=True, refinement_steps=0)
x = solver.solve(b)
```

`factorize(force_cpu=None)` (the default) retains the current selection.
`factorize(force_cpu=False)` selects Metal again. Switching releases the old
numerical factors and allocates the chosen backend's factors; a failed switch
leaves solves disabled until a successful factorization.
