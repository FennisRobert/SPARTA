# CPU backend

`Config { force_cpu: true, ..Config::default() }` selects CPU factorization and
solves. The default remains the GPU; failures do not switch backends.

Wide LU fronts take the blocked route in `blocked_lu.rs` (`?getrf`, two
triangular solves, one Schur GEMM), as wide LDLᵀ fronts take `blocked.rs`;
`panel.rs` is the fallback for exactly singular blocks and for
`lu_diag_threshold > 0`.

The backend is compiled once per value type from this one source: `cpu_z`
complex128, `cpu_c` complex64, `cpu_d` float64, `cpu_s` float32 (see
`src/scalar.rs`). `Solver::factor_values` picks the copy from the
values' type. The examples below use complex128.
The CPU path never creates a Metal context.

```rust
use sparta::{Analysis, Complex64, Config, Solver};
let config = Config { force_cpu: true, ..Config::default() };
let plan = Analysis::ldlt(2, &[0, 2, 3], &[0, 1, 1], &config)?;
let mut solver = Solver::new(plan)?;
solver.factor_c128(&[
    Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0),
    Complex64::new(3.0, 0.0),
])?;
let x = solver.solve_c128(&[Complex64::new(1.0, 0.0); 2])?;
# Ok::<(), sparta::Error>(())
```

In Python:

```python
solver = sparta.analyze(A, force_cpu=True, refinement_steps=0).factorize()
x = solver.solve(b)
X = solver.solve_many(B)
solver.factorize(A_new)  # same sparsity pattern
```

Keep `A` and `b` in double precision to retain their original precision. CPU
Python factors in float64 for a real matrix and complex128 for a complex one,
and returns that precision. Scaling and iterative refinement remain available;
`refinement_steps=0` tests the direct factors without corrections. Rust's existing
complex64 methods also work on CPU, but round their inputs/outputs to complex64;
use `factor_c128`, `solve_c128`, and `solve_many_c128` for end-to-end complex128.
The benchmark accepts `--force-cpu` (also `--force_cpu`):

```sh
cargo run --release --example bench -- --force-cpu test_matrices/em_m2_sym1.sparta
```

It factors in complex128; `--single` with `--force-cpu` factors in complex64,
the precision of the hybrid's CPU half.

The C/C++ ABI exposes the GPU only: complex64, and float32 through
`sparta_factor_real`.

## Shared architecture

Analysis, ordering, supernodes, child scatter maps, level order and reusable
front offsets are unchanged. Input remains lower CSC for transpose-symmetric
LDLᵀ, full CSC for LU. Nothing conjugates. Fully summed columns alone may pivot;
pivots are not delayed to a parent. Completed L columns and LU U12 are copied
into permanent storage before a later level can reuse the frontal workspace.

Small fronts follow Metal's serial complete diagonal/off-diagonal search for
LDLᵀ and restricted partial row pivoting for LU. A 2×2 LDLᵀ pivot can cross a
panel boundary and is always consumed whole.

Larger LDLᵀ fronts eliminate their whole fully summed block through LAPACK's
Bunch–Kaufman `zsytrf` and then subtract the entire Schur complement as one
rank-`nc` product (`blocked.rs`). This is the same pivot test the cooperative
Metal kernels apply, over the same rows: the search was already confined to the
fully summed block, which is exactly the block LAPACK sees. `zsytrf` interleaves
its interchanges with its elimination steps, so `Front::adopt` replays them onto
the columns already written to recover the single-permutation form the solve and
the child scatter maps assume. A front LAPACK reports as exactly singular is
handed back untouched and factored by `panel.rs` with its rescue floor, so
perturbation counting is unchanged.

`panel.rs` forms CPU panel columns over the complete front; it is the LU path
and the LDLᵀ fallback. It does not split fully summed rows and trailing rows
into separate GPU dispatches. `update.rs` packs L and D*Lᵀ (or U), then updates
the whole remaining front once per panel. Metal's strip/Schur dispatch
separation and multi-panel Schur deferral are not needed here. Large
updates go to whichever BLAS `build.rs` linked -- Accelerate on Apple, or the
library `SPARTA_BLAS_LIB` names -- always with an ordinary transpose, never a
conjugate transpose. Smaller updates, and builds with no BLAS at all, use Rust
loops. `SPARTA_BLAS_LIB` takes a comma-separated list, so reference netlib
links as `SPARTA_BLAS_LIB=blas,lapack`.

Only the lower triangle of an LDLᵀ front is assembled and only the lower
triangle of its Schur complement is formed. Child maps ascend, so a child's
lower triangle lands in its parent's, and the entry map already reflects every
LDLᵀ entry below the diagonal. The two kernels that permute rows and columns in
place — the serial narrow-front kernel and the `panel.rs` fallback — restore a
full square first.

Fronts in one level are independent and their workspace slots are disjoint, so
`level.rs` runs every level of two or more fronts across a Rayon pool. A front
holding more than a thread's share of its level's work also splits its own rows
and row blocks into tasks, which threads done with the smaller fronts steal; a
level of one front, the root's, runs that front alone. Rayon owns the parallelism and Accelerate is set
single-threaded per worker, except inside `zsytrf`, whose serial diagonal chain
it splits better from the inside. That needs a per-thread BLAS thread count,
which Accelerate (`BLASSetThreading`) and MKL (`MKL_Set_Num_Threads_Local`)
have and OpenBLAS does not. OpenBLAS's `openblas_set_num_threads` is
process-wide, so `dense::set_global_threading` sets it between levels instead,
when no thread is inside BLAS: one thread for a spread level, the size of the
Rayon pool it runs on for a lone front, so `RAYON_NUM_THREADS` or `threads`
bounds OpenBLAS too and an
`OPENBLAS_NUM_THREADS` meant for another library in the process is overridden.
Its `?sytrf` therefore runs on one thread inside a spread level. The per-thread
controls are only taken from the vendor `SPARTA_BLAS_LIB` names, so an MKL
loaded by something else in the process (PARDISO, say) is not mistaken for the
library SPARTA calls.

`Config::threads` (`src/threads.rs`) gives a solver a Rayon pool of its own;
`SPARTA_NUM_THREADS` overrides it. A solver on its own pool never hands a
product to the library to thread, since Accelerate's and MKL's own threads
would not respect the pool's size: a lone front splits its work over the pool
like a front in a spread level. The controls are resolved by a query that sets
nothing (`dense::controllable`): a lookup that probed by setting the calling
thread to `Machine` would leave that thread threading its BLAS calls inside the
spread levels of the process's first factorization.

On a Ryzen 5 3600XT (Zen 2, 6 cores, Windows) with an 84k complex-symmetric
EMerge patch antenna system, complex128 LDLᵀ took 1.52 s with every level run
one front at a time, 0.55 s with levels spread under OpenBLAS (conda), and
0.65–0.76 s against MKL (conda `mkl_rt`). Compiling for `x86-64-v3` (AVX2,
FMA) and using 6 rather than 12 Rayon threads each made no measurable
difference.

Factor columns are stored row-major per node (`nr` rows of `nc` values), the
layout a front's rows copy into without a transpose; wide LDLᵀ fronts write
them there directly. The hybrid keeps them column-major, the layout the GPU
reads. The solve is multifrontal like the factorization: each node hands its
trailing update to its parent rather than scattering it into rows another node
owns, so the nodes of a level solve concurrently, and every right-hand side of
a block is carried through one pass over the factors (through GEMM from four
right-hand sides). This preserves sparse storage and the numerical algorithm,
not bitwise arithmetic order or GPU scheduling.

`SPARTA_CPU_DUMP=1` prints one line per level of a CPU factorization: depth,
`seq` or `par`, front count, wall seconds, GFLOP and the first front's rows.

## Precision and settings

All CPU front, factor, panel, diagonal, RHS and output scalars are the copy's
value type `C`, and thresholds use its real type `R`. The scaled 2×2
determinant threshold is `R::EPSILON`; dead pivots are rescued at
`sqrt(R::EPSILON)` times the front scale. Finite nonzero 1×1 pivots are not
blanket-floored. Perturbations are counted, and failed factorizations invalidate
the factors until a later successful refactor. Destroy releases storage permanently.
Higher precision does not eliminate singularity or conditioning limits.

The independent CPU settings are available in Rust `Config` and in Python
`analyze` and `Config`:

| Setting | Default | Effect |
|---|---:|---|
| `force_cpu` | false | Explicitly select the CPU backend |
| `cpu_panel` | 32 | Pivots before a trailing update, panel path only |
| `cpu_block_rows` | 96 | Rows per block of the trailing update, and the unit of work when one front is split across threads |
| `cpu_blas_min_rows` | 64 | Minimum trailing rows before the panel path calls BLAS |

`cpu_block_rows` was measured on an M4 Pro `curl_curl_30^3`: 64 and 128 each
cost about 5% against 96, and 256 about 20%. The others are conservative
starting settings, not a measured optimum for every Mac.
`small_front_rows`, ordering, amalgamation and `lu_diag_threshold` remain shared.
GPU-specific dispatch fields have no effect on CPU numerical execution. Analysis
still validates the shared configuration and retains its 32-bit layout limits.

Complex buffers take twice their GPU byte size. Analysis byte counts reflect the
selected backend, so `panel_bytes()` reports zero for CPU. Each Rayon worker
grows its own dense scratch to the largest front it happens to take, which is
outside those counts. Byte counts are
not a complete process peak-memory estimate.

`factor_seconds()` reports CPU factor wall time (GPU command time on Metal).
`factor_device_seconds()` remains GPU-only and reports zero for CPU. Pivot counts,
pivot ranges, multiple RHS columns, refactor and destroy work on both paths.

Python's `quality` retains pivot ranges and perturbation counts on CPU. Its
Metal-calibrated pivot-ratio verdict cutoffs do not apply to complex128; without
a measured residual or perturbations the CPU verdict is `unknown`.

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
