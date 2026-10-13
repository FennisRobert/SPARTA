# Complex128 CPU backend

`Config { force_cpu: true, ..Config::default() }` selects CPU factorization and
solves. The default remains Metal complex64; failures do not switch backends.
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

Keep `A` and `b` in complex128 to retain their original precision. CPU Python
outputs are always complex128. Scaling and iterative refinement remain available;
`refinement_steps=0` tests the direct factors without corrections. Rust's existing
complex64 methods also work on CPU, but round their inputs/outputs to complex64;
use `factor_c128`, `solve_c128`, and `solve_many_c128` for end-to-end complex128.
The benchmark accepts `--force-cpu` (also `--force_cpu`):

```sh
cargo run --release --example bench -- --force-cpu test_matrices/em_m2_sym1.sparta
```

The C/C++ ABI continues to expose its existing complex64 GPU interface.

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
`level.rs` runs a level across a Rayon pool when it holds at least one front per
thread, and otherwise runs one front at a time with that front's own rows and
row blocks spread instead. Rayon owns the parallelism and Accelerate is set
single-threaded per worker, except inside `zsytrf`, whose serial diagonal chain
it splits better from the inside. That needs a per-thread BLAS thread count,
which Accelerate (`BLASSetThreading`) and MKL (`MKL_Set_Num_Threads_Local`)
have and OpenBLAS does not: `openblas_set_num_threads` is process-wide, so
`dense::set_threading` refuses it and `level::schedule` keeps wide fronts off
the spread path instead. An x86 build against MKL should therefore beat the
same build against OpenBLAS for reasons unrelated to their GEMM kernels;
measure both before concluding anything about either. Multiple RHS columns solve in parallel with
Rayon. Forward contributions are accumulated without GPU atomics. This preserves
sparse storage and the numerical algorithm, not bitwise arithmetic order or GPU
scheduling.

## Precision and settings

All CPU front, factor, panel, diagonal, RHS and output scalars are complex128.
The scaled 2×2 determinant threshold is `f64::EPSILON`; dead pivots are rescued
at `sqrt(f64::EPSILON)` times the front scale. Finite nonzero 1×1 pivots are not
blanket-floored. Perturbations are counted, and failed factorizations invalidate
the factors until a later successful refactor. Destroy releases storage permanently.
Higher precision does not eliminate singularity or conditioning limits.

The independent CPU settings are available in Rust `Config` and in Python
`analyze` and `Config`:

| Setting | Default | Effect |
|---|---:|---|
| `force_cpu` | false | Explicitly select CPU complex128 |
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
