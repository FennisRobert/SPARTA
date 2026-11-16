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

## Supernodal LDLᵀ

`Config::supernodal` (`supernodal` in Python, on by default) replaces the
multifrontal LDLᵀ factorization with a right-looking supernodal one
(`supernodal.rs`). Every node's `nr × nc` block of the factor store is cleared
and given its original entries up front. A node is then factored in place --
`?sytrf` on the fully summed block and the off-diagonal solve, or the serial
kernel on a temporary front for narrow nodes and exactly singular blocks --
and its update `L21 D L21ᵀ` is formed one row block at a time by GEMM into a
per-thread buffer and added straight into the ancestors' columns, under
striped locks because nodes of one level can share an ancestor. No front is
cleared or extend-added, which was most of the multifrontal CPU time on large
EMerge systems, and the frontal workspace is not allocated. The factor comes
out in the multifrontal layout, so the solve, refinement and pivot reporting
are shared. LU and the hybrid's column-major solvers stay multifrontal.

### Real subtrees of a complex LDLᵀ

A complex system whose imaginary parts are confined to a few unknowns, such
as port and boundary unknowns in an FEM model, has a mostly real factor. A
node's `L` block is real when none of its own entries, and none of any
descendant's, has a nonzero imaginary part: every update it receives then
comes from real nodes. Those nodes form whole subtrees, found at every
factorization from the values (`real_nodes` in `mod.rs`), so the ordering,
the fill and the plan do not change.

The complex copies (`cpu_z`, `cpu_c`) keep a `RealPart` from the real copy of
the same precision (`cpu_d`, `cpu_s`). It runs first: the supernodal
factorization over the real parts of every entry, restricted to the real
nodes, in its own real factor store. That leaves each complex node's block
holding the real parts of its entries plus every update the real nodes sent
it. The complex half starts each complex node from that block plus the
imaginary parts of its entries and factors only the complex nodes. Real
nodes' `L` blocks stay in the real store, and the solve multiplies them into
the complex right-hand sides as real matrices: a row-major block of `r`
complex columns is a block of `2r` real ones (`solve.rs`, `Block::Real`).
Only the pivots are copied into the complex store.

Measured on EMerge systems, complex128, M4 Pro, where 0.4-1.6% of unknowns
touch a complex entry. Factor times are a cold first factorization:

| System | real share of factor flops | factor | peak memory |
|---|---|---|---|
| em_m1 | 36% | 0.143 → 0.126 s | |
| em_m2 | 90% | 0.040 → 0.030 s | |
| em_bandcond | 92% | 0.407 → 0.222 s | 2.49 → 1.79 GB |
| em_m4 | 43% | 2.85 → 2.07 s | 7.57 → 6.64 GB |

The real half's dense work runs about four times faster than in complex
arithmetic. What keeps the overall gain smaller is the scatter of updates
into ancestors, which is bandwidth-bound and only halves, and the top of the
tree, which every complex unknown reaches and which stays complex. Moving the
complex unknowns later in the ordering keeps more of the tree real, but on
these systems the extra fill costs more than it saves.

## Shared architecture

Analysis, ordering, supernodes, child scatter maps, level order and reusable
front offsets are unchanged. Input remains lower CSC for transpose-symmetric
LDLᵀ, full CSC for LU. Nothing conjugates. Fully summed columns alone may pivot;
pivots are not delayed to a parent. A front may therefore be left with a tiny
pivot, typically when its zero-diagonal column's 2×2 partner sits in another
front. `Config::static_pivot` (default `1e-12`, τ) raises a nonzero LDLᵀ pivot below
τ·min(max|A|, colmax) to that size, keeping its phase, where colmax is the
largest entry below it in its column. The column is scanned only for a pivot
already below τ·max|A|, so healthy fronts pay nothing, and a small pivot with
nothing below it (a badly scaled diagonal) is kept. The blocked
`?sytrf` cannot, so a front where it took such a pivot is refactored by the
panel kernel. Exact zeros keep their front-relative `sqrt(eps)` rescue. Completed L columns and LU U12 are copied
into permanent storage before a later level can reuse the frontal workspace.

Small fronts follow Metal's serial complete diagonal/off-diagonal search for
LDLᵀ and restricted partial row pivoting for LU. A 2×2 LDLᵀ pivot can cross a
panel boundary and is always consumed whole.

Larger LDLᵀ fronts eliminate their whole fully summed block through the
Bunch–Kaufman `?sytrf` (SpartaBLAS's, in LAPACK's conventions) and then
subtract the entire Schur complement as one rank-`nc` product (`blocked.rs`).
This is the same pivot test the cooperative Metal kernels apply, over the same
rows: the search was already confined to the fully summed block, which is
exactly the block `?sytrf` sees. It interleaves its interchanges with its
elimination steps, so `Front::adopt` replays them onto the columns already
written to recover the single-permutation form the solve and the child scatter
maps assume. A front `?sytrf` reports as exactly singular is
handed back untouched and factored by `panel.rs` with its rescue floor, so
perturbation counting is unchanged.

`panel.rs` forms CPU panel columns over the complete front; it is the LU path
and the LDLᵀ fallback. It does not split fully summed rows and trailing rows
into separate GPU dispatches. `update.rs` packs L and D*Lᵀ (or U), then updates
the whole remaining front once per panel. Metal's strip/Schur dispatch
separation and multi-panel Schur deferral are not needed here. Large
updates go to SpartaBLAS (`crates/sparta-blas`, through `dense.rs`), always
with an ordinary transpose, never a conjugate transpose. Updates below
`cpu_blas_min_rows` use Rust loops.

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
level of one front, the root's, runs that front alone. Rayon owns all the
parallelism. SpartaBLAS starts no threads of its own and has a per-thread
setting (`dense::set_threading`): every worker of a spread level sets
`Caller`, so its products stay on that thread, and a lone front sets
`Machine`, so its products split over the Rayon pool it runs on. Because the
split uses that pool, `RAYON_NUM_THREADS`, `Config::threads` and
`SPARTA_NUM_THREADS` bound the dense work as well.

Letting the products of a spread level's fronts split as well (nested Rayon
parallelism) was measured and is not used: on em_m4 (M4 Pro, complex128) it
factored in 2.6-3.1 s against 2.1-2.6 s. So was reserving the SME matrix units
for a few threads and running the rest on NEON: 2.2-3.1 s against 2.08-2.13 s
for SME everywhere. Scratch buffers that live in thread-locals are taken out
for a call and put back after, never borrowed across a dense call, because
Rayon may run another task on the same thread while a split product waits.

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

## Sparse right-hand sides and projections

`sparse.rs` serves `Solver::solve_sparse` and `Solver::solve_projected`
(`W^T A^-1 B`) on `LDL^T` factors.

- **The reach.** A column of `B` with nonzeros in a few unknowns has
  `y = L^-1 P^T b` zero outside the fronts on the paths from those unknowns'
  fronts to the root. Columns are grouped eight at a time by their first
  permuted unknown, which nested dissection makes a proxy for their subtree.
- **The sweep.** Each group runs the full solve's elimination over the
  group's reach in level order: `Solver::eliminate`, factored out of
  `forward_node`, so real-split blocks, 2x2 pivots and BLAS behave
  identically. Values are held only for the reach's pivot rows.
- **The projection.** It is `(L^-1 P^T W)^T D^-1 (L^-1 P^T B)` for the plain
  transpose: one `gemm_tn` per front two groups share. With `W = B` one
  triangle of group pairs is formed and written to both places, so `Z` is
  exactly symmetric.
- **`solve_sparse`.** It scatters the reaches into the permuted block and
  finishes with the full diagonal and backward sweeps.

The reach comes from the analysis, which is valid because pivots never leave
their front. Delayed pivoting would have to update it. LU and the GPU take
the dense route in `crate::Solver`.

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
| `cpu_blas_min_rows` | 64 | Minimum trailing rows before the panel path calls the dense kernels |

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
