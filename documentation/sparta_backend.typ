#set document(title: "Inside SPARTA", author: "SPARTA project")
#set page(paper: "a4", margin: (x: 23mm, y: 22mm), numbering: "1", header: context {
  if counter(page).get().first() > 1 { text(size: 9pt, fill: rgb("596979"))[SPARTA / Backend manual] }
})
#set text(font: "Libertinus Serif", size: 11pt, lang: "en")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#set outline(indent: auto)
#set table(inset: 7pt, stroke: 0.4pt + rgb("cbd5df"))
#show heading: set text(fill: rgb("193b55"))
#show raw.where(block: true): set text(size: 9pt)
#let note(title, body) = block(width: 100%, inset: 11pt, fill: rgb("eef4f7"), radius: 3pt)[
  #strong(title) #linebreak() #body
]
#let source(body) = block(above: 5pt)[#text(size: 9pt, fill: rgb("596979"))[Code to read: #body]]

#align(center)[
  #v(26mm)
  #text(size: 34pt, weight: "bold", fill: rgb("193b55"))[Inside SPARTA]
  #v(5mm)
  #text(size: 19pt)[From sparse elimination to Metal kernels]
  #v(12mm)
  #text(size: 12pt)[A backend instruction manual]
  #v(3mm)
  Working-tree edition · 14 September 2026
]
#v(18mm)
This manual starts where a basic direct solver ends. If you have implemented Gaussian elimination, stored a sparse matrix, and solved a triangular system, you already know the central arithmetic. SPARTA reorganizes that arithmetic so that sparse dependencies turn into batches of dense work on an Apple GPU.

The aim is to explain what happens to a matrix from entry to exit, why the major data structures exist, and how the kernels cooperate. You do not need prior knowledge of multifrontal methods or GPU programming.

This describes the implementation in this repository, including its current restrictions. It is not a specification for every sparse direct solver. Source pointers name files and functions relative to the SPARTA crate; they are useful places to continue reading after each explanation. Constants describe this edition and may change with tuning.

#note[Reading route][Read sections 1–5 for the solver mathematics and sparse structure. Sections 6–10 explain the GPU execution and memory design. Sections 11–13 connect that machinery to accuracy, Python use, and performance diagnosis. The final sections provide a source map and a glossary.]
#pagebreak()
#outline(title: [Contents], depth: 1)
#pagebreak()

= What SPARTA solves

SPARTA solves $A x = b$ for sparse, *complex-symmetric* matrices:
$ A^T = A. $
A transpose swaps row and column indices. It does not conjugate complex values. For example,
$ A = mat(2, 1 + i; 1 + i, 3) $
is complex-symmetric. A Hermitian matrix instead satisfies $A^H = A$, where $H$ includes conjugation. SPARTA does not implement that different factorization. This distinction matters directly in electromagnetic matrices: inserting a conjugate into a Schur update changes the system being solved.

The factorization is an $L D L^T$ factorization with permutations. $L$ is lower triangular with an implicit unit diagonal. $D$ is block diagonal: its blocks are either one scalar or a symmetric $2 times 2$ matrix. The permutations first reduce fill and later move numerical pivots into suitable positions. Conceptually, for the combined permutation matrix $P$,
$ P A P^T approx L D L^T. $
The approximation accounts for rounding and, when needed, the rescue mechanism described in section 12. The implementation stores a global ordering and local pivot permutations separately rather than constructing a dense $P$.

All factorization and substitution arithmetic uses *complex64*: two 32-bit floats, eight bytes per complex value. In Rust this type is named `Complex32`; in NumPy it is `complex64`; in Metal it is a `float2`. Those names describe the same precision here. The GPU does not compute complex128 factors.

#table(columns: (1fr, 2fr),
  [*CPU responsibility*], [*GPU responsibility*],
  [Validate and canonicalize the input.], [Assemble dense fronts from the input and child contributions.],
  [Order the graph and build the symbolic plan.], [Choose pivots, form factor columns, and apply dense updates.],
  [Allocate buffers and submit a precomputed schedule.], [Perform forward, diagonal, and backward substitution.],
  [Convert input precision; reorder and return vectors.], [Keep numerical factors resident between calls.],
)

There is no CPU numerical fallback and no CPU/GPU dense-operation cutoff in this standalone solver. The different numerical routes discussed below are all Metal routes. Analysis itself can run without a GPU; factorization requires one.
#source[`src/lib.rs`, `src/gpu.rs`, `python/sparta/__init__.py`.]

#pagebreak(weak: true)
= Starting from one elimination step

== The arithmetic you already know

Suppose we partition a symmetric matrix into its first pivot and everything remaining:
$ A = mat(d, v^T; v, B). $
If $d$ is usable, the multipliers are $l = v / d$. Eliminating this variable leaves
$ S = B - l d l^T = B - (v v^T) / d. $
$S$ is the *Schur complement*: the smaller system that contains the effect of the eliminated variable. This is exactly the trailing update inside ordinary Gaussian elimination.

Consider the real-valued example below; real entries make the arithmetic easier to follow, but the same steps apply to complex-symmetric entries:
$ A = mat(4, 2, 0; 2, 5, 1; 0, 1, 3). $
The first pivot is $4$, the multipliers are $(1/2, 0)^T$, and the remaining matrix is
$ mat(5, 1; 1, 3) - mat(1, 0; 0, 0) = mat(4, 1; 1, 3). $
The next pivot is $4$, its multiplier is $1/4$, and the last pivot is $11/4$. Thus
$ L = mat(1, 0, 0; 1/2, 1, 0; 0, 1/4, 1), quad
  D = mat(4, 0, 0; 0, 4, 0; 0, 0, 11/4). $

SPARTA performs this same elimination, but does not hold and repeatedly scan one enormous dense matrix. It assembles only the dense pieces needed by the sparse dependency structure.

== Why two-by-two pivots exist

A zero diagonal does not prove that a symmetric matrix is singular. The matrix
$ D = mat(0, 1; 1, 0) $
is invertible, although neither diagonal entry works as the first scalar pivot. Taking both variables together gives a valid $2 times 2$ pivot.

For a block pivot, a row of two multipliers is obtained by solving
$ l D = v. $
The trailing update still has exactly the same form, $B <- B - L D L^T$. Only the width of the pivot changes. This is why SPARTA carries separate diagonal entries, off-diagonal entries of two-by-two blocks, and flags recording where those blocks begin.

== Fill is created by elimination

If a variable connects to two other variables, eliminating it can connect those two even when their original matrix entry was zero. Algebraically, the product $v v^T / d$ can have a nonzero there. Graphically, elimination connects the remaining neighbors of the eliminated vertex.

These new entries are called *fill*. A sparse solver must plan room for them. Counting only the nonzeros of the input matrix tells you neither the memory needed for the factors nor the amount of elimination work.

= From a mesh graph to a useful ordering

== Nested dissection: leave the interface until last

Think of the sparsity pattern as an undirected graph. A vertex is an unknown; an off-diagonal nonzero connects two unknowns. A *separator* is a set of vertices whose removal separates the graph into independent pieces.

Nested dissection orders the interior vertices of those pieces first and the separator last. It applies the same idea recursively inside each piece. In a mesh, this resembles eliminating the interiors of subdomains before solving for their shared boundaries. The benefit is both less fill and independent subproblems that can run together.

#align(center)[#block(inset: 10pt, stroke: 0.6pt + rgb("cbd5df"))[
  Left interior #h(6mm) | separator | #h(6mm) Right interior \
  eliminate first #h(6mm) eliminate last #h(6mm) eliminate first
]]

The setting `ordering="scalpel"`, the only one, selects the independent Rust/Rayon SCALPEL backend (Sparse Coarsening And Local Partitioning Elimination Logic). SCALPEL compresses structural twins, contracts heavy-edge matches, refines weighted vertex separators, and recursively orders independent subgraphs in parallel. It feeds the rest of symbolic analysis. See `documentation/SCALPEL.md` for timing and fill comparisons.

Small subgraphs switch to approximate minimum degree (AMD), which cheaply favors eliminating low-degree vertices. The current default switch is 200 vertices. AMD is therefore still part of the algorithm, although the public solver no longer offers AMD as a top-level ordering option. Independent connected components and sufficiently large recursive sides can use CPU parallelism; the recursion uses Rayon, with a 4096-vertex threshold for forking the sides.

== Elimination tree and postorder

The *elimination tree* records how elimination dependencies flow toward later variables. At column level, its parent relation can be understood as pointing to the first later variable reached by that column's filled factor structure. Independent branches may be eliminated separately; their effects eventually meet at an ancestor.

A *postorder* numbers children before parents. SPARTA composes this traversal with the fill-reducing ordering so that the later symbolic and numerical passes can work from descendants toward ancestors.

The permutation convention is worth learning once:
```text
perm[new_index] = original_index
perm_inv[original_index] = new_index
```
This means a permuted right-hand side is gathered as `b[perm[i]]`. Returning the answer performs the reverse scatter.

== Column counts and supernodes

Symbolic analysis computes filled column counts without carrying matrix values through numerical elimination. The implementation uses the Gilbert–Ng–Peyton column-count method. Its result helps recognize consecutive columns with compatible factor structure.

A *supernode* groups such columns so they can be processed as a dense block. Fundamental supernodes follow structural rules: a consecutive parent-child chain, matching column counts after one elimination, and a single-child condition. SPARTA then *amalgamates* some neighboring small supernodes: it merges them to reduce per-front overhead.

Merging is a trade: it may reserve and process some extra zero entries, but avoids many tiny pieces of assembly, pivoting, and substitution. The default `nemin` is 16; this is a merge criterion, not a guarantee that every node has 16 columns. The actual rules include structural chain merges, adjacency requirements, and a cap on large root merges. An automatic strategy can use a biased postorder to place desirable child merges immediately before their parents. The optional relaxed-merge machinery exists in the symbolic module but is not enabled by the default plan.

#note[Why ordering changes GPU behavior][Ordering determines more than a nonzero count. It determines how many independent fronts exist, their heights, their elimination widths, and the sizes of the dense updates. A wider tree can expose many independent small jobs. A narrow tree can concentrate the work into fewer, much larger jobs. Neither a high GPU utilization percentage nor a small input matrix alone tells you which ordering is faster.]
#source[`src/symbolic/mod.rs`: `symbolic_factorize`; `elimination_tree.rs`, `postorder.rs`, `column_counts.rs`, `supernode.rs`; `crates/scalpel/src/lib.rs`.]

= The front: a local dense elimination problem

A supernode names the variables eliminated together. Its *front* is the dense matrix needed to eliminate them. SPARTA uses two essential dimensions:

#table(columns: (auto, 1fr),
  [`nr` / $m$], [Total number of rows and columns in the square front.],
  [`nc` / $p$], [Number of fully summed variables eliminated here.],
  [$t = m - p$], [Number of trailing variables whose contribution goes upward.],
)

“Fully summed” means all descendant contributions needed for those variables have arrived. The trailing variables are present because they couple to the variables being eliminated, but their complete equations belong to ancestors. They cannot yet be eliminated here.

Partition the assembled front as
$ F = mat(F_(11), F_(21)^T; F_(21), F_(22)). $
The first block has size $p times p$ and the trailing block has size $t times t$. Ignoring permutation notation for this paragraph, the local work is
$ F_(11) = L_(11) D L_(11)^T, quad
  F_(21) = L_(21) D L_(11)^T, $
followed by
$ C = F_(22) - L_(21) D L_(21)^T. $
SPARTA retains the $L$ columns and $D$ for future solves. It sends $C$, the *contribution block*, toward the parent. At a root there are no remaining variables, so the contribution block is empty.

A front with `nr=1000, nc=8` is tall but has few pivots. Its large trailing block receives a rank-8 update. A front with `nr=1000, nc=600` has many sequential pivot decisions and many panels of dense updates. Both have the same square storage size, but they are very different GPU workloads. This is why front height alone is a poor routing rule.

== Where a front's entries come from

Assembly starts with zero storage, places the original matrix entries owned by this supernode, then adds each child's contribution block. A child-to-parent map tells the kernel where each child trailing row belongs in the parent. For instance, if a child's trailing rows are global rows `[7, 11]` and the parent's rows are `[5, 7, 9, 11]`, the map is `[1, 3]` using zero-based positions.

The map embeds the child's entire two-by-two contribution into those parent positions. Several children may add into the same position. Original entries, however, have a unique owning supernode determined by their earlier permuted column. An original separator entry is inserted by its owner rather than repeatedly by every descendant that touches the separator.

The plan reconstructs complete front row sets from the ordered input and child row sets. It does not use the symbolic supernode's provisional row list as if it were a complete numerical front. This distinction is essential for correct assembly.
#source[`src/plan.rs`: `Plan::analyze`, `Node`, `Child`, `Entry`; `initialize_large` and the assembly portion of `factor_nodes` in `src/kernels.metal`.]

= One matrix, from Python to solution

```python
analysis = sparta.analyze(A, ordering="scalpel")
factor = analysis.factor()
x = factor.solve(b)

# New values, identical stored lower-triangle pattern:
factor.refactor(A_next)
x_next = factor.solve(b_next)
```

The first call converts the sparse input to canonical lower CSC, runs ordering and symbolic analysis, and builds the front layouts and assembly maps. CSC means compressed sparse column: `indptr[j:j+2]` gives the range for column $j$ in `indices` and `data`.

Creating the factor also creates its Metal context: buffers, compiled kernel pipelines, and dispatch tables. It converts the values to complex64 and runs the numerical factorization. The factors remain in those buffers. Calling `solve` copies in a permuted right-hand side, executes three substitution stages, and returns the unpermuted result.

`refactor` reuses this allocation and the symbolic plan. It repeats numerical assembly and elimination with the new values; it does not update the old factors by a low-rank formula. Numerical pivots are chosen again. This reuse is particularly valuable for a frequency sweep, where geometry and connectivity stay fixed while material or frequency-dependent values change.

A new sparsity pattern needs new analysis. Even stored zeros participate in the pattern contract. Removing a stored zero can therefore make a matrix unsuitable for an existing plan although its mathematical value has not changed.
#source[`python/sparta/__init__.py`: `Analysis`, `Factor`; `python/src/lib.rs`; `src/gpu.rs`: `Solver::new`, `factor`, `solve`.]

= A short introduction to Metal execution

A *kernel* is a function dispatched over many GPU threads. A *threadgroup* is a cooperating group of threads: they can share scratch memory and synchronize at barriers. A *SIMD group* is a smaller set of lanes that execute together and support operations such as summing their values or finding a maximum. A thread is not a dedicated physical GPU core; the hardware schedules many groups onto its execution resources.

SPARTA uses three kinds of numerical storage inside kernels:

#table(columns: (auto, 1fr),
  [*Device buffers*], [Long-lived arrays holding fronts, factors, maps, values, and vectors.],
  [*Threadgroup memory*], [Scratch shared by the cooperating threads of one group, such as staged factor rows.],
  [*Registers*], [A thread's local accumulators and temporary values. Reusing them avoids repeated buffer traffic.],
)

A barrier coordinates only threads in its own group. It cannot wait for some unrelated group to finish a child front. SPARTA therefore expresses dependencies between fronts and stages through the dispatch schedule. Threads do not spin waiting for other fronts.

Within a pivot group, a maximum search first reduces across each SIMD group, then combines the few group results through shared scratch. This hierarchical reduction avoids a long sequence of full-threadgroup reduction barriers. Index ties prefer the lowest index. Backward substitution uses the same pattern for complex sums.

The bridge compiles Metal Shading Language 3.0 with fast math disabled. The complex arithmetic is explicit `float2` arithmetic; multiplying two complex values forms their real and imaginary components. The current update kernels do not call MPS, BLAS, or SIMD-group matrix-multiply instructions.
#source[`src/metal.m`: pipeline creation and `encode`; `src/kernels.metal`: `mul`, `cdiv`, `reduce_argmax`, `reduce_sum`.]

= Three factorization routes

With the current default constants, every front follows one of these routes:

#table(columns: (1fr, 1fr, 1.6fr),
  [*Shape*], [*Kernel*], [*Execution*],
  [`nr <= 32`], [`factor_nodes`], [One GPU thread does the complete front.],
  [`nr > 32`, `nc <= 32`], [`factor_narrow`], [One 128-thread group takes all pivots; a separate tiled update drains the contribution.],
  [`nr > 32`, `nc > 32`], [`factor_panel`], [One 128-thread group forms a panel; tiled kernels apply its deferred updates. Repeat.],
)

== Short fronts: keep overhead small

`factor_nodes` assembles, searches for pivots, forms multipliers, and updates the whole front in one thread. Many fronts share a dispatch. This is reasonable only while each front is small: cooperation and extra dispatches can cost more than the few arithmetic operations saved.

The short path searches the remaining fully summed block for its largest diagonal and off-diagonal magnitudes. It uses their comparison to choose a scalar or two-by-two pivot. This is a broader search than the column-and-row Bunch–Kaufman search in the cooperative paths. Changing routes is therefore not a promise of identical pivots or bitwise-identical results.

== Narrow fronts: cooperate without repeated launches

`initialize_large` first assembles cooperative fronts. `factor_narrow` then takes all their pivots within a single group. After each pivot it updates the remaining fully summed columns, including their rows below the pivot block, so the next pivot search sees current data.

It leaves the trailing-by-trailing contribution block untouched until all pivots have been formed. `update_schur` then applies the accumulated block in one drain. This keeps thin per-pivot updates inside the group instead of launching a separate grid for each of them.

== Wide fronts: panels of pivots

A wide front has enough work to benefit from splitting pivot formation and dense updates. `factor_panel` accumulates nominally 16 pivot columns. A two-by-two pivot can extend that to 17, so the workspace has 17 columns. Two-by-two blocks are never split just to satisfy the nominal panel boundary.

After the panel is formed, `update_strip` updates the not-yet-eliminated fully summed columns. `update_schur` updates the trailing-by-trailing block. `advance_panel` marks the pending contribution as applied. The next panel can then begin.

#note[Panel width is not front width][A supernode may eliminate hundreds of variables, in multiple panels of 16 columns. A panel is an arithmetic batching choice inside that supernode. A tile is a spatial subdivision of the output of an update. These are three different levels of grouping.]
#source[`src/plan.rs`: `small_front_rows`, `NARROW_COLS`, `PANEL`, `PANEL_COLS`, `is_wide`; `factor_nodes`, `factor_narrow`, `factor_panel`.]

= Pivoting and the deferred-update invariant

== Finding a numerical pivot

The cooperative paths use a Bunch–Kaufman-style decision. At the next column, compare the diagonal magnitude to the largest off-diagonal candidate in that column. If it is sufficiently strong, take the diagonal. Otherwise inspect the candidate's row and decide whether to keep the current diagonal, swap in the candidate diagonal, or take a two-by-two block. The comparison constant in the kernels is approximately 0.6403882.

Candidates are restricted to the remaining fully summed variables. A large coupling to a trailing variable does not make that variable eligible for elimination. SPARTA does not delay pivots to a parent or dynamically enlarge fronts to accommodate such a decision.

Symmetric pivoting swaps both a row and a column. It also updates the node's local permutation so that subsequent substitution associates factor entries with the right global variables. A two-by-two pivot's off-diagonal belongs to $D$; the corresponding internal lower-triangular position is set to zero rather than interpreted as an $L$ multiplier.

== Why the wide front needs a separate panel buffer

Blocking introduces a subtle problem. At the start of a panel, the trailing front has had every earlier panel applied. During panel formation, new pivot columns need up-to-date values, but the bulk update is intentionally postponed.

SPARTA keeps the remaining front *uniformly stale with respect to the current panel*. Updated candidate columns are formed separately in a column-major workspace. “Stale” here has a precise meaning: the missing updates are known and deliberately deferred, not accidentally absent.

To form a candidate column $j$, the panel routine subtracts the contributions of its already completed pivots:
$ w_j = F_(:,j) - W D_("done") W_(j,:)^T. $
Here $W$ represents the completed factor columns of this panel over the active rows. The actual code folds $D$ into a short coefficient vector before taking row dot products.

Suppose the pivot search selects a column outside the current panel. If that column alone had already been updated in place, swapping it into the panel would leave an inconsistent mixture of updated and unupdated columns in the front. With the separate workspace, symmetric swaps act on a consistent remaining front; the already formed workspace rows are swapped correspondingly, and the selected column can be formed correctly.

The second column of a proposed two-by-two pivot must subtract only *completed* pivot contributions. Its first column is still part of the raw pivot block at that moment, not yet an $L$ column. The separate `count` and `dest` arguments in `form_column` express this distinction.

At panel close, the completed factor columns are published into the front. The two update kernels consume them, and the remaining matrix becomes current again. The narrow route performs its column-and-row search through `take_pivot`; the wide route implements the corresponding search on deferred columns inside `factor_panel`. They are related algorithms, not one shared function, and rounding order can differ.
#source[`src/kernels.metal`: `take_pivot`, `form_column`, `apply_block_diagonal`, `swap_panel_rows`, `factor_panel`.]

= Why the dense update is fast

== Change how often the output is touched

A scalar trailing update is an outer product. For each pivot it reads and writes much of the same trailing matrix. Even if the GPU has abundant arithmetic capacity, this can spend most of its time moving bytes.

A blocked update accumulates multiple pivots:
$ C <- C - L_R D L_C^T. $
$L_R$ and $L_C$ are the factor rows needed by one output tile. The output tile is read and written once for the accumulated block rather than once per pivot. Input-panel reads, synchronization, and edge handling still cost time; blocking does not make all memory traffic disappear.

This is the central distinction between “many GPU cores” and an efficient GPU algorithm. The cores need reused operands and enough independent output work. Sending many tiny, isolated outer products can leave the machine limited by launch overhead and bandwidth.

== Inside one Schur tile

`update_schur` dispatches a $16 times 16$ threadgroup: 256 threads. Each thread holds a $4 times 4$ block of complex output accumulators in registers. Together the group covers a $64 times 64$ output tile.

For each chunk of pivot columns:

+ Threads cooperatively load two blocks of factor rows into threadgroup memory.
+ One side is multiplied by the block diagonal $D$ while loading. Two-by-two flags select the paired-column formula.
+ Each thread reuses four values from each side to update its sixteen accumulators.
+ A barrier allows the group to reuse the staging arrays for the next chunk.
+ The group finally writes the valid lower-triangle entries and mirrors them without conjugation.

With a chunk width of 16, the two staging arrays occupy $2 times 64 times 16 times 8 = 16384$ bytes. Partial tiles use bounds checks, so front dimensions need not be multiples of 64. Chunk boundaries avoid splitting two-by-two pivots.

`update_strip` uses a smaller $16 times 16$ output tile with one complex accumulator per thread. The strip is the region whose columns still belong to the fully summed block; the Schur tile origin instead starts at row and column `nc`. These two regions must cover the required trailing update without overlapping writes.

== Do not dispatch work that has already finished

The CPU sorts wide fronts by decreasing elimination width. At a given panel step, fronts that may still have work form a prefix of that list. Their Schur tiles are arranged in matching prefixes.

Strip tiles are sorted by tile column. After nominal panel $q$, the first $16(q+1)$ columns have already been processed, so the still-needed tile columns form a contiguous suffix. The bridge dispatches that suffix instead of launching all strip tiles and relying on each group to discover it has no work.

The schedule is conservative because a two-by-two pivot can advance an extra column. GPU state still checks the actual progress and suppresses any remaining no-work cases. The static dispatch audit verifies coverage and counts for the guaranteed panel advance; it is not a trace of every actual numerical pivot.
#source[`src/kernels.metal`: `fold_diagonal`, `update_strip`, `update_schur`; `src/plan.rs`: `dispatch_plan`, `dispatch_audit`; `examples/dispatch_audit.rs`.]

= Scheduling and unified memory

== Levels provide the dependency boundary

Leaves have level zero. A parent has one plus the maximum level of its children. At each level, the bridge submits short fronts, cooperative assembly, narrow factorization and its drain, then the wide panels and their updates. It finally copies that level's factor columns to permanent storage.

Fronts in one batch are independent and can run concurrently, but successive dispatches use a serial compute encoder. The schedule does not overlap arbitrary stages from arbitrary levels. A parent waits for the required earlier dispatches; this also conservatively delays some ready work behind unrelated work.

The CPU does not read back every pivot choice. `FrontState` carries progress on the GPU. Command buffers are committed after each 64 wide-panel passes to bound command size. They stay ordered on the same queue, without a CPU wait between those chunks. At the end of the public factor or solve call, the bridge waits for completion before returning or reading results. GPU execution is asynchronous during submission, while the public API is synchronous.

== Shared memory is not free memory traffic

All numerical buffers use `MTLResourceStorageModeShared`. CPU and GPU access shared allocations in the unified-memory system. SPARTA still copies the caller's values into its input buffer, and copies vectors into and out of its vector buffer. It is not operating directly on arbitrary NumPy array storage.

The important saving is that fronts, factor columns, pivot state, and child contributions remain in these GPU-accessible allocations throughout the numerical work. There is no per-front trip back to CPU factorization. Nevertheless, assembly, factor copying, and updates all consume memory bandwidth. Unified memory removes the need for a separate device-memory copy at each handoff; it does not remove reads, writes, ordering, or allocation costs.

== Recycle fronts, keep factors

Each front reserves a full row-major square, addressed as `offset + i*nr + j`. A completed front contains both factor columns and the child's contribution for its parent. Storing every such square for the whole tree would waste substantial memory.

SPARTA separates their lifetimes:

#table(columns: (1fr, 2fr),
  [*Storage*], [*Lifetime*],
  [Front square], [From its own level through its parent's level, inclusive.],
  [Permanent factor columns], [Until the factor is refactored or destroyed; used by every solve.],
  [Wide-front panel workspace], [Reserved in the solver allocation; reused for that front's successive panels.],
)

Before a level finishes, `store_factors` copies its `nc` columns into permanent column-major storage, addressed as `loffset + k*nr + i`. This layout suits substitution's loop over rows of one factor column. $D$, local permutations, and two-by-two flags are separate arrays. The full front square can be recycled after its parent has consumed the contribution.

The allocator visits levels, places larger fronts first, and finds the first available gap large enough. It merges adjacent free gaps. A parent and its children must have disjoint storage: the parent zeros its own square before reading the children. Fronts within the same level must also be disjoint. Even a child that finished several levels earlier remains live until its parent has assembled it.

Approximate dominant storage is
$ 8 ("front workspace elements" + sum_s m_s p_s + 17 sum_(s " wide") m_s) " bytes", $
plus the original values, row maps, dispatch records, $D$, permutations, status, and vectors. The bridge checks individual allocations against the device's maximum buffer length and their total against its recommended working-set size. The planner also has 32-bit descriptor limits. `max_front=None` removes a user-imposed front-height cap; it does not remove these limits.

Assembly and factor copying use one group per front, as wide as the pipeline permits up to 1024 threads. They are flat strided sweeps that can use that width. A root with only one front still provides only one assembly group; the later tiled update can provide many more groups.
#source[`src/plan.rs`: `assign_front_offsets`; `src/gpu.rs`: buffer sizes; `src/metal.m`: `begin`, `encode`, `sparta_v1_run`; `store_factors`.]

= Solving with the factors

Once the factors exist, a solve is much cheaper than factorization. In conceptual permuted coordinates it is three operations:
$ L y = P b, quad D z = y, quad L^T u = z, quad x = P^T u. $
SPARTA implements local pivot mappings through `global_row` rather than creating separately permuted copies for every front.

== Forward: leaves toward roots

One 256-thread group handles a front. It first solves that front's fully summed triangular block, stepping through the pivots in order. Threads update different remaining rows, with a barrier between pivots.

It then applies the solved values to the trailing rows. Those values belong to ancestors. Two sibling fronts may contribute to the same ancestor row, so the final subtraction uses floating-point atomics on the real and imaginary components. Each thread accumulates a whole row dot product before the atomic subtraction, rather than issuing an atomic for every multiplier. Solved values are staged in chunks of 256 to reduce repeated indirect loads.

== Diagonal: independent block solves

`solve_diagonal` uses one GPU thread per front to divide by scalar pivots or solve two-by-two blocks. Different fronts own disjoint pivot variables, so they can perform this stage independently. Two-by-two arithmetic scales its block before forming the determinant, as in factorization.

== Backward: roots toward leaves

One 256-thread group per front works in reverse column order. Threads form parts of a dot product, reduce their sum, and update the current fully summed variable. Ancestor values are already final. The kernel writes only its own variables, so no atomics are needed here.

The remaining sequential pivot loops are real dependencies. A GPU cannot simply solve every unknown at once. Tree parallelism provides concurrent fronts; row-wise parallelism accelerates work within a front.

Because forward atomic additions can occur in different orders, repeated solves need not be bitwise identical. Floating-point addition is order dependent. This does not itself imply an assembly race, but numerical comparisons should use tolerances and residuals rather than exact equality.
#source[`src/kernels.metal`: `global_row`, `atomic_subtract`, `solve_forward`, `solve_diagonal`, `solve_backward`; `src/gpu.rs`: `Solver::solve`.]

= Precision, pivot rescue, and refinement

== What success does and does not mean

A completed factorization means the implemented checks allowed elimination to finish. It is not a proof of full rank in exact arithmetic, an estimate of the condition number, or a guarantee of complex128 accuracy.

Scalar zero pivots, unusable two-by-two blocks, and nonfinite values are different situations. SPARTA scales a proposed two-by-two block by its largest magnitude and checks the magnitude of the resulting determinant against `PIVOT_EPS = 1e-7`. This is a local numerical test; it does not determine the exact rank of the original system.

The current kernels can rescue some breakdowns. An exactly dead scalar pivot can be replaced by `PIVOT_FLOOR = 3.45e-4` times the largest magnitude found in the current front storage. A wholly zero front has no nonzero scale and still fails. A two-by-two block with an unusable determinant is demoted to a scalar pivot: a nonzero first diagonal is retained; a dead one is replaced if possible. The partner is considered again at the next step.

`perturbed_pivots` counts these rescue events. In particular, demoting a two-by-two block increments the counter even when its nonzero scalar diagonal is retained. The count is therefore not a norm of the matrix change and not simply the number of replaced scalar values. Ordinary small but nonzero scalar pivots are not all raised to a threshold.

Treat nonzero rescue counts as a reason to check the original system carefully. The factors may be useful as an approximate inverse or preconditioner, but useful accuracy or convergence is not guaranteed merely by the rescue succeeding. Nonfinite arithmetic or an unrescuable pivot produces an explicit error. A failed native factorization invalidates its factor until another succeeds; a child failure also prevents its parent from being factored normally.

== Residual versus solution error

For the original high-precision matrix, compute
$ r = b - A x, quad rho = norm(r)_2 / norm(b)_2. $
For a zero right-hand side, use an absolute residual or another explicitly chosen scale instead. The relative residual measures satisfaction of the equations. It does not directly measure the error in $x$: an ill-conditioned matrix can turn a small residual into a much larger solution error. Likewise, an error in an S-parameter is a derived observable, not a substitute for the linear-system residual.

Returning a complex128 array after complex64 substitution merely widens the stored numbers. It cannot recover digits lost in factorization. Keep the original complex128 matrix and right-hand side when checking accuracy; a residual evaluated only against the narrowed matrix answers a different question.

== Refinement exists as an example, not an automatic API feature

`examples/refine.rs` demonstrates mixed-precision refinement outside the core solver:

+ Keep the running solution and original matrix in complex128 on the CPU.
+ Compute $r = b - A x$ in complex128.
+ Scale and narrow the residual; solve a correction equation with the existing complex64 GPU factors.
+ Widen and rescale the correction, add it to $x$ in complex128, and recompute the residual.

The example normalizes each residual by its norm before narrowing, then scales the correction back. This helps preserve a very small residual's significant digits during conversion. It does not improve the quality of the approximate factors themselves.

Refinement can stagnate or diverge if those factors are too inaccurate. Residual monitoring and a stopping policy remain necessary. The current Python `solve` and `refactor` do not run this loop, and diagnostics explicitly report `refinement=False`.
#source[`src/kernels.metal`: `PIVOT_EPS`, `PIVOT_FLOOR`, `floor_pivot`, pivot branches; `src/gpu.rs`: failure decoding and `perturbed_pivots`; `examples/refine.rs`.]

= Practical options and measurements

The Python settings are deliberately small:

#table(columns: (auto, 1fr),
  [`ordering="scalpel"`], [Parallel Rust SCALPEL, the only ordering.],
  [`max_front=None`], [No configured height cap. A positive integer rejects plans with larger fronts; it does not split them.],
  [`triangle="lower"`], [Use the lower triangle and interpret it symmetrically. Full input is accepted, but its upper values do not define the solve.],
  [`triangle="full"`], [Require exact equality with the transpose before extracting the lower triangle.],
)

RSLAB options such as `dense_backend`, `panel_nb`, and its GPU environment switches do not configure this solver. `SPARTA_SMALL_FRONT` and `SPARTA_NEMIN` are compile-time overrides read through Rust's `option_env!`; they are not runtime Python settings. Other panel and tile constants live in the source and are mirrored across languages.

`Analysis.info` reports dimensions, front count, tree levels, largest front, frontal workspace bytes, and factor bytes. Its current `large_fronts` count means fronts above the short-front threshold, including narrow cooperative fronts. Its `gpu_bytes` is `Analysis::device_bytes`: every buffer the backend allocates, sized for one right-hand side. A wider solve grows the solution buffer, and on LU the scratch beside it, to `8 n nrhs` bytes each.

`factor_device_seconds` sums the GPU command durations for the latest factorization. It excludes Python preparation, input conversion, context creation, and host-side waiting outside those device intervals. It is not end-to-end wall time and is not the last solve time. Repeated `analysis.factor()` calls create fresh factor contexts; `factor.refactor()` is the path for reusing them.

`solve_many` currently loops over columns and submits one solve per right-hand side. It is not a batched triangular kernel. Python calls hold the GIL, and factor objects must be used on the thread that created them.

== Interpreting a slow case

#table(columns: (1fr, 1.6fr),
  [*Observation*], [*Mechanism to investigate*],
  [Many tiny fronts], [Assembly and per-front work may dominate. Examine amalgamation and route counts.],
  [Tall fronts with small `nc`], [Low update rank despite large squares; narrow-route work and assembly traffic matter.],
  [A few large root fronts], [Tiled updates have parallelism, but single-group assembly and serial pivot dependencies can limit throughput.],
  [Large device time reduction but little wall-time gain], [Measure conversion, allocation, analysis, and repeated context creation separately.],
  [High memory use], [Inspect live frontal workspace, permanent factors, panels, and the remaining allocation total.],
  [Poor residual or rescue events], [Investigate numerical accuracy before accepting a timing win.],
)

Use `examples/bench.rs` for staged timings and high-precision residuals, `dispatch_audit` for planned update coverage, and `plan_digest` when checking whether a symbolic change altered the numerical work. A fast solve on one mesh does not calibrate every tree shape. Compare the same matrix values, ordering, reuse policy, precision, and residual criteria.
#source[`python/sparta/__init__.py`: `Settings`, `info`, `diagnostics`, `solve_many`; `src/plan.rs`: introspection; `examples/`.]

= A guided source tour

Read the code in the following order if you want to make a backend change:

#table(columns: (1.1fr, 1.9fr),
  [*File or directory*], [*Question it answers*],
  [`src/gpu.rs`], [What does one factor/solve call own, validate, submit, and return?],
  [`src/plan.rs`], [Where does every front row come from, where is it stored, and which jobs run together?],
  [`src/metal.m`], [In which order are the kernels submitted, and when may the CPU reuse the buffers?],
  [`src/kernels.metal`], [What arithmetic and synchronization happen inside each job? Start with `factor_nodes`, then narrow, panel, and tiled updates.],
  [`src/symbolic/`], [How do ordering, tree traversal, column counts, and merging produce supernodes?],
  [`crates/scalpel/`], [How does the graph become a nested-dissection ordering?],
  [`python/`], [Which matrix forms, dtypes, reuse patterns, and options does the caller actually get?],
)

The cross-language data records are small arrays of integers rather than pointer-rich Rust objects. `Node` describes dimensions and offsets; `Child` locates a child and its assembly map; `Entry` locates an original value and its front position; `UpdateTile` names a front and tile coordinates. `FrontState` stores progress and rescue counts. Their Rust layouts must match Metal, and the bridge's level and step records must match the plan as well.

Buffer slots group naturally into metadata (nodes, rows, maps, entries and dispatch IDs), numerical input and workspaces (values, fronts, panels), persistent factors ($L$, $D$, permutations and flags), and execution state (status, progress and vectors). Slot 14 is special: dispatch-specific start/count data is supplied with `setBytes`. The fixed binding table in `gpu.rs` and `metal.m` is the detailed reference when changing the interface.

== Invariants to preserve

+ Every input entry has one owner; every child trailing row maps into its parent.
+ Parents read complete child contributions, and no live front overwrites another.
+ A level's factor columns are copied out before its front storage can be recycled.
+ Deferred panel arithmetic never mixes inconsistent update states in the remaining matrix.
+ Two-by-two blocks remain paired in pivot formation, updates, storage flags, and diagonal solves.
+ Tile outputs are disjoint; only forward substitution's shared ancestor contributions use numerical atomics.
+ Every thread in a cooperating group reaches the required barriers; early exits around barriers must be group-uniform.
+ Rust/Objective-C/Metal record layouts and tuning constants agree.

The intended validation layers are symbolic and allocation tests in Rust, dispatch-coverage audits, and Metal numerical comparisons against complex128 references. The checked-in tests include front-size boundaries, child contributions, two-by-two pivots, and refactorization. In this working-tree edition, some tests still use removed APIs; their presence is not evidence that the current full suite passes. Use actual validation output when judging a change.

= Glossary

#table(columns: (auto, 1fr),
  [*Amalgamation*], [Merging small supernodes to trade extra dense work for less per-node overhead.],
  [*Assembly*], [Adding owned original entries and mapped child contributions into a dense front.],
  [*Contribution block*], [The Schur complement passed from a front to its parent.],
  [*Elimination tree*], [The structure describing how elimination dependencies flow toward ancestors.],
  [*Fill*], [Entries created by elimination that were not present in the original pattern.],
  [*Front*], [A local dense square containing variables to eliminate and variables they still couple to.],
  [*Fully summed*], [All required descendant contributions have arrived; these variables may be eliminated here.],
  [*Panel*], [A consecutive batch of pivot columns accumulated before a bulk update.],
  [*Pivot*], [A scalar or two-by-two block used to eliminate one or two variables.],
  [*Postorder*], [A tree traversal that places descendants before their parents.],
  [*Rank-k update*], [An update expressed as a product through k factor columns; not a claim that the whole matrix has rank k.],
  [*Schur complement*], [The remaining matrix after eliminating a selected block of variables.],
  [*Supernode*], [A group of columns eliminated together using dense storage and kernels.],
  [*Symbolic analysis*], [Pattern-only planning of ordering, dependencies, fill structure, and storage.],
  [*Threadgroup*], [GPU threads that can cooperate through shared scratch and barriers.],
  [*Tile*], [A small rectangular portion of an output matrix assigned to one update group.],
)
