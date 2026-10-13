# SCALPEL

**S**parse **C**oarsening **A**nd **L**ocal **P**artitioning **E**limination **L**ogic.
A sharp surgical tool for graph dissections.

Deterministic, parallel Rust nested dissection for sparse matrix column ordering.
The crate depends only on Rayon, forbids unsafe Rust, and has no METIS, Metal,
SPARTA, or Python dependency.

## Standalone API

```rust
use scalpel::{order, Options};

// Undirected path 0--1--2, stored in CSR, with both edge directions.
let ptr = [0, 1, 3, 4];
let adj = [1, 0, 2, 1];
let permutation = order(&ptr, &adj, &Options::default()).unwrap();
```

`permutation[new] = old`. Rows must be sorted, unique, and in range. Both
edge directions are required. Diagonal entries are accepted and ignored.
Malformed inputs return an error. Empty and disconnected graphs are supported.
For symmetric matrices, CSR and CSC describe the same graph. For unsymmetric
matrices, construct the structural union `A | A^T` first. This is symmetric
nested dissection, not a column-only `A^T A` ordering such as COLAMD.

Use `RAYON_NUM_THREADS=8`, or call `order` within
`rayon::ThreadPool::install` to choose a private pool. Identical options and
input produce identical permutations regardless of worker count or scheduling.

## Algorithm

1. Validate CSR, drop self loops, and compress exactly identical closed
   neighborhoods. Hashes only filter candidates; full equality confirms every
   merge. Compressed vertices carry the number of original columns.
2. Separate disconnected components before partitioning.
3. Build a weighted multilevel hierarchy with degree-prioritized heavy-edge
   matching and summed parallel-edge weights. Reject ineffective contractions.
4. Search deterministic double-sweep BFS level separators from several
   seeds. Refine with weighted greedy moves and bounded FM passes. FM accepts
   temporary losses and restores the best balanced prefix. Its move queue is
   an indexed 4-ary heap holding one entry per vertex and side, and a move
   that would pull a locked vertex is dropped for the rest of the pass.
   Optionally, add candidates cut along the lowest Laplacian eigenvectors of
   the coarsest graph (`spectral_candidates`).
5. Project and refine at each finer resolution. Compare a direct fine-graph
   candidate to protect against poor contraction choices. By default the best
   coarse candidate is chosen before projection; `fine_selection` projects
   every candidate and chooses on the full graph. Exact bipartite vertex
   covers refine boundaries: matching for unit weights, minimum cut for weighted
   compressed graphs.
6. Repeatedly widen the separator to include every neighbor of a separator
   vertex, block-trim the one-sided vertices back out, and re-run the exact
   cover on the result. Widening gives the cut a corridor several layers thick
   instead of the current boundary alone, so it reaches partitions that
   single-vertex moves cannot. A cycle that fails to lower the cost is
   discarded and ends the loop.
7. Recursively order the independent children with Rayon `join`, then append
   the separator. Small leaves use bitset-based exact weighted minimum fill
   with a halo: each leaf's neighbors in ancestor separators count toward
   degrees and fill but are never eliminated, so leaf vertices that border a
   separator are ordered last. Twin groups expand into consecutive original
   columns at the end.

The separator invariant is that no edge joins the two children. Both children
must fit the configured weight bound. When no useful balanced split is found,
static degree ordering terminates that branch; it does not call METIS. This
fallback bounds recursion but is not a strong ordering for general expanders.

Parallelism covers candidate searches and independent subtrees/components.
Input validation, twin compression and the construction of each contracted
graph are split over rows, and the direct fine-graph candidate runs alongside
the coarse candidates. Heavy-edge matching and each individual refinement pass
remain serial within a task. This is not a complete implementation of mt-Metis's
cooperative parallel coarsening or segmented FM scheme.

Minimum fill is exact on each leaf and its halo, including compressed weights,
except that fill between two halo vertices is not counted: those join ancestor
separators that are dense after elimination anyway. Separator vertices use
static degree order; the algorithm does not simulate their filled graph.

## Options

| Option | Default | Effect |
|---|---:|---|
| `leaf_size` | 32 | Maximum bitset leaf size, 1 through 256 |
| `parallel_threshold` | 4096 | Avoid task overhead below this subgraph size |
| `separator_trials` | 4 | More initial candidates cost more time |
| `refine_all_candidates` | false | Compare large-graph candidates after full refinement; more CPU work |
| `imbalance` | 0.8 | Maximum fraction of total vertex weight in either child |
| `half_levels` | false | Two-ended distance-difference separators instead of single-root level sets |
| `balance_quality` | true | Score partitions by separator weight over the child-weight product, not separator weight alone |
| `wide_cycles` | 4 | Widen-and-re-minimise refinement cycles per bisection; 0 disables them |
| `coarsen_threshold` | 512 | Stop contracting below this size; 0 disables contraction |
| `refinement_passes` | 4 | Maximum greedy sweeps per resolution |
| `fm_passes` | 4 | Maximum FM sweeps per multilevel resolution; 0 is faster but often increases fill |
| `one_sided_refinement` | false | PMORSy-style moves toward the initially smaller child in each pass |
| `compress` | true | Merge exact structural twins |
| `cover_refinement` | true | Exact weighted cover improvement at separator boundaries |
| `spectral_candidates` | 0 | Extra initial candidates from Laplacian eigenvectors, on graphs of at least `parallel_threshold` vertices |
| `fine_selection` | false | Choose among coarse candidates after projecting each to the full graph; more CPU work |

The 4096-vertex task cutoff and 32-vertex leaf size are initial settings, not
hardware-independent optima. See [measurements](../../documentation/SCALPEL.md)
for the refinement tradeoff. Balance applies to weighted columns, not equal
numbers of compressed vertices. The weighted minimum-cut network covers the
separator and one adjacent partition boundary; `wide_cycles` widens that
corridor by growing the separator before each cut, which is a coarser device
than building a corridor of a chosen width directly.

## Literature

Original Rust implementation from the algorithm descriptions, with no source
code copied from METIS, mt-Metis, or other ordering libraries:

- LaSalle and Karypis, *Efficient Nested Dissection for Multicore
  Architectures*: independent candidate searches, recursive task parallelism,
  greedy and FM separator refinement.
- Karypis and Kumar, *A Fast and High Quality Multilevel Scheme for Partitioning
  Irregular Graphs*: heavy-edge contraction and coarse-to-fine refinement.
- Ashcraft, *Compressed Graphs and the Minimum Degree Algorithm*: exact
  closed-neighborhood compression.
- Hendrickson and Rothberg, *Improving the Run Time and Quality of Nested
  Dissection Ordering*: weighted compression, relaxed separator balance, and
  combining dissection with minimum-degree leaves.
- Ashcraft and Liu, *Applications of the Dulmage-Mendelsohn Decomposition and
  Network Flow to Graph Bisection Improvement*: bipartite vertex-cover
  improvement. SCALPEL implements unit-weight matching and weighted bipartite
  minimum cut, not the full Dulmage-Mendelsohn decomposition.
- Pirova, Meyerov, Kozinov and Lebedev, *PMORSy: parallel sparse matrix ordering
  software for fill-in minimization*, sections 3-4: optional refinement toward
  the partition that was smaller at the start of each pass, plus task
  granularity control. SCALPEL does not reproduce the full PMORSy implementation
  or its per-matrix parameter tuning.
- Ashcraft, Duff, Hogg, Scott and Thorne, *Nested dissection revisited*,
  RAL-TR-2016-004, sections 2 and 6: the `cost1` evaluation function, a balance
  parameter that tolerates large imbalance, and the Algorithm 6 refinement cycle
  with the Algorithm 8 block trimming.
  SCALPEL substitutes its exact weighted vertex cover for the paper's fine
  trimming and Ford-Fulkerson step, and does not implement the balance-augmented
  network capacities of section 6.1.3 or SPRAL ND's banded Fiduccia-Mattheyses.
- Pellegrini, Roman and Amestoy, *Hybridizing nested dissection and halo
  approximate minimum degree for efficient sparse matrix ordering*,
  Concurrency: Practice and Experience 12 (2000): ordering a leaf with its
  ancestor-separator neighbors as a halo that is never eliminated. SCALPEL
  applies the idea with exact minimum fill on small bitset leaves, not with
  approximate minimum degree.
- Pothen, Simon and Liou, *Partitioning sparse matrices with eigenvectors of
  graphs*, SIAM J. Matrix Anal. Appl. 11 (1990): cuts along the Fiedler
  vector. SCALPEL computes several low eigenvectors by Lanczos on the
  coarsest graph and tries cut directions in their span.

## Tests

`cargo test -p scalpel` runs without a GPU or SPARTA's build script. Tests cover
malformed CSR, empty graphs, cliques, stars, paths, disconnected components,
random graphs, weighted twin meshes, separator validity and balance, leaf fill,
and deterministic output across thread counts. Minimum covers are checked
against exhaustive enumeration of every 3-by-3 bipartite graph, with unit and
unequal weights. Weighted halo leaf orderings are checked against explicit set
elimination, and the indexed heap against an ordered map of live keys. Both refinement modes and full candidate selection are tested.
