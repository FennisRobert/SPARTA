# SCALPEL ordering and A/B measurement

**SCALPEL** means **S**parse **C**oarsening **A**nd **L**ocal **P**artitioning
**E**limination **L**ogic: a sharp surgical tool for graph dissections.

SCALPEL lives in `crates/scalpel`. Its only external dependency is Rayon.
SCALPEL is SPARTA's only ordering. METIS, the ordering SPARTA used before, was
removed in October 2026 on the measurements below: with four or more threads
SCALPEL orders faster on every system, with fill within a few percent. The
METIS columns in the tables below are that baseline, kept as the record; the
commands that produced them no longer run on current code.

## Selecting the backend

Rust: `Analysis::ldlt` for lower symmetric CSC and `Analysis::lu` for full CSC
both order with SCALPEL, tuned through `Config::scalpel`.

Python: `sparta.analyze(a)`. Analysis uses the same ordering for LU and LDLT.
The selected ordering is retained across refactors and printed in verbose
statistics. Reordering requires a new analysis.

Standalone: `scalpel::order(&ptr, &adj, &options)` returns `perm[new] = old`.
Use `scalpel::Options` for advanced tuning. This API validates sorted symmetric
CSR including reverse edges. The SPARTA adapter uses the default options.

## Reproducible comparisons

Ordering time and symbolic quality, with no GPU required:

```sh
cargo run --release --example ordering_bench -- \
  test_matrices/em_m1_sym1.sparta test_matrices/em_m2_sym1.sparta \
  --curl-curl 20 --threads 1,4,8 --repeat 5
```

The harness constructs the symmetric pattern before timing, warms SCALPEL once
per thread count, and prints the median of repeated calls. SCALPEL's
validation and compression are included. Thread-pool construction is excluded. Exact scalar symbolic `nnz(L)` and `sum(count[j]^2)`
are computed outside the timed section. The latter is a work estimate, not
measured GPU time. Supernode amalgamation/padding is not included in `nnz(L)`.

`--fm 0` disables FM and gives a faster, lower-quality variant. `--fm 2` is the
default. `--coarsen 0` disables multilevel contraction; `--trials N` selects the
number of initial candidates. `--wide N` sets the number of wide-separator
refinement cycles, and `--wide 0` disables them. `--half-levels` selects the
half-level set initial partition, `--imbalance F` sets the maximum fraction of
weight in one child, `--one-sided` applies the PMORSy rule, and `--full-trials`
compares fully refined large-graph candidates. Changing options can improve one
matrix and make another worse.

Full analysis, factorization and solve:

```sh
RAYON_NUM_THREADS=8 cargo run --release --example bench -- test_matrices/em_m1_sym1.sparta
```

Run these from a macOS environment with Metal access. Compare analysis time,
factor/solve time, factor storage, workspace, residual, and perturbed pivots.
For repeated refactorization, amortize ordering over the number of matrices.

## Measurements: 25 September 2026

Apple M4 Pro (14 cores), macOS arm64, optimized release build, `ordering_bench`
with five repetitions after warm-up, default options. Other processes held
about 1.5 cores during the run. Times are local measurements, not performance
guarantees.

| Matrix | Columns | METIS ms | SCALPEL 1 worker ms | 4 workers ms | 8 workers ms | SCALPEL nnz(L) |
|---|---:|---:|---:|---:|---:|---:|
| em_m1_sym1 | 109,370 | 501.0 | 570.8 | 186.4 | 143.3 | 24,202,867 |
| em_m2_sym1 | 68,690 | 245.7 | 296.5 | 97.1 | 72.1 | 8,269,387 |
| em_m3_asym1 | 86,006 | 455.7 | 460.0 | 155.5 | 111.1 | 40,630,601 |
| curl_curl_20^3 | 24,000 | 145.3 | 163.9 | 61.1 | 48.9 | 13,801,172 |
| curl_curl_30^3 | 81,000 | 578.1 | 613.1 | 196.7 | 141.3 | 84,348,316 |

The permutations are bit-identical to those of 20 September; only their cost
changed. Against the tables below, four workers are 1.8x to 1.9x faster and
eight workers 1.6x to 2.3x faster. One worker is still slower than METIS on
every example, by 1% on em_m3_asym1 and by up to 21% elsewhere. The changes
behind this:

- FM keeps each vertex's neighbor weight per child up to date as labels change,
  so a gain costs O(1) instead of a pass over the vertex's neighbors. It pushes
  a heap entry only when a gain changes, and packs entries into one `u128`.
- Contraction orders vertices by counting sort and merges coarse rows through a
  dense marker array instead of sorting duplicate edges; rows are built in
  parallel.
- Input validation checks symmetry from one half of the matrix in parallel;
  twin compression finds twins and builds compressed rows in parallel.
- The weighted min-cut network is one flat array instead of a vector per node.
- The direct fine-graph candidate runs alongside the coarse candidates.

## Measurements: 20 September 2026

Local macOS arm64 execution, optimized release build, five repetitions after
warm-up. Times are local measurements, not performance guarantees. All rows use
the default options.

| Matrix | Columns | METIS ms | SCALPEL 1 worker ms | 4 workers ms | 8 workers ms | METIS nnz(L) | SCALPEL nnz(L) | Fill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| em_m1_sym1 | 109,370 | 487.589 | 859.292 | 335.168 | 303.417 | 23,726,935 | 24,202,867 | +2.0% |
| em_m2_sym1 | 68,690 | 239.677 | 441.570 | 174.340 | 140.372 | 8,317,869 | 8,269,387 | -0.6% |
| em_m3_asym1 | 86,006 | 440.590 | 695.551 | 292.638 | 253.522 | 39,628,413 | 40,630,601 | +2.5% |
| curl_curl_20^3 | 24,000 | 141.052 | 247.640 | 118.190 | 77.098 | 14,445,289 | 13,801,172 | -4.5% |

Three further curl-curl sizes, eight workers, three repetitions:

| Matrix | Columns | METIS ms | SCALPEL ms | METIS nnz(L) | SCALPEL nnz(L) | Fill |
|---|---:|---:|---:|---:|---:|---:|
| curl_curl_16^3 | 12,288 | 66.588 | 51.396 | 5,276,261 | 5,331,287 | +1.0% |
| curl_curl_24^3 | 41,472 | 267.502 | 147.087 | 31,610,177 | 32,153,022 | +1.7% |
| curl_curl_30^3 | 81,000 | 557.696 | 273.196 | 83,554,005 | 84,348,316 | +1.0% |

Eight workers give ordering speedups of 1.61x, 1.71x, 1.74x and 1.83x on the
first table and 1.30x, 1.82x and 2.04x on the second. Fill now ranges from 4.5%
below METIS to 2.5% above it. One SCALPEL worker remains slower than METIS on
every example. Output and fill are identical across worker counts.

The work estimates for METIS/SCALPEL are respectively 14.37/15.01 billion,
2.02/1.97 billion, 66.37/68.53 billion, and 17.59/15.65 billion. These estimates
must not be presented as measured factorization slowdowns. The GPU's front
shapes and dispatches also matter.

The `em_m1_sym2`, `em_m2_sym2` and `em_m3_asym2` matrices share their sparsity
patterns with the `*_sym1`/`*_asym1` files and produce identical orderings, so
they add no structural coverage to the tables above.

### What changed on 20 September 2026

Three settings from Ashcraft, Duff, Hogg, Scott and Thorne, *Nested dissection
revisited* replaced the earlier defaults. Measured on
em_m1/em_m3/curl-curl 20 cubed with eight workers, applied cumulatively:

| Configuration | em_m1 | em_m3 | curl_curl_20^3 |
|---|---:|---:|---:|
| METIS | 23,726,935 | 39,628,413 | 14,445,289 |
| Previous defaults | 27,591,011 | 50,785,513 | 15,150,023 |
| plus `cost1` scoring | 25,900,739 | 45,491,025 | 14,075,667 |
| plus `imbalance` 0.8 | 25,268,719 | 41,229,269 | 13,907,848 |
| plus 4 wide cycles | 24,202,867 | 40,630,601 | 13,801,172 |

1. **`cost1` scoring** (section 2, equation 2.4). Candidate partitions are now
   ranked by separator weight divided by the product of the child weights
   rather than by separator weight alone, so a split is not accepted merely for
   having a thin separator. This is the `balance_quality` option, previously
   present but off by default. The paper reports `cost1` as both better and
   cheaper than its `cost2` alternative, which SCALPEL does not implement.
2. **A balance parameter that tolerates real imbalance** (section 2, equation
   2.3 and section 8.4). SPRAL ND defaults to `imb <= 4.0`, that is a child of
   at most 80% of the weight; SCALPEL previously capped a child at 65%. Table
   8.3 of the paper shows 4.0 and 5.0 giving the lowest fill on most of its test
   set. A sweep over 0.65/0.70/0.75/0.80 here agreed, so `imbalance` is now 0.8.
3. **The wide-separator refinement cycle** (Algorithm 6, with the block
   trimming of Algorithm 8). This is the new code, in `crates/scalpel/src/wide.rs`.
   Each cycle grows the separator to include every neighbour of a separator
   vertex, block-trims the one-sided vertices back out, re-runs SCALPEL's exact
   weighted vertex cover on the result, and applies FM. A cycle whose result
   does not lower the cost is discarded and stops the loop.

The third item is the one that closes most of the remaining gap on the EMerge
matrices. The reason is that the exact cover was previously confined to the
current separator and one adjacent boundary layer: it could only find the best
cut inside a corridor one vertex thick, which is a local optimum that no single
vertex move escapes. Widening first gives the same exact cut a corridor several
layers thick. This addresses the "wider flow corridors" limitation that earlier
versions of this document named as a suspected source of the quality gap.

A sweep of `wide_cycles` over 0/1/2/4/6 showed fill still improving slightly at
4 and flat at 6, with no measurable time cost beyond two cycles because the loop
exits as soon as a cycle fails to improve. The default is 4.

### Options that did not become defaults

`half_levels` (the paper's Algorithm 4 half-level set initial partition) was
re-measured with `cost1` and `imbalance` 0.8 enabled. It improved em_m1 to
23.97 million entries, slightly better than METIS, but made em_m2, em_m3 and
curl-curl worse. Section 8.2 of the paper reports the same unpredictability
between its half-level set and level set methods and offers no rule for
choosing in advance. It remains available as an option and off by default.

The paper's fine trimming (Algorithm 9) and Ford-Fulkerson minimal-separator
step (Algorithm 10) are not implemented, because SCALPEL's existing exact
weighted vertex cover already computes an optimal minimal separator for one
side of the boundary. The balance-augmented network capacities of section 6.1.3
and SPRAL ND's banded Fiduccia-Mattheyses are also not implemented, so partition
balance is enforced only by the cost function, not inside the cut itself.

### Why fill differs

Eliminating a vertex joins its remaining neighbors. A different permutation
therefore creates different fill edges. Separator size, separator placement,
child balance, and ordering within leaves/separators all influence the result.
A separator placed late accumulates contributions from both subtrees, making
its size and placement particularly consequential.

SCALPEL and METIS do not implement identical heuristics: SCALPEL uses its own
matching tie breaks, BFS candidates, bounded greedy/FM search, exact weighted
minimum-degree leaves, and static separator order. It still does not model
ancestor-separator fill in leaf degrees or reproduce METIS's mature search
policy. The remaining few percent on em_m1 and em_m3 is not attributed to any
one of these. The gap is not nondeterminism from Rayon: the permutation is
invariant across worker counts.

### Refinement tradeoff

On em_m1, an earlier greedy-only configuration ordered in roughly 177 ms with
four workers, but produced 42.3 million factor entries versus METIS's 23.7
million. Bounded FM refinement substantially reduced that penalty. The wide
cycles cost roughly 70 ms of the em_m1 ordering time at eight workers, measured
as the difference between `--wide 0` at 220 ms and `--wide 4` at 290 ms, in
exchange for removing about 1.07 million factor entries.

`--full-trials` refines all large-graph candidates before choosing one. It is an
optional quality/time tradeoff, not the default.

### PMORSy heuristic

Pirova et al., *PMORSy: parallel sparse matrix ordering software for fill-in
minimization*, section 3, restrict primitive moves to the partition that was
smaller at the beginning of a refinement pass. SCALPEL exposes this through
`Options::one_sided_refinement` and the benchmark's `--one-sided` flag. The
restriction applies to greedy and FM passes; exact cover improvement remains a
separate phase. This is an adaptation of the documented rule, not a complete
reimplementation of PMORSy. It was measured against the pre-September-20
defaults and has not been re-tuned against the current ones; two-sided
refinement remains the default.

## Validation

- Standalone crate tests cover graph validation, weighted compression,
  parallel determinism, separator invariants, exact leaf behavior, and exhaustive
  minimum-cover checks for all 512 simple 3-by-3 bipartite graphs with both
  unit and unequal weights, plus an independent explicit-fill leaf oracle.
- The wide-separator cycle is covered by four tests in `wide.rs`: expansion
  grows the separator and keeps both children, expansion refuses a growth that
  would swallow a child, block trimming leaves no separator vertex that touches
  only one child, and a full cycle never returns a partition worse than its
  input.
- SPARTA integration tests check both ordering selectors and both LU/LDLT
  analysis paths; explicit SCALPEL matches the default plan digest.
- Rust and Python numerical tests for both backends are included. They require
  Metal and skip when no device is available.
- `sh validate.sh` runs workspace Rust tests, builds release examples and the
  Python extension, then runs the Metal numerical suite. On 20 September 2026
  the 63 workspace Rust tests pass and the Metal suite runs 51 tests with three
  failures and three errors.
- Those six are unrelated to ordering and reproduce with the SCALPEL defaults
  reverted to their previous values. Three are `TypeError` in test-local `Analysis`
  doubles that do not accept the ordering argument added to the native
  constructor. `test_symmetry.test_roundoff_and_override` fails on a path
  expectation, `LDLT` where it wants `LU`. The two `test_lu_real` failures are
  on `em_m2_sym1` model 2, where SPARTA's own residual is 2e-7 against a stored
  reference whose residual is 1.6e-3, so the 7.5% disagreement is with the
  reference, not with the factorization. All six run on the METIS path.

See [the crate README](../crates/scalpel/README.md) for the seven-paper
algorithm provenance and explicit limits of what has been implemented.
