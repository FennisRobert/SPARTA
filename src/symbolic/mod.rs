//! Symbolic analysis: fill-reducing ordering, elimination tree, and supernode
//! partition.
//!
//! This is the pipeline the numeric phase depends on, and nothing more. It
//! orders the matrix, postorders the elimination tree, counts factor columns,
//! and groups columns into supernodes. Ordering is delegated to the independent
//! parallel Rust SCALPEL crate.
//!
//! Original SPARTA work. The supernode merge rule follows SPRAL SSIDS (BSD-3). See LICENSE-THIRD-PARTY and crates/scalpel/README.md.

pub mod column_counts;
pub mod elimination_tree;
pub mod postorder;
pub mod supernode;

use rayon::prelude::*;

/// Output columns per parallel task in [`permute_pattern`]. Chosen so a
/// 100k-column system yields a few dozen tasks: large enough to amortize task
/// overhead, small enough to balance across cores when column lengths vary.
const PERMUTE_BLOCK: usize = 2048;

/// Column count below which the symbolic passes stay on the calling thread.
/// Under this size the Rayon task overhead exceeds the work being spread.
///
/// Not a [`crate::Config`] setting: this is a task-granularity constant tuned
/// to core count and cache, not a choice an application makes. It has a flat
/// optimum and changing it does not change any result.
pub(crate) const PARALLEL_COLUMNS: usize = 4096;

use column_counts::column_counts_gnp;
use elimination_tree::{EliminationTree, NO_PARENT};
use postorder::{biased_postorder, postorder};
pub use supernode::{AmalgamationStrategy, RelaxAmalgamation, Supernode, SupernodeParams};

/// Sparsity pattern of a square matrix in compressed-column form.
#[derive(Clone, Debug)]
pub struct CscPattern {
    pub n: usize,
    /// n+1 column offsets. 32-bit: the plan's descriptors are `u32` too, so a
    /// pattern that does not fit here could not be factorized anyway.
    pub col_ptr: Vec<u32>,
    /// Row indices, ascending within each column.
    pub row_idx: Vec<u32>,
}

impl CscPattern {
    /// Borrows the row indices of column `j`, ascending.
    #[inline]
    pub fn column(&self, j: usize) -> &[u32] {
        &self.row_idx[self.col_ptr[j] as usize..self.col_ptr[j + 1] as usize]
    }
}

/// Selects the nested-dissection backend used before factorization.
///
/// SCALPEL is the only one. The setting stays so a future ordering has a
/// place to go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderingMethod {
    /// Independent parallel Rust nested dissection.
    Scalpel,
}

/// What the numeric phase consumes: the permutation and the supernode
/// partition of the permuted matrix.
pub struct SymbolicFactorization {
    /// `perm[new] = old`: the fill-reducing order composed with a postorder.
    pub perm: Vec<usize>,
    /// `perm_inv[old] = new`.
    pub perm_inv: Vec<usize>,
    /// Supernodes in children-first order.
    pub supernodes: Vec<Supernode>,
    /// The symmetrized input pattern under `perm`, rows ascending per column.
    /// Retained because the frontal layout needs the same permuted adjacency
    /// the column counts were taken from, and recomputing it is not free.
    pub permuted: CscPattern,
}

#[derive(Debug)]
pub struct SymbolicError(pub String);

impl std::fmt::Display for SymbolicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SymbolicError {}

/// Mirrors a lower-triangle pattern into the full symmetric pattern the
/// ordering and elimination tree need.
///
/// `n` is the dimension and `col_ptr`/`row_idx` the lower triangle in CSC,
/// with each column's rows sorted ascending and unique. Every output column is
/// sorted: its entries above the diagonal are scattered in increasing source
/// column order, then its own lower entries are copied after them, so no
/// per-column sort is needed.
pub fn symmetric_pattern(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> CscPattern {
    let mut counts = vec![0u32; n];
    for j in 0..n {
        let column = &row_idx[col_ptr[j]..col_ptr[j + 1]];
        debug_assert!(column.windows(2).all(|w| w[0] < w[1]));
        counts[j] += column.len() as u32;
        for &i in column {
            if i != j {
                counts[i] += 1;
            }
        }
    }
    let mut ptr = vec![0u32; n + 1];
    for j in 0..n {
        ptr[j + 1] = ptr[j] + counts[j];
    }
    let mut rows = vec![0u32; ptr[n] as usize];
    // `at[i]` is the next free slot among column i's entries above the diagonal.
    let mut at = ptr[..n].to_vec();
    for j in 0..n {
        let column = &row_idx[col_ptr[j]..col_ptr[j + 1]];
        for &i in column {
            if i != j {
                rows[at[i] as usize] = j as u32;
                at[i] += 1;
            }
        }
        // Columns before j have all been scattered, so column j's upper part
        // is complete and its lower part follows it.
        let lower = &mut rows[at[j] as usize..ptr[j + 1] as usize];
        for (slot, &i) in lower.iter_mut().zip(column) {
            *slot = i as u32;
        }
    }
    CscPattern {
        n,
        col_ptr: ptr,
        row_idx: rows,
    }
}

/// Applies a symmetric permutation to a pattern. `perm[new] = old`.
#[doc(hidden)]
pub fn permute_pattern(pattern: &CscPattern, perm: &[usize]) -> CscPattern {
    let n = pattern.n;
    let mut inv = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        inv[old] = new;
    }
    let mut ptr = vec![0u32; n + 1];
    for new in 0..n {
        let old = perm[new];
        ptr[new + 1] = ptr[new] + (pattern.col_ptr[old + 1] - pattern.col_ptr[old]);
    }
    let mut rows = vec![0u32; ptr[n] as usize];
    // `ptr` already fixes where every output column lands, so the columns own
    // disjoint slices of `rows` and can be filled and sorted independently.
    // Blocks of columns, rather than single columns, keep the task count near
    // the core count instead of near n.
    let mut blocks = Vec::with_capacity(n.div_ceil(PERMUTE_BLOCK));
    let mut rest = rows.as_mut_slice();
    for first in (0..n).step_by(PERMUTE_BLOCK) {
        let last = (first + PERMUTE_BLOCK).min(n);
        let (head, tail) = rest.split_at_mut((ptr[last] - ptr[first]) as usize);
        blocks.push((first, last, head));
        rest = tail;
    }
    let fill = |&mut (first, last, ref mut block): &mut (usize, usize, &mut [u32])| {
        let base = ptr[first];
        for new in first..last {
            let old = perm[new];
            let src = pattern.column(old);
            let dst = &mut block[(ptr[new] - base) as usize..(ptr[new + 1] - base) as usize];
            for (slot, &i) in dst.iter_mut().zip(src) {
                *slot = inv[i as usize] as u32;
            }
            dst.sort_unstable();
        }
    };
    if n >= PARALLEL_COLUMNS {
        blocks.par_iter_mut().for_each(fill);
    } else {
        blocks.iter_mut().for_each(fill);
    }
    CscPattern {
        n,
        col_ptr: ptr,
        row_idx: rows,
    }
}

fn invert(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (new, &old) in perm.iter().enumerate() {
        inv[old] = new;
    }
    inv
}

#[doc(hidden)]
pub fn order(pattern: &CscPattern, config: &crate::Config) -> Result<Vec<usize>, SymbolicError> {
    match config.ordering {
        OrderingMethod::Scalpel => {
            scalpel::order(&pattern.col_ptr, &pattern.row_idx, &config.scalpel)
                .map_err(|e| SymbolicError(e.to_string()))
        }
    }
}

/// Orders the matrix and partitions it into supernodes.
///
/// The pipeline is: fill-reducing ordering, elimination tree of the ordered
/// pattern, postorder composed into the permutation, column counts, then
/// supernode detection. When the amalgamation strategy resolves to `Renumber`,
/// the tree is postordered a second time with merge-desirable children placed
/// adjacent to their parents so the supernode adjacency test can succeed.
pub fn symbolic_factorize(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    params: &SupernodeParams,
    config: &crate::Config,
) -> Result<SymbolicFactorization, SymbolicError> {
    let full = symmetric_pattern(n, col_ptr, row_idx);
    crate::profile::mark("symmetric_pattern");

    let first = order(&full, config)?;
    crate::profile::mark("order");
    // Building the tree from a materialised pattern beats reading `full`
    // through the permutation: the union-find traversal then walks each column
    // in memory order instead of chasing `perm_inv` for every entry. Measured
    // on em_m1_sym1, the copy plus the faster traversal costs less than the
    // in-place version it replaces. The staged copy dies before the final one
    // is built, so peak memory is unchanged.
    let staged = permute_pattern(&full, &first);
    crate::profile::mark("stage_pattern");
    let first_etree = EliminationTree::from_pattern(&staged);
    drop(staged);
    crate::profile::mark("elimination_tree");

    // Compose the ordering with a postorder of its elimination tree, so that a
    // supernode's columns are contiguous.
    let (post, post_inv) = postorder(&first_etree);
    let mut perm: Vec<usize> = post.iter().map(|&p| first[p]).collect();
    crate::profile::mark("postorder");
    let mut permuted = permute_pattern(&full, &perm);
    crate::profile::mark("permute_pattern");
    let mut etree = EliminationTree {
        parent: (0..n)
            .map(|new| {
                let p = first_etree.parent[post[new]];
                if p == NO_PARENT {
                    NO_PARENT
                } else {
                    post_inv[p as usize] as u32
                }
            })
            .collect(),
        n,
    };
    let mut counts = column_counts_gnp(&permuted, &etree);
    crate::profile::mark("column_counts");

    let mut effective = params.clone();
    if matches!(effective.amalgamation_strategy, AmalgamationStrategy::Auto) {
        effective.amalgamation_strategy = supernode::pick_amalgamation_strategy(&etree);
    }
    if matches!(
        effective.amalgamation_strategy,
        AmalgamationStrategy::Renumber
    ) {
        let bias = supernode::predict_merges(&etree, &counts, &effective);
        if bias.iter().any(|&b| b) {
            let (post2, post2_inv) = biased_postorder(&etree, &bias);
            perm = post2.iter().map(|&p| perm[p]).collect();
            crate::profile::mark("amalgam/postorder");
            permuted = permute_pattern(&full, &perm);
            crate::profile::mark("amalgam/permute_pattern");
            etree = EliminationTree {
                parent: (0..n)
                    .map(|new| {
                        let p = etree.parent[post2[new]];
                        if p == NO_PARENT {
                            NO_PARENT
                        } else {
                            post2_inv[p as usize] as u32
                        }
                    })
                    .collect(),
                n,
            };
            // Both postorders are topological orders of the same tree, so they
            // are equivalent orderings with one filled graph: the column
            // counts only move with their columns.
            counts = post2.iter().map(|&p| counts[p]).collect();
            crate::profile::mark("amalgam/column_counts");
        }
    }

    let supernodes = supernode::find_supernodes(&etree, &counts, &effective);
    let perm_inv = invert(&perm);
    crate::profile::mark("supernodes");
    Ok(SymbolicFactorization {
        perm,
        perm_inv,
        supernodes,
        permuted,
    })
}

#[cfg(test)]
pub(crate) mod testing {
    use super::CscPattern;

    /// Minimal lower-triangle triplet builder, enough for the ported unit tests
    /// of the ordering and supernode passes. Values are ignored; only the
    /// pattern matters to everything under test.
    pub(crate) struct CscMatrix {
        n: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
    }

    impl CscMatrix {
        pub(crate) fn from_triplets(
            n: usize,
            rows: &[usize],
            cols: &[usize],
            _vals: &[f64],
        ) -> Result<Self, &'static str> {
            let mut columns: Vec<Vec<usize>> = vec![Vec::new(); n];
            for (&i, &j) in rows.iter().zip(cols) {
                if i >= n || j >= n {
                    return Err("index out of range");
                }
                if i < j {
                    return Err("entries must be lower triangular");
                }
                columns[j].push(i);
            }
            let (mut col_ptr, mut row_idx) = (vec![0usize], Vec::new());
            for column in &mut columns {
                column.sort_unstable();
                column.dedup();
                row_idx.extend(column.iter().copied());
                col_ptr.push(row_idx.len());
            }
            Ok(Self {
                n,
                col_ptr,
                row_idx,
            })
        }

        pub(crate) fn symmetric_pattern(&self) -> CscPattern {
            super::symmetric_pattern(self.n, &self.col_ptr, &self.row_idx)
        }
    }
}
