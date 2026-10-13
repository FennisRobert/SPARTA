//! Supernode partition: which columns share a frontal matrix.
//!
//! Two stages. *Fundamental* supernodes are the columns whose row structures
//! are genuinely nested, so grouping them stores no explicit zeros.
//! *Amalgamation* then merges neighbouring small groups, accepting a few
//! explicit zeros to get fewer and wider dense fronts — which is what the GPU
//! wants, since a wide front is one large matrix multiply instead of several
//! narrow ones.
//!
//! The merge rule follows SPRAL SSIDS (`core_analyse.f90`, BSD-3-Clause,
//! (c) Science and Technology Facilities Council).

use super::elimination_tree::{EliminationTree, NO_PARENT};

/// Parameters controlling supernode amalgamation.
///
/// Symbolic phase only: ordering, amalgamation and small-leaf grouping.
/// Scaling and other numeric-time concerns live on the numeric factor options.
#[derive(Clone)]
pub struct SupernodeParams {
    /// Smallest supernode worth leaving alone. Two neighbouring groups that
    /// are both narrower than this get merged even though it costs fill.
    ///
    /// Default 32, the SPRAL SSIDS default; MUMPS documents 5.
    pub nemin: usize,
    /// How sibling merges are ordered. See [`AmalgamationStrategy`].
    pub amalgamation_strategy: AmalgamationStrategy,
    /// Optional fill-tolerant widening, applied only at scale. `None` keeps
    /// amalgamation to the structural rules above.
    pub relax: Option<RelaxAmalgamation>,
}

/// Fill-tolerant amalgamation limits.
///
/// The PARDISO/MUMPS relaxed-amalgamation lever: trade a little explicit-zero
/// fill for much wider dense fronts, and so higher-rank Schur updates.
#[derive(Clone, Copy)]
pub struct RelaxAmalgamation {
    /// Hard ceiling on the merged supernode's column count.
    pub max_width: usize,
    /// How many explicit zero rows a single merge may introduce.
    pub max_extra_rows: usize,
}

/// Relaxed amalgamation is ignored below this many columns, so small problems
/// — and the supernode-structure tests — keep the structural rules exactly.
pub const RELAX_MIN_N: usize = 1024;

/// Order in which a parent's children are offered for merging.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum AmalgamationStrategy {
    /// Pick from the elimination tree's shape. Never observed downstream:
    /// [`pick_amalgamation_strategy`] resolves it before use.
    #[default]
    Auto,
    /// Walk supernodes in increasing order. On a parent with several children
    /// only the highest-numbered one is column-adjacent, so at most one merge
    /// happens per parent.
    Adjacency,
    /// Walk supernodes in decreasing order, parent before children. Each merge
    /// pulls the parent's first column down to the newly merged child's, which
    /// exposes the next child down for merging. Paired with the merge-biased
    /// postorder this lets every predicted merge actually land.
    Renumber,
}

impl Default for SupernodeParams {
    fn default() -> Self {
        SupernodeParams {
            nemin: 32,
            amalgamation_strategy: AmalgamationStrategy::default(),
            relax: None,
        }
    }
}

/// One frontal matrix: a block of columns eliminated together.
pub struct Supernode {
    /// First eliminated column, in the postordered numbering.
    pub first_col: usize,
    /// Number of eliminated columns. They are `first_col .. first_col + ncol`.
    pub ncol: usize,
    /// Height of the frontal matrix, always at least `ncol`. The first `ncol`
    /// rows are the eliminated columns; the rest are the contribution block.
    pub nrow: usize,
    /// Child supernodes.
    pub children: Vec<usize>,
    /// Budget for delayed pivots arriving from descendants at numeric time.
    ///
    /// The numeric phase enforces `n_delayed_in <= delayed_capacity` and, on
    /// overflow, either reports it or falls back to static perturbation.
    /// `usize::MAX` means unbounded — the behaviour before the budget existed,
    /// and what any code constructing a `Supernode` by hand will observe.
    pub delayed_capacity: usize,
}

/// Above this fraction of branching internal nodes, prefer [`Renumber`].
///
/// Below it the tree is a path or near-path, where `Renumber` over-merges.
/// Chosen from an elimination-tree shape probe over the known-answer matrices:
/// the one case that `Renumber` loses sits at 0.002 and the next-lowest, which
/// it wins, at 0.20, so 0.05 sits comfortably in the gap.
///
/// [`Renumber`]: AmalgamationStrategy::Renumber
pub const AUTO_MULTI_CHILD_FRAC_THRESHOLD: f64 = 0.05;

/// Resolve [`AmalgamationStrategy::Auto`] from the elimination tree's shape.
///
/// Returns [`Adjacency`] for path-like trees and [`Renumber`] otherwise; never
/// returns `Auto`. One `O(n)` pass, once per `symbolic_factorize`.
///
/// [`Adjacency`]: AmalgamationStrategy::Adjacency
/// [`Renumber`]: AmalgamationStrategy::Renumber
pub fn pick_amalgamation_strategy(etree: &EliminationTree) -> AmalgamationStrategy {
    let n = etree.n;
    if n == 0 {
        return AmalgamationStrategy::Adjacency;
    }

    let mut child_count = vec![0u32; n];
    for &p in &etree.parent {
        if p != NO_PARENT {
            child_count[p as usize] += 1;
        }
    }

    let leaves = child_count.iter().filter(|&&c| c == 0).count();
    let internal = n - leaves;
    if internal == 0 {
        // Isolated nodes only: nothing to amalgamate either way.
        return AmalgamationStrategy::Adjacency;
    }

    let branching = child_count.iter().filter(|&&c| c >= 2).count();
    if (branching as f64 / internal as f64) < AUTO_MULTI_CHILD_FRAC_THRESHOLD {
        AmalgamationStrategy::Adjacency
    } else {
        AmalgamationStrategy::Renumber
    }
}

/// Fundamental supernodes, before any amalgamation.
pub(crate) struct FundamentalSupernodes {
    /// First column of each supernode.
    pub(crate) snode_starts: Vec<usize>,
    /// Columns in each supernode.
    pub(crate) snode_ncols: Vec<usize>,
    /// Supernode holding the elimination-tree parent of this supernode's last
    /// column, or `None` at a root.
    pub(crate) snode_parent: Vec<Option<usize>>,
}

/// Group columns into fundamental supernodes.
///
/// Column `j` continues the group ending at `j - 1` when all three hold:
///
/// 1. `j` is the elimination-tree parent of `j - 1`;
/// 2. `j` has one fewer nonzero than `j - 1` — the same rows, minus the column
///    just eliminated;
/// 3. `j - 1` is `j`'s only child, so no other subtree contributes rows.
///
/// Together these say the two columns have nested row structures, so storing
/// them as one dense block adds no explicit zeros.
pub(crate) fn find_fundamental_supernodes(
    etree: &EliminationTree,
    col_counts: &[usize],
) -> FundamentalSupernodes {
    let n = etree.n;
    if n == 0 {
        return FundamentalSupernodes {
            snode_starts: Vec::new(),
            snode_ncols: Vec::new(),
            snode_parent: Vec::new(),
        };
    }

    let mut child_count = vec![0u32; n];
    for &p in &etree.parent {
        if p != NO_PARENT {
            child_count[p as usize] += 1;
        }
    }

    let mut snode_of_col = vec![0usize; n];
    let mut snode_starts = vec![0usize];
    for j in 1..n {
        let continues = etree.parent[j - 1] == j as u32
            && col_counts[j] + 1 == col_counts[j - 1]
            && child_count[j] == 1;
        if continues {
            snode_of_col[j] = snode_of_col[j - 1];
        } else {
            snode_of_col[j] = snode_starts.len();
            snode_starts.push(j);
        }
    }

    let count = snode_starts.len();
    let mut snode_ncols = vec![0usize; count];
    for &s in &snode_of_col {
        snode_ncols[s] += 1;
    }

    let mut snode_parent = vec![None; count];
    for s in 0..count {
        let last_col = snode_starts[s] + snode_ncols[s] - 1;
        let p = etree.parent[last_col];
        if p != NO_PARENT {
            snode_parent[s] = Some(snode_of_col[p as usize]);
        }
    }

    FundamentalSupernodes {
        snode_starts,
        snode_ncols,
        snode_parent,
    }
}

/// Marker in the merge forest for a supernode that has not been absorbed.
const NOT_MERGED: u32 = u32::MAX;

/// Group that has absorbed `s`, following the chain to its end.
///
/// Path halving keeps repeated lookups near-constant; merge chains are short
/// but a long amalgamated run would otherwise be walked from the bottom on
/// every visit.
#[inline]
fn absorbing_group(merged_into: &mut [u32], s: usize) -> usize {
    let mut node = s;
    loop {
        let up = merged_into[node];
        if up == NOT_MERGED {
            return node;
        }
        let up2 = merged_into[up as usize];
        if up2 == NOT_MERGED {
            return up as usize;
        }
        merged_into[node] = up2;
        node = up2 as usize;
    }
}

/// Partition the columns into supernodes and amalgamate the small ones.
///
/// Returns supernodes in postorder — children before parents — which is what
/// the plan builder assumes.
///
/// Amalgamation merges a child group into its parent when their columns are
/// adjacent and one of three rules applies:
///
/// - **trivial chain** — the parent is a single column whose row structure is
///   the child's minus the eliminated column, so the merge is free;
/// - **size** — both groups are narrower than `nemin`, so the fill is worth
///   paying to avoid two tiny fronts;
/// - **relaxed** — the merged group stays within `relax.max_width` and adds at
///   most `relax.max_extra_rows` explicit zero rows. Only above
///   [`RELAX_MIN_N`].
///
/// A merge is rejected when it would grow a root supernode past its cap; see
/// the note on `root_cap` below.
pub fn find_supernodes(
    etree: &EliminationTree,
    col_counts: &[usize],
    params: &SupernodeParams,
) -> Vec<Supernode> {
    let n = etree.n;
    if n == 0 {
        return Vec::new();
    }

    // Relaxed amalgamation applies only at scale, and implies the Renumber
    // order: bushy trees only merge once the columns have been renumbered.
    let relax = if n >= RELAX_MIN_N { params.relax } else { None };
    let relax_rows = relax.map_or(0, |r| r.max_extra_rows);

    let fund = find_fundamental_supernodes(etree, col_counts);
    let snode_parent = fund.snode_parent;
    let count = fund.snode_starts.len();
    let mut first_col = fund.snode_starts;
    let mut ncols = fund.snode_ncols;
    let mut merged_into = vec![NOT_MERGED; count];

    // Frontal height, kept exact across merges.
    //
    // Within a fundamental supernode the column structures are nested, so the
    // first column's count is the height. After a merge that stops being true:
    // the group's first column is the child's, which does not carry the rows
    // only the parent touches.
    //
    // The union still has a closed form. In an elimination tree a parent's row
    // structure contains its child's minus the columns the child eliminates
    // (Liu 1990), so the merged row set is the child's own column block —
    // dense, and disjoint from the parent's rows — together with the parent's:
    //
    //     merged_nrow = child_ncol + parent_nrow
    //
    // which composes along a chain in either iteration order, because a group
    // that later merges upward carries its whole accumulated `ncol` with it.
    let mut nrow: Vec<usize> = (0..count)
        .map(|s| col_counts[first_col[s]].max(ncols[s]))
        .collect();

    // Above this size, cap how wide a root supernode may grow. A very wide
    // root front is effectively dense and also catches every delayed pivot
    // from the subtree below it — the worst combination for memory. Small
    // problems amalgamate freely; `nemin` is the right constraint there.
    const ROOT_CAP_MIN_N: usize = 1024;
    let root_cap = if n >= ROOT_CAP_MIN_N {
        (n / 20).min(2048)
    } else {
        usize::MAX
    };

    let reverse =
        relax.is_some() || matches!(params.amalgamation_strategy, AmalgamationStrategy::Renumber);

    for step in 0..count {
        let s = if reverse { count - 1 - step } else { step };

        let Some(parent) = snode_parent[s] else {
            continue;
        };
        if merged_into[s] != NOT_MERGED {
            continue; // already absorbed
        }
        let group_p = absorbing_group(&mut merged_into, parent);
        if group_p == s {
            continue;
        }

        // Merging is only legal when the child's columns run straight into the
        // parent's. Otherwise the merged group would not be a contiguous block
        // of the numbering, and everything downstream — row construction,
        // assembly, factor storage, the solve gather and scatter — would claim
        // columns belonging to other supernodes.
        //
        // Under a postordered numbering a parent's columns follow all of its
        // descendants', so at most one child of a branching parent is
        // adjacent: the one ending at `parent_first - 1`. An arrow matrix is
        // the extreme case, where only the last child qualifies. The Renumber
        // strategy exists to work around exactly this.
        let child_first = first_col[s];
        let child_ncol = ncols[s];
        let parent_first = first_col[group_p];
        if child_first + child_ncol != parent_first {
            continue;
        }

        let parent_ncol = ncols[group_p];
        let merged_ncol = child_ncol + parent_ncol;
        let child_last = child_first + child_ncol - 1;

        let trivial_chain =
            parent_ncol == 1 && col_counts[parent_first] + 1 == col_counts[child_last];
        let size_based = child_ncol < params.nemin && parent_ncol < params.nemin;
        let relaxed = relax.is_some_and(|r| {
            let extra = col_counts[child_last].saturating_sub(1 + col_counts[parent_first]);
            merged_ncol <= r.max_width && extra <= relax_rows
        });

        let at_root = snode_parent[group_p].is_none();
        let over_root_cap = at_root && merged_ncol > root_cap;

        if (trivial_chain || size_based || relaxed) && !over_root_cap {
            merged_into[s] = group_p as u32;
            ncols[group_p] = merged_ncol;
            // Adjacency guarantees the child sits below the parent, so the
            // merged range is [child_first, parent_first + parent_ncol).
            first_col[group_p] = child_first;
            // The merged front gains exactly the child's own column block;
            // every other child row is already in the parent's row set.
            nrow[group_p] += child_ncol;
        }
    }

    let mut supernodes: Vec<Supernode> = Vec::new();
    let mut renumbered = vec![0usize; count];
    for s in 0..count {
        if merged_into[s] != NOT_MERGED {
            continue;
        }
        renumbered[s] = supernodes.len();
        supernodes.push(Supernode {
            first_col: first_col[s],
            ncol: ncols[s],
            nrow: nrow[s].max(ncols[s]),
            children: Vec::new(),
            // Patched with a real estimate once the tree is wired.
            delayed_capacity: usize::MAX,
        });
    }

    for s in 0..count {
        if merged_into[s] != NOT_MERGED {
            continue;
        }
        if let Some(parent) = snode_parent[s] {
            let group_p = absorbing_group(&mut merged_into, parent);
            if group_p != s {
                let child = renumbered[s];
                supernodes[renumbered[group_p]].children.push(child);
            }
        }
    }

    supernodes
}

/// Which columns want to be merged into their parent supernode.
///
/// Applies the same trivial-chain and size rules as [`find_supernodes`], but
/// *without* the adjacency requirement. The caller feeds the result to
/// [`biased_postorder`] to renumber the columns so that the merges it predicts
/// become adjacent, and therefore legal, on the second pass.
///
/// The result is per column rather than per supernode so the caller can bias
/// elimination-tree nodes directly.
///
/// [`biased_postorder`]: super::postorder::biased_postorder
pub(crate) fn predict_merges(
    etree: &EliminationTree,
    col_counts: &[usize],
    params: &SupernodeParams,
) -> Vec<bool> {
    let n = etree.n;
    let mut wants_merge = vec![false; n];
    if n == 0 {
        return wants_merge;
    }

    let fund = find_fundamental_supernodes(etree, col_counts);
    for s in 0..fund.snode_starts.len() {
        let Some(parent) = fund.snode_parent[s] else {
            continue;
        };
        let child_ncol = fund.snode_ncols[s];
        let parent_ncol = fund.snode_ncols[parent];
        let child_first = fund.snode_starts[s];
        let child_last = child_first + child_ncol - 1;
        let parent_first = fund.snode_starts[parent];

        let trivial_chain =
            parent_ncol == 1 && col_counts[parent_first] + 1 == col_counts[child_last];
        let size_based = child_ncol < params.nemin && parent_ncol < params.nemin;

        if trivial_chain || size_based {
            for flag in wants_merge.iter_mut().skip(child_first).take(child_ncol) {
                *flag = true;
            }
        }
    }

    wants_merge
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::column_counts::column_counts;
    use crate::symbolic::testing::CscMatrix;

    #[test]
    fn test_supernodes_tridiagonal() {
        // Tridiagonal 4x4: col_counts = [2, 2, 2, 1]
        // Columns 2,3 form a fundamental supernode (parent[2]=3, counts[3]+1=counts[2])
        // Columns 0 and 1 are singletons
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        // With nemin=1, we get 3 supernodes: {0}, {1}, {2,3}
        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);
        assert_eq!(snodes.len(), 3);

        let total_cols: usize = snodes.iter().map(|s| s.ncol).sum();
        assert_eq!(total_cols, 4);
    }

    #[test]
    fn test_supernodes_tridiagonal_amalgamated() {
        // With large nemin, all singletons should be amalgamated into one
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 32,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // All 4 columns should be amalgamated into 1 supernode
        let total_cols: usize = snodes.iter().map(|s| s.ncol).sum();
        assert_eq!(total_cols, 4);
        assert_eq!(snodes.len(), 1);
    }

    #[test]
    fn test_supernodes_dense() {
        // Dense 3x3: col_counts = [3, 2, 1]
        // Fundamental: column 1 chains into column 0 (parent[0]=1, counts[1]=counts[0]-1)
        // Column 2 chains into column 1 (parent[1]=2, counts[2]=counts[1]-1)
        // So all 3 columns form one fundamental supernode
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // Should be 1 supernode with 3 columns (fundamental)
        assert_eq!(snodes.len(), 1);
        assert_eq!(snodes[0].ncol, 3);
        assert_eq!(snodes[0].nrow, 3);
        assert_eq!(snodes[0].nrow - snodes[0].ncol, 0); // no contribution block
    }

    #[test]
    fn test_supernodes_block_diagonal() {
        // Two 2x2 dense blocks: two independent supernodes
        let m = CscMatrix::from_triplets(4, &[0, 1, 1, 2, 3, 3], &[0, 0, 1, 2, 2, 3], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // Two fundamental supernodes of size 2
        assert_eq!(snodes.len(), 2);
        assert_eq!(snodes[0].ncol, 2);
        assert_eq!(snodes[1].ncol, 2);
    }

    #[test]
    fn test_supernodes_diagonal_no_amalg() {
        // Diagonal 4x4 with nemin=1: 4 singletons, no merging possible
        let m = CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0; 4]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // Each column is independent (no parents), so 4 supernodes
        assert_eq!(snodes.len(), 4);
    }

    #[test]
    fn test_supernodes_total_columns() {
        // For any matrix, the total columns across all supernodes should equal n
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        for nemin in [1, 5, 32] {
            let params = SupernodeParams {
                nemin,
                ..Default::default()
            };
            let snodes = find_supernodes(&etree, &counts, &params);
            let total: usize = snodes.iter().map(|s| s.ncol).sum();
            assert_eq!(total, 5, "nemin={}: total columns {} != 5", nemin, total);
        }
    }

    #[test]
    fn test_supernode_children_valid() {
        // Verify all child indices are valid
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        for (i, s) in snodes.iter().enumerate() {
            for &child in &s.children {
                assert!(child < snodes.len(), "invalid child index");
                assert!(
                    child < i,
                    "child {} should come before parent {} in postorder",
                    child,
                    i
                );
            }
        }
    }
}
