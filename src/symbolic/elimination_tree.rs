//! Elimination tree of a symmetric sparsity pattern.
//!
//! Written from George & Liu, *Computer Solution of Large Sparse Positive
//! Definite Systems* (1981) ch. 5, and Liu, *The role of elimination trees in
//! sparse factorization*, SIAM J. Matrix Anal. Appl. 11(1):134-172 (1990).

use super::CscPattern;

/// Value in [`EliminationTree::parent`] marking a node with no parent.
pub const NO_PARENT: u32 = u32::MAX;

/// Elimination tree of a symmetric matrix `A`.
///
/// `parent[j]` is the smallest `i > j` with `L(i, j) != 0`, or [`NO_PARENT`]
/// when column `j` has no such entry. Because the parent is always a higher
/// column, children always precede their parent in index order — which is why
/// every derived quantity below falls out of a single ascending loop.
///
/// The structure is a forest: one tree per connected component of `A`.
///
/// Parents are `u32`. An `Option<usize>` would cost 16 bytes a node — `usize`
/// has no spare bit pattern, so the niche optimisation does not apply — against
/// 4 here. The symbolic phase is bound by memory latency, and this array is
/// walked repeatedly.
#[derive(Debug, Clone)]
pub struct EliminationTree {
    /// Parent of each node, or [`NO_PARENT`]. Always `parent[j] > j`.
    pub parent: Vec<u32>,
    /// Number of nodes.
    pub n: usize,
}

/// Children of every node, in one flat array.
///
/// One allocation pair rather than the `n` separate `Vec`s a
/// `Vec<Vec<usize>>` would need, and the same layout the GPU scheduler walks.
#[derive(Debug, Clone)]
pub struct Children {
    ptr: Vec<u32>,
    idx: Vec<u32>,
}

impl Children {
    /// Children of `node`, ascending.
    #[inline]
    pub fn of(&self, node: usize) -> &[u32] {
        &self.idx[self.ptr[node] as usize..self.ptr[node + 1] as usize]
    }

    /// Number of children of `node`.
    #[inline]
    pub fn count(&self, node: usize) -> usize {
        (self.ptr[node + 1] - self.ptr[node]) as usize
    }

    /// Index of `node`'s first child in the flat array.
    #[inline]
    pub fn start(&self, node: usize) -> u32 {
        self.ptr[node]
    }

    /// One past `node`'s last child in the flat array.
    #[inline]
    pub fn end(&self, node: usize) -> u32 {
        self.ptr[node + 1]
    }
}

impl EliminationTree {
    /// Build the elimination tree of `pattern`, which must hold both triangles.
    pub fn from_pattern(pattern: &CscPattern) -> Self {
        Self::build(pattern.n, |j| pattern.column(j).iter().map(|&r| r as usize))
    }

    /// Build the elimination tree of the permuted pattern `PᵀAP`, where
    /// `perm[new] = old` and `perm_inv[old] = new`.
    ///
    /// Reads column `perm[j]` and maps its rows through `perm_inv` on the way
    /// past, so the permuted pattern never has to be materialised. The tree is
    /// a function of the pattern alone, so the result matches
    /// `from_pattern(&permute_pattern(pattern, perm))` exactly.
    pub fn from_permuted_pattern(pattern: &CscPattern, perm: &[usize], perm_inv: &[usize]) -> Self {
        Self::build(pattern.n, |j| {
            pattern
                .column(perm[j])
                .iter()
                .map(|&r| perm_inv[r as usize])
        })
    }

    /// Liu's union-find construction, over an abstract view of the columns.
    ///
    /// `column(j)` yields the row indices of column `j` in any order; entries
    /// at or below the diagonal are ignored.
    ///
    /// Columns are absorbed left to right. `reached[v]` records the column that
    /// most recently absorbed `v`, so climbing that chain from a row index `i`
    /// lands on the highest node of `i`'s partial tree. The first time a node
    /// is reached from a later column, that column becomes its parent; every
    /// node passed on the way is re-pointed straight at `j`, which keeps the
    /// climb near-constant and gives the overall `O(nnz · α(n))`.
    fn build<I: Iterator<Item = usize>>(n: usize, column: impl Fn(usize) -> I) -> Self {
        let mut parent = vec![NO_PARENT; n];
        let mut reached = vec![NO_PARENT; n];

        for j in 0..n {
            let jx = j as u32;
            for i in column(j) {
                if i >= j {
                    continue;
                }
                let mut v = i;
                // Climb, flattening the chain onto `j` as we pass.
                while reached[v] != NO_PARENT && reached[v] != jx {
                    let next = reached[v] as usize;
                    reached[v] = jx;
                    v = next;
                }
                // `v` is the top of i's tree. Untouched by j means j is its parent.
                if reached[v] == NO_PARENT {
                    parent[v] = jx;
                    reached[v] = jx;
                }
            }
        }

        EliminationTree { parent, n }
    }

    /// Nodes with no parent, ascending.
    pub fn roots(&self) -> Vec<usize> {
        (0..self.n)
            .filter(|&j| self.parent[j] == NO_PARENT)
            .collect()
    }

    /// Children of every node, ascending within each parent.
    pub fn children(&self) -> Children {
        let n = self.n;
        let mut ptr = vec![0u32; n + 1];
        for &p in &self.parent {
            if p != NO_PARENT {
                ptr[p as usize] += 1;
            }
        }
        // Exclusive prefix sum: ptr[i] becomes the start of i's block, and
        // ptr[n] the total number of children across the forest.
        let mut run = 0u32;
        for slot in ptr.iter_mut() {
            let count = *slot;
            *slot = run;
            run += count;
        }

        let mut cursor = ptr.clone();
        let mut idx = vec![0u32; run as usize];
        // Ascending j fills each parent's block in ascending order.
        for (j, &p) in self.parent.iter().enumerate() {
            if p != NO_PARENT {
                idx[cursor[p as usize] as usize] = j as u32;
                cursor[p as usize] += 1;
            }
        }

        Children { ptr, idx }
    }

    /// Nodes in each subtree, counting the node itself.
    pub fn subtree_sizes(&self) -> Vec<u32> {
        let mut size = vec![1u32; self.n];
        for j in 0..self.n {
            let p = self.parent[j];
            if p != NO_PARENT {
                size[p as usize] += size[j];
            }
        }
        size
    }

    /// Longest path from each node down to a leaf. Leaves are zero.
    ///
    /// This is the quantity the GPU dispatch schedule calls a level: every node
    /// at the same height has all of its children complete, so a height class
    /// can be dispatched as one batch.
    pub fn heights(&self) -> Vec<u32> {
        let mut height = vec![0u32; self.n];
        for j in 0..self.n {
            let p = self.parent[j];
            if p != NO_PARENT {
                height[p as usize] = height[p as usize].max(height[j] + 1);
            }
        }
        height
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::permute_pattern;
    use crate::symbolic::testing::CscMatrix;

    /// Lower-triangle triplets -> full symmetric pattern.
    fn pattern(n: usize, rows: &[usize], cols: &[usize]) -> crate::symbolic::CscPattern {
        let vals = vec![1.0; rows.len()];
        CscMatrix::from_triplets(n, rows, cols, &vals)
            .unwrap()
            .symmetric_pattern()
    }

    fn tridiagonal(n: usize) -> crate::symbolic::CscPattern {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
            }
        }
        pattern(n, &rows, &cols)
    }

    #[test]
    fn empty_matrix() {
        let pat = crate::symbolic::CscPattern {
            n: 0,
            col_ptr: vec![0],
            row_idx: Vec::new(),
        };
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(etree.n, 0);
        assert!(etree.parent.is_empty());
        assert!(etree.roots().is_empty());
        assert!(etree.subtree_sizes().is_empty());
        assert!(etree.heights().is_empty());
    }

    #[test]
    fn diagonal_is_all_roots() {
        let pat = pattern(4, &[0, 1, 2, 3], &[0, 1, 2, 3]);
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(etree.parent, vec![NO_PARENT; 4]);
        assert_eq!(etree.roots(), vec![0, 1, 2, 3]);
        assert_eq!(etree.subtree_sizes(), vec![1, 1, 1, 1]);
        assert_eq!(etree.heights(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn tridiagonal_is_a_chain() {
        let etree = EliminationTree::from_pattern(&tridiagonal(5));
        assert_eq!(etree.parent, vec![1, 2, 3, 4, NO_PARENT]);
        assert_eq!(etree.roots(), vec![4]);
        assert_eq!(etree.subtree_sizes(), vec![1, 2, 3, 4, 5]);
        assert_eq!(etree.heights(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn dense_is_a_chain() {
        // Every column below the diagonal is full, so the factor is full.
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for j in 0..4 {
            for i in j..4 {
                rows.push(i);
                cols.push(j);
            }
        }
        let etree = EliminationTree::from_pattern(&pattern(4, &rows, &cols));
        assert_eq!(etree.parent, vec![1, 2, 3, NO_PARENT]);
    }

    #[test]
    fn arrow_makes_a_star_then_chain() {
        // Column 0 couples to every other column; eliminating it fills in
        // completely, so 1..4 chain upward behind it.
        let rows = vec![0, 1, 2, 3, 4, 1, 2, 3, 4];
        let cols = vec![0, 0, 0, 0, 0, 1, 2, 3, 4];
        let etree = EliminationTree::from_pattern(&pattern(5, &rows, &cols));
        assert_eq!(etree.parent, vec![1, 2, 3, 4, NO_PARENT]);
    }

    #[test]
    fn disconnected_blocks_give_a_forest() {
        // Two independent 2x2 blocks.
        let rows = vec![0, 1, 1, 2, 3, 3];
        let cols = vec![0, 0, 1, 2, 2, 3];
        let etree = EliminationTree::from_pattern(&pattern(4, &rows, &cols));
        assert_eq!(etree.parent, vec![1, NO_PARENT, 3, NO_PARENT]);
        assert_eq!(etree.roots(), vec![1, 3]);
        assert_eq!(etree.subtree_sizes(), vec![1, 2, 1, 2]);
        assert_eq!(etree.heights(), vec![0, 1, 0, 1]);
    }

    #[test]
    fn children_are_flat_and_ascending() {
        // Star: 0, 1, 2 all children of 3.
        let rows = vec![0, 3, 1, 3, 2, 3, 3];
        let cols = vec![0, 0, 1, 1, 2, 2, 3];
        let etree = EliminationTree::from_pattern(&pattern(4, &rows, &cols));
        assert_eq!(etree.parent, vec![3, 3, 3, NO_PARENT]);

        let children = etree.children();
        assert_eq!(children.of(3), &[0, 1, 2]);
        assert_eq!(children.count(3), 3);
        for leaf in 0..3 {
            assert!(children.of(leaf).is_empty());
            assert_eq!(children.count(leaf), 0);
        }
        assert_eq!(etree.heights(), vec![0, 0, 0, 1]);
    }

    #[test]
    fn children_cover_every_non_root_exactly_once() {
        let etree = EliminationTree::from_pattern(&tridiagonal(64));
        let children = etree.children();
        let mut seen = vec![0usize; etree.n];
        for node in 0..etree.n {
            let list = children.of(node);
            assert!(list.windows(2).all(|w| w[0] < w[1]), "children must ascend");
            for &c in list {
                seen[c as usize] += 1;
                assert_eq!(etree.parent[c as usize], node as u32);
            }
        }
        for (j, &count) in seen.iter().enumerate() {
            let expected = usize::from(etree.parent[j] != NO_PARENT);
            assert_eq!(count, expected, "node {j}");
        }
    }

    #[test]
    fn parent_is_always_a_higher_column() {
        let etree = EliminationTree::from_pattern(&tridiagonal(128));
        for j in 0..etree.n {
            let p = etree.parent[j];
            if p != NO_PARENT {
                assert!(p as usize > j, "parent[{j}] = {p} must exceed {j}");
            }
        }
    }

    #[test]
    fn permuted_build_matches_permuting_first() {
        // `from_permuted_pattern` skips materialising P^T A P. It must agree
        // with building the tree of the materialised pattern.
        let rows = vec![0, 1, 2, 5, 1, 3, 2, 4, 3, 5, 4, 5, 5];
        let cols = vec![0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5];
        let full = pattern(6, &rows, &cols);

        for perm in [
            vec![0, 1, 2, 3, 4, 5],
            vec![5, 4, 3, 2, 1, 0],
            vec![2, 0, 4, 1, 5, 3],
        ] {
            let mut perm_inv = vec![0usize; 6];
            for (new, &old) in perm.iter().enumerate() {
                perm_inv[old] = new;
            }
            let direct = EliminationTree::from_permuted_pattern(&full, &perm, &perm_inv);
            let materialised = EliminationTree::from_pattern(&permute_pattern(&full, &perm));
            assert_eq!(direct.parent, materialised.parent, "perm {perm:?}");
        }
    }

    #[test]
    fn subtree_sizes_sum_to_n_over_the_roots() {
        let etree = EliminationTree::from_pattern(&tridiagonal(50));
        let sizes = etree.subtree_sizes();
        let total: u32 = etree.roots().iter().map(|&r| sizes[r]).sum();
        assert_eq!(total as usize, etree.n);
    }
}
