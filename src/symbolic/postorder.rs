//! Postorder traversals of the elimination tree.
//!
//! A postorder numbers every node after all of its descendants, which is what
//! makes a subtree a contiguous range of columns — the property the supernode
//! partition and the column counts both rely on.

use super::elimination_tree::EliminationTree;

// Total children examined while deciding sibling order. Test instrument: the
// ordering is decided once per node, so this must stay linear in `n`.
#[cfg(test)]
thread_local! {
    static SORT_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Postorder of the elimination tree.
///
/// Returns `(order, inverse)` with `order[k]` the node visited at position `k`
/// and `inverse[node]` that node's position.
///
/// Siblings are visited smallest subtree first, which keeps the live set of
/// frontal matrices small: a subtree's contribution has to be held from the
/// moment it finishes until its parent assembles it, so finishing the cheap
/// ones first leaves less on the stack while the expensive one runs.
pub fn postorder(etree: &EliminationTree) -> (Vec<usize>, Vec<usize>) {
    traverse(etree, None)
}

/// Postorder that places merge-desirable children next to their parent.
///
/// Identical to [`postorder`], except that among a parent's children those
/// marked in `bias` are visited last, so their columns end up immediately
/// before the parent's. `find_supernodes` will only merge a child into its
/// parent when their columns are adjacent, so this is what lets a predicted
/// merge actually happen.
///
/// Within each group siblings keep the smallest-subtree-first order, so
/// `biased_postorder(etree, &vec![false; n])` equals `postorder(etree)`.
pub fn biased_postorder(etree: &EliminationTree, bias: &[bool]) -> (Vec<usize>, Vec<usize>) {
    debug_assert_eq!(
        bias.len(),
        etree.n,
        "biased_postorder bias length must equal etree.n"
    );
    traverse(etree, Some(bias))
}

/// Iterative postorder with the sibling order decided up front.
///
/// The visit order of every child block is computed once, into one flat array,
/// before the walk starts. The walk itself then carries only a cursor per
/// stack frame, so a node with `c` children costs `O(c log c)` once rather
/// than re-sorting on each of its `c + 1` visits. That re-sorting is what used
/// to make a star-shaped tree — one root over `n - 1` leaves, the shape a
/// bordered system produces — cost `O(n² log n)` here.
fn traverse(etree: &EliminationTree, bias: Option<&[bool]>) -> (Vec<usize>, Vec<usize>) {
    let n = etree.n;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let children = etree.children();
    let sizes = etree.subtree_sizes();

    // Sort key: biased children last, then smallest subtree first, then node
    // number. The final term makes the order total, so the traversal is
    // reproducible regardless of how the sort breaks ties.
    let key = |c: u32| {
        let late = bias.is_some_and(|b| b[c as usize]);
        (late, sizes[c as usize], c)
    };

    // One flat visit array, each parent's block ordered in place.
    let mut visit: Vec<u32> = Vec::with_capacity(n);
    let mut block: Vec<u32> = Vec::new();
    for node in 0..n {
        block.clear();
        block.extend_from_slice(children.of(node));
        #[cfg(test)]
        SORT_WORK.with(|w| w.set(w.get() + block.len()));
        block.sort_unstable_by_key(|&c| key(c));
        visit.extend_from_slice(&block);
    }

    // Roots have no parent to sit beside, so they only take the size order.
    let mut roots: Vec<u32> = (0..n as u32)
        .filter(|&j| etree.parent[j as usize] == super::elimination_tree::NO_PARENT)
        .collect();
    roots.sort_unstable_by_key(|&r| (sizes[r as usize], r));

    // `cursor[node]` walks that node's block in `visit`.
    let mut cursor: Vec<u32> = (0..n).map(|node| children.start(node)).collect();

    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<usize> = Vec::with_capacity(64);

    for root in roots {
        stack.push(root as usize);
        while let Some(&node) = stack.last() {
            let at = cursor[node];
            if at < children.end(node) {
                cursor[node] = at + 1;
                stack.push(visit[at as usize] as usize);
            } else {
                order.push(node);
                stack.pop();
            }
        }
    }

    let mut inverse = vec![0usize; n];
    for (position, &node) in order.iter().enumerate() {
        inverse[node] = position;
    }
    (order, inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::elimination_tree::NO_PARENT;
    use crate::symbolic::testing::CscMatrix;

    #[test]
    fn test_postorder_tridiagonal() {
        // Chain: 0->1->2->3. Postorder should be [0, 1, 2, 3].
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, inv) = postorder(&etree);

        assert_eq!(order.len(), 4);
        // In a chain, postorder visits from leaf to root
        assert_eq!(order, vec![0, 1, 2, 3]);

        // Verify inverse
        for (k, &node) in order.iter().enumerate() {
            assert_eq!(inv[node], k);
        }
    }

    #[test]
    fn test_postorder_valid_topological_order() {
        // For any matrix: every child appears before its parent in postorder
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, inv) = postorder(&etree);

        assert_eq!(order.len(), 5);

        // Verify topological property: parent appears after child
        for j in 0..5 {
            if etree.parent[j] != NO_PARENT {
                let p = etree.parent[j] as usize;
                assert!(
                    inv[j] < inv[p],
                    "child {} (pos {}) should appear before parent {} (pos {})",
                    j,
                    inv[j],
                    p,
                    inv[p]
                );
            }
        }
    }

    #[test]
    fn test_postorder_diagonal() {
        // Forest of singletons: any order is a valid postorder
        let m = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0; 3]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, _) = postorder(&etree);

        assert_eq!(order.len(), 3);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_postorder_inverse_roundtrip() {
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, inv) = postorder(&etree);

        // order[inv[j]] == j for all j
        for j in 0..4 {
            assert_eq!(order[inv[j]], j);
        }
        // inv[order[k]] == k for all k
        for k in 0..4 {
            assert_eq!(inv[order[k]], k);
        }
    }

    #[test]
    fn test_postorder_empty() {
        let etree = EliminationTree {
            parent: Vec::new(),
            n: 0,
        };
        let (order, inv) = postorder(&etree);
        assert!(order.is_empty());
        assert!(inv.is_empty());
    }

    /// Build a star elimination tree: nodes `0..n-1` are leaves whose only
    /// parent is the last node `n-1` (the root). This is the etree of an
    /// arrow/bordered matrix whose dense border sits at the *trailing*
    /// index (`A[n-1, i] != 0` for every `i < n-1`) - exactly the shape
    /// AMD produces for the dense-border KKT rows in this codebase's tests.
    fn star_etree(n: usize) -> EliminationTree {
        // Lower-triangle: diagonal + a dense trailing column n-1.
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i); // diagonal
            if i < n - 1 {
                rows.push(n - 1);
                cols.push(i); // (row n-1, col i): border in the lower triangle
            }
        }
        let vals = vec![1.0; rows.len()];
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let pat = m.symmetric_pattern();
        EliminationTree::from_pattern(&pat)
    }

    /// The previous `postorder`
    /// re-cloned and re-sorted `children[node]` on every stack visit, so a
    /// node with `c` children (on top of the stack `c+1` times) paid
    /// O(c^2*log c). On a star etree (one root with `n-1` children) that is
    /// O(n^2*log n) - quadratic - in the default symbolic pipeline.
    ///
    /// Reproduction is deterministic via the `SORT_WORK` counter (total
    /// child-list elements materialized across all per-node sorts), so no
    /// flaky wall-clock timing is needed. Pre-fix the root's `(n-1)`-element
    /// child list is materialized `n` times -> `~n^2` elements. Post-fix it is
    /// materialized exactly once -> `~n` elements. The assertion `work <= 4*n`
    /// fails on the quadratic version and passes on the linear fix.
    #[test]
    fn test_postorder_star_sort_work_is_linear() {
        let n = 2000;
        let etree = star_etree(n);

        // Sanity: this really is a star (root n-1, all others its children).
        assert_eq!(etree.children().count(n - 1), n - 1);
        assert_eq!(etree.roots(), vec![n - 1]);

        SORT_WORK.with(|w| w.set(0));
        let (order, inv) = postorder(&etree);
        let work = SORT_WORK.with(|w| w.get());

        // Output correctness still holds (every child before its parent).
        assert_eq!(order.len(), n);
        for j in 0..n {
            if etree.parent[j] != NO_PARENT {
                let p = etree.parent[j] as usize;
                assert!(inv[j] < inv[p], "child {j} must precede parent {p}");
            }
        }

        // The fix: child-sorting work is linear, not quadratic. The old
        // sort-on-every-visit code materializes ~n^2 elements here.
        assert!(
            work <= 4 * n,
            "postorder sort work {work} exceeds the linear bound {} (n={n}); \
             the O(n^2*log n) sort-on-every-stack-visit regression is back",
            4 * n
        );
    }
}
