use super::elimination_tree::{EliminationTree, NO_PARENT};
use super::CscPattern;

/// Compute the number of nonzeros in each column of the Cholesky factor L.
///
/// Uses elimination graph simulation: process columns left to right,
/// maintaining the fill pattern. For column j, `L[:,j]` contains:
/// - The diagonal entry (j,j)
/// - All original entries (i,j) with i > j
/// - All fill entries propagated from earlier columns
///
/// For indefinite factorization (LDL^T), the fill pattern is the same as
/// Cholesky - pivoting changes values but not structure (ignoring delayed
/// pivots, which are Phase 2).
///
/// Input `pattern` should be the full symmetric pattern (both triangles).
///
/// Returns a vector of length n where `counts[j]` is the number of nonzeros
/// in column j of L (including the diagonal).
///
/// Test-only reference oracle: the production pipeline uses
/// [`column_counts_gnp`] exclusively. This O(n^2) simulation is kept solely so
/// the tests here and in `supernode.rs` can cross-check GNP against a
/// first-principles implementation that shares no lineage with it.
#[cfg(test)]
pub fn column_counts(pattern: &CscPattern, _etree: &EliminationTree) -> Vec<usize> {
    let n = pattern.n;
    if n == 0 {
        return Vec::new();
    }

    // Simulate the elimination to compute the exact fill pattern.
    // For each column j, track the set of row indices i > j that will
    // have nonzeros in L[:,j].
    //
    // When column j is eliminated, for every pair of rows (i1, i2) in
    // L[:,j] with i1 > j and i2 > j, a fill entry is created at (max(i1,i2), min(i1,i2)).
    // These fill entries propagate to subsequent columns.

    // Build adjacency: for each column j, the set of rows > j
    let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, col_j) in col_rows.iter_mut().enumerate() {
        for &i in pattern.column(j) {
            let i = i as usize;
            if i > j {
                col_j.push(i);
            }
        }
        col_j.sort_unstable();
        col_j.dedup();
    }

    let mut counts = vec![1usize; n]; // diagonal always present

    for j in 0..n {
        let rows = std::mem::take(&mut col_rows[j]);
        counts[j] += rows.len();

        // Propagate fill: all rows in this column become connected.
        // The minimum row index inherits all other row indices.
        if rows.len() > 1 {
            let min_row = rows[0]; // smallest row > j
            for &row in &rows[1..] {
                // Add row to column min_row's pattern (if not already present)
                if !col_rows[min_row].contains(&row) {
                    col_rows[min_row].push(row);
                }
            }
            col_rows[min_row].sort_unstable();
            col_rows[min_row].dedup();
        }
    }

    counts
}

/// Compute the total number of nonzeros in L from column counts.
pub fn total_factor_nnz(counts: &[usize]) -> usize {
    counts.iter().sum()
}

/// Gilbert-Ng-Peyton column counts, `O(nnz(A) * alpha(nnz(A), n))`.
///
/// Number of nonzeros in each column of the Cholesky factor `L`, including the
/// diagonal. `pattern` must hold both triangles; `etree` must be the
/// elimination tree of `pattern`.
///
/// The counts are obtained without forming `L`: each vertex gets a small signed
/// weight, chosen so that `cc(v)` is the sum of the weights over `v`'s subtree,
/// and a single postorder sweep accumulates them.
///
/// Weights come from the row subtrees `Tr[u]` (the vertices whose elimination
/// updates row `u`). Decomposing `Tr[u]` into the tree paths that join
/// successive leaves gives, per row subtree, `+1` at every leaf, `-1` at every
/// least common ancestor of two successive leaves, and `-1` at `parent(u)`;
/// everything else cancels. Only the leaves matter, which is the skeleton-graph
/// reduction: the entries of column `u` that sit inside an earlier entry's
/// subtree contribute nothing.
///
/// The sweep visits vertices in postorder and, at vertex `w`, walks the
/// *higher*-numbered neighbours of `w` — that is, the row subtrees in which `w`
/// participates. Because the sweep is a postorder, each row subtree therefore
/// sees its own vertices in increasing order, so one array of "newest leaf so
/// far" per row subtree is enough to recognise the next leaf and to name the
/// pair whose least common ancestor needs the `-1`. That ancestor is exactly
/// what a disjoint-set structure reports if every visited vertex is merged into
/// its parent as the sweep leaves it: the representative of an already-merged
/// vertex is its lowest ancestor that the sweep has not yet reached, and for a
/// vertex of `T[w]` that is `lca(_, w)`. One pass over the pattern, one `find`
/// per skeleton edge.
///
/// The disjoint-set structure uses path halving without union by rank, which
/// the paper measures as the best-performing variant.
pub fn column_counts_gnp(pattern: &CscPattern, etree: &EliminationTree) -> Vec<usize> {
    let n = pattern.n;
    if n == 0 {
        return Vec::new();
    }
    // The working arrays are index-typed; `u32` halves their cache footprint
    // and covers every problem that fits in memory. `usize` is the fallback so
    // the contract holds for any `n`.
    if n < u32::MAX as usize {
        gnp::<u32>(pattern, etree)
    } else {
        gnp::<usize>(pattern, etree)
    }
}

/// Index width for the working arrays of [`column_counts_gnp`].
trait Idx: Copy + Ord {
    fn from_usize(v: usize) -> Self;
    fn get(self) -> usize;
}

impl Idx for u32 {
    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v as u32
    }
    #[inline(always)]
    fn get(self) -> usize {
        self as usize
    }
}

impl Idx for usize {
    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        v
    }
    #[inline(always)]
    fn get(self) -> usize {
        self
    }
}

/// Representative of `x`'s set: its lowest ancestor the sweep has not passed.
///
/// Path halving - point every other node on the walk at its grandparent.
#[inline]
fn find<I: Idx>(link: &mut [I], mut x: usize) -> usize {
    loop {
        let up = link[x].get();
        if up == x {
            return x;
        }
        let up2 = link[up].get();
        if up2 == up {
            return up;
        }
        link[x] = I::from_usize(up2);
        x = up2;
    }
}

fn gnp<I: Idx>(pattern: &CscPattern, etree: &EliminationTree) -> Vec<usize> {
    let n = pattern.n;
    let parent = &etree.parent;

    // --- Postorder ---------------------------------------------------------
    //
    // Computed here rather than taken from the caller: the function is
    // specified to work for any numbering, and it costs two O(n) sweeps.
    //
    // `parent[j] > j` means children always precede their parent in index
    // order, so subtree sizes accumulate in one ascending pass, and the
    // positions those sizes imply can then be handed down in one descending
    // pass. No recursion and no per-node child lists.
    let mut size = vec![I::from_usize(1); n];
    for j in 0..n {
        if parent[j] != NO_PARENT {
            let p = parent[j] as usize;
            size[p] = I::from_usize(size[p].get() + size[j].get());
        }
    }

    // Each subtree owns a contiguous block of positions and its root takes the
    // last of them. `free[v]` is one past the block still unclaimed below `v`,
    // so visiting `v`'s children from the largest down hands out the blocks in
    // ascending order. Roots are laid out first, ascending, to match.
    let mut pos = vec![I::from_usize(0); n];
    let mut free = vec![I::from_usize(0); n];
    let mut next_root_block = 0usize;
    for j in 0..n {
        if parent[j] == NO_PARENT {
            let at = next_root_block + size[j].get() - 1;
            pos[j] = I::from_usize(at);
            free[j] = I::from_usize(at);
            next_root_block = at + 1;
        }
    }
    for j in (0..n).rev() {
        if parent[j] != NO_PARENT {
            let p = parent[j] as usize;
            let at = free[p].get() - 1;
            pos[j] = I::from_usize(at);
            free[j] = I::from_usize(at);
            free[p] = I::from_usize(at + 1 - size[j].get());
        }
    }
    drop(free);

    // Position-indexed views: the vertex at each position, the first position
    // of its subtree, and its parent's position (`n` marks a root).
    let mut post = vec![I::from_usize(0); n];
    let mut first = vec![I::from_usize(0); n];
    let mut up = vec![I::from_usize(n); n];
    for j in 0..n {
        let k = pos[j].get();
        post[k] = I::from_usize(j);
        first[k] = I::from_usize(k + 1 - size[j].get());
        if parent[j] != NO_PARENT {
            up[k] = pos[parent[j] as usize];
        }
    }
    drop(size);

    // --- Weights -----------------------------------------------------------
    //
    // Every row subtree contributes `+1` at its own root and `-1` at that
    // root's parent; both are known up front.
    let mut wt = vec![1i64; n];
    for &parent_pos in up.iter() {
        let p = parent_pos.get();
        if p != n {
            wt[p] -= 1;
        }
    }

    // `newest[u]` is the most recent leaf seen in row subtree `u`, stored one
    // based so that zero means "none yet". A vertex `w` is a leaf of `Tr[u]`
    // exactly when no earlier entry of column `u` lies in `T[w]`; since the
    // entries arrive in postorder and `T[w]` is the position range
    // `first[w] ..= pos[w]`, that is the single test `newest[u] <= first[w]`.
    // Testing against the newest *leaf* rather than the newest entry is
    // equivalent, because the newest leaf always lies in the newest entry's
    // subtree.
    let mut newest = vec![I::from_usize(0); n];
    let mut link: Vec<I> = (0..n).map(I::from_usize).collect();

    let col_ptr = &pattern.col_ptr;
    let row_idx = &pattern.row_idx;

    for k in 0..n {
        let w = post[k].get();
        let first_k = first[k].get();

        // Closing pair of row subtree `w`: its last leaf against `w` itself,
        // whose least common ancestor is `w`.
        if newest[k].get() != 0 {
            wt[k] -= 1;
        }

        // Neighbours of `w` above it in the elimination tree. Row indices
        // ascend and the elimination tree orders ancestors after descendants,
        // so these are the tail of the column.
        let lo = col_ptr[w] as usize;
        let hi = col_ptr[w + 1] as usize;
        let tail = lo + row_idx[lo..hi].partition_point(|&v| (v as usize) <= w);
        for &v in &row_idx[tail..hi] {
            let v = v as usize;
            let u = pos[v].get();
            let prev = newest[u].get();
            if prev <= first_k {
                wt[k] += 1;
                if prev != 0 {
                    let q = find(&mut link, prev - 1);
                    wt[q] -= 1;
                }
                newest[u] = I::from_usize(k + 1);
            }
        }

        // The sweep is done with `k`; merge it into its parent so later finds
        // report ancestors of `k` instead of `k`.
        let p = up[k].get();
        if p != n {
            link[k] = I::from_usize(p);
        }
    }

    // --- Accumulate ---------------------------------------------------------
    //
    // cc(v) is the sum of the weights over T[v]; children precede parents.
    let mut counts = vec![0usize; n];
    for k in 0..n {
        let c = wt[k];
        counts[post[k].get()] = c as usize;
        let p = up[k].get();
        if p != n {
            wt[p] += c;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::testing::CscMatrix;

    #[test]
    fn test_column_counts_diagonal() {
        // Diagonal matrix: each column of L has exactly 1 nonzero (the diagonal)
        let m = CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0; 4]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![1, 1, 1, 1]);
        assert_eq!(total_factor_nnz(&counts), 4);
    }

    #[test]
    fn test_column_counts_tridiagonal() {
        // Tridiagonal 4x4: L has entries on diagonal and one subdiagonal
        // Column 0: rows 0, 1 -> count = 2
        // Column 1: rows 1, 2 -> count = 2
        // Column 2: rows 2, 3 -> count = 2
        // Column 3: row 3      -> count = 1
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![2, 2, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 7);
    }

    #[test]
    fn test_column_counts_dense() {
        // Dense 3x3: L is full lower triangle
        // Column 0: rows 0, 1, 2 -> count = 3
        // Column 1: rows 1, 2    -> count = 2
        // Column 2: row 2        -> count = 1
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![3, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 6); // n*(n+1)/2
    }

    #[test]
    fn test_column_counts_arrow() {
        // Arrow 5x5: column 0 has entries at rows 0-4, others are diagonal
        // Eliminating column 0 creates fill among rows 1-4
        // Column 0: rows 0,1,2,3,4 -> count = 5
        // Column 1: rows 1,2,3,4 (fill from col 0) -> count = 4
        // Column 2: rows 2,3,4 -> count = 3
        // Column 3: rows 3,4 -> count = 2
        // Column 4: row 4 -> count = 1
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

        assert_eq!(counts, vec![5, 4, 3, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 15); // fully dense: n*(n+1)/2
    }

    #[test]
    fn test_column_counts_block_diagonal() {
        // Two 2x2 dense blocks: no fill between blocks
        // [a b 0 0]
        // [b c 0 0]
        // [0 0 d e]
        // [0 0 e f]
        let m = CscMatrix::from_triplets(4, &[0, 1, 1, 2, 3, 3], &[0, 0, 1, 2, 2, 3], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![2, 1, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 6);
    }

    // --- GNP parity with the reference oracle ------------------------------
    //
    // Each reuses the exact pattern from the reference tests above and asserts
    // column_counts_gnp returns the same vector as column_counts.

    #[test]
    fn gnp_matches_reference_diagonal() {
        let m = CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0; 4]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_tridiagonal() {
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_dense() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_arrow() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_block_diagonal() {
        let m = CscMatrix::from_triplets(4, &[0, 1, 1, 2, 3, 3], &[0, 0, 1, 2, 2, 3], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_banded() {
        // 7x7 banded: entries on diagonal, sub-diagonal, and 2 below diagonal.
        // Produces a non-trivial row-subtree pattern (more interesting than chain).
        let rows = vec![0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 6];
        let cols = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6];
        let m = CscMatrix::from_triplets(7, &rows, &cols, &vec![1.0; rows.len()]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    // --- Randomised cross-checks against the first-principles oracle --------

    fn lcg(state: &mut u64) -> usize {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as usize
    }

    /// Dense-ish random symmetric patterns, always with a diagonal.
    #[test]
    fn gnp_matches_reference_random_dense() {
        let mut state = 12345u64;
        for trial in 0..3000 {
            let n = 1 + lcg(&mut state) % 30;
            let density = 1 + lcg(&mut state) % 40;
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            for j in 0..n {
                for i in j..n {
                    if i == j || lcg(&mut state) % 100 < density {
                        rows.push(i);
                        cols.push(j);
                    }
                }
            }
            let m = CscMatrix::from_triplets(n, &rows, &cols, &vec![1.0; rows.len()]).unwrap();
            let pat = m.symmetric_pattern();
            let etree = EliminationTree::from_pattern(&pat);
            assert_eq!(
                column_counts_gnp(&pat, &etree),
                column_counts(&pat, &etree),
                "trial {trial}, n = {n}, density = {density}%"
            );
        }
    }

    /// Sparse random patterns with no diagonal, so most are reducible and the
    /// elimination tree is a forest rather than a single tree.
    #[test]
    fn gnp_matches_reference_random_forest() {
        let mut state = 999u64;
        for trial in 0..3000 {
            let n = 1 + lcg(&mut state) % 40;
            let edges = lcg(&mut state) % (2 * n + 1);
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            for _ in 0..edges {
                let a = lcg(&mut state) % n;
                let b = lcg(&mut state) % n;
                rows.push(a.max(b));
                cols.push(a.min(b));
            }
            if rows.is_empty() {
                continue;
            }
            let m = match CscMatrix::from_triplets(n, &rows, &cols, &vec![1.0; rows.len()]) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let pat = m.symmetric_pattern();
            let etree = EliminationTree::from_pattern(&pat);
            assert_eq!(
                column_counts_gnp(&pat, &etree),
                column_counts(&pat, &etree),
                "trial {trial}, n = {n}"
            );
        }
    }

    #[test]
    fn gnp_empty_matrix() {
        let pat = CscPattern {
            n: 0,
            col_ptr: vec![0],
            row_idx: Vec::new(),
        };
        let etree = EliminationTree {
            parent: Vec::new(),
            n: 0,
        };
        assert!(column_counts_gnp(&pat, &etree).is_empty());
    }
}
