//! Leaf ordering: exact halo minimum fill on small subgraphs, on bitset rows.
//! This path is hot, so rows are flat `u64` words rather than sets.

use crate::graph::Graph;

/// Filled adjacency of a leaf and its halo. Local vertices `0..n` are the
/// leaf's own; `n..` are halo vertices, which are never eliminated.
/// `rows[v * words..][..words]` is interior vertex `v`'s current neighbor set.
/// `planes[k * words..][..words]` marks the vertices whose weight has bit `k`
/// set, so the weight of any vertex set is a sum of shifted popcounts.
struct Leaf {
    n: usize,
    words: usize,
    rows: Vec<u64>,
    weight: Vec<usize>,
    planes: Vec<u64>,
}
impl Leaf {
    /// Builds the leaf `g` of `root`, adding each `root` neighbor outside `g`
    /// as a halo vertex. All such neighbors lie in ancestor separators.
    fn new(g: &Graph, root: &Graph) -> Self {
        let n = g.len();
        // Local index of each `root` vertex: interior `v`, halo `n + h` with
        // halo vertices in ascending `root` order. `g` is induced by `root`,
        // so the `root` rows give every interior and halo edge.
        let (rows, words, halo) = crate::cover::with_local(root.len(), |map| {
            for (v, &r) in g.ids.iter().enumerate() {
                map[r] = v as u32;
            }
            let mut halo: Vec<u32> = Vec::new();
            for &r in &g.ids {
                halo.extend(root.neighbors(r).iter().filter(|&&x| map[x as usize] == u32::MAX));
            }
            halo.sort_unstable();
            halo.dedup();
            for (h, &x) in halo.iter().enumerate() {
                map[x as usize] = (n + h) as u32;
            }
            let words = (n + halo.len()).div_ceil(64);
            let mut rows = vec![0u64; n * words];
            for (v, &r) in g.ids.iter().enumerate() {
                let row = &mut rows[v * words..(v + 1) * words];
                for &x in root.neighbors(r) {
                    let x = map[x as usize];
                    row[x as usize / 64] |= 1 << (x % 64);
                }
            }
            for r in g.ids.iter().copied().chain(halo.iter().map(|&x| x as usize)) {
                map[r] = u32::MAX;
            }
            (rows, words, halo)
        });
        let weight: Vec<usize> = g
            .weight
            .iter()
            .copied()
            .chain(halo.iter().map(|&r| root.weight[r as usize]))
            .collect();
        let bits = usize::BITS - weight.iter().max().unwrap_or(&0).leading_zeros();
        let mut planes = vec![0u64; bits as usize * words];
        for (x, &wx) in weight.iter().enumerate() {
            for k in 0..bits as usize {
                if wx >> k & 1 == 1 {
                    planes[k * words + x / 64] |= 1 << (x % 64);
                }
            }
        }
        Self { n, words, rows, weight, planes }
    }
    /// Borrows the neighbor bitset of interior vertex `v`.
    fn row(&self, v: usize) -> &[u64] {
        &self.rows[v * self.words..(v + 1) * self.words]
    }
    /// Sums vertex weights of the bits in `bits`, which is word `j` of a row.
    #[inline]
    fn mass(&self, j: usize, bits: u64) -> usize {
        self.planes
            .iter()
            .skip(j)
            .step_by(self.words)
            .enumerate()
            .map(|(k, &plane)| ((bits & plane).count_ones() as usize) << k)
            .sum()
    }
    /// Mask of the interior vertices held in word `j`.
    fn interior(&self, j: usize) -> u64 {
        match (self.n.saturating_sub(j * 64)).min(64) {
            64 => !0,
            k => (1u64 << k) - 1,
        }
    }
    /// Returns (fill, degree) for eliminating interior vertex `v` now. Fill is
    /// the weighted count of missing edges between `v`'s neighbors that have at
    /// least one interior end; halo-halo pairs join a separator that is dense
    /// after elimination anyway. Degree is the neighbors' total weight.
    fn key(&self, v: usize) -> (usize, usize) {
        let nv = self.row(v);
        let degree = (0..self.words).map(|j| self.mass(j, nv[j])).sum();
        let mut fill = 0;
        for w in 0..self.n.div_ceil(64) {
            let mut bits = nv[w] & self.interior(w);
            while bits != 0 {
                let u = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let nu = self.row(u);
                // Interior partners above `u` only, so each such pair counts once.
                let missing: usize = (0..self.words)
                    .map(|j| {
                        let above = match j.cmp(&(u / 64)) {
                            std::cmp::Ordering::Less => 0,
                            std::cmp::Ordering::Equal => (!0u64 << (u % 64)) << 1,
                            std::cmp::Ordering::Greater => !0,
                        };
                        self.mass(j, nv[j] & !nu[j] & (above | !self.interior(j)))
                    })
                    .sum();
                fill += self.weight[u] * missing;
            }
        }
        (fill, degree)
    }
    /// Eliminates interior vertex `v`, making its neighbors a clique. Returns
    /// the interior vertices whose key may have changed as a bitset of words:
    /// `v`'s neighbors and their neighbors.
    fn eliminate(&mut self, v: usize) -> Vec<u64> {
        let words = self.words;
        let nv = self.row(v).to_vec();
        let mut dirty = vec![0u64; self.n.div_ceil(64)];
        for w in 0..dirty.len() {
            let mut bits = nv[w] & self.interior(w);
            while bits != 0 {
                let u = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let row = &mut self.rows[u * words..(u + 1) * words];
                for j in 0..words {
                    row[j] |= nv[j];
                }
                row[u / 64] &= !(1 << (u % 64));
                row[v / 64] &= !(1 << (v % 64));
                for (d, r) in dirty.iter_mut().zip(row.iter()) {
                    *d |= r;
                }
                dirty[w] |= 1 << (u % 64);
            }
        }
        dirty
    }
}

/// Orders leaf `g`, a subgraph of `root` of at most 256 vertices, by exact
/// minimum fill with the halo of ancestor-separator neighbors (Pellegrini,
/// Roman and Amestoy's halo orderings). Ties go to lower degree, then to the
/// lower `root` identifier. Returns `root` identifiers in elimination order.
pub(crate) fn order(g: &Graph, root: &Graph) -> Vec<usize> {
    let n = g.len();
    let mut leaf = Leaf::new(g, root);
    let mut key: Vec<_> = (0..n).map(|v| leaf.key(v)).collect();
    let mut alive = vec![true; n];
    let mut result = Vec::with_capacity(n);
    for _ in 0..n {
        let v = (0..n)
            .filter(|&v| alive[v])
            .min_by_key(|&v| (key[v], g.ids[v]))
            .unwrap();
        result.push(g.ids[v]);
        alive[v] = false;
        let dirty = leaf.eliminate(v);
        for (w, &bits) in dirty.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let x = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if x < n && alive[x] {
                    key[x] = leaf.key(x);
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    /// Bitset fill, halo and weighted keys agree with explicit set elimination.
    fn weighted_halo_leaf_matches_set_oracle() {
        for seed in 1..20usize {
            let (total, n) = (70, 45);
            let mut ptr = vec![0];
            let mut adj = Vec::new();
            for v in 0..total {
                for u in 0..total {
                    if v != u && ((v.min(u) * 13 + v.max(u) * 17 + seed) % 11) < 3 {
                        adj.push(u);
                    }
                }
                ptr.push(adj.len());
            }
            let mut root = Graph::checked(&ptr, &adj).unwrap();
            root.weight = (0..total).map(|v| 1 + (v + seed) % 4).collect();
            let inside: Vec<usize> = (0..n).collect();
            let g = root.induced(&inside, &root.local_map(&[inside.clone()]), |u| u < n);
            let w = &root.weight;
            let mut neighbors: Vec<BTreeSet<usize>> = (0..total)
                .map(|v| root.neighbors(v).iter().map(|&u| u as usize).collect())
                .collect();
            let key = |neighbors: &[BTreeSet<usize>], v: usize| {
                let mut fill = 0;
                for &a in &neighbors[v] {
                    for &b in &neighbors[v] {
                        if a < b && (a < n || b < n) && !neighbors[a].contains(&b) {
                            fill += w[a] * w[b];
                        }
                    }
                }
                (fill, neighbors[v].iter().map(|&u| w[u]).sum::<usize>(), v)
            };
            let mut alive = vec![true; n];
            let mut oracle = Vec::new();
            for _ in 0..n {
                let v = (0..n)
                    .filter(|&v| alive[v])
                    .min_by_key(|&v| key(&neighbors, v))
                    .unwrap();
                oracle.push(v);
                alive[v] = false;
                let adjacent: Vec<_> = neighbors[v].iter().copied().collect();
                for &u in &adjacent {
                    neighbors[u].remove(&v);
                    for &x in &adjacent {
                        if u != x {
                            neighbors[u].insert(x);
                        }
                    }
                }
            }
            assert_eq!(order(&g, &root), oracle);
        }
    }
}
