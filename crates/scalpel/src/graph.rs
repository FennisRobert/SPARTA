use crate::Error;
use rayon::prelude::*;

/// Rows handled by one parallel task in whole-graph passes. Small enough to
/// balance 14 workers on a 50k-vertex graph, large enough that task overhead
/// stays below the per-row work.
pub(crate) const ROWS_PER_TASK: usize = 2048;

/// Fills CSR payload `out` in parallel: row `v` is `out[ptr[v]..ptr[v + 1]]`
/// and is written by `f(v, row)`. Blocks of `ROWS_PER_TASK` rows run as tasks.
pub(crate) fn fill_rows<T: Send>(ptr: &[usize], out: &mut [T], f: impl Fn(usize, &mut [T]) + Sync) {
    let n = ptr.len() - 1;
    let mut blocks = Vec::with_capacity(n / ROWS_PER_TASK + 1);
    let mut rest = out;
    let mut first = 0;
    while first < n {
        let end = (first + ROWS_PER_TASK).min(n);
        let (block, tail) = rest.split_at_mut(ptr[end] - ptr[first]);
        blocks.push((first, end, block));
        rest = tail;
        first = end;
    }
    blocks.into_par_iter().for_each(|(first, end, block)| {
        let base = ptr[first];
        for v in first..end {
            f(v, &mut block[ptr[v] - base..ptr[v + 1] - base]);
        }
    });
}

/// A CSR index type accepted by [`crate::order`].
/// Implemented for `usize` and `u32` so a caller can hand over either width
/// without first copying its arrays.
pub trait Index: Copy + Ord + Send + Sync {
    /// Widens this index to a `usize` offset.
    fn get(self) -> usize;
}
impl Index for usize {
    /// Returns the value unchanged.
    fn get(self) -> usize {
        self
    }
}
impl Index for u32 {
    /// Widens to the platform offset type.
    fn get(self) -> usize {
        self as usize
    }
}

/// Owned CSR with vertex/edge weights and original identifiers. Rows are
/// sorted in graphs built by `checked`, `induced` and compression; contracted
/// graphs leave them in merge order. Neighbor
/// indices are `u32` (vertex counts are checked below 2^32) so that the
/// neighbor scans that dominate every pass move half the memory.
pub(crate) struct Graph {
    pub ptr: Vec<usize>,
    pub adj: Vec<u32>,
    pub edge: Vec<usize>,
    pub weight: Vec<usize>,
    pub ids: Vec<usize>,
}
impl Graph {
    /// Checks sorted symmetric CSR slices `ptr` and `adj`, removing self loops.
    /// Rows are checked and copied in parallel. Symmetry is proved from one
    /// half: every entry above the diagonal must have its mirror, and the two
    /// halves must hold equally many entries. Because rows are unique, the
    /// mirror map is then a bijection from the upper to the lower half.
    pub fn checked<T: Index>(ptr: &[T], adj: &[T]) -> Result<Self, Error> {
        if ptr.is_empty()
            || ptr[0].get() != 0
            || ptr.last().map(|v| v.get()) != Some(adj.len())
            || ptr.windows(2).any(|w| w[0] > w[1])
        {
            return Err(Error("invalid CSR offsets"));
        }
        let n = ptr.len() - 1;
        let row = |v: usize| &adj[ptr[v].get()..ptr[v + 1].get()];
        let malformed = (0..n).into_par_iter().with_min_len(ROWS_PER_TASK).any(|v| {
            row(v).iter().any(|&u| u.get() >= n) || row(v).windows(2).any(|w| w[0] >= w[1])
        });
        if malformed {
            return Err(Error("CSR rows must be sorted, unique, and in range"));
        }
        // Per row: (entries above the diagonal, entries below, missing mirrors).
        let (upper, lower, missing) = (0..n)
            .into_par_iter()
            .with_min_len(ROWS_PER_TASK)
            .map(|v| {
                let r = row(v);
                let below = r.partition_point(|u| u.get() < v);
                let above = r.len() - r[below..].partition_point(|u| u.get() <= v) - below;
                let missing = r[r.len() - above..]
                    .iter()
                    .filter(|&&u| row(u.get()).binary_search_by(|p| p.get().cmp(&v)).is_err())
                    .count();
                (above, below, missing)
            })
            .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        if missing != 0 || upper != lower {
            return Err(Error("graph must contain both directions of each edge"));
        }
        let mut p = Vec::with_capacity(n + 1);
        p.push(0);
        for v in 0..n {
            let diagonal = row(v).binary_search_by(|u| u.get().cmp(&v)).is_ok();
            p.push(p[v] + row(v).len() - usize::from(diagonal));
        }
        let mut a = vec![0; p[n]];
        fill_rows(&p, &mut a, |v, out| {
            for (slot, u) in out
                .iter_mut()
                .zip(row(v).iter().map(|u| u.get()).filter(|&u| u != v))
            {
                *slot = u as u32;
            }
        });
        Ok(Self {
            edge: vec![1; a.len()],
            ptr: p,
            adj: a,
            weight: vec![1; n],
            ids: (0..n).collect(),
        })
    }
    /// Returns the number of vertices in this graph.
    pub fn len(&self) -> usize {
        self.weight.len()
    }
    /// Borrows adjacent vertex indices for local vertex `v`.
    pub fn neighbors(&self, v: usize) -> &[u32] {
        &self.adj[self.ptr[v]..self.ptr[v + 1]]
    }
    /// Returns local vertices in ascending (degree, index) order by counting sort.
    pub fn by_degree(&self) -> Vec<usize> {
        let n = self.len();
        let max = (0..n)
            .map(|v| self.ptr[v + 1] - self.ptr[v])
            .max()
            .unwrap_or(0);
        let mut start = vec![0usize; max + 2];
        for v in 0..n {
            start[self.ptr[v + 1] - self.ptr[v] + 1] += 1;
        }
        for d in 1..start.len() {
            start[d] += start[d - 1];
        }
        let mut order = vec![0; n];
        for v in 0..n {
            let d = self.ptr[v + 1] - self.ptr[v];
            order[start[d]] = v;
            start[d] += 1;
        }
        order
    }
    /// Finds connected components using an iterative queue (no graph-depth stack).
    pub fn components(&self) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.len()];
        let mut components = Vec::new();
        for root in 0..self.len() {
            if seen[root] {
                continue;
            }
            seen[root] = true;
            let mut queue = vec![root];
            let mut at = 0;
            while at < queue.len() {
                let v = queue[at];
                at += 1;
                for &u in self.neighbors(v) {
                    let u = u as usize;
                    if !seen[u] {
                        seen[u] = true;
                        queue.push(u);
                    }
                }
            }
            components.push(queue);
        }
        components
    }
    /// Maps parent vertices to local indices in disjoint vertex `groups`.
    pub fn local_map(&self, groups: &[Vec<usize>]) -> Vec<u32> {
        let mut map = vec![u32::MAX; self.len()];
        for group in groups {
            for (i, &v) in group.iter().enumerate() {
                map[v] = i as u32;
            }
        }
        map
    }
    /// Copies the subgraph on `vertices`, with parent-to-local indices in `map`.
    /// `inside(u)` tells whether parent vertex `u`, a neighbor of one of
    /// `vertices`, is itself one of them; rows keep their parent order.
    pub fn induced(&self, vertices: &[usize], map: &[u32], inside: impl Fn(usize) -> bool) -> Self {
        let mut ptr = Vec::with_capacity(vertices.len() + 1);
        ptr.push(0);
        let bound: usize = vertices.iter().map(|&v| self.ptr[v + 1] - self.ptr[v]).sum();
        let mut adj = Vec::with_capacity(bound);
        let mut edge = Vec::with_capacity(bound);
        for &v in vertices {
            for k in self.ptr[v]..self.ptr[v + 1] {
                let u = self.adj[k] as usize;
                if inside(u) {
                    adj.push(map[u]);
                    edge.push(self.edge[k]);
                }
            }
            ptr.push(adj.len());
        }
        Self {
            ptr,
            adj,
            edge,
            weight: vertices.iter().map(|&v| self.weight[v]).collect(),
            ids: vertices.iter().map(|&v| self.ids[v]).collect(),
        }
    }
}
