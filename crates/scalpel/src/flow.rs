use crate::graph::Graph;

/// Residual arc for an integral-capacity flow network. Capacities are vertex
/// weights, bounded by the total weight, which fits in `u32` like the vertex
/// count; the saturating "infinite" capacity is `u32::MAX`.
#[derive(Clone, Copy, Default)]
struct Arc {
    to: u32,
    reverse: u32,
    capacity: u32,
}

/// Flow network in CSR form: the arcs leaving node `v` are
/// `arcs[start[v]..start[v + 1]]`, in the order they were added.
struct Network {
    start: Vec<u32>,
    arcs: Vec<Arc>,
}

/// Improves labels `s` between separator and child `side` by a weighted min cut.
/// Vertex weights in `g` are capacities; no recursion depends on graph depth.
/// `s` is changed in place and `w`, the partition weights, updated to match;
/// each changed vertex is appended to `undo` with its old label. Work is
/// proportional to the separator and its `side` boundary.
pub(crate) fn improve(g: &Graph, s: &mut [u8], side: u8, w: &mut [usize; 3], undo: &mut Vec<(u32, u8)>) {
    let n = g.len();
    let separator: Vec<u32> = (0..n as u32).filter(|&v| s[v as usize] == 2).collect();
    if separator.is_empty() {
        return;
    }
    crate::cover::with_local(n, |map| {
        // One scan records each separator vertex's `side` neighbors, in row
        // order, as `cut[first[i]..first[i + 1]]` for separator vertex `i`.
        // Network vertices are the separator and those neighbors, numbered in
        // ascending local order.
        let mut first = Vec::with_capacity(separator.len() + 1);
        let mut cut: Vec<u32> = Vec::new();
        let mut vertices = separator.clone();
        first.push(0);
        for &v in &separator {
            for &u in g.neighbors(v as usize) {
                if s[u as usize] == side {
                    cut.push(u);
                    if map[u as usize] == u32::MAX {
                        map[u as usize] = 0;
                        vertices.push(u);
                    }
                }
            }
            first.push(cut.len());
        }
        vertices.sort_unstable();
        for (i, &v) in vertices.iter().enumerate() {
            map[v as usize] = i as u32;
        }
        let source = vertices.len();
        let sink = source + 1;
        // Counting pass sizes each node's arc list; the fill pass then adds arcs in
        // the same sequence, so each list keeps its insertion order.
        let mut start = vec![0u32; sink + 2];
        let mut at = 0;
        for (i, &v) in vertices.iter().enumerate() {
            if s[v as usize] == 2 {
                let row = &cut[first[at]..first[at + 1]];
                at += 1;
                start[source + 1] += 1;
                start[i + 1] += 1 + row.len() as u32;
                for &u in row {
                    start[map[u as usize] as usize + 1] += 1;
                }
            } else {
                start[i + 1] += 1;
                start[sink + 1] += 1;
            }
        }
        for k in 0..=sink {
            start[k + 1] += start[k];
        }
        let mut net = Network {
            arcs: vec![Arc::default(); start[sink + 1] as usize],
            start,
        };
        let mut fill = net.start[..=sink].to_vec();
        let mut at = 0;
        for (i, &v) in vertices.iter().enumerate() {
            let wv = g.weight[v as usize] as u32;
            if s[v as usize] == 2 {
                net.add(&mut fill, source, i, wv);
                for &u in &cut[first[at]..first[at + 1]] {
                    net.add(&mut fill, i, map[u as usize] as usize, u32::MAX);
                }
                at += 1;
            } else {
                net.add(&mut fill, i, sink, wv);
            }
        }
        for &v in &vertices {
            map[v as usize] = u32::MAX;
        }
        let reachable = net.max_flow(source, sink);
        for (i, &v) in vertices.iter().enumerate() {
            let v = v as usize;
            let to = if s[v] == 2 {
                if reachable[i] { 1 - side } else { 2 }
            } else if reachable[i] {
                2
            } else {
                continue;
            };
            if to != s[v] {
                undo.push((v as u32, s[v]));
                w[s[v] as usize] -= g.weight[v];
                w[to as usize] += g.weight[v];
                s[v] = to;
            }
        }
    });
}

impl Network {
    /// Adds forward/reverse arcs from `u` to `v` with integral `capacity`.
    /// `fill[x]` is the next free arc slot of node `x`.
    #[inline]
    fn add(&mut self, fill: &mut [u32], u: usize, v: usize, capacity: u32) {
        let (a, b) = (fill[u], fill[v]);
        fill[u] += 1;
        fill[v] += 1;
        self.arcs[a as usize] = Arc {
            to: v as u32,
            reverse: b,
            capacity,
        };
        self.arcs[b as usize] = Arc {
            to: u as u32,
            reverse: a,
            capacity: 0,
        };
    }

    /// Computes a maximum flow by Dinic's algorithm, returning the vertices
    /// reachable from `source` in the final residual network. `source` and
    /// `sink` are distinct node indices; blocking paths use a heap stack.
    fn max_flow(&mut self, source: usize, sink: usize) -> Vec<bool> {
        const UNSEEN: u32 = u32::MAX;
        let n = self.start.len() - 1;
        let mut queue: Vec<u32> = Vec::with_capacity(n);
        let mut level = vec![UNSEEN; n];
        let mut next = vec![0u32; n];
        let mut path: Vec<u32> = Vec::new();
        loop {
            level.fill(UNSEEN);
            level[source] = 0;
            queue.clear();
            queue.push(source as u32);
            let mut at = 0;
            while at < queue.len() {
                let v = queue[at] as usize;
                at += 1;
                let lv = level[v] + 1;
                for e in &self.arcs[self.start[v] as usize..self.start[v + 1] as usize] {
                    if e.capacity > 0 && level[e.to as usize] == UNSEEN {
                        level[e.to as usize] = lv;
                        queue.push(e.to);
                    }
                }
            }
            if level[sink] == UNSEEN {
                return level.iter().map(|&x| x != UNSEEN).collect();
            }
            next.copy_from_slice(&self.start[..n]);
            path.clear();
            let mut v = source;
            loop {
                if v == sink {
                    let amount = path
                        .iter()
                        .map(|&k| self.arcs[k as usize].capacity)
                        .min()
                        .unwrap();
                    for &k in &path {
                        let arc = &mut self.arcs[k as usize];
                        arc.capacity -= amount;
                        let reverse = arc.reverse as usize;
                        self.arcs[reverse].capacity += amount;
                    }
                    path.clear();
                    v = source;
                }
                let end = self.start[v + 1];
                let want = level[v].wrapping_add(1);
                let mut k = next[v];
                while k < end {
                    let e = &self.arcs[k as usize];
                    if e.capacity > 0 && level[e.to as usize] == want {
                        break;
                    }
                    k += 1;
                }
                next[v] = k;
                if k == end {
                    level[v] = UNSEEN;
                    if let Some(k) = path.pop() {
                        let arc = self.arcs[k as usize];
                        v = self.arcs[arc.reverse as usize].to as usize;
                        next[v] = k + 1;
                    } else {
                        break;
                    }
                } else {
                    path.push(k);
                    v = self.arcs[k as usize].to as usize;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    /// Verifies every 3-by-3 weighted graph against exhaustive vertex covers.
    fn weighted_covers_match_exhaustive_oracle() {
        for mask in 0..512 {
            let mut ptr = vec![0];
            let mut adj = Vec::new();
            for v in 0..6 {
                for u in 0..6 {
                    let (l, r) = (v.min(u), v.max(u));
                    if l < 3 && r >= 3 && mask & (1 << (l * 3 + r - 3)) != 0 {
                        adj.push(u);
                    }
                }
                ptr.push(adj.len());
            }
            let mut g = Graph::checked(&ptr, &adj).unwrap();
            g.weight = vec![1, 3, 2, 3, 1, 2];
            let mut s = [2, 2, 2, 0, 0, 0];
            improve(&g, &mut s, 0, &mut [6, 0, 6], &mut Vec::new());
            let actual: usize = (0..6).filter(|&v| s[v] == 2).map(|v| g.weight[v]).sum();
            let oracle = (0..64u32)
                .filter(|cover| {
                    (0..3).all(|l| {
                        (3..6).all(|r| {
                            mask & (1 << (l * 3 + r - 3)) == 0 || cover & ((1 << l) | (1 << r)) != 0
                        })
                    })
                })
                .map(|cover| {
                    (0..6)
                        .filter(|&v| cover & (1 << v) != 0)
                        .map(|v| g.weight[v])
                        .sum::<usize>()
                })
                .min()
                .unwrap();
            assert_eq!(actual, oracle);
            for v in 0..6 {
                for &u in g.neighbors(v) {
                    let u = u as usize;
                    assert!(s[v] == 2 || s[u] == 2 || s[v] == s[u]);
                }
            }
        }
    }
}
