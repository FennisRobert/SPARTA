use crate::graph::{fill_rows, Graph, ROWS_PER_TASK};
use rayon::prelude::*;

/// Compresses exactly equal closed neighborhoods of sorted graph `g`.
/// Returns weighted CSR and original vertex groups only if compression occurs.
/// Hashes, twin searches and compressed rows are computed in parallel; only the
/// group numbering, which depends on every earlier vertex, is a serial pass.
pub(crate) fn compress(g: &Graph) -> Option<(Graph, Groups)> {
    let n = g.len();
    let hash: Vec<_> = (0..n)
        .into_par_iter()
        .with_min_len(ROWS_PER_TASK)
        .map(|v| {
            g.neighbors(v)
                .iter()
                .fold(mix(v), |h, &u| h.wrapping_add(mix(u as usize)))
        })
        .collect();
    // First earlier neighbor with an identical closed neighborhood, if any.
    let twin: Vec<usize> = (0..n)
        .into_par_iter()
        .with_min_len(ROWS_PER_TASK)
        .map(|v| {
            let row = g.neighbors(v);
            row.iter()
                .map(|&u| u as usize)
                .take_while(|&u| u < v)
                .find(|&u| {
                    hash[v] == hash[u]
                        && row.len() == g.neighbors(u).len()
                        && closed(g, v).eq(closed(g, u))
                })
                .unwrap_or(usize::MAX)
        })
        .collect();
    // Group ids in order of first member; members listed by counting sort.
    let mut map = vec![0u32; n];
    let mut m = 0;
    for v in 0..n {
        map[v] = if twin[v] == usize::MAX {
            m += 1;
            (m - 1) as u32
        } else {
            map[twin[v]]
        };
    }
    if m == n {
        return None;
    }
    let mut start = vec![0usize; m + 1];
    for &c in &map {
        start[c as usize + 1] += 1;
    }
    for i in 0..m {
        start[i + 1] += start[i];
    }
    let weight: Vec<usize> = start.windows(2).map(|w| w[1] - w[0]).collect();
    let mut members = vec![0; n];
    let mut next = start[..m].to_vec();
    for (v, &c) in map.iter().enumerate() {
        let c = c as usize;
        members[next[c]] = v;
        next[c] += 1;
    }
    let groups = Groups { start, members };
    // Blocks of compressed rows are built in parallel into flat buffers.
    let blocks: Vec<(Vec<u32>, Vec<usize>)> = (0..m.div_ceil(ROWS_PER_TASK))
        .into_par_iter()
        .map(|b| {
            let (mut adj, mut ptr) = (Vec::new(), Vec::new());
            for i in b * ROWS_PER_TASK..((b + 1) * ROWS_PER_TASK).min(m) {
                let at = adj.len();
                let first = groups.members[groups.start[i]];
                // Twins are adjacent to each other, so the first member's row
                // lists the others; they map to `i` and are dropped.
                adj.extend(
                    g.neighbors(first)
                        .iter()
                        .map(|&u| map[u as usize])
                        .filter(|&u| u as usize != i),
                );
                adj[at..].sort_unstable();
                let mut len = at;
                for k in at..adj.len() {
                    if k == at || adj[k] != adj[len - 1] {
                        adj[len] = adj[k];
                        len += 1;
                    }
                }
                adj.truncate(len);
                ptr.push(len);
            }
            (adj, ptr)
        })
        .collect();
    let mut ptr = Vec::with_capacity(m + 1);
    ptr.push(0);
    let mut adj = Vec::with_capacity(blocks.iter().map(|b| b.0.len()).sum());
    for (block, ends) in &blocks {
        let base = adj.len();
        adj.extend_from_slice(block);
        ptr.extend(ends.iter().map(|&e| base + e));
    }
    let mut edge = vec![0; adj.len()];
    fill_rows(&ptr, &mut edge, |i, out| {
        for (e, &u) in out.iter_mut().zip(&adj[ptr[i]..ptr[i + 1]]) {
            *e = weight[i] * weight[u as usize];
        }
    });
    let c = Graph {
        ptr,
        adj,
        edge,
        ids: (0..m).collect(),
        weight,
    };
    Some((c, groups))
}

/// Original vertices of each compressed vertex, in ascending order:
/// group `i` is `members[start[i]..start[i + 1]]`.
pub(crate) struct Groups {
    pub start: Vec<usize>,
    pub members: Vec<usize>,
}
impl Groups {
    /// Borrows the original vertices merged into compressed vertex `i`.
    pub fn of(&self, i: usize) -> &[usize] {
        &self.members[self.start[i]..self.start[i + 1]]
    }
}

/// Iterates the sorted closed neighborhood of local vertex `v` in graph `g`.
fn closed(g: &Graph, v: usize) -> impl Iterator<Item = u32> + '_ {
    let row = g.neighbors(v);
    let at = row.partition_point(|&u| (u as usize) < v);
    row[..at]
        .iter()
        .copied()
        .chain(std::iter::once(v as u32))
        .chain(row[at..].iter().copied())
}

/// Mixes a vertex index `v` into an order-independent neighborhood fingerprint.
fn mix(v: usize) -> u64 {
    let mut x = (v as u64).wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
