use crate::graph::Graph;
use std::cell::RefCell;

thread_local! {
    /// Parent-to-local index map reused across calls on this thread: every
    /// entry is `u32::MAX` between calls, so no call pays O(n) to clear it.
    static LOCAL: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` with this thread's cleared parent-to-local map for `n` vertices.
/// `f` must reset every entry it sets back to `u32::MAX` before returning.
pub(crate) fn with_local<R>(n: usize, f: impl FnOnce(&mut [u32]) -> R) -> R {
    LOCAL.with(|cell| {
        let mut map = cell.borrow_mut();
        if map.len() < n {
            map.resize(n, u32::MAX);
        }
        f(&mut map[..n])
    })
}

/// Replaces the boundary between separator and `side` with a minimum vertex cover.
/// `g` must have unit weights; labels `s` use 0/1 for children and 2 for the
/// separator and are changed in place. `w` holds the partition weights and is
/// updated to match; each changed vertex is appended to `undo` with its old label.
/// Work is proportional to the separator and its `side` boundary, which are
/// renumbered locally. Matching and alternating paths are iterative.
pub(crate) fn improve(g: &Graph, s: &mut [u8], side: u8, w: &mut [usize; 3], undo: &mut Vec<(u32, u8)>) {
    let n = g.len();
    let left: Vec<u32> = (0..n as u32).filter(|&v| s[v as usize] == 2).collect();
    if left.is_empty() {
        return;
    }
    with_local(n, |map| {
        // Local nodes: separator vertices first, then `side` neighbors as found.
        // `edges[start[i]..start[i + 1]]` are left node `i`'s neighbors in `side`,
        // in the order of `g`'s row.
        let mut nodes = left.clone();
        for (i, &v) in left.iter().enumerate() {
            map[v as usize] = i as u32;
        }
        let mut start = Vec::with_capacity(left.len() + 1);
        let mut edges: Vec<u32> = Vec::new();
        start.push(0);
        for &v in &left {
            for &u in g.neighbors(v as usize) {
                if s[u as usize] == side {
                    let slot = &mut map[u as usize];
                    if *slot == u32::MAX {
                        *slot = nodes.len() as u32;
                        nodes.push(u);
                    }
                    edges.push(*slot);
                }
            }
            start.push(edges.len());
        }
        for &v in &nodes {
            map[v as usize] = u32::MAX;
        }
        let (nl, m) = (left.len(), nodes.len());
        let row = |i: usize| &edges[start[i]..start[i + 1]];
        const FREE: u32 = u32::MAX;
        let mut mate = vec![FREE; m];
        // A greedy warm start leaves only a small number of augmenting paths.
        for i in 0..nl {
            if let Some(&u) = row(i).iter().find(|&&u| mate[u as usize] == FREE) {
                mate[i] = u;
                mate[u as usize] = i as u32;
            }
        }
        let mut stamp = vec![0u32; nl];
        let mut parent = vec![0u32; nl];
        let mut queue: Vec<u32> = Vec::new();
        for root in 0..nl {
            if mate[root] != FREE {
                continue;
            }
            let tag = root as u32 + 1;
            queue.clear();
            queue.push(root as u32);
            stamp[root] = tag;
            let mut at = 0;
            let mut end = None;
            'search: while at < queue.len() {
                let v = queue[at] as usize;
                at += 1;
                for &u in row(v) {
                    let next = mate[u as usize];
                    if next == FREE {
                        end = Some((v, u));
                        break 'search;
                    }
                    if stamp[next as usize] != tag {
                        stamp[next as usize] = tag;
                        parent[next as usize] = v as u32;
                        queue.push(next);
                    }
                }
            }
            if let Some((mut v, mut u)) = end {
                loop {
                    let old = mate[v];
                    mate[v] = u;
                    mate[u as usize] = v as u32;
                    if v == root {
                        break;
                    }
                    u = old;
                    v = parent[v] as usize;
                }
            }
        }
        // Konig's theorem: (left \ reachable) union (right intersect reachable).
        let mut seen = vec![false; m];
        queue.clear();
        for i in 0..nl {
            if mate[i] == FREE {
                seen[i] = true;
                queue.push(i as u32);
            }
        }
        let mut at = 0;
        while at < queue.len() {
            let v = queue[at] as usize;
            at += 1;
            for &u in row(v) {
                if mate[v] == u || seen[u as usize] {
                    continue;
                }
                seen[u as usize] = true;
                let back = mate[u as usize];
                if back != FREE && !seen[back as usize] {
                    seen[back as usize] = true;
                    queue.push(back);
                }
            }
        }
        for (i, &v) in nodes.iter().enumerate() {
            let v = v as usize;
            let to = if i < nl {
                if seen[i] { 1 - side } else { 2 }
            } else if seen[i] {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    /// Compares matching cover cardinality against exhaustive covers on 3-by-3 graphs.
    fn covers_match_exhaustive_oracle() {
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
            let g = Graph::checked(&ptr, &adj).unwrap();
            let mut s = [2, 2, 2, 0, 0, 0];
            improve(&g, &mut s, 0, &mut [3, 0, 3], &mut Vec::new());
            let actual = s.iter().filter(|&&x| x == 2).count();
            let oracle = (0..64u32)
                .filter(|cover| {
                    (0..3).all(|l| {
                        (3..6).all(|r| {
                            mask & (1 << (l * 3 + r - 3)) == 0 || cover & ((1 << l) | (1 << r)) != 0
                        })
                    })
                })
                .map(u32::count_ones)
                .min()
                .unwrap() as usize;
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
