use crate::{
    graph::{Graph, ROWS_PER_TASK},
    Options,
};
use rayon::prelude::*;

/// Finds a separator of `g` with options `o`; labels are left=0, right=1, separator=2.
pub(crate) fn bisect(g: &Graph, o: &Options) -> Vec<u8> {
    let mut hierarchy = Vec::new();
    let mut current = g;
    if o.coarsen_threshold != 0 {
        while current.len() > o.coarsen_threshold {
            let (coarse, map) = contract(current);
            if coarse.len() * 10 > current.len() * 9 {
                break;
            }
            hierarchy.push((coarse, map));
            current = &hierarchy.last().unwrap().0;
        }
    }
    let full_trials = o.refine_all_candidates && g.len() >= o.parallel_threshold;
    let spectral = if o.spectral_candidates > 0 && g.len() >= o.parallel_threshold {
        crate::spectral::candidates(current, o, o.spectral_candidates)
    } else {
        Vec::new()
    };
    let trial = |i: usize| {
        let mut labels = if i < o.separator_trials {
            initial(current, i, o)
        } else {
            spectral[i - o.separator_trials].clone()
        };
        refine(current, &mut labels, o);
        if !hierarchy.is_empty() {
            crate::refine::fm(current, &mut labels, o);
        }
        if full_trials {
            finish(g, project(g, &hierarchy, labels, o), o)
        } else {
            labels
        }
    };
    let parallel = g.len() >= o.parallel_threshold;
    let coarse = || {
        let candidates: Vec<_> = if parallel {
            (0..o.separator_trials + spectral.len()).into_par_iter().map(trial).collect()
        } else {
            (0..o.separator_trials + spectral.len()).map(trial).collect()
        };
        if !full_trials && o.fine_selection && !hierarchy.is_empty() {
            let fine: Vec<_> = if parallel {
                candidates.into_par_iter().map(|c| project(g, &hierarchy, c, o)).collect()
            } else {
                candidates.into_iter().map(|c| project(g, &hierarchy, c, o)).collect()
            };
            return fine.into_iter().min_by_key(|s| score(g, s, o)).unwrap();
        }
        let best = candidates
            .into_iter()
            .min_by_key(|s| score(if full_trials { g } else { current }, s, o))
            .unwrap();
        if full_trials {
            best
        } else {
            project(g, &hierarchy, best, o)
        }
    };
    // A direct fine-graph candidate protects mesh geometry from contraction
    // noise. It does not depend on the coarse candidates, so both run at once.
    let direct = || {
        (!hierarchy.is_empty()).then(|| {
            let mut direct = initial(g, 0, o);
            refine(g, &mut direct, o);
            direct
        })
    };
    let (mut best, direct) = if parallel {
        rayon::join(coarse, direct)
    } else {
        (coarse(), direct())
    };
    if let Some(direct) = direct {
        if score(g, &direct, o) < score(g, &best, o) {
            best = direct;
        }
    }
    finish(g, best, o)
}

/// Projects `best` through `hierarchy` to graph `g`, refining with options `o`.
fn project(
    g: &Graph,
    hierarchy: &[(Graph, Vec<u32>)],
    mut best: Vec<u8>,
    o: &Options,
) -> Vec<u8> {
    for level in (0..hierarchy.len()).rev() {
        let fine = if level == 0 {
            g
        } else {
            &hierarchy[level - 1].0
        };
        best = hierarchy[level].1.iter().map(|&c| best[c as usize]).collect();
        refine(fine, &mut best, o);
        crate::refine::fm(fine, &mut best, o);
    }
    best
}

/// Applies optional minimum-cover improvement to candidate `best` in graph `g`.
fn finish(g: &Graph, best: Vec<u8>, o: &Options) -> Vec<u8> {
    if !o.cover_refinement {
        return best;
    }
    let mut best = minimize(g, best, o);
    crate::wide::cycle(g, &mut best, o);
    best
}

/// Replaces each separator/child boundary of `best` in `g` with an exact minimum
/// weighted vertex cover, keeping a side only when it lowers the cost under `o`.
/// Alternating sides lets a cut found on one side expose a better cut on the other.
pub(crate) fn minimize(g: &Graph, mut best: Vec<u8>, o: &Options) -> Vec<u8> {
    let unit = g.weight.iter().all(|&w| w == 1);
    let mut w = weights(g, &best);
    let mut key = quality(w, o);
    let mut undo = Vec::new();
    // `settled[side]`: the last cover on `side` was rolled back and nothing has
    // changed since, so a repeat would see the same labels and be rolled back.
    let mut settled = [false; 2];
    for side in [0u8, 1, 0, 1] {
        if settled[side as usize] {
            continue;
        }
        // Improve in place; a side that does not lower the cost is rolled back.
        let mut trial = w;
        undo.clear();
        if unit {
            crate::cover::improve(g, &mut best, side, &mut trial, &mut undo);
        } else {
            crate::flow::improve(g, &mut best, side, &mut trial, &mut undo);
        }
        let trial_key = quality(trial, o);
        if trial_key < key {
            (w, key) = (trial, trial_key);
            settled = [false; 2];
        } else {
            for &(v, old) in undo.iter().rev() {
                best[v as usize] = old;
            }
            settled[side as usize] = true;
        }
    }
    best
}

/// Rates labels `s` by feasibility, separator weight, then child imbalance in `g`.
pub(crate) fn score(g: &Graph, s: &[u8], o: &Options) -> (bool, u64, usize) {
    quality(weights(g, s), o)
}

/// Ranks partition weights `w` using balance constraints and objective in `o`.
/// Positive finite floating-point cost bits preserve total ordering in integer keys.
pub(crate) fn quality(w: [usize; 3], o: &Options) -> (bool, u64, usize) {
    let limit = (w.iter().sum::<usize>() as f64 * o.imbalance).ceil() as usize;
    let cost = if o.balance_quality {
        w[2] as f64 / (w[0].max(1) as f64 * w[1].max(1) as f64)
    } else {
        w[2] as f64
    };
    (
        w[0] == 0 || w[1] == 0 || w[0].max(w[1]) > limit,
        cost.to_bits(),
        w[0].abs_diff(w[1]),
    )
}

/// Sums vertex weights in graph `g` by partition labels `s`.
pub(crate) fn weights(g: &Graph, s: &[u8]) -> [usize; 3] {
    let mut w = [0; 3];
    for (&side, &wv) in s.iter().zip(&g.weight) {
        w[side as usize] += wv;
    }
    w
}

/// Computes unweighted hop distances in connected `g` from local vertex `root`.
/// Also returns the BFS visit order, whose tail is the farthest level.
fn distances(g: &Graph, root: usize) -> (Vec<u32>, Vec<u32>) {
    let mut d = vec![u32::MAX; g.len()];
    let mut q = Vec::with_capacity(g.len());
    q.push(root as u32);
    d[root] = 0;
    let mut at = 0;
    while at < q.len() {
        let v = q[at] as usize;
        at += 1;
        let next = d[v] + 1;
        for &u in g.neighbors(v) {
            let du = &mut d[u as usize];
            if *du == u32::MAX {
                *du = next;
                q.push(u);
            }
        }
    }
    (d, q)
}

/// Like [`distances`], and also returns each vertex's largest neighbor
/// distance (its own for an isolated vertex). When `v` is scanned, every
/// neighbor has a distance already or receives `d[v] + 1` in that scan.
fn distances_and_reach(g: &Graph, root: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = g.len();
    let mut d = vec![u32::MAX; n];
    let mut reach = vec![0u32; n];
    let mut q = Vec::with_capacity(n);
    q.push(root as u32);
    d[root] = 0;
    let mut at = 0;
    while at < q.len() {
        let v = q[at] as usize;
        at += 1;
        let next = d[v] + 1;
        let mut far = d[v];
        for &u in g.neighbors(v) {
            let du = &mut d[u as usize];
            if *du == u32::MAX {
                *du = next;
                q.push(u);
            }
            far = far.max(*du);
        }
        reach[v] = far;
    }
    (d, q, reach)
}

/// Builds a balanced BFS level separator for deterministic trial `trial` of `g`.
fn initial(g: &Graph, trial: usize, o: &Options) -> Vec<u8> {
    let n = g.len();
    let seed = trial.wrapping_mul(2654435761) % n;
    let (d, q) = distances(g, seed);
    let root = farthest(g, &d, &q);
    let (mut d, q, mut max_neighbor) = distances_and_reach(g, root);
    let mut depth = d[*q.last().unwrap() as usize] as usize;
    let width = if o.half_levels {
        let (opposite, oq) = distances(g, farthest(g, &d, &q));
        let shift = opposite[*oq.last().unwrap() as usize];
        for v in 0..n {
            d[v] = d[v] + shift - opposite[v];
        }
        depth = *d.iter().max().unwrap() as usize;
        // The shifted distances are no BFS levels, so reach needs a full pass.
        max_neighbor = (0..n)
            .map(|v| {
                g.neighbors(v)
                    .iter()
                    .map(|&u| d[u as usize])
                    .max()
                    .unwrap_or(d[v])
            })
            .collect();
        2
    } else {
        1
    };
    let mut levels = vec![0; depth + width];
    let mut redundant = vec![0; levels.len()];
    let mut total = 0;
    for (v, &level) in d.iter().enumerate() {
        let (level, wv) = (level as usize, g.weight[v]);
        total += wv;
        levels[level] += wv;
        let reach = max_neighbor[v] as usize;
        let first = level.saturating_sub(width - 1);
        for (cut, slot) in redundant.iter_mut().enumerate().take(level + 1).skip(first) {
            if reach < cut + width {
                *slot += wv;
            }
        }
    }
    let mut left = 0;
    let mut best = (true, u64::MAX, usize::MAX);
    let mut cut = 0;
    for level in 0..=levels.len() - width {
        let wide: usize = levels[level..level + width].iter().sum();
        let right = total - left - wide;
        let key = quality([left + redundant[level], right, wide - redundant[level]], o);
        if key < best {
            best = key;
            cut = level;
        }
        left += levels[level];
    }
    d.iter()
        .zip(&max_neighbor)
        .map(|(&level, &reach)| {
            let (level, reach) = (level as usize, reach as usize);
            if level < cut || (level < cut + width && reach < cut + width) {
                0
            } else if level >= cut + width {
                1
            } else {
                2
            }
        })
        .collect()
}

/// Chooses an extremal vertex of `g` from BFS distances `d` and visit order
/// `q`, breaking ties by lower degree, then lower index. Only the last level,
/// the tail of `q`, can hold the maximum distance.
fn farthest(g: &Graph, d: &[u32], q: &[u32]) -> usize {
    let far = d[*q.last().unwrap() as usize];
    q.iter()
        .rev()
        .take_while(|&&v| d[v as usize] == far)
        .map(|&v| v as usize)
        .max_by_key(|&v| (usize::MAX - g.neighbors(v).len(), usize::MAX - v))
        .unwrap()
}

/// Greedily moves separator vertices and pulls opposite neighbors in `g`.
/// `s` is changed in place; `o` caps sweeps and enforces child weight balance.
fn refine(g: &Graph, s: &mut [u8], o: &Options) {
    let mut w = weights(g, s);
    let limit = ((w.iter().sum::<usize>()) as f64 * o.imbalance).ceil() as usize;
    let n = g.len();
    for pass in 0..o.refinement_passes {
        let mut changed = false;
        let first = if w[0] <= w[1] { 0 } else { 1 };
        for side in [first, 1 - first]
            .into_iter()
            .take(if o.one_sided_refinement { 1 } else { 2 })
        {
            let opposite = 1 - side;
            for k in 0..n {
                let v = if pass % 2 == 0 { k } else { n - 1 - k };
                let wv = g.weight[v];
                if s[v] != 2 || w[side] + wv > limit {
                    continue;
                }
                // Stops as soon as the pulled weight rules the move out.
                let mut pulled = 0;
                for &u in g.neighbors(v) {
                    if s[u as usize] as usize == opposite {
                        pulled += g.weight[u as usize];
                        if pulled > wv {
                            break;
                        }
                    }
                }
                if pulled > wv || pulled == w[opposite] {
                    continue;
                }
                if pulled == wv && w[side] + wv >= w[opposite] {
                    continue;
                }
                s[v] = side as u8;
                w[side] += wv;
                w[2] -= wv;
                for &u in g.neighbors(v) {
                    let u = u as usize;
                    if s[u] as usize == opposite {
                        s[u] = 2;
                        w[opposite] -= g.weight[u];
                        w[2] += g.weight[u];
                    }
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// Contracts heavy-edge matches in `g`; returns coarse CSR and fine-to-coarse map.
/// Vertices are matched serially in ascending (degree, index) order; the
/// coarse rows are then built in parallel.
fn contract(g: &Graph) -> (Graph, Vec<u32>) {
    const NONE: u32 = u32::MAX;
    let n = g.len();
    let mut map = vec![NONE; n];
    let mut members: Vec<(u32, u32)> = Vec::with_capacity(n);
    let max_weight = (g.weight.iter().sum::<usize>() / 32).max(2);
    for v in g.by_degree() {
        if map[v] != NONE {
            continue;
        }
        let wv = g.weight[v];
        let (mut mate, mut key) = (NONE, (0, 0, 0));
        // Same winner as a max over (edge, fewer neighbors, lower index).
        for k in g.ptr[v]..g.ptr[v + 1] {
            let u = g.adj[k] as usize;
            if map[u] != NONE || wv + g.weight[u] > max_weight {
                continue;
            }
            let candidate = (
                g.edge[k],
                usize::MAX - (g.ptr[u + 1] - g.ptr[u]),
                usize::MAX - u,
            );
            if mate == NONE || candidate > key {
                (mate, key) = (u as u32, candidate);
            }
        }
        let id = members.len() as u32;
        map[v] = id;
        if mate != NONE {
            map[mate as usize] = id;
        }
        members.push((v as u32, mate));
    }
    let m = members.len();
    let weight: Vec<usize> = members
        .iter()
        .map(|&(v, mate)| {
            g.weight[v as usize]
                + if mate == NONE {
                    0
                } else {
                    g.weight[mate as usize]
                }
        })
        .collect();
    // Blocks of coarse rows are merged in parallel. Per worker, `stamp[c] == id`
    // marks coarse neighbor `c` as already in row `id`, whose summed edge weight
    // accumulates in `sum[c]`. Rows stay in merge order, unsorted: every user of
    // a contracted graph breaks ties by vertex keys, never by position in a row.
    let build = |(stamp, sum): &mut (Vec<u32>, Vec<usize>), b: usize| {
        let first = b * ROWS_PER_TASK;
        let block = &members[first..(first + ROWS_PER_TASK).min(m)];
        let bound: usize = block
            .iter()
            .map(|&(v, mate)| {
                let deg = |x: u32| g.ptr[x as usize + 1] - g.ptr[x as usize];
                deg(v) + if mate == NONE { 0 } else { deg(mate) }
            })
            .sum();
        let (mut adj, mut edge) = (Vec::with_capacity(bound), Vec::with_capacity(bound));
        let mut ends = Vec::with_capacity(block.len());
        for (id, &(v, mate)) in (first as u32..).zip(block) {
            let at = adj.len();
            for u in std::iter::once(v).chain((mate != NONE).then_some(mate)) {
                let u = u as usize;
                for k in g.ptr[u]..g.ptr[u + 1] {
                    let c = map[g.adj[k] as usize];
                    if c == id {
                        continue;
                    }
                    let c = c as usize;
                    if stamp[c] == id {
                        sum[c] += g.edge[k];
                    } else {
                        stamp[c] = id;
                        sum[c] = g.edge[k];
                        adj.push(c as u32);
                    }
                }
            }
            edge.extend(adj[at..].iter().map(|&c| sum[c as usize]));
            ends.push(adj.len());
        }
        (adj, edge, ends)
    };
    let count = m.div_ceil(ROWS_PER_TASK);
    let scratch = || (vec![NONE; m], vec![0; m]);
    let blocks: Vec<(Vec<u32>, Vec<usize>, Vec<usize>)> = if count >= 4 {
        (0..count)
            .into_par_iter()
            .map_init(scratch, build)
            .collect()
    } else {
        let mut own = scratch();
        (0..count).map(|b| build(&mut own, b)).collect()
    };
    let total = blocks.iter().map(|b| b.0.len()).sum();
    let mut coarse = Graph {
        ptr: Vec::with_capacity(m + 1),
        adj: Vec::with_capacity(total),
        edge: Vec::with_capacity(total),
        weight,
        ids: (0..m).collect(),
    };
    coarse.ptr.push(0);
    for (adj, edge, ends) in &blocks {
        let base = coarse.adj.len();
        coarse.adj.extend_from_slice(adj);
        coarse.edge.extend_from_slice(edge);
        coarse.ptr.extend(ends.iter().map(|&e| base + e));
    }
    (coarse, map)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Checks every split, projection and refinement for graph `g`.
    fn check(g: &Graph) {
        let o = Options {
            coarsen_threshold: 32,
            ..Options::default()
        };
        let s = bisect(g, &o);
        for v in 0..g.len() {
            for &u in g.neighbors(v) {
                let u = u as usize;
                assert!(!(s[v] < 2 && s[u] < 2 && s[v] != s[u]));
            }
        }
        assert!(!score(g, &s, &o).0);
    }
    #[test]
    /// Exercises weighted projection and separator validity on a rectangular mesh.
    fn mesh_separator_is_balanced_and_disconnects() {
        let n = 400;
        let mut ptr = vec![0];
        let mut adj = Vec::new();
        for v in 0..n {
            let mut row = Vec::new();
            if v % 20 > 0 {
                row.push(v - 1);
            }
            if v % 20 < 19 {
                row.push(v + 1);
            }
            if v >= 20 {
                row.push(v - 20);
            }
            if v + 20 < n {
                row.push(v + 20);
            }
            row.sort_unstable();
            adj.extend(row);
            ptr.push(adj.len());
        }
        check(&Graph::checked(&ptr, &adj).unwrap());
    }
}
