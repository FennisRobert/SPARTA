//! Wide-separator refinement: SPRAL ND Algorithm 6 with the block trimming of
//! Algorithm 8. Widening a separator and re-minimising it escapes local minima
//! that single-vertex moves cannot reach, because the minimum cut is recomputed
//! over a corridor several layers thick instead of over the current boundary.

use crate::graph::Graph;
use crate::separator::{minimize, quality, score, weights};
use crate::Options;

/// Grows separator `s` of `g` to also hold every neighbor of a separator vertex.
/// Returns `None` when the growth empties a child and leaves no valid partition.
fn expand(g: &Graph, s: &[u8]) -> Option<Vec<u8>> {
    let mut wide = s.to_vec();
    for (v, &side) in s.iter().enumerate() {
        if side == 2 {
            for &u in g.neighbors(v) {
                wide[u as usize] = 2;
            }
        }
    }
    let split = wide.contains(&0) && wide.contains(&1);
    split.then_some(wide)
}

/// Reports whether `v` has a neighbor in child 0 and in child 1 of `g` under `s`.
fn touching(g: &Graph, s: &[u8], v: usize) -> [bool; 2] {
    // Separator neighbors set the unused third flag; stops once both are seen.
    let mut touch = [false; 3];
    for &u in g.neighbors(v) {
        touch[s[u as usize] as usize] = true;
        if touch[0] && touch[1] {
            break;
        }
    }
    [touch[0], touch[1]]
}

/// Trims separator `s` of `g` by moving whole one-sided blocks into a child.
/// Each sweep moves the block with the lower cost under `o`; separator vertices
/// with no neighbor in either child are placed last, one at a time.
/// Only vertices with no neighbor in the receiving child's opposite are moved,
/// so `s` stays a valid separator throughout.
fn block_trim(g: &Graph, s: &mut [u8], o: &Options) {
    let mut w = weights(g, s);
    let mut isolated = Vec::new();
    // Separator vertices in ascending order; a sweep only ever removes some.
    let mut separator: Vec<usize> = (0..g.len()).filter(|&v| s[v] == 2).collect();
    let mut block = [Vec::new(), Vec::new()];
    loop {
        block[0].clear();
        block[1].clear();
        isolated.clear();
        separator.retain(|&v| s[v] == 2);
        for &v in &separator {
            match touching(g, s, v) {
                [true, false] => block[0].push(v),
                [false, true] => block[1].push(v),
                [false, false] => isolated.push(v),
                [true, true] => {}
            }
        }
        let pick = (0..2)
            .filter(|&side| !block[side].is_empty())
            .map(|side| {
                let moved: usize = block[side].iter().map(|&v| g.weight[v]).sum();
                let mut trial = w;
                trial[side] += moved;
                trial[2] -= moved;
                (quality(trial, o), side, moved)
            })
            .min();
        let Some((_, side, moved)) = pick else { break };
        for &v in &block[side] {
            s[v] = side as u8;
        }
        w[side] += moved;
        w[2] -= moved;
    }
    // Moving an isolated vertex can attach the next one to a child, so each is
    // re-examined at the moment it is moved rather than from the stale scan.
    for &v in &isolated {
        let side = match touching(g, s, v) {
            [true, true] => continue,
            [true, false] => 0,
            [false, true] => 1,
            [false, false] => {
                let cost = |side: usize| {
                    let mut trial = w;
                    trial[side] += g.weight[v];
                    trial[2] -= g.weight[v];
                    quality(trial, o)
                };
                usize::from(cost(1) < cost(0))
            }
        };
        s[v] = side as u8;
        w[side] += g.weight[v];
        w[2] -= g.weight[v];
    }
}

/// Refines labels `s` of `g` by repeated widening and exact re-minimisation.
/// `o` bounds the number of cycles; the loop stops at the first cycle that
/// fails to lower the cost, and `s` is left at the best partition seen.
pub(crate) fn cycle(g: &Graph, s: &mut Vec<u8>, o: &Options) {
    let mut best = score(g, s, o);
    for _ in 0..o.wide_cycles {
        let Some(mut candidate) = expand(g, s) else {
            break;
        };
        block_trim(g, &mut candidate, o);
        candidate = minimize(g, candidate, o);
        crate::refine::fm(g, &mut candidate, o);
        let key = score(g, &candidate, o);
        if key >= best {
            break;
        }
        best = key;
        *s = candidate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a `rows` by `cols` grid graph with unit weights.
    fn grid(rows: usize, cols: usize) -> Graph {
        let mut ptr = vec![0];
        let mut adj = Vec::new();
        for v in 0..rows * cols {
            let mut row = Vec::new();
            if v % cols > 0 {
                row.push(v - 1);
            }
            if v % cols < cols - 1 {
                row.push(v + 1);
            }
            if v >= cols {
                row.push(v - cols);
            }
            if v + cols < rows * cols {
                row.push(v + cols);
            }
            row.sort_unstable();
            adj.extend(row);
            ptr.push(adj.len());
        }
        Graph::checked(&ptr, &adj).unwrap()
    }

    /// Asserts no edge of `g` joins the two children under labels `s`.
    fn assert_separates(g: &Graph, s: &[u8]) {
        for v in 0..g.len() {
            for &u in g.neighbors(v) {
                let u = u as usize;
                assert!(s[v] == 2 || s[u] == 2 || s[v] == s[u]);
            }
        }
    }

    #[test]
    /// Widening a grid column separator keeps both children non-empty.
    fn expand_grows_separator_and_keeps_children() {
        let g = grid(8, 8);
        let s: Vec<u8> = (0..64)
            .map(|v| match v % 8 {
                0..=2 => 0,
                3 => 2,
                _ => 1,
            })
            .collect();
        let wide = expand(&g, &s).unwrap();
        assert!(wide.iter().filter(|&&x| x == 2).count() > s.iter().filter(|&&x| x == 2).count());
        assert!(wide.contains(&0) && wide.contains(&1));
    }

    #[test]
    /// Expansion reports failure when the separator would swallow a child.
    fn expand_rejects_swallowed_child() {
        let g = grid(3, 3);
        let mut s = vec![2u8; 9];
        s[0] = 0;
        s[8] = 1;
        assert!(expand(&g, &s).is_none());
    }

    #[test]
    /// Block trimming reduces a wide separator to one touching both children.
    fn block_trim_yields_minimal_separator() {
        let g = grid(10, 10);
        let s: Vec<u8> = (0..100)
            .map(|v| {
                if v % 10 < 4 {
                    0
                } else if v % 10 > 5 {
                    1
                } else {
                    2
                }
            })
            .collect();
        let mut wide = expand(&g, &s).unwrap();
        let o = Options::default();
        block_trim(&g, &mut wide, &o);
        assert_separates(&g, &wide);
        for v in 0..g.len() {
            if wide[v] == 2 {
                assert_eq!(
                    touching(&g, &wide, v),
                    [true, true],
                    "vertex {v} is redundant"
                );
            }
        }
    }

    #[test]
    /// A full cycle never returns a partition worse than the one it was given.
    fn cycle_never_regresses() {
        let g = grid(16, 16);
        let o = Options::default();
        let start: Vec<u8> = (0..256)
            .map(|v| {
                if v % 16 < 7 {
                    0
                } else if v % 16 > 8 {
                    1
                } else {
                    2
                }
            })
            .collect();
        let before = score(&g, &start, &o);
        let mut s = start;
        cycle(&g, &mut s, &o);
        assert!(score(&g, &s, &o) <= before);
        assert_separates(&g, &s);
    }
}
