//! Division of one factorization between the CPU and the GPU.
//!
//! The CPU takes whole subtrees of the elimination tree and the GPU everything
//! else. A subtree needs nothing from outside itself, so the two can factor at
//! the same time until the GPU reaches the first front whose child is a CPU
//! subtree root. That level is the cut: the GPU factors its share of the levels
//! below it while the CPU factors its subtrees, the CPU's results are copied in,
//! and the GPU factors the levels from the cut up.
use crate::Analysis;

/// Which nodes the CPU factors, and where the GPU has to wait for them.
pub(crate) struct Split {
    /// `cpu[node]` is true for every node of a CPU subtree.
    pub cpu: Vec<bool>,
    /// The CPU subtree roots, whose contribution blocks the GPU assembles.
    pub roots: Vec<u32>,
    /// First level at which a GPU front assembles a CPU subtree root.
    pub cut: usize,
    /// Fraction of the factorization work the CPU subtrees carry.
    pub share: f64,
    /// Fraction of the work in GPU fronts from the cut up, which runs after
    /// the hand-over.
    pub above: f64,
}

impl Analysis {
    /// Choose CPU subtrees carrying about `fraction` of the work, or `None`
    /// when there is none to choose.
    ///
    /// Work is the real flop count of each front's elimination.
    pub(crate) fn hybrid_split(&self, fraction: f64) -> Option<Split> {
        let count = self.nodes.len();
        if fraction <= 0.0 || fraction >= 1.0 || count == 0 {
            return None;
        }
        let mut level_of = vec![0usize; count];
        for (l, level) in self.levels.iter().enumerate() {
            for &id in level {
                level_of[id as usize] = l;
            }
        }
        let mut parent = vec![u32::MAX; count];
        for (p, node) in self.nodes.iter().enumerate() {
            let first = node.children as usize;
            for child in &self.children[first..first + node.count as usize] {
                parent[child.node as usize] = p as u32;
            }
        }
        // Children sit on lower levels than their parents, so one pass in
        // level order accumulates every subtree.
        let mut subtree: Vec<f64> = self.nodes.iter().map(|d| front_flops(d.nr, d.nc)).collect();
        for level in &self.levels {
            for &id in level {
                if parent[id as usize] != u32::MAX {
                    let w = subtree[id as usize];
                    subtree[parent[id as usize] as usize] += w;
                }
            }
        }
        let total: f64 = (0..count).filter(|&i| parent[i] == u32::MAX).map(|i| subtree[i]).sum();
        let mut level_work = vec![0.0; self.levels.len()];
        for (l, level) in self.levels.iter().enumerate() {
            level_work[l] = level
                .iter()
                .map(|&id| front_flops(self.nodes[id as usize].nr, self.nodes[id as usize].nc))
                .sum::<f64>()
                / total;
        }
        // A subtree can go to the CPU for cut `c` when its root is below `c` and
        // its parent is at or above it: then every CPU node is factored before
        // the GPU reaches the cut, and the root's contribution block slot is
        // still reserved when it is copied in. Per cut, candidates are taken
        // largest first while they fit within 1.1 times the target. The cut is
        // the one whose selection misses the target least, counting a quarter
        // of the GPU work from the cut up, which follows the hand-over.
        let (mut roots, mut taken, mut cut, mut after) = (Vec::new(), 0.0, 0, 0.0);
        let mut best = f64::INFINITY;
        let mut above = 0.0;
        for c in (1..self.levels.len()).rev() {
            above += level_work[c];
            let mut open: Vec<u32> = (0..count as u32)
                .filter(|&i| {
                    let p = parent[i as usize];
                    level_of[i as usize] < c && p != u32::MAX && level_of[p as usize] >= c
                })
                .collect();
            open.sort_by(|a, b| subtree[*b as usize].total_cmp(&subtree[*a as usize]));
            let (mut chosen, mut sum) = (Vec::new(), 0.0);
            for &id in &open {
                let w = subtree[id as usize] / total;
                if sum + w <= 1.1 * fraction {
                    chosen.push(id);
                    sum += w;
                }
            }
            let score = (sum - fraction).abs() + ABOVE * above;
            if !chosen.is_empty() && score < best {
                (roots, taken, cut, after, best) = (chosen, sum, c, above, score);
            }
        }
        if roots.is_empty() {
            return None;
        }
        let mut cpu = vec![false; count];
        for &root in &roots {
            cpu[root as usize] = true;
        }
        // Top-down, so a node is marked once its parent is.
        for level in self.levels.iter().rev() {
            for &id in level {
                if parent[id as usize] != u32::MAX && cpu[parent[id as usize] as usize] {
                    cpu[id as usize] = true;
                }
            }
        }
        Some(Split {
            cpu,
            roots,
            cut,
            share: taken,
            above: after,
        })
    }
}

/// Weight of the GPU work from the cut up against a miss of the target share
/// in choosing the cut. That work runs after the hand-over with the CPU idle,
/// but a cut low enough to carry much of it is also one with many small
/// candidate subtrees, which the CPU factors more efficiently than a few large
/// ones. 0.25 picked the fastest of the cuts tried on the 887k-unknown EMerge
/// sweep (M4 Pro); a full makespan model with the GPU's average rate picked
/// worse ones, because the CPU's rate depends on the subtrees it gets.
const ABOVE: f64 = 0.25;

/// Real flops of one complex LDL^T front with `nr` rows and `nc` pivots: the
/// lower-triangle update of every pivot, eight real flops per complex
/// multiply-add. An LU front updates the whole square, twice this for every
/// shape, so the same weights give LU plans the same shares.
fn front_flops(nr: u32, nc: u32) -> f64 {
    let (r, c) = (nr as f64, nc as f64);
    4.0 * c * (r * r - r * c + c * c / 3.0)
}
