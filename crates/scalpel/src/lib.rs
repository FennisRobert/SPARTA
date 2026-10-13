//! SCALPEL: Sparse Coarsening And Local Partitioning Elimination Logic.
//! Independent, deterministic parallel nested dissection. No GPU dependency.
//! Input is an undirected CSR graph; output uses `permutation[new] = old`.
#![forbid(unsafe_code)]
mod compress;
mod cover;
mod flow;
mod graph;
mod heap;
mod leaf;
mod refine;
mod separator;
mod spectral;
mod wide;

pub use graph::Index;

use graph::Graph;
use rayon::prelude::*;

/// Controls ordering cost and separator quality. Counts are numbers of vertices.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// Maximum subgraph size for exact minimum-degree leaf elimination (1..=256).
    pub leaf_size: usize,
    /// Minimum subgraph size for parallel searches and recursion.
    pub parallel_threshold: usize,
    /// Independent separator candidates; more candidates trade time for quality.
    pub separator_trials: usize,
    /// Refine every large-graph candidate before selection; costs more CPU work.
    pub refine_all_candidates: bool,
    /// Maximum fraction of vertex weight in either child (0.5..=0.9).
    pub imbalance: f64,
    /// Use two-ended distance-difference separators (SPRAL ND half-level sets).
    pub half_levels: bool,
    /// Score feasible partitions by separator weight divided by child-weight product.
    pub balance_quality: bool,
    /// Maximum greedy separator improvement sweeps at each resolution.
    pub refinement_passes: usize,
    /// FM passes per resolution; 0 selects greedy-only refinement.
    pub fm_passes: usize,
    /// PMORSy-style refinement: fix moves toward the initially smaller child per pass.
    pub one_sided_refinement: bool,
    /// Contract graphs above this size before finding a separator; 0 disables it.
    pub coarsen_threshold: usize,
    /// Merge vertices with identical closed neighborhoods before dissection.
    pub compress: bool,
    /// Improve separator boundaries with exact weighted bipartite vertex covers.
    pub cover_refinement: bool,
    /// Widen-and-re-minimise refinement cycles per bisection; 0 disables them.
    pub wide_cycles: usize,
    /// Extra initial candidates cut along the graph's lowest Laplacian
    /// eigenvectors, on graphs of at least `parallel_threshold` vertices; 0 disables them.
    pub spectral_candidates: usize,
    /// Project every coarse candidate to the full graph and choose there,
    /// instead of choosing on the coarsest graph; costs more CPU work.
    pub fine_selection: bool,
}
impl Default for Options {
    /// Returns conservative task sizes; use benchmarks to tune for a workload.
    fn default() -> Self {
        Self {
            leaf_size: 32,
            parallel_threshold: 4096,
            separator_trials: 4,
            refine_all_candidates: false,
            imbalance: 0.8,
            half_levels: false,
            balance_quality: true,
            refinement_passes: 4,
            fm_passes: 4,
            one_sided_refinement: false,
            coarsen_threshold: 512,
            compress: true,
            cover_refinement: true,
            wide_cycles: 4,
            spectral_candidates: 0,
            fine_selection: false,
        }
    }
}

/// Invalid graph or options, reported before starting ordering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub &'static str);
impl std::fmt::Display for Error {
    /// Formats the error into the supplied formatter.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Error {}

/// Orders an undirected graph with both directions present in sorted CSR rows.
/// `ptr` contains n+1 offsets; `adj` contains unique zero-based vertex indices.
/// Both may be `usize` or `u32`, so a 32-bit caller need not widen its arrays.
/// Diagonal entries are accepted and ignored. `options` controls work and balance.
/// Returns `permutation[new] = old`. Run inside `rayon::ThreadPool::install` to
/// select a private thread pool, or set `RAYON_NUM_THREADS` for the global pool.
pub fn order<T: Index>(ptr: &[T], adj: &[T], options: &Options) -> Result<Vec<usize>, Error> {
    if options.leaf_size == 0
        || options.leaf_size > 256
        || options.parallel_threshold == 0
        || options.separator_trials == 0
        || !options.imbalance.is_finite()
        || !(0.5..=0.9).contains(&options.imbalance)
        || (options.coarsen_threshold != 0 && options.coarsen_threshold < 32)
    {
        return Err(Error("invalid SCALPEL options"));
    }
    // FM heap entries pack a local vertex index into 32 bits.
    if ptr.len() > u32::MAX as usize {
        return Err(Error("SCALPEL supports fewer than 2^32 - 1 vertices"));
    }
    let graph = Graph::checked(ptr, adj)?;
    if options.compress {
        if let Some((compressed, groups)) = compress::compress(&graph) {
            drop(graph);
            return Ok(recurse(&compressed, &compressed, options)
                .into_iter()
                .flat_map(|v| groups.of(v).iter().copied())
                .collect());
        }
    }
    Ok(recurse(&graph, &graph, options))
}

/// Orders local CSR `g`, a subgraph of `root`, using `o`. Returns `root`
/// vertex identifiers. Leaves read their ancestor-separator neighbors from `root`.
fn recurse(g: &Graph, root: &Graph, o: &Options) -> Vec<usize> {
    let n = g.len();
    if n <= o.leaf_size {
        return leaf::order(g, root);
    }
    let components = g.components();
    if components.len() > 1 {
        // Batch singleton output without spawning one task per isolated vertex.
        let mut isolated = Vec::new();
        let mut groups = Vec::new();
        for c in components {
            if c.len() == 1 {
                isolated.push(g.ids[c[0]]);
            } else {
                groups.push(c);
            }
        }
        let map = g.local_map(&groups);
        // A component holds every neighbor of its vertices.
        let run = |c: &Vec<usize>| recurse(&g.induced(c, &map, |_| true), root, o);
        let parts: Vec<_> = if n >= o.parallel_threshold {
            groups.par_iter().map(run).collect()
        } else {
            groups.iter().map(run).collect()
        };
        isolated.extend(parts.into_iter().flatten());
        return isolated;
    }
    let labels = separator::bisect(g, o);
    debug_assert!((0..n).all(|v| g
        .neighbors(v)
        .iter()
        .all(|&u| labels[v] == 2 || labels[u as usize] == 2 || labels[v] == labels[u as usize])));
    let mut groups = [Vec::new(), Vec::new(), Vec::new()];
    for (v, &side) in labels.iter().enumerate() {
        groups[side as usize].push(v);
    }
    let limit = (g.weight.iter().sum::<usize>() as f64 * o.imbalance).ceil() as usize;
    if groups[0].is_empty()
        || groups[1].is_empty()
        || groups[..2]
            .iter()
            .any(|group| group.iter().map(|&v| g.weight[v]).sum::<usize>() > limit)
    {
        // Dense/expander graphs may have no useful balanced separator. Bound work
        // and recursion depth instead of repeatedly peeling a single vertex.
        let mut vertices: Vec<_> = (0..n).collect();
        vertices.sort_unstable_by_key(|&v| (g.neighbors(v).len(), g.ids[v]));
        return vertices.into_iter().map(|v| g.ids[v]).collect();
    }
    let map = g.local_map(&groups);
    // No edge joins the children, so a neighbor is in a child iff it has its label.
    let child = |k: u8| {
        let c = g.induced(&groups[k as usize], &map, |u| labels[u] == k);
        recurse(&c, root, o)
    };
    let (mut left, right) = if n >= o.parallel_threshold {
        rayon::join(|| child(0), || child(1))
    } else {
        (child(0), child(1))
    };
    left.extend(right);
    // Ancestor updates make separators dense; degree tie breaking is cheap.
    groups[2].sort_unstable_by_key(|&v| (g.neighbors(v).len(), g.ids[v]));
    left.extend(groups[2].iter().map(|&v| g.ids[v]));
    left
}
