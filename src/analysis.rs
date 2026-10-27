use crate::symbolic::{
    symbolic_factorize, CscPattern, OrderingMethod, Supernode, SupernodeParams,
    SymbolicFactorization, PARALLEL_COLUMNS,
};
use crate::Config;
use crate::{Error, Result};
use rayon::prelude::*;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Node {
    /// Element offset into the reusable frontal workspace.
    pub offset: u32,
    pub nr: u32,
    pub nc: u32,
    pub rows: u32,
    pub children: u32,
    pub count: u32,
    pub entries: u32,
    pub nentries: u32,
    pub first: u32,
    /// Element offset into the permanent factor store. The node's `nc` L
    /// columns live there column-major with leading dimension `nr`.
    ///
    /// On the LU path this one block holds three things at once, because the
    /// first `nc` columns of a finished front are exactly `L11` below the
    /// diagonal, `U11` above it, and `L21` beneath. Only `U12` needs a second
    /// block, which is what keeps LU factor storage at its `nc*(2*nr - nc)`
    /// minimum instead of two full `nr*nc` copies.
    pub loffset: u32,
    /// Element offset into the panel workspace, for wide fronts only. Holds the
    /// panel's updated columns while the front itself stays uniformly stale.
    pub woffset: u32,
    /// Element offset into the `U12` store, LU only: the node's `nc` rows of
    /// the trailing columns, row-major with leading dimension `nr - nc`. Zero
    /// and unread on the LDL^T path, where `U` is `L^T` and needs no storage.
    pub uoffset: u32,
    /// Element offset, in the same workspace as `offset`, of this node's
    /// contribution block: the trailing `nr - nc` square that the parent
    /// assembles.
    ///
    /// Either a slot of its own, when `store_factors` compacted the block out
    /// of the front so the whole `nr*nr` slot could be recycled a level early,
    /// or a position inside the front itself when it did not -- see
    /// [`COMPACT_PERCENT`]. `sld` says which, and the parent reads the block
    /// through both without caring. A fully summed front has no block and
    /// neither field is read.
    pub soffset: u32,
    /// Leading dimension of the block at `soffset`: `nr - nc` once compacted,
    /// `nr` while it is still strided inside the front.
    pub sld: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Child {
    pub node: u32,
    pub map: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    pub source: u32,
    pub position: u32,
}
/// Which factorization a plan was built for.
///
/// The symbolic phase is identical either way -- both paths eliminate the same
/// symmetric pattern, in the same order, with the same fronts and the same
/// assembly maps -- so the difference is confined to how much of each front is
/// stored, which tiles the trailing update covers, and which kernels run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Complex-symmetric `LDL^T`. Input is the lower triangle; the front is
    /// mirrored about its diagonal and `U` is never formed.
    Ldlt,
    /// Unsymmetric `LU` with partial pivoting. Input is the whole matrix and
    /// both triangles are stored, but the *pattern* is still symmetrized, so
    /// the elimination tree, the fronts and the assembly maps are unchanged.
    Lu,
}
/// Immutable symbolic boundary. Input is sorted, unique, lower-triangle CSC.
///
/// Frontal matrices share one reusable workspace: a front is live only from its
/// own level until its parent's level, so peak workspace tracks the assembly
/// tree's live set rather than the sum over every node. The `nc` factor columns
/// each node contributes to the solve are copied out to a separate permanent
/// store, which is what makes the reuse legal.
#[derive(Clone)]
pub struct Analysis {
    /// The settings this plan was built with. Front classification and
    /// dispatch read them back, so a plan always reports itself consistently
    /// with how it was analyzed.
    pub(crate) config: Config,
    pub(crate) kind: Kind,
    pub(crate) n: usize,
    pub(crate) nnz: usize,
    pub(crate) nodes: Vec<Node>,
    pub(crate) rows: Vec<u32>,
    pub(crate) children: Vec<Child>,
    pub(crate) maps: Vec<u32>,
    pub(crate) entries: Vec<Entry>,
    pub(crate) levels: Vec<Vec<u32>>,
    pub(crate) perm: Vec<u32>,
    /// Peak elements live in the frontal workspace at any one level.
    pub(crate) elements: usize,
    /// Elements in the permanent factor store.
    pub(crate) factor_elements: usize,
    /// Elements in the wide-front panel workspace.
    pub(crate) panel_elements: usize,
    /// Elements in the `U12` store. Zero on the LDL^T path.
    pub(crate) upper_elements: usize,
}
/// Shape checks common to both entry points: square, nonempty, well-formed
/// column pointers, and a usable front bound.
fn shape(n: usize, ptr: &[usize], rows: &[usize], max_front: usize, what: &str) -> Result<()> {
    if n == 0
        || n > u32::MAX as usize
        || ptr.len() != n + 1
        || ptr[0] != 0
        || ptr[n] != rows.len()
        || ptr.windows(2).any(|w| w[0] > w[1])
    {
        return Err(Error::Invalid(what.into()));
    }
    if max_front == 0 {
        return Err(Error::Invalid("max_front must be at least 1".into()));
    }
    Ok(())
}

/// Lower triangle of `pattern(A) | pattern(A^T)`, which is what the symmetric
/// symbolic phase consumes. Sorted and duplicate-free by construction: each
/// column is built by merging two already-sorted lists.
fn symmetrized_lower(n: usize, ptr: &[usize], rows: &[usize]) -> (Vec<usize>, Vec<usize>) {
    // Row j of the lower triangle of A^T is column j of A above the diagonal,
    // transposed; collect those first so each column can be merged in one pass.
    let mut counts = vec![0usize; n];
    for j in 0..n {
        for &i in &rows[ptr[j]..ptr[j + 1]] {
            if i < j {
                counts[i] += 1;
            }
        }
    }
    let mut at = vec![0usize; n + 1];
    for j in 0..n {
        at[j + 1] = at[j] + counts[j];
    }
    let mut mirrored = vec![0usize; at[n]];
    let mut fill = at[..n].to_vec();
    for j in 0..n {
        for &i in &rows[ptr[j]..ptr[j + 1]] {
            if i < j {
                mirrored[fill[i]] = j;
                fill[i] += 1;
            }
        }
    }
    let mut out_ptr = Vec::with_capacity(n + 1);
    let mut out_rows = Vec::with_capacity(rows.len());
    out_ptr.push(0);
    for j in 0..n {
        // Both sides are ascending: `rows` because the input is sorted, and
        // `mirrored[j]` because it was filled in increasing column order.
        let own = rows[ptr[j]..ptr[j + 1]].iter().copied().filter(|&i| i >= j);
        let other = mirrored[at[j]..at[j + 1]].iter().copied();
        let (mut own, mut other) = (own.peekable(), other.peekable());
        loop {
            let take = match (own.peek(), other.peek()) {
                (None, None) => break,
                (Some(&a), Some(&b)) => {
                    if a <= b {
                        other.next_if_eq(&a);
                        own.next()
                    } else {
                        other.next()
                    }
                }
                (Some(_), None) => own.next(),
                (None, Some(_)) => other.next(),
            };
            out_rows.push(take.unwrap());
        }
        out_ptr.push(out_rows.len());
    }
    (out_ptr, out_rows)
}

fn narrow(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::Unsupported("32-bit descriptor limit".into()))
}

/// Largest a contribution block may be, as a percentage of the front it comes
/// out of, for compacting it to be worth the copy.
///
/// Compaction frees `nr*nr - tail*tail` for the level between a node and its
/// parent, and costs `2*tail*tail` of memory traffic in `store_factors`. When
/// `tail` approaches `nr` -- a wide separator over a thin pivot block, which is
/// what curl-curl fronts look like -- it frees almost nothing and still pays in
/// full, so those fronts keep their block in place and are simply held until
/// the parent has assembled, exactly as before.
///
/// Peak workspace and compaction traffic against this cutoff, from the
/// liveness model, on `em_bandcond_sym1` and a 30^3 curl-curl grid:
///
/// ```text
///   cutoff   bandcond peak / traffic     curl30 peak / traffic
///     none     0.881 GiB /  0.00 GiB     1.039 GiB /  0.00 GiB
///      70%     0.830 GiB /  2.75 GiB     1.028 GiB /  0.47 GiB
///      80%     0.765 GiB /  5.15 GiB     1.024 GiB /  1.26 GiB
///     100%     0.764 GiB /  5.99 GiB     1.023 GiB / 35.84 GiB
/// ```
///
/// 80% takes essentially all of the memory there is to take on both, and is
/// where curl-curl stops paying for it: going on to 100% costs it another
/// 34 GiB of traffic to save 1 MiB. Measured factor time on a 30^3 curl-curl
/// grid went 1.041s -> 1.380s at 100% and back to 1.065s here; on
/// `em_bandcond_sym1`, 0.322s -> 0.353s and back to 0.345s.
///
/// What the whole mechanism is worth, at this cutoff, against holding every
/// front to its parent -- peak workspace and factor time:
///
/// ```text
///   em_m2_sym1        0.105 -> 0.080 GiB    0.0455 -> 0.0466 s
///   em_m1_sym1        0.273 -> 0.209 GiB    0.1174 -> 0.1207 s
///   em_bandcond_sym1  1.004 -> 0.790 GiB    0.3218 -> 0.3451 s
///   curl_curl_30^3    1.115 -> 1.108 GiB    1.0408 -> 1.0650 s
/// ```
///
/// So it is worth about a fifth of the frontal workspace on the EMerge systems
/// and nothing at all on curl-curl, whose fronts are all separator: those keep
/// their blocks in place and pay only for the ones near the cutoff.
const COMPACT_PERCENT: usize = 80;

/// Whether this front's contribution block is worth compacting out of it.
fn worth_compacting(nr: usize, nc: usize) -> bool {
    let tail = nr - nc;
    tail > 0 && tail * tail * 100 <= COMPACT_PERCENT * nr * nr
}

/// Returns free slots to the gap list and hands trailing space back to the
/// bump pointer. Gaps stay sorted by address and maximally coalesced.
fn reclaim(gaps: &mut Vec<(usize, usize)>, freed: &mut Vec<(usize, usize)>, top: &mut usize) {
    if freed.is_empty() {
        return;
    }
    gaps.append(freed);
    gaps.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(gaps.len());
    for gap in gaps.drain(..) {
        match merged.last_mut() {
            Some(last) if last.0 + last.1 == gap.0 => last.1 += gap.1,
            _ => merged.push(gap),
        }
    }
    *gaps = merged;
    while let Some(&(start, len)) = gaps.last() {
        if start + len != *top {
            break;
        }
        *top = start;
        gaps.pop();
    }
}

/// First-fit over the coalesced gap list, falling back to the bump pointer.
fn place(gaps: &mut Vec<(usize, usize)>, top: &mut usize, size: usize) -> usize {
    match gaps.iter().position(|gap| gap.1 >= size) {
        Some(i) if gaps[i].1 == size => gaps.remove(i).0,
        Some(i) => {
            let start = gaps[i].0;
            gaps[i] = (start + size, gaps[i].1 - size);
            start
        }
        None => {
            let start = *top;
            *top += size;
            start
        }
    }
}

/// Moves every wide front to the last level its parent allows, and leaves the
/// small and narrow fronts where they are. Returns the new level of each node.
///
/// A level runs its wide fronts' panel steps in lockstep, as many as its
/// widest front needs, so the factorization pays the sum over levels of the
/// widest front's steps. Levels count height above the leaves, which on an
/// unbalanced tree puts separators of the same depth, and similar width, on
/// different levels, each paying its own steps. Counting down from the root
/// instead lines them up. Factorization device time on the M4 Pro is
/// 3.5 ms/GFLOP plus 86 us per panel step, within 1% on the four EMerge
/// systems, so a step saved is worth 0.025 GFLOP. Steps and device time, before
/// and after: `em_m2_sym1` 135 -> 123, -10%; `em_m1_sym1` 285 -> 257, -6%;
/// `em_bandcond_sym1` 392 -> 338, -1.3%. Moving the small fronts as well
/// costs more in assembly than the steps save: `em_m2_sym1` +8%.
fn defer_wide_fronts(
    nodes: &[Node],
    parent: &[u32],
    levels: &[usize],
    nlevels: usize,
    config: &Config,
) -> Vec<usize> {
    let mut late = levels.to_vec();
    // Parents come after their children, so a parent is placed first.
    for s in (0..nodes.len()).rev() {
        let node = nodes[s];
        if is_wide(node.nr, node.nc, config.small_front_rows, config.narrow_cols) {
            late[s] = match parent[s] {
                u32::MAX => nlevels - 1,
                up => late[up as usize] - 1,
            };
        }
    }
    late
}

/// Panel steps the factorization runs under `levels`: per level, the steps of
/// its widest front, as `dispatch_plan` schedules them.
fn panel_steps(nodes: &[Node], levels: &[usize], nlevels: usize, config: &Config) -> u32 {
    let mut steps = vec![0u32; nlevels];
    for (node, &l) in nodes.iter().zip(levels) {
        if is_wide(node.nr, node.nc, config.small_front_rows, config.narrow_cols) {
            steps[l] = steps[l].max(node.nc.div_ceil(config.panel));
        }
    }
    steps.iter().sum()
}

/// Places every frontal matrix, and the contribution block it leaves behind, in
/// one shared workspace, and returns the peak element count.
///
/// A front is live for the level that factors it. What happens next depends on
/// [`worth_compacting`]: either `store_factors` compacts the trailing block
/// into a slot of its own and the whole `nr*nr` front is recycled from the next
/// level, or the block stays strided inside the front and the front is held
/// until the parent has assembled it -- which is what every front used to do.
///
/// A level is allocated in two phases, because that is how it runs. Assembly
/// comes first and consumes the children's blocks; compaction comes last and
/// writes this level's. So the blocks a level *reads* are dead by the time the
/// blocks it *writes* are placed, and the two can share storage. Allocating
/// both at once instead -- the obvious reading of "live at this level" -- holds
/// two generations of blocks at the peak and is *worse* than never compacting
/// at all: measured 1.12 GiB against the 1.00 GiB of holding whole fronts, on
/// `em_bandcond_sym1`.
///
/// What the phase split buys on that system is 1.00 GiB -> 0.77 GiB, and on
/// `em_m1_sym1` 0.27 GiB -> 0.21 GiB.
///
/// Level-ordered first-fit with coalescing, largest request first within a
/// phase. On 3D curl-curl problems this lands under 20% above the liveness
/// lower bound while costing well under a millisecond per thousand nodes; a
/// global greedy closes that gap by ~1% and costs three orders of magnitude
/// more time, and best-fit is a wash -- it wins on one EMerge system and loses
/// on another.
fn assign_front_offsets(
    nodes: &mut [Node],
    levels: &[usize],
    parent: &[u32],
    nlevels: usize,
) -> Result<usize> {
    let mut by_level: Vec<Vec<u32>> = vec![Vec::new(); nlevels];
    for (s, &l) in levels.iter().enumerate() {
        by_level[l].push(s as u32);
    }
    // Front slots, free from the level after the one that factored them.
    let mut free_front: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nlevels + 1];
    // Slots holding a contribution block -- a compacted one, or the front it
    // was left inside -- free once the level that assembles it has done so,
    // which is before that same level compacts its own. Hence two phases.
    let mut free_block: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nlevels + 1];
    let mut gaps: Vec<(usize, usize)> = Vec::new();
    let (mut top, mut peak) = (0usize, 0usize);
    let mut order: Vec<u32> = Vec::new();
    // The level that assembles this node's block, and so frees it. A root is
    // assembled by nobody and frees at its own level.
    let consumed = |s: u32| match parent[s as usize] {
        u32::MAX => levels[s as usize],
        up => levels[up as usize],
    };
    for level in 0..nlevels {
        reclaim(&mut gaps, &mut free_front[level], &mut top);
        order.clone_from(&by_level[level]);
        order.sort_unstable_by_key(|&s| (std::cmp::Reverse(nodes[s as usize].nr), s));
        for &s in &order {
            let node = nodes[s as usize];
            let (nr, nc) = (node.nr as usize, node.nc as usize);
            let start = place(&mut gaps, &mut top, nr * nr);
            let node = &mut nodes[s as usize];
            node.offset = narrow(start)?;
            if worth_compacting(nr, nc) {
                free_front[level + 1].push((start, nr * nr));
            } else {
                // Left in place: the block is the trailing square of the front,
                // so the front itself has to survive until the parent reads it.
                node.soffset = narrow(if nr > nc { start + nc * nr + nc } else { start })?;
                node.sld = nr as u32;
                free_block[consumed(s)].push((start, nr * nr));
            }
        }
        peak = peak.max(top);
        // Phase two: this level's assembly is done with its children's blocks.
        reclaim(&mut gaps, &mut free_block[level], &mut top);
        order.sort_unstable_by_key(|&s| {
            let node = nodes[s as usize];
            (std::cmp::Reverse(node.nr - node.nc), s)
        });
        for &s in &order {
            let node = nodes[s as usize];
            let (nr, nc) = (node.nr as usize, node.nc as usize);
            if !worth_compacting(nr, nc) {
                continue;
            }
            let tail = nr - nc;
            let start = place(&mut gaps, &mut top, tail * tail);
            let node = &mut nodes[s as usize];
            node.soffset = narrow(start)?;
            node.sld = tail as u32;
            free_block[consumed(s)].push((start, tail * tail));
        }
        peak = peak.max(top);
    }
    Ok(peak)
}

/// Input columns per parallel task when building the assembly map. Like
/// [`crate::symbolic::PARALLEL_COLUMNS`], a granularity constant rather than a
/// [`crate::Config`] setting: it changes no result and has a flat optimum.
const ENTRY_BLOCK: usize = 4096;

/// Resolves one input entry to its owning node and its permuted coordinates.
/// `i`/`j` are the input row/column and `perm_inv` maps input to permuted
/// indices. On the LDL^T path an entry the permutation lifted above the
/// diagonal is reflected back down, because only one triangle is stored; the
/// LU path keeps it, since reflecting it would transpose the matrix.
/// Returns `None` when no node owns the resolved column.
#[inline]
fn locate(
    kind: Kind,
    perm_inv: &[usize],
    owner: &[usize],
    i: usize,
    j: usize,
) -> Option<(usize, usize, usize)> {
    let (mut r, mut c) = (perm_inv[i], perm_inv[j]);
    if kind == Kind::Ldlt && r < c {
        std::mem::swap(&mut r, &mut c);
    }
    // A node owns the entries it eliminates: those in its fully summed
    // columns, and -- above the diagonal, LU only -- those in its fully summed
    // rows. The other index is a front row either way, because the pattern the
    // fronts were built from is symmetric.
    let s = if r >= c { owner[c] } else { owner[r] };
    (s != usize::MAX).then_some((s, r, c))
}

/// Builds the flat assembly map: every stored input entry paired with its slot
/// in its owning front. Returns the entries grouped by node and the per-node
/// count, so the caller can fill in each node descriptor.
///
/// Parallel over fixed blocks of input columns. Block boundaries and the
/// node-major, block-minor output order do not depend on the thread count, so
/// the result is byte-for-byte the order a single serial scan would produce.
/// The scatter targets are disjoint by construction -- each (block, node) pair
/// owns one contiguous run whose start the prefix sum fixed in advance -- so
/// the relaxed atomic slots are each written exactly once.
fn build_assembly_entries(
    kind: Kind,
    n: usize,
    ptr: &[usize],
    rows: &[usize],
    sym: &SymbolicFactorization,
    owner: &[usize],
    front_rows: &[Vec<u32>],
) -> Result<(Vec<Entry>, Vec<u32>)> {
    let (perm_inv, supernodes) = (&sym.perm_inv, &sym.supernodes);
    let nodes = supernodes.len();
    let bounds: Vec<(usize, usize)> = (0..n)
        .step_by(ENTRY_BLOCK)
        .map(|first| (first, (first + ENTRY_BLOCK).min(n)))
        .collect();
    let parallel = n >= PARALLEL_COLUMNS;

    // Pass 1: how many entries each block contributes to each node. This walks
    // the input without the binary search the placement pass needs.
    let mut counts = vec![0u32; bounds.len() * nodes];
    let count = |(row, &(first, last)): (&mut [u32], &(usize, usize))| -> Result<()> {
        for j in first..last {
            for &i in &rows[ptr[j]..ptr[j + 1]] {
                let (s, _, _) = locate(kind, perm_inv, owner, i, j)
                    .ok_or_else(|| Error::Invalid("unowned symbolic column".into()))?;
                row[s] += 1;
            }
        }
        Ok(())
    };
    if parallel {
        counts
            .par_chunks_mut(nodes)
            .zip(bounds.par_iter())
            .try_for_each(count)?;
    } else {
        counts
            .chunks_mut(nodes)
            .zip(bounds.iter())
            .try_for_each(count)?;
    }

    // Grouped by node, and within a node by block, which is ascending input
    // column order.
    let mut start = vec![0u32; bounds.len() * nodes];
    let mut per_node = vec![0u32; nodes];
    let mut total = 0usize;
    for s in 0..nodes {
        let first = total;
        for b in 0..bounds.len() {
            start[b * nodes + s] = narrow(total)?;
            total += counts[b * nodes + s] as usize;
        }
        per_node[s] = narrow(total - first)?;
    }

    // Pass 2: place each entry. `source` and `position` pack into one slot so
    // a single relaxed store publishes both.
    let slots: Vec<AtomicU64> = (0..total).map(|_| AtomicU64::new(0)).collect();
    let place = |(cursor, &(first, last)): (&mut [u32], &(usize, usize))| -> Result<()> {
        for j in first..last {
            for (index, &i) in rows.iter().enumerate().take(ptr[j + 1]).skip(ptr[j]) {
                let (s, r, c) = locate(kind, perm_inv, owner, i, j)
                    .ok_or_else(|| Error::Invalid("unowned symbolic column".into()))?;
                let node = &supernodes[s];
                let front = &front_rows[s];
                let find = |v: usize| {
                    front
                        .binary_search(&(v as u32))
                        .map_err(|_| Error::Invalid("input absent from symbolic front".into()))
                };
                let (local_row, local_col) = if r >= c {
                    (find(r)?, c - node.first_col)
                } else {
                    (r - node.first_col, find(c)?)
                };
                let position = (local_row * front.len() + local_col) as u64;
                slots[cursor[s] as usize].store(((index as u64) << 32) | position, Relaxed);
                cursor[s] += 1;
            }
        }
        Ok(())
    };
    if parallel {
        start
            .par_chunks_mut(nodes)
            .zip(bounds.par_iter())
            .try_for_each(place)?;
    } else {
        start
            .chunks_mut(nodes)
            .zip(bounds.iter())
            .try_for_each(place)?;
    }

    let entries = slots
        .into_iter()
        .map(|slot| {
            let packed = slot.into_inner();
            Entry {
                source: (packed >> 32) as u32,
                position: packed as u32,
            }
        })
        .collect();
    Ok((entries, per_node))
}

/// Computes the complete ascending row list of one front, plus the local index
/// within it of every child's trailing row, concatenated in child order.
///
/// `s` names the node, `adjacency` is the permuted symmetrized pattern, and
/// `front_rows` must already hold the finished rows of every child of `s`.
/// `max_front` caps the number of rows a front may have.
fn build_front(
    s: usize,
    supernodes: &[Supernode],
    adjacency: &CscPattern,
    front_rows: &[Vec<u32>],
    max_front: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let node = &supernodes[s];
    let end = node.first_col + node.ncol;
    let cut = end as u32;
    let mut trailing = Vec::new();
    for c in node.first_col..end {
        let col = adjacency.column(c);
        // Rows ascend, so the trailing block is a suffix of the column.
        trailing.extend_from_slice(&col[col.partition_point(|&r| r < cut)..]);
    }
    for &child in &node.children {
        trailing.extend(
            front_rows[child][supernodes[child].ncol..]
                .iter()
                .copied()
                .filter(|&r| r >= cut),
        );
    }
    trailing.sort_unstable();
    trailing.dedup();
    let mut complete: Vec<u32> = (node.first_col as u32..cut).collect();
    complete.extend(trailing);
    let nr = complete.len();
    if nr > max_front {
        return Err(Error::Unsupported(format!(
            "node {s} has front {nr}, limit {max_front}; configured front bound exceeded"
        )));
    }
    if node.ncol == 0 || nr < node.ncol || complete.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::Invalid("invalid symbolic front".into()));
    }
    let mut maps = Vec::new();
    for &child in &node.children {
        for row in &front_rows[child][supernodes[child].ncol..] {
            let local = complete
                .binary_search(row)
                .map_err(|_| Error::Invalid("child contribution missing from parent".into()))?;
            maps.push(local as u32);
        }
    }
    Ok((complete, maps))
}

/// Every front's rows, every node's child maps, and every node's tree level,
/// each indexed by node.
type Fronts = (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<usize>);

/// Builds every front's rows and child maps, and each node's tree level.
///
/// A node unions its children's trailing rows, so it cannot start until they
/// finish; nodes sharing an elimination-tree level are independent and are
/// built together. Results are placed by node index, so the output does not
/// depend on the order the level's tasks happened to complete.
fn build_fronts(
    supernodes: &[Supernode],
    adjacency: &CscPattern,
    max_front: usize,
    parallel: bool,
) -> Result<Fronts> {
    let count = supernodes.len();
    let mut levels = vec![0usize; count];
    let mut by_level: Vec<Vec<usize>> = Vec::new();
    for (s, node) in supernodes.iter().enumerate() {
        for &child in &node.children {
            if child >= s {
                return Err(Error::Invalid(
                    "symbolic nodes must be children-first".into(),
                ));
            }
            levels[s] = levels[s].max(levels[child] + 1);
        }
        while by_level.len() <= levels[s] {
            by_level.push(Vec::new());
        }
        by_level[levels[s]].push(s);
    }
    let mut front_rows: Vec<Vec<u32>> = vec![Vec::new(); count];
    let mut child_maps: Vec<Vec<u32>> = vec![Vec::new(); count];
    for level in &by_level {
        let build = |&s: &usize| build_front(s, supernodes, adjacency, &front_rows, max_front);
        let built: Vec<(Vec<u32>, Vec<u32>)> = if parallel {
            level.par_iter().map(build).collect::<Result<_>>()?
        } else {
            level.iter().map(build).collect::<Result<_>>()?
        };
        for (&s, (rows, maps)) in level.iter().zip(built) {
            front_rows[s] = rows;
            child_maps[s] = maps;
        }
    }
    Ok((front_rows, child_maps, levels))
}

impl Analysis {
    /// Analyzes a complex-symmetric matrix for the `LDL^T` path.
    ///
    /// `n` is the dimension. `ptr` and `rows` are the **lower triangle** in
    /// CSC: `ptr` holds `n + 1` column offsets and `rows` holds row indices,
    /// sorted and unique within each column, each at or below the diagonal.
    /// `config` carries every tuning choice; see [`Config`].
    ///
    /// The plan keeps a copy of `config`, so a later factorization and the
    /// Metal kernels it compiles use exactly the settings analyzed with.
    pub fn ldlt(n: usize, ptr: &[usize], rows: &[usize], config: &Config) -> Result<Self> {
        config.check().map_err(Error::Invalid)?;
        // The whole analysis runs on the configured threads, which also keeps
        // the profiler's marks, recorded per thread, on one thread.
        crate::threads::Pool::new(config.threads)?.run(|| Self::ldlt_here(n, ptr, rows, config))
    }

    /// [`Analysis::ldlt`] on the calling thread's pool.
    fn ldlt_here(n: usize, ptr: &[usize], rows: &[usize], config: &Config) -> Result<Self> {
        let max_front = config.max_front.unwrap_or(usize::MAX);
        crate::profile::start();
        shape(
            n,
            ptr,
            rows,
            max_front,
            "nonempty square lower CSC required",
        )?;
        for j in 0..n {
            let col = &rows[ptr[j]..ptr[j + 1]];
            if col.iter().any(|&i| i < j || i >= n) || col.windows(2).any(|w| w[0] >= w[1]) {
                return Err(Error::Invalid(
                    "CSC rows must be sorted, unique, and lower triangular".into(),
                ));
            }
        }
        crate::profile::mark("validate");
        Self::build(Kind::Ldlt, n, ptr, rows, ptr, rows, max_front, config)
    }

    /// Analyzes an unsymmetric matrix for the `LU` path, with partial pivoting.
    ///
    /// `n` is the dimension. `ptr` and `rows` are the **whole** matrix in CSC,
    /// sorted and unique within each column. `config` carries every tuning
    /// choice; see [`Config`].
    ///
    /// The symbolic phase is the symmetric one, run on
    /// `pattern(A) | pattern(A^T)`. That is the standard treatment for a
    /// pattern that is symmetric or nearly so, which is what the frontal
    /// method wants anyway: a front is already a dense square block, so
    /// squaring up the pattern costs nothing structurally and buys an
    /// elimination tree, an assembly map and a front-placement pass shared
    /// verbatim with the `LDL^T` path. It is the wrong analysis for a strongly
    /// unsymmetric pattern, which would be padded out to its symmetrized fill;
    /// SPARTA does not target those.
    ///
    /// Pivoting is by row interchange within the fully summed rows. Pivots are
    /// not delayed to the parent when that block offers nothing acceptable;
    /// SPARTA perturbs instead and reports the count through
    /// [`Solver::perturbed_pivots`](crate::Solver::perturbed_pivots).
    pub fn lu(n: usize, ptr: &[usize], rows: &[usize], config: &Config) -> Result<Self> {
        config.check().map_err(Error::Invalid)?;
        crate::threads::Pool::new(config.threads)?.run(|| Self::lu_here(n, ptr, rows, config))
    }

    /// [`Analysis::lu`] on the calling thread's pool.
    fn lu_here(n: usize, ptr: &[usize], rows: &[usize], config: &Config) -> Result<Self> {
        let max_front = config.max_front.unwrap_or(usize::MAX);
        crate::profile::start();
        shape(n, ptr, rows, max_front, "nonempty square CSC required")?;
        for j in 0..n {
            let col = &rows[ptr[j]..ptr[j + 1]];
            if col.iter().any(|&i| i >= n) || col.windows(2).any(|w| w[0] >= w[1]) {
                return Err(Error::Invalid(
                    "CSC rows must be sorted, unique, and within the matrix".into(),
                ));
            }
        }
        crate::profile::mark("validate");
        let (sym_ptr, sym_rows) = symmetrized_lower(n, ptr, rows);
        crate::profile::mark("symmetrize_lu");
        Self::build(
            Kind::Lu,
            n,
            &sym_ptr,
            &sym_rows,
            ptr,
            rows,
            max_front,
            config,
        )
    }

    /// Shared symbolic path. `sym_*` is the lower triangle of the symmetrized
    /// pattern, which decides the fronts; `ptr`/`rows` is the caller's own CSC,
    /// which decides only where each input value is scattered.
    #[allow(clippy::too_many_arguments)]
    fn build(
        kind: Kind,
        n: usize,
        sym_ptr: &[usize],
        sym_rows: &[usize],
        ptr: &[usize],
        rows: &[usize],
        max_front: usize,
        config: &Config,
    ) -> Result<Self> {
        narrow(rows.len())?;
        let params = SupernodeParams {
            nemin: config.nemin,
            ..Default::default()
        };
        let small_front = config.small_front_rows;
        let sym = symbolic_factorize(n, sym_ptr, sym_rows, &params, config)
            .map_err(|e| Error::Invalid(e.to_string()))?;
        let mut p = Self {
            config: config.clone(),
            kind,
            n,
            nnz: rows.len(),
            nodes: Vec::new(),
            rows: Vec::new(),
            children: Vec::new(),
            maps: Vec::new(),
            entries: Vec::new(),
            levels: Vec::new(),
            perm: sym.perm.iter().map(|&v| v as u32).collect(),
            elements: 0,
            factor_elements: 0,
            panel_elements: 0,
            upper_elements: 0,
        };
        // The permuted adjacency comes from the symbolic phase, which already
        // built it to take the column counts. Exported supernode row_indices
        // are not the completed numerical fronts.
        let adjacency = &sym.permuted;
        let (front_rows, child_maps, levels) =
            build_fronts(&sym.supernodes, adjacency, max_front, n >= PARALLEL_COLUMNS)?;
        crate::profile::mark("front_rows");
        let mut owner = vec![usize::MAX; n];
        let mut parent = vec![u32::MAX; sym.supernodes.len()];
        for (s, node) in sym.supernodes.iter().enumerate() {
            let end = node.first_col + node.ncol;
            let complete = &front_rows[s];
            let nr = complete.len();
            let descriptor = Node {
                offset: 0, // Assigned by the workspace allocator below.
                nr: nr as u32,
                nc: node.ncol as u32,
                rows: narrow(p.rows.len())?,
                children: narrow(p.children.len())?,
                count: narrow(node.children.len())?,
                entries: 0,
                nentries: 0,
                first: node.first_col as u32,
                loffset: narrow(p.factor_elements)?,
                woffset: narrow(p.panel_elements)?,
                uoffset: narrow(p.upper_elements)?,
                soffset: 0, // Both assigned by the workspace allocator below.
                sld: 0,
            };
            p.factor_elements = p
                .factor_elements
                .checked_add(node.ncol * nr)
                .ok_or_else(|| Error::Unsupported("factor store overflow".into()))?;
            narrow(p.factor_elements)?;
            if kind == Kind::Lu {
                // U12 only: L11, U11 and L21 are all inside the nr x nc block
                // already counted above, which is the whole of the front's
                // first nc columns. Total per node is therefore
                // nc*(2*nr - nc), the minimum an LU factor can occupy.
                p.upper_elements = p
                    .upper_elements
                    .checked_add(node.ncol * (nr - node.ncol))
                    .ok_or_else(|| Error::Unsupported("upper store overflow".into()))?;
                narrow(p.upper_elements)?;
            }
            if is_wide(nr as u32, node.ncol as u32, small_front, config.narrow_cols) {
                p.panel_elements = p
                    .panel_elements
                    .checked_add(nr * config.panel_cols() as usize)
                    .ok_or_else(|| Error::Unsupported("panel workspace overflow".into()))?;
                narrow(p.panel_elements)?;
            }
            p.rows.extend_from_slice(complete);
            owner[node.first_col..end].fill(s);
            let mut at = 0;
            for &child in &node.children {
                parent[child] = s as u32;
                p.children.push(Child {
                    node: child as u32,
                    map: narrow(p.maps.len())?,
                });
                let len = front_rows[child].len() - sym.supernodes[child].ncol;
                p.maps.extend_from_slice(&child_maps[s][at..at + len]);
                at += len;
            }
            while p.levels.len() <= levels[s] {
                p.levels.push(Vec::new());
            }
            p.levels[levels[s]].push(s as u32);
            p.nodes.push(descriptor);
        }
        crate::profile::mark("descriptors+maps");
        let nlevels = p.levels.len();
        p.elements = assign_front_offsets(&mut p.nodes, &levels, &parent, nlevels)?;
        // Deferring spends front workspace, which on a large system is the
        // scarce resource, so it is taken only for a material saving: at least
        // 5% of the panel steps for at most 25% more workspace. Measured:
        // `em_m1_sym1` 10% fewer steps for 17.6% more, `em_bandcond_sym1` 14%
        // for 5.7%, `em_m2_sym1` 9% for none. `em_m3_asym1` would save 0.7% of
        // its steps for 20.7% more, and keeps height levels.
        let late = defer_wide_fronts(&p.nodes, &parent, &levels, nlevels, config);
        let (before, after) = (
            panel_steps(&p.nodes, &levels, nlevels, config),
            panel_steps(&p.nodes, &late, nlevels, config),
        );
        if after + before / 20 <= before {
            let mut nodes = p.nodes.clone();
            let elements = assign_front_offsets(&mut nodes, &late, &parent, nlevels)?;
            if elements <= p.elements + p.elements / 4 {
                p.nodes = nodes;
                p.elements = elements;
                for level in &mut p.levels {
                    level.clear();
                }
                for (s, &l) in late.iter().enumerate() {
                    p.levels[l].push(s as u32);
                }
            }
        }
        crate::profile::mark("front_offsets");
        let (entries, per_node) =
            build_assembly_entries(kind, n, ptr, rows, &sym, &owner, &front_rows)?;
        let mut at = 0u32;
        for (s, &count) in per_node.iter().enumerate() {
            p.nodes[s].entries = at;
            p.nodes[s].nentries = count;
            at += count;
        }
        p.entries = entries;
        crate::profile::mark("assembly_entries");
        crate::profile::report(match config.ordering {
            OrderingMethod::Scalpel => "scalpel",
        });
        Ok(p)
    }
    /// Nonzeros in the symbolic factor: the numbers a solve fundamentally has
    /// to store, before any of this implementation's choices.
    ///
    /// LDL^T counts one triangle including D; LU counts L+U with the diagonal
    /// once. Each front contributes its `nc` columns down to row `nr`, less the
    /// strictly upper corner of the diagonal block, which is structurally zero.
    ///
    /// [`Analysis::factor_bytes`] exceeds `8 *` this, because the factor store
    /// keeps each node's block as a full `nc` by `nr` rectangle rather than
    /// packing that triangular corner away -- about a tenth, in exchange for a
    /// leading dimension the solve can index without a division.
    pub fn factor_nonzeros(&self) -> usize {
        let lower: usize = self
            .nodes
            .iter()
            .map(|node| {
                let c = node.nc as usize;
                c * node.nr as usize - c * c.saturating_sub(1) / 2
            })
            .sum();
        if self.kind == Kind::Lu {
            2 * lower - self.n
        } else {
            lower
        }
    }
    /// Symbolic factor envelope entries divided by input stored entries.
    /// LDLT counts one triangle including D; LU counts L+U with diagonal once.
    pub fn fill_ratio(&self) -> f64 {
        self.factor_nonzeros() as f64 / self.nnz.max(1) as f64
    }

    pub fn dimension(&self) -> usize {
        self.n
    }
    pub fn supernodes(&self) -> usize {
        self.nodes.len()
    }
    pub fn levels(&self) -> usize {
        self.levels.len()
    }
    /// `(nr, nc)` of every front on the cooperative path.
    pub fn large_nodes(&self) -> Vec<(usize, usize)> {
        self.nodes
            .iter()
            .filter(|d| d.nr > self.config.small_front_rows)
            .map(|d| (d.nr as usize, d.nc as usize))
            .collect()
    }
    /// `(nr, nc)` of every front, grouped by factorization level from the
    /// leaves (level 0) to the root.
    pub fn level_fronts(&self) -> Vec<Vec<(usize, usize)>> {
        self.levels
            .iter()
            .map(|level| {
                level
                    .iter()
                    .map(|&id| {
                        let d = &self.nodes[id as usize];
                        (d.nr as usize, d.nc as usize)
                    })
                    .collect()
            })
            .collect()
    }
    pub fn max_front_size(&self) -> usize {
        self.nodes.iter().map(|n| n.nr as usize).max().unwrap_or(0)
    }
    pub fn large_fronts(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.nr > self.config.small_front_rows)
            .count()
    }
    /// Select CPU complex128 (`true`) or Metal complex64 (`false`) for future solvers.
    /// The symbolic layout is unchanged; solvers already created keep their backend.
    pub fn set_force_cpu(&mut self, force_cpu: bool) {
        self.config.force_cpu = force_cpu;
    }
    /// Select the hybrid CPU share, `Config::hybrid_cpu`, for solvers created
    /// from this analysis from now on. Solvers already created keep theirs.
    pub fn set_hybrid_cpu(&mut self, share: f64) {
        self.config.hybrid_cpu = share;
    }
    /// Bytes per numerical scalar for the selected backend.
    fn scalar_bytes(&self) -> usize {
        if self.config.force_cpu { 16 } else { 8 }
    }
    /// Peak bytes live in the reusable frontal workspace.
    pub fn front_bytes(&self) -> usize {
        self.elements * self.scalar_bytes()
    }
    /// Which factorization this plan drives.
    pub fn kind(&self) -> Kind {
        self.kind
    }
    /// Bytes of factor retained for the solve across the whole assembly tree.
    ///
    /// On the LU path this includes `U12`, so it runs up to twice the LDL^T
    /// figure for the same pattern -- exactly `nc*(2*nr - nc)` against `nc*nr`
    /// per front, which is the price of not being able to read `U` off `L`.
    pub fn factor_bytes(&self) -> usize {
        (self.factor_elements + self.upper_elements) * self.scalar_bytes()
    }
    /// The `U12` share of [`Analysis::factor_bytes`]. Zero on the LDL^T path.
    pub fn upper_bytes(&self) -> usize {
        self.upper_elements * self.scalar_bytes()
    }
    /// Persistent GPU panel workspace bytes; CPU scratch is allocated per front.
    pub fn panel_bytes(&self) -> usize {
        if self.config.force_cpu {
            0
        } else {
            self.panel_elements * 8
        }
    }
    /// Total device bytes a GPU [`crate::Solver`] allocates for this plan, or
    /// zero when it is set to the CPU backend.
    ///
    /// Every buffer the backend creates, not only the two big ones. The factor
    /// store and the frontal workspace dominate, but they are not the whole
    /// bill: the assembly map is `8*nnz` on its own, and it is copied to the
    /// device alongside the input values, the row lists, the child maps and the
    /// dispatch tables. On a 457k EMerge system that remainder is 177 MiB
    /// against 1.55 GiB of factors and workspace, so quoting the two large
    /// buffers alone understates a solve by about an eighth.
    ///
    /// The total holds one right-hand side. The first solve of a wider block
    /// grows the solution buffer, and on LU the pivot-order scratch beside it,
    /// to `8 * n * nrhs` bytes each and keeps that capacity.
    #[cfg(any(test, sparta_metal, sparta_vulkan))]
    pub fn device_bytes(&self) -> usize {
        if self.config.force_cpu {
            return 0;
        }
        let d = self.dispatch_plan();
        // Mirrors the buffer table in `gpu::metal::Context::new`, which is the
        // one place that decides what is allocated.
        let descriptors = std::mem::size_of_val(self.nodes.as_slice())
            + std::mem::size_of_val(self.rows.as_slice())
            + std::mem::size_of_val(self.children.as_slice())
            + std::mem::size_of_val(self.maps.as_slice())
            + std::mem::size_of_val(self.entries.as_slice())
            + std::mem::size_of_val(d.ids.as_slice())
            + std::mem::size_of_val(d.tiles.as_slice())
            + std::mem::size_of_val(d.schur.as_slice())
            + self.nodes.len() * std::mem::size_of::<FrontState>()
            + self.nodes.len() * 4;
        // Input values, one right-hand side, and the per-column pivot records.
        let vectors = self.nnz * 8 + self.n * (8 * 3 + 4 * 2);
        descriptors + vectors + self.front_bytes() + self.factor_bytes() + self.panel_bytes()
    }
    /// Zero: this build carries no GPU backend, so nothing is allocated on a
    /// device.
    #[cfg(not(any(test, sparta_metal, sparta_vulkan)))]
    pub fn device_bytes(&self) -> usize {
        0
    }
}

#[cfg(any(test, sparta_metal, sparta_vulkan))]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DispatchLevel {
    pub small_start: u32,
    pub small_count: u32,
    pub large_start: u32,
    pub large_count: u32,
    pub steps: u32,
    pub tile_start: u32,
    pub tile_count: u32,
    /// Index into the step table of this level's first pivot step.
    pub step_start: u32,
    pub schur_start: u32,
    pub schur_count: u32,
    /// Fronts factored completely inside one threadgroup, and their Schur
    /// tiles, which are drained once rather than on the panel cadence.
    pub narrow_start: u32,
    pub narrow_count: u32,
    pub narrow_schur_start: u32,
    pub narrow_schur_count: u32,
    /// Rows in this level's largest front. The solve sweeps parallelise over a
    /// front's rows, so the host matches its threadgroup width to this rather
    /// than dispatching one fixed width for the whole tree.
    pub max_rows: u32,
    /// Widest fully summed block among this level's wide fronts. `factor_panel`
    /// now sweeps only that block -- `factor_panel_tail` takes the rows below
    /// it -- so its useful width follows this rather than the front height, and
    /// dispatching it at one fixed width would leave most lanes with no row to
    /// own on a level of narrow fronts.
    pub max_cols: u32,
    /// Most children any front in this level has. Child Schur blocks are
    /// accumulated into overlapping regions of the parent, so two children of
    /// one front cannot be applied at the same time; the host issues one
    /// dispatch per child ordinal and this says how many that is. A level of
    /// leaves has none and costs no dispatch at all.
    pub max_children: u32,
}
/// How much of a level is still live at one pivot step. Large fronts are
/// ordered by decreasing column count, so the nodes that still have a pivot to
/// take at step `k` are always a prefix, and so are their update tiles.
#[cfg(any(test, sparta_metal, sparta_vulkan))]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepCounts {
    pub nodes: u32,
    /// First strip tile of this step, relative to the level's tile range. Strip
    /// tiles are ordered by tile column, and after panel `q` the front is
    /// current through column `16*(q+1)`, so the tiles that can still write are
    /// exactly those with `col > q` -- a suffix.
    pub tile_start: u32,
    pub tiles: u32,
    pub schur: u32,
}
#[cfg(any(test, sparta_metal, sparta_vulkan))]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct UpdateTile {
    pub node: u32,
    pub row: u32,
    pub col: u32,
}
/// Panel width: pivots accumulated before one rank-`nb` Schur update.
/// The frontal geometry lives in [`crate::Config`]: `panel`, `schur_block`,
/// `strip_block`, `narrow_cols`, `pivot_threads` and `schur_chunk`, with
/// `schur_tile`, `strip_rows`, `strip_span` and `panel_cols` derived from
/// them. A plan stores the values it was built with and `Solver::new` compiles
/// the kernels from the same ones, so the two cannot disagree.
///
/// Reports whether front `nr` by `nc` takes the wide cooperative path.
/// `small_front` and `narrow` come from the plan's configuration.
pub(crate) fn is_wide(nr: u32, nc: u32, small_front: u32, narrow: u32) -> bool {
    nr > small_front && nc > narrow
}
/// Mirror of the Metal-side per-front progress record. Rust only allocates the
/// buffer and asserts the layout; the GPU owns every field.
#[cfg(any(test, sparta_metal, sparta_vulkan))]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
// Only a backend's dispatch loop constructs one.
#[cfg_attr(not(sparta_metal), allow(dead_code))]
pub(crate) struct FrontState {
    /// Pivots taken so far.
    pub next: u32,
    /// First column and width of the most recent pivot.
    pub start: u32,
    pub width: u32,
    /// First pivot whose Schur contribution is still pending.
    pub panel: u32,
    /// Pivots this front replaced with a perturbation rather than failing on.
    pub perturbed: u32,
}
// Read by a backend's dispatch loop; a build without one still builds it so
// `dispatch_audit` and the analysis tests stay available everywhere.
#[cfg(any(test, sparta_metal, sparta_vulkan))]
#[cfg_attr(not(sparta_metal), allow(dead_code))]
pub(crate) struct DispatchPlan {
    pub ids: Vec<u32>,
    pub levels: Vec<DispatchLevel>,
    /// Tiles covering the fully summed columns, updated once per pivot so that
    /// later pivot searches see current data.
    pub tiles: Vec<UpdateTile>,
    /// Tiles covering the Schur complement, drained on the cadence set by
    /// `schur_batch()` as a single rank-`next - panel` product, and always once
    /// more at the end of the level. Coordinates are relative to the Schur
    /// origin.
    pub schur: Vec<UpdateTile>,
    pub steps: Vec<StepCounts>,
}
impl Analysis {
    /// Schur-complement tiles for one node, relative to row and column `nc`.
    ///
    /// LDL^T keeps the lower triangle and mirrors each write; LU has to cover
    /// the whole square, which is a little under twice as many tiles for the
    /// same amount of arithmetic per element.
    #[cfg(any(test, sparta_metal, sparta_vulkan))]
    fn push_schur_tiles(&self, schur: &mut Vec<UpdateTile>, s: u32) {
        let node = self.nodes[s as usize];
        let tiles = (node.nr - node.nc).div_ceil(self.config.schur_tile());
        for row in 0..tiles {
            let last = match self.kind {
                Kind::Ldlt => row,
                Kind::Lu => tiles - 1,
            };
            for col in 0..=last {
                schur.push(UpdateTile { node: s, row, col });
            }
        }
    }
    /// Drop every symbolic array this plan owns, keeping only its dimensions.
    ///
    /// The plan is unusable afterwards: `dispatch_plan` and the storage
    /// accessors would describe a tree that is no longer here. It exists so a
    /// caller can hand the memory back at a point of its choosing rather than
    /// waiting for the value to go out of scope.
    pub fn release(&mut self) {
        self.nodes = Vec::new();
        self.rows = Vec::new();
        self.children = Vec::new();
        self.maps = Vec::new();
        self.entries = Vec::new();
        self.levels = Vec::new();
        self.perm = Vec::new();
    }
    /// The part of this plan a GPU solver still reads once its backend exists:
    /// the dimensions, the configuration, and the permutation.
    ///
    /// Everything else -- the node table, row lists, child maps and assembly
    /// entries -- is copied into device storage when the backend is created and
    /// never read from the host again. On unified memory keeping it as well is
    /// simply the same arrays in RAM twice: 91 MiB of the 176 MiB of
    /// descriptors on a 457k EMerge system, 47 MiB on a 40^3 curl-curl grid.
    ///
    /// The result describes no tree, so everything that walks one --
    /// [`Analysis::dispatch_plan`], [`Analysis::fill_ratio`],
    /// [`Analysis::digest`], the storage accessors -- is meaningless on it.
    /// It exists to be owned by a solver, not to be inspected.
    pub(crate) fn runtime_only(&self) -> Self {
        Self {
            config: self.config.clone(),
            kind: self.kind,
            n: self.n,
            nnz: self.nnz,
            perm: self.perm.clone(),
            nodes: Vec::new(),
            rows: Vec::new(),
            children: Vec::new(),
            maps: Vec::new(),
            entries: Vec::new(),
            levels: Vec::new(),
            elements: self.elements,
            factor_elements: self.factor_elements,
            panel_elements: self.panel_elements,
            upper_elements: self.upper_elements,
        }
    }
    #[cfg(any(test, sparta_metal, sparta_vulkan))]
    pub(crate) fn dispatch_plan(&self) -> DispatchPlan {
        self.dispatch_plan_for(None)
    }
    /// The dispatch tables for the nodes `keep` marks, or for every node when
    /// `keep` is `None`.
    ///
    /// Levels keep their positions, so a level whose nodes are all dropped is
    /// still present, with nothing to dispatch. That is what lets a hybrid
    /// factorization hand the GPU its share of the fronts under the same level
    /// numbers the full plan uses.
    #[cfg(any(test, sparta_metal, sparta_vulkan))]
    pub(crate) fn dispatch_plan_for(&self, keep: Option<&[bool]>) -> DispatchPlan {
        let mut ids = Vec::new();
        let mut levels = Vec::new();
        let mut tiles = Vec::new();
        let mut schur = Vec::new();
        let mut steps_table = Vec::new();
        for full in &self.levels {
            let level: Vec<u32> = full
                .iter()
                .copied()
                .filter(|&s| keep.is_none_or(|k| k[s as usize]))
                .collect();
            let small_start = ids.len() as u32;
            ids.extend(
                level
                    .iter()
                    .copied()
                    .filter(|&s| self.nodes[s as usize].nr <= self.config.small_front_rows),
            );
            let small_count = ids.len() as u32 - small_start;
            // Narrow fronts factor completely in one dispatch, so they carry no
            // pivot tiles and their Schur tiles are drained once.
            let narrow_start = ids.len() as u32;
            let narrow_schur_start = schur.len() as u32;
            for &s in level.iter().filter(|&&s| {
                let node = self.nodes[s as usize];
                node.nr > self.config.small_front_rows && node.nc <= self.config.narrow_cols
            }) {
                ids.push(s);
                self.push_schur_tiles(&mut schur, s);
            }
            let narrow_count = ids.len() as u32 - narrow_start;
            let narrow_schur_count = schur.len() as u32 - narrow_schur_start;
            let large_start = ids.len() as u32;
            let tile_start = tiles.len() as u32;
            let schur_start = schur.len() as u32;
            let step_start = steps_table.len() as u32;
            // Decreasing column count, so a step's live nodes are a prefix and
            // finished fronts stop being dispatched instead of returning early.
            let mut large: Vec<u32> = level
                .iter()
                .copied()
                .filter(|&s| {
                    let node = self.nodes[s as usize];
                    node.nr > self.config.small_front_rows && node.nc > self.config.narrow_cols
                })
                .collect();
            large.sort_by_key(|&s| std::cmp::Reverse(self.nodes[s as usize].nc));
            let mut steps = 0;
            // Schur tiles per node, in the same order, so prefixes of both agree.
            let mut per_node = Vec::with_capacity(large.len());
            // (step after which this tile is dead, tile). See the key below.
            let mut level_tiles: Vec<(u32, UpdateTile)> = Vec::new();
            for &s in &large {
                ids.push(s);
                let node = self.nodes[s as usize];
                // One step per panel, not per pivot. A panel always advances at
                // least PANEL columns unless it reaches nc, so this many suffice
                // even when a 2x2 pivot carries one past the boundary.
                steps = steps.max(node.nc.div_ceil(self.config.panel));
                let before_schur = schur.len();
                // Rows are counted in STRIP_ROWS blocks and columns in 16s, so
                // a tile is `STRIP_ROWS x 16` and row block `r` reaches column
                // block `STRIP_SPAN * (r + 1) - 1`.
                let columns = node.nc.div_ceil(16);
                let all = node.nr.div_ceil(self.config.strip_rows());
                // A tile stops writing at the earlier of three steps: the panel
                // passing its last column, the panel passing its last row, and
                // the front running out of fully summed columns altogether. The
                // first two are its own edges in 16-column units; the third is
                // shared by every tile of the front and is what keeps a
                // finished front's tiles out of later steps, which the row edge
                // alone is too coarse to guarantee once a row block spans
                // several column blocks.
                let dead = |row: u32, col: u32| {
                    col.min(self.config.strip_span() * (row + 1) - 1)
                        .min(columns - 1)
                };
                match self.kind {
                    // Fully summed columns: every row, columns below the
                    // diagonal. The upper half comes free from the mirrored
                    // write inside the kernel.
                    Kind::Ldlt => {
                        for row in 0..all {
                            for col in 0..columns.min(self.config.strip_span() * (row + 1)) {
                                level_tiles
                                    .push((dead(row, col), UpdateTile { node: s, row, col }));
                            }
                        }
                    }
                    // Nothing is mirrored, so the strip is the whole cross: the
                    // fully summed columns over every row, and the fully summed
                    // rows over the trailing columns. The second arm is what
                    // the next panel's triangular solve reads, so it is on the
                    // panel cadence like the first.
                    Kind::Lu => {
                        for row in 0..all {
                            for col in 0..columns {
                                level_tiles
                                    .push((dead(row, col), UpdateTile { node: s, row, col }));
                            }
                        }
                        let wide = node.nr.div_ceil(16);
                        for row in 0..node.nc.div_ceil(self.config.strip_rows()) {
                            for col in columns..wide {
                                level_tiles
                                    .push((dead(row, col), UpdateTile { node: s, row, col }));
                            }
                        }
                    }
                }
                // Schur complement, addressed relative to row/column nc.
                self.push_schur_tiles(&mut schur, s);
                per_node.push((
                    node.nc.div_ceil(self.config.panel),
                    (schur.len() - before_schur) as u32,
                ));
            }
            // Order the level's strip tiles by that key so each step can
            // dispatch a contiguous suffix: every tile with a key of `q` or
            // less writes nothing after step `q`, so the live set is exactly
            // the tiles with a larger key. No separate node filter is needed --
            // a finished front's tiles all carry its own last step as their
            // key and drop out together.
            level_tiles.sort_by_key(|&(k, _)| k);
            let columns = level_tiles.last().map_or(0, |&(k, _)| k as usize + 1);
            // starts[c] is the index of the first tile with key >= c.
            let mut starts = vec![0u32; columns + 2];
            for &(k, _) in &level_tiles {
                starts[k as usize + 1] += 1;
            }
            for c in 0..starts.len() - 1 {
                starts[c + 1] += starts[c];
            }
            tiles.extend(level_tiles.iter().map(|&(_, t)| t));
            let mut live = per_node.len();
            let mut live_schur: u32 = per_node.iter().map(|&(_, t)| t).sum();
            for step in 0..steps {
                while live > 0 && per_node[live - 1].0 <= step {
                    live -= 1;
                    live_schur -= per_node[live].1;
                }
                let first = starts[(step as usize + 1).min(columns + 1)];
                steps_table.push(StepCounts {
                    nodes: live as u32,
                    tile_start: first,
                    tiles: level_tiles.len() as u32 - first,
                    schur: live_schur,
                });
            }
            levels.push(DispatchLevel {
                small_start,
                small_count,
                large_start,
                large_count: ids.len() as u32 - large_start,
                steps,
                tile_start,
                tile_count: tiles.len() as u32 - tile_start,
                step_start,
                schur_start,
                schur_count: schur.len() as u32 - schur_start,
                narrow_start,
                narrow_count,
                narrow_schur_start,
                narrow_schur_count,
                max_rows: level
                    .iter()
                    .map(|&s| self.nodes[s as usize].nr)
                    .max()
                    .unwrap_or(0),
                // Only the cooperative fronts are assembled by the child
                // dispatch; the small serial path does its own extend-add.
                max_cols: level
                    .iter()
                    .map(|&s| &self.nodes[s as usize])
                    .filter(|node| {
                        is_wide(
                            node.nr,
                            node.nc,
                            self.config.small_front_rows,
                            self.config.narrow_cols,
                        )
                    })
                    .map(|node| node.nc)
                    .max()
                    .unwrap_or(0),
                max_children: level
                    .iter()
                    .map(|&s| &self.nodes[s as usize])
                    .filter(|node| node.nr > self.config.small_front_rows)
                    .map(|node| node.count)
                    .max()
                    .unwrap_or(0),
            });
        }
        DispatchPlan {
            ids,
            levels,
            tiles,
            schur,
            steps: steps_table,
        }
    }
}

/// Counts of dispatched versus output-producing GPU work for one plan.
///
/// Counts are static: they use the panel schedule's guaranteed advance of
/// `PANEL` columns per step. A 2x2 pivot straddling a panel boundary advances
/// one further, which retires a little more work than counted here, so the
/// "useful" figures are upper bounds and the waste figures are lower bounds.
#[cfg(any(test, sparta_metal, sparta_vulkan))]
pub struct DispatchAudit {
    /// Kernel dispatches encoded across the whole factorization.
    pub dispatches: usize,
    /// Command buffers submitted; the bridge closes one every 64 panel passes.
    pub command_buffers: usize,
    /// `factor_panel` threadgroups: one per live wide front per panel step.
    pub panel_groups: usize,
    /// `update_strip` threadgroups actually dispatched, taken from the step
    /// table, and how many can write an output, counted independently from
    /// front geometry. The column-ordered suffix schedule is supposed to make
    /// these equal; a gap means the schedule is launching dead groups again.
    pub strip_groups: usize,
    pub useful_strip_groups: usize,
    /// `update_schur` threadgroups, over every path that drains one: the
    /// once-per-level narrow drain, the batched drains inside the panel loop,
    /// and the mandatory final drain that closes each level. The final drain
    /// covers every tile of the level, so unlike the others not all of its
    /// groups write -- `writing_schur_groups` is the subset that does.
    pub schur_groups: usize,
    /// Schur threadgroups that actually have a pending range to apply. The gap
    /// to `schur_groups` is what the unconditional final drain costs.
    pub writing_schur_groups: usize,
    pub small_fronts: usize,
    pub worst_small_front: usize,
    /// Serial per-thread element traffic in the small-front kernel: it zeroes
    /// and rescans the whole nr x nr front regardless of how few columns it
    /// eliminates.
    pub small_fixed_ops: usize,
    /// Complex multiply-adds of actual trailing-update work.
    pub update_flops: usize,
    /// Update work inside the fully summed block, applied one panel at a time
    /// so that later pivot searches see current data.
    pub pivot_block_flops: usize,
    /// Update work on the Schur complement. Invariant under the drain cadence:
    /// batching changes how often the block is read and written, never how many
    /// products are evaluated.
    pub schur_flops: usize,
}

#[cfg(any(test, sparta_metal, sparta_vulkan))]
impl Analysis {
    /// Pivots accumulated before each blocked Schur update.
    pub fn panel_width(&self) -> usize {
        self.config.panel as usize
    }
    /// How front assembly's work is distributed, for tuning its dispatch.
    ///
    /// Returns the histogram of children per front, indexed by child count, and
    /// the complex-entry totals of the two passes assembly is made of: the
    /// front clear, which writes `nr * nr` per front, and the child extend-add,
    /// which reads a child Schur block and a parent entry and writes the parent
    /// back for each element. The second is the one whose ordering constrains
    /// how widely assembly can be dispatched, so its share decides whether
    /// splitting it further is worth the dispatches.
    pub fn assembly_profile(&self) -> (Vec<usize>, u64, u64) {
        let widest = self.nodes.iter().map(|n| n.count as usize).max().unwrap_or(0);
        let mut histogram = vec![0usize; widest + 1];
        let mut cleared = 0u64;
        let mut extended = 0u64;
        for node in &self.nodes {
            histogram[node.count as usize] += 1;
            cleared += u64::from(node.nr) * u64::from(node.nr);
            let first = node.children as usize;
            for child in &self.children[first..first + node.count as usize] {
                let cn = &self.nodes[child.node as usize];
                let tail = u64::from(cn.nr - cn.nc);
                extended += tail * tail;
            }
        }
        (histogram, cleared, extended)
    }

    /// How the wide-front panel factorization's work splits by row band.
    ///
    /// `factor_panel` forms each panel column over every row of the front, but
    /// only rows inside the fully summed block feed the pivot search. Rows
    /// below it are a triangular solve against a block diagonal the search has
    /// already fixed, so they carry no serial dependency and could in principle
    /// be spread over as many threadgroups as there are rows.
    ///
    /// Returns (wide fronts, work inside the fully summed block, work below it,
    /// rows below it summed over fronts), where work is counted in complex
    /// multiply-accumulates: forming panel column `t` costs one per row per
    /// completed column, which over a front of `nr` rows and `nc` columns at
    /// panel width `nb` totals `rows * nc * (nb-1) / 2`.
    pub fn panel_profile(&self) -> (usize, u64, u64, u64) {
        let nb = u64::from(self.config.panel);
        let (mut fronts, mut inside, mut below, mut rows_below) = (0, 0u64, 0u64, 0u64);
        for node in &self.nodes {
            if !is_wide(
                node.nr,
                node.nc,
                self.config.small_front_rows,
                self.config.narrow_cols,
            ) {
                continue;
            }
            let (nr, nc) = (u64::from(node.nr), u64::from(node.nc));
            let per_row = nc * (nb - 1) / 2;
            fronts += 1;
            inside += nc * per_row;
            below += (nr - nc) * per_row;
            rows_below += nr - nc;
        }
        (fronts, inside, below, rows_below)
    }

    pub fn dispatch_audit(&self) -> DispatchAudit {
        self.dispatch_audit_at(self.config.schur_batch)
    }
    /// The same walk at an explicit cadence, so a test can compare schedules
    /// without mutating the environment out from under a parallel test thread.
    pub(crate) fn dispatch_audit_at(&self, batch: u32) -> DispatchAudit {
        let d = self.dispatch_plan();
        let mut audit = DispatchAudit {
            dispatches: 0,
            command_buffers: 1,
            panel_groups: 0,
            strip_groups: 0,
            useful_strip_groups: 0,
            schur_groups: 0,
            writing_schur_groups: 0,
            small_fronts: self
                .nodes
                .iter()
                .filter(|n| n.nr <= self.config.small_front_rows)
                .count(),
            worst_small_front: self
                .nodes
                .iter()
                .filter(|n| n.nr <= self.config.small_front_rows)
                .map(|n| n.nr as usize)
                .max()
                .unwrap_or(0),
            small_fixed_ops: self
                .nodes
                .iter()
                .filter(|n| n.nr <= self.config.small_front_rows)
                .map(|n| 2 * n.nr as usize * n.nr as usize)
                .sum(),
            // LDL^T updates one triangle and mirrors it; LU updates the whole
            // square, so every region below is close to twice the size for the
            // same factorization.
            update_flops: self
                .nodes
                .iter()
                .map(|n| {
                    let (nr, nc) = (n.nr as usize, n.nc as usize);
                    match self.kind {
                        Kind::Ldlt => nc * nr * nr / 2,
                        Kind::Lu => nc * nr * nr,
                    }
                })
                .sum(),
            pivot_block_flops: self
                .nodes
                .iter()
                .map(|d| {
                    let (nr, nc) = (d.nr as usize, d.nc as usize);
                    match self.kind {
                        // Per pivot k: columns (k, nc), rows [j, nr).
                        Kind::Ldlt => (0..nc)
                            .map(|k| ((k + 1)..nc).map(|j| nr - j).sum::<usize>())
                            .sum::<usize>(),
                        // Per pivot k the whole cross: the remaining fully
                        // summed columns over every row below k, and the
                        // remaining fully summed rows over the trailing columns.
                        Kind::Lu => (0..nc)
                            .map(|k| (nc - k - 1) * (nr - k - 1) + (nc - k - 1) * (nr - nc))
                            .sum::<usize>(),
                    }
                })
                .sum(),
            schur_flops: self
                .nodes
                .iter()
                .map(|d| {
                    let (nr, nc) = (d.nr as usize, d.nc as usize);
                    let tail = nr - nc;
                    match self.kind {
                        Kind::Ldlt => nc * tail * (tail + 1) / 2,
                        Kind::Lu => nc * tail * tail,
                    }
                })
                .sum(),
        };
        // Strip tiles, counted per node. The dispatched set at a step is a
        // prefix over the level's nodes, so summing each node over its own
        // steps gives the same total as walking the step table.
        for node in self.nodes.iter().filter(|n| {
            is_wide(
                n.nr,
                n.nc,
                self.config.small_front_rows,
                self.config.narrow_cols,
            )
        }) {
            let (nr, nc) = (node.nr as usize, node.nc as usize);
            let span = self.config.strip_span() as usize;
            let (rows, columns) = (
                nr.div_ceil(self.config.strip_rows() as usize),
                nc.div_ceil(16),
            );
            let steps = nc.div_ceil(self.config.panel as usize);
            // Row block `r` is behind the panel once `span * (r + 1) <= first`.
            let live_rows = |first: usize, of: usize| of.saturating_sub(first / span);
            for q in 0..steps {
                // After panel q the front is current through column
                // PANEL*(q+1); a tile is live only if it owns a column past it.
                let done = (q + 1) * self.config.panel as usize;
                if done >= nc {
                    continue; // Front finished: every tile is dead.
                }
                let first = done / 16;
                audit.useful_strip_groups += match self.kind {
                    Kind::Ldlt => (0..rows)
                        .map(|r| columns.min(span * (r + 1)).saturating_sub(first))
                        .sum::<usize>(),
                    // A tile is live when the earlier of its two edges is past
                    // the panel. The cross is `rows x columns` down the fully
                    // summed columns plus `nc` rows' worth across the trailing
                    // ones, where only the row edge can bite.
                    Kind::Lu => {
                        let cf = columns.saturating_sub(first);
                        let arm1: usize = (0..rows)
                            .map(|r| if span * (r + 1) <= first { 0 } else { cf })
                            .sum();
                        let fs = nc.div_ceil(self.config.strip_rows() as usize);
                        arm1 + live_rows(first, fs) * (nr.div_ceil(16) - columns)
                    }
                };
            }
        }
        // Dispatch and command-buffer accounting, mirroring `sparta_v1_run`.
        // The Schur drain follows the cadence, so this walk has to follow it too
        // or the counts describe a schedule that is not the one running.
        let batch = batch as usize;
        let mut passes = 0usize;
        for level in &d.levels {
            audit.dispatches += usize::from(level.small_count > 0)
                + usize::from(level.narrow_count + level.large_count > 0)
                + usize::from(level.narrow_count > 0) // factor_narrow
                + usize::from(level.narrow_schur_count > 0)
                + usize::from(level.narrow_count > 0) // advance_panel
                + usize::from(level.small_count + level.narrow_count + level.large_count > 0);
            // The narrow path takes every pivot in one dispatch and drains once,
            // on no cadence at all. Omitting it here used to understate the
            // Schur total by the whole narrow contribution.
            audit.schur_groups += level.narrow_schur_count as usize;
            audit.writing_schur_groups += level.narrow_schur_count as usize;
            for step in 0..level.steps {
                let live = d.steps[(level.step_start + step) as usize];
                if live.nodes == 0 {
                    break;
                }
                audit.panel_groups += live.nodes as usize;
                audit.strip_groups += live.tiles as usize;
                audit.dispatches += 2 + usize::from(live.tiles > 0);
                if batch != 0 && (step as usize + 1) % batch == 0 {
                    audit.schur_groups += live.schur as usize;
                    audit.writing_schur_groups += live.schur as usize;
                    audit.dispatches += usize::from(live.schur > 0) + 1;
                }
                passes += 1;
                if passes == 64 {
                    passes = 0;
                    audit.command_buffers += 1;
                }
            }
            // The final drain covers every Schur tile of the level whether or
            // not it owes anything, because a front that took its last panel
            // mid-batch is in no later step's live prefix.
            audit.schur_groups += level.schur_count as usize;
            audit.dispatches +=
                usize::from(level.schur_count > 0) + usize::from(level.large_count > 0);
        }
        // Of those final-drain tiles, the ones with a pending range. A front
        // still owes a drain exactly when its panel count is not a whole number
        // of batches; at batch 0 every front owes its only drain.
        for node in self.nodes.iter().filter(|n| {
            is_wide(
                n.nr,
                n.nc,
                self.config.small_front_rows,
                self.config.narrow_cols,
            )
        }) {
            let steps = (node.nc as usize).div_ceil(self.config.panel as usize);
            if batch != 0 && steps % batch == 0 {
                continue;
            }
            let q = ((node.nr - node.nc) as usize).div_ceil(self.config.schur_tile() as usize);
            audit.writing_schur_groups += match self.kind {
                Kind::Ldlt => q * (q + 1) / 2,
                Kind::Lu => q * q,
            };
        }
        audit
    }
}

#[cfg(test)]
/// A bare plan with the given front shapes and level partition. Only the
/// fields `dispatch_plan` reads are populated; there is no matrix behind it.
fn fixture(kind: Kind, n: usize, shapes: &[(u32, u32)], levels: Vec<Vec<u32>>) -> Analysis {
    let mut nodes = Vec::new();
    let mut offset = 0;
    for &(nr, nc) in shapes {
        nodes.push(Node {
            offset,
            nr,
            nc,
            rows: 0,
            children: 0,
            count: 0,
            entries: 0,
            nentries: 0,
            first: 0,
            loffset: 0,
            woffset: 0,
            uoffset: 0,
            soffset: 0,
            sld: nr,
        });
        offset += nr * nr;
    }
    Analysis {
        config: Config::default(),
        kind,
        n,
        nnz: 0,
        nodes,
        rows: vec![],
        children: vec![],
        maps: vec![],
        entries: vec![],
        levels,
        perm: vec![],
        elements: offset as usize,
        factor_elements: 0,
        panel_elements: 0,
        upper_elements: 0,
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;

    /// A plan with wide fronts of deliberately awkward column counts: one an
    /// exact multiple of PANEL, one a multiple of 4 panels, and two that leave
    /// a short final panel. Those are the shapes where a batched drain can drop
    /// a contribution, because a front that takes its last panel mid-batch is
    /// in no later step's live prefix.
    fn cadence_plan(kind: Kind) -> Analysis {
        fixture(
            kind,
            297,
            &[(512, 64), (400, 100), (380, 96), (300, 37)],
            vec![vec![0, 1], vec![2, 3]],
        )
    }

    /// The Schur drain runs on its own cadence, so `FrontState.panel` is no
    /// longer simply "the previous panel". Whatever the cadence, each panel's
    /// rank-k block must reach the Schur complement exactly once, and nothing
    /// may still be pending when the level ends -- the parent assembles this
    /// front's Schur block in the next level, and the workspace is recycled
    /// from there on.
    ///
    /// This walks the same schedule `sparta_v1_run` encodes, tracking the
    /// `panel`/`next` pair the GPU keeps in `FrontState`.
    #[test]
    fn every_schur_contribution_drains_exactly_once_at_any_cadence() {
        // The cadence is a property of the schedule, not of the arithmetic, so
        // it has to hold for both factorizations. LU only changes how many
        // tiles each drain covers.
        for kind in [Kind::Ldlt, Kind::Lu] {
            let p = cadence_plan(kind);
            let d = p.dispatch_plan();
            for batch in [1usize, 2, 3, 4, 5, 8, 16, 0] {
                // Columns of each front already applied to its Schur complement.
                let mut panel = vec![0u32; p.nodes.len()];
                // How many times each column's contribution was drained.
                let mut applied: Vec<Vec<u32>> =
                    p.nodes.iter().map(|n| vec![0u32; n.nc as usize]).collect();
                for level in &d.levels {
                    let large = &d.ids[level.large_start as usize
                        ..(level.large_start + level.large_count) as usize];
                    // next[s] is the pivot count factor_panel has published.
                    let mut next = vec![0u32; p.nodes.len()];
                    for step in 0..level.steps as usize {
                        let live = d.steps[level.step_start as usize + step];
                        if live.nodes == 0 {
                            break;
                        }
                        for &s in &large[..live.nodes as usize] {
                            // A panel advances PANEL columns unless it hits nc.
                            next[s as usize] = p.nodes[s as usize]
                                .nc
                                .min((step as u32 + 1) * Config::default().panel);
                        }
                        if batch != 0 && (step + 1) % batch == 0 {
                            for &s in &large[..live.nodes as usize] {
                                for c in panel[s as usize]..next[s as usize] {
                                    applied[s as usize][c as usize] += 1;
                                }
                                panel[s as usize] = next[s as usize];
                            }
                        }
                    }
                    // The mandatory final drain, over every front of the level.
                    for &s in large {
                        for c in panel[s as usize]..next[s as usize] {
                            applied[s as usize][c as usize] += 1;
                        }
                        panel[s as usize] = next[s as usize];
                    }
                }
                for (s, node) in p.nodes.iter().enumerate() {
                    if !is_wide(
                        node.nr,
                        node.nc,
                        p.config.small_front_rows,
                        p.config.narrow_cols,
                    ) {
                        continue;
                    }
                    assert_eq!(
                        panel[s], node.nc,
                        "batch {batch}: node {s} ended the level with a pending Schur drain"
                    );
                    assert!(
                        applied[s].iter().all(|&c| c == 1),
                        "batch {batch}: node {s} drained columns {:?} times, not once each",
                        applied[s]
                    );
                }
            }
        }
    }

    /// The audit is the second, independent route to the same counts, so it has
    /// to follow the cadence too. Fewer drains can only mean fewer groups, the
    /// final drain can only add non-writing ones, and the arithmetic -- the
    /// thing batching does not reduce -- must not move at all.
    #[test]
    fn cadence_reduces_schur_groups_without_changing_the_work() {
        for kind in [Kind::Ldlt, Kind::Lu] {
            let p = cadence_plan(kind);
            let baseline = p.dispatch_audit_at(1);
            let mut previous = baseline.schur_groups;
            for batch in [2u32, 4, 8, 0] {
                let audit = p.dispatch_audit_at(batch);
                // Non-increasing rather than strictly decreasing: once the batch
                // exceeds the widest front's panel count nothing drains mid-loop
                // any more, and every further widening is the same schedule.
                assert!(
                    audit.schur_groups <= previous,
                    "batch {batch}: {} groups, more than {previous}",
                    audit.schur_groups
                );
                assert!(audit.writing_schur_groups <= audit.schur_groups);
                assert_eq!(audit.schur_flops, baseline.schur_flops);
                assert_eq!(audit.pivot_block_flops, baseline.pivot_block_flops);
                assert_eq!(audit.panel_groups, baseline.panel_groups);
                assert_eq!(audit.strip_groups, baseline.strip_groups);
                previous = audit.schur_groups;
            }
            assert!(
                previous < baseline.schur_groups,
                "{kind:?}: final-only drained {previous} groups, no fewer than the {} of cadence 1",
                baseline.schur_groups
            );
        }
    }

    #[test]
    fn dispatch_partitions_levels_and_covers_each_lower_entry_once() {
        let small = Config::default().small_front_rows;
        let wide = Config::default().narrow_cols + 1;
        // One serial front, one narrow cooperative front, then two wide fronts
        // whose pivot and Schur tiles must together cover the update region
        // exactly once -- the lower triangle for LDL^T, the whole square for LU.
        for kind in [Kind::Ldlt, Kind::Lu] {
            let p = fixture(
                kind,
                27,
                &[(small, 2), (small + 1, 3), (286, wide), (512, wide + 8)],
                vec![vec![0, 1, 2], vec![3]],
            );
            let d = p.dispatch_plan();
            // Each level lists serial fronts, then narrow, then wide.
            assert_eq!(d.ids, vec![0, 1, 2, 3]);
            // Every step's live counts are prefixes that shrink as fronts finish.
            for level in &d.levels {
                let table =
                    &d.steps[level.step_start as usize..(level.step_start + level.steps) as usize];
                assert!(table.windows(2).all(|w| w[0].nodes >= w[1].nodes));
                // Strip tiles are a shrinking suffix: each step drops the column
                // block the previous panel just factored, so the range start only
                // moves forward and the count only falls.
                assert!(table.windows(2).all(|w| w[0].tile_start <= w[1].tile_start));
                assert!(table.windows(2).all(|w| w[0].tiles >= w[1].tiles));
                for step in table {
                    assert_eq!(step.tile_start + step.tiles, level.tile_count);
                }
                // Step 0 already excludes the first column block, which panel 0 has
                // just finished, so it dispatches strictly less than the whole level.
                assert!(table[0].tiles < level.tile_count);
                assert_eq!(table[0].nodes, level.large_count);
            }
            assert_eq!(
                (
                    d.levels[0].small_count,
                    d.levels[0].narrow_count,
                    d.levels[0].large_count,
                    d.levels[0].steps
                ),
                (1, 1, 1, wide.div_ceil(Config::default().panel))
            );
            assert_eq!(
                (
                    d.levels[1].small_count,
                    d.levels[1].narrow_count,
                    d.levels[1].large_count,
                    d.levels[1].steps
                ),
                (0, 0, 1, (wide + 8).div_ceil(Config::default().panel))
            );
            for (l, level) in d.levels.iter().enumerate() {
                let jobs = &d.tiles
                    [level.tile_start as usize..(level.tile_start + level.tile_count) as usize];
                let schur = &d.schur
                    [level.schur_start as usize..(level.schur_start + level.schur_count) as usize];
                assert!(jobs
                    .iter()
                    .chain(schur)
                    .all(|t| p.levels[l].contains(&t.node)));
                for &s in &d.ids
                    [level.large_start as usize..(level.large_start + level.large_count) as usize]
                {
                    let node = p.nodes[s as usize];
                    let (n, nc) = (node.nr as usize, node.nc as usize);
                    let mut seen = vec![0u8; n * n];
                    // The strip: for LDL^T every row over the columns below nc and
                    // below the diagonal, the upper half coming from the kernel's
                    // mirrored write. For LU the whole cross, because nothing is
                    // mirrored -- the same tiles plus the fully summed rows over
                    // the trailing columns.
                    let in_strip = |i: usize, j: usize| match kind {
                        Kind::Ldlt => j < nc && j <= i,
                        Kind::Lu => i < nc || j < nc,
                    };
                    let tall = Config::default().strip_rows() as usize;
                    for tile in jobs.iter().filter(|t| t.node == s) {
                        for i in tile.row as usize * tall..(tile.row as usize + 1) * tall {
                            for j in tile.col as usize * 16..(tile.col as usize + 1) * 16 {
                                if i < n && j < n && in_strip(i, j) {
                                    seen[i * n + j] += 1;
                                }
                            }
                        }
                    }
                    // Schur tiles are addressed relative to the Schur origin.
                    let edge = Config::default().schur_tile() as usize;
                    for tile in schur.iter().filter(|t| t.node == s) {
                        for i in tile.row as usize * edge..(tile.row as usize + 1) * edge {
                            for j in tile.col as usize * edge..(tile.col as usize + 1) * edge {
                                let keep = match kind {
                                    Kind::Ldlt => j <= i,
                                    Kind::Lu => nc + j < n,
                                };
                                if nc + i < n && keep {
                                    seen[(nc + i) * n + nc + j] += 1;
                                }
                            }
                        }
                    }
                    // Together the two lists tile the update region exactly once.
                    for i in 0..n {
                        for j in 0..n {
                            let want = u8::from(match kind {
                                Kind::Ldlt => j <= i,
                                Kind::Lu => true,
                            });
                            assert_eq!(
                                seen[i * n + j],
                                want,
                                "{kind:?} node {s} entry ({i},{j}) covered {} times",
                                seen[i * n + j]
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn each_level_owns_one_contiguous_id_range() {
        // store_factors dispatches over a level's small and large nodes as a
        // single range, so the two partitions must stay adjacent in `ids`.
        let small = Config::default().small_front_rows;
        let p = fixture(
            Kind::Ldlt,
            22,
            &[
                (small + 32, 2),
                (small + 97, Config::default().narrow_cols + 5),
                (small, 4),
                (512, Config::default().narrow_cols + 1),
                (small - 12, 1),
            ],
            vec![vec![0, 1, 2], vec![3, 4]],
        );
        let d = p.dispatch_plan();
        let mut next = 0;
        for (l, level) in d.levels.iter().enumerate() {
            // store_factors and both solve sweeps dispatch over a level's three
            // classes as one range, and assembly over the last two, so the
            // partitions must stay adjacent and in this order.
            assert_eq!(level.small_start, next);
            assert_eq!(level.narrow_start, level.small_start + level.small_count);
            assert_eq!(level.large_start, level.narrow_start + level.narrow_count);
            let total = level.small_count + level.narrow_count + level.large_count;
            assert_eq!(total as usize, p.levels[l].len());
            let range = &d.ids[level.small_start as usize..(level.small_start + total) as usize];
            let mut seen: Vec<u32> = range.to_vec();
            seen.sort_unstable();
            let mut want = p.levels[l].clone();
            want.sort_unstable();
            assert_eq!(seen, want, "level {l} ids must cover exactly that level");
            next += total;
        }
        assert_eq!(next as usize, d.ids.len());
    }

    #[test]
    fn metal_descriptor_layouts_are_packed_u32_records() {
        assert_eq!(std::mem::size_of::<Node>(), 56);
        // Seventeen u32 fields, the last being max_children. metal.m's SpartaLevel
        // mirrors this field for field; changing one without the other is what
        // this guards.
        assert_eq!(std::mem::size_of::<DispatchLevel>(), 68);
        assert_eq!(std::mem::size_of::<StepCounts>(), 16);
        assert_eq!(std::mem::size_of::<FrontState>(), 20);
        assert_eq!(std::mem::size_of::<UpdateTile>(), 12);
        assert_eq!(std::mem::align_of::<DispatchLevel>(), 4);
    }
}

#[cfg(test)]
mod allocator_tests {
    use super::*;

    /// Lower-triangle CSC pattern of a 3D complex curl-curl operator.
    fn curl_curl(k: usize) -> (usize, Vec<usize>, Vec<usize>) {
        let (n, ptr, rows, _) = sparta_matgen::curl_curl(&[k, k, k], 1.0, 0.1);
        (n, ptr, rows)
    }

    fn parents(p: &Analysis) -> Vec<u32> {
        let mut parent = vec![u32::MAX; p.nodes.len()];
        for (s, node) in p.nodes.iter().enumerate() {
            let first = node.children as usize;
            for child in &p.children[first..first + node.count as usize] {
                parent[child.node as usize] = s as u32;
            }
        }
        parent
    }

    fn level_of(p: &Analysis) -> Vec<usize> {
        let mut level = vec![0usize; p.nodes.len()];
        for (l, nodes) in p.levels.iter().enumerate() {
            for &s in nodes {
                level[s as usize] = l;
            }
        }
        level
    }

    /// Every workspace slot live in phase `phase`, as half-open element ranges.
    ///
    /// A level runs in two phases and the allocator exploits both, so liveness
    /// is counted in half-levels: phase `2*l` is level `l`'s assembly and
    /// factorization, phase `2*l + 1` its store and compaction.
    ///
    /// A front is live for its own level, both phases. The block it compacts
    /// into is live from the phase that writes it -- alongside the front it is
    /// read from -- through the phase in which the parent assembles it.
    fn live_slots(p: &Analysis, parent: &[u32], level: &[usize], phase: usize) -> Vec<(usize, usize)> {
        let mut live = Vec::new();
        for s in 0..p.nodes.len() {
            let node = p.nodes[s];
            let (nr, nc) = (node.nr as usize, node.nc as usize);
            let own = level[s];
            let last = match parent[s] {
                u32::MAX => own,
                up => level[up as usize],
            };
            let o = node.offset as usize;
            if !worth_compacting(nr, nc) {
                // The block was left inside the front, so the front is what
                // has to survive until the parent assembles.
                if (2 * own..=2 * last).contains(&phase) {
                    live.push((o, o + nr * nr));
                }
                continue;
            }
            if phase / 2 == own {
                live.push((o, o + nr * nr));
            }
            if (2 * own + 1..=2 * last).contains(&phase) {
                let b = node.soffset as usize;
                live.push((b, b + (nr - nc) * (nr - nc)));
            }
        }
        live
    }

    #[test]
    fn concurrent_slots_never_share_workspace() {
        let (n, ptr, rows) = curl_curl(8);
        let p = Analysis::ldlt(n, &ptr, &rows, &Config::default()).unwrap();
        let (parent, level) = (parents(&p), level_of(&p));
        assert!(p.nodes.len() > 100, "expected a nontrivial tree");
        for phase in 0..2 * p.levels.len() {
            let mut live = live_slots(&p, &parent, &level, phase);
            live.sort_unstable();
            for pair in live.windows(2) {
                assert!(
                    pair[0].1 <= pair[1].0,
                    "phase {phase}: slots {:?} and {:?} overlap",
                    pair[0],
                    pair[1]
                );
            }
            if let Some(last) = live.last() {
                assert!(last.1 <= p.elements, "slot runs past the workspace peak");
            }
        }
    }

    /// What a GPU solver keeps after its backend has copied the descriptors.
    /// Everything dropped here is in device storage by then, so retaining it
    /// would be the same arrays in RAM twice.
    #[test]
    fn runtime_only_keeps_the_permutation_and_drops_the_tree() {
        let (n, ptr, rows) = curl_curl(6);
        let p = Analysis::ldlt(n, &ptr, &rows, &Config::default()).unwrap();
        let kept = p.runtime_only();
        assert_eq!(kept.perm, p.perm);
        assert_eq!((kept.n, kept.nnz), (p.n, p.nnz));
        assert_eq!(kept.elements, p.elements);
        assert_eq!(kept.factor_elements, p.factor_elements);
        assert!(!p.entries.is_empty(), "expected a nontrivial assembly map");
        for len in [
            kept.nodes.len(),
            kept.rows.len(),
            kept.children.len(),
            kept.maps.len(),
            kept.entries.len(),
            kept.levels.len(),
        ] {
            assert_eq!(len, 0);
        }
    }

    /// A root is fully summed, so it contributes nothing and needs no slot.
    /// The allocator relies on this: it never gives a root a contribution slot,
    /// and `store_factors` would otherwise compact into an unassigned offset.
    #[test]
    fn roots_are_fully_summed() {
        let (n, ptr, rows) = curl_curl(6);
        let p = Analysis::ldlt(n, &ptr, &rows, &Config::default()).unwrap();
        let parent = parents(&p);
        let mut roots = 0;
        for (s, node) in p.nodes.iter().enumerate() {
            if parent[s] == u32::MAX {
                roots += 1;
                assert_eq!(node.nr, node.nc, "root {s} has a contribution block");
            }
        }
        assert!(roots > 0, "expected at least one root");
    }

    /// The parent assembles each child's contribution slot while its own front
    /// is live, so those two must not overlap. The child's *front* may well be
    /// reused by then, which is the whole point of compacting.
    #[test]
    fn every_parent_is_disjoint_from_its_children() {
        let (n, ptr, rows) = curl_curl(6);
        let p = Analysis::ldlt(n, &ptr, &rows, &Config::default()).unwrap();
        for node in &p.nodes {
            let first = node.children as usize;
            let ps = node.offset as usize;
            let pe = ps + node.nr as usize * node.nr as usize;
            for child in &p.children[first..first + node.count as usize] {
                let c = p.nodes[child.node as usize];
                let tail = (c.nr - c.nc) as usize;
                if tail == 0 {
                    continue;
                }
                let cs = c.soffset as usize;
                // Spans the strided extent, which is the block itself once
                // compacted and part of the child's front when not.
                let ce = cs + (tail - 1) * c.sld as usize + tail;
                assert!(
                    pe <= cs || ce <= ps,
                    "parent {ps}..{pe} overlaps child block {cs}..{ce}"
                );
            }
        }
    }

    #[test]
    fn factor_store_is_packed_and_disjoint() {
        let (n, ptr, rows) = curl_curl(6);
        let p = Analysis::ldlt(n, &ptr, &rows, &Config::default()).unwrap();
        let mut next = 0usize;
        for node in &p.nodes {
            assert_eq!(node.loffset as usize, next);
            next += node.nc as usize * node.nr as usize;
        }
        assert_eq!(next, p.factor_elements);
    }

    #[test]
    fn reuse_beats_holding_every_front_resident() {
        let (n, ptr, rows) = curl_curl(8);
        let p = Analysis::ldlt(n, &ptr, &rows, &Config::default()).unwrap();
        let resident: usize = p.nodes.iter().map(|d| d.nr as usize * d.nr as usize).sum();
        // Check actual reuse, not an ordering-specific ratio. The ordering may
        // change the tree shape; disjointness/lifetimes are tested separately above.
        assert!(
            p.elements < resident,
            "workspace {} must be smaller than the {resident} all-resident total",
            p.elements
        );
    }
}

impl Analysis {
    /// Order-sensitive digest of everything the numeric phase consumes.
    ///
    /// Two plans with the same digest drive identical GPU work, which makes this
    /// the regression guard for changes to the symbolic phase: the ordering, the
    /// supernode partition, and every derived map must come out byte-identical.
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x100000001b3);
        };
        eat(self.n as u64);
        eat(self.nnz as u64);
        eat(self.elements as u64);
        eat(self.factor_elements as u64);
        eat(self.panel_elements as u64);
        for v in &self.perm {
            eat(*v as u64);
        }
        for d in &self.nodes {
            for v in [
                d.offset, d.nr, d.nc, d.rows, d.children, d.count, d.entries, d.nentries, d.first,
                d.loffset, d.woffset, d.soffset, d.sld,
            ] {
                eat(v as u64);
            }
        }
        for v in &self.rows {
            eat(*v as u64);
        }
        for c in &self.children {
            eat(c.node as u64);
            eat(c.map as u64);
        }
        for v in &self.maps {
            eat(*v as u64);
        }
        for e in &self.entries {
            eat(e.source as u64);
            eat(e.position as u64);
        }
        for level in &self.levels {
            eat(level.len() as u64);
            for v in level {
                eat(*v as u64);
            }
        }
        h
    }
}

#[cfg(test)]
mod lu_assembly_tests {
    use super::*;

    #[test]
    fn full_csc_scatter_preserves_orientation_and_owns_each_entry_once() {
        // Numerically asymmetric values will occupy this structurally symmetric
        // ring plus chord. Verify coordinates, rather than multiplying by a
        // mirrored test matrix that could hide an upper/lower transpose bug.
        let n = 7;
        let mut ptr = vec![0];
        let mut rows = vec![];
        let mut original = vec![];
        for j in 0..n {
            for i in 0..n {
                if i == j
                    || (i + 1) % n == j
                    || (j + 1) % n == i
                    || (i == 0 && j == 3)
                    || (i == 3 && j == 0)
                {
                    rows.push(i);
                    original.push((i, j));
                }
            }
            ptr.push(rows.len());
        }
        let p = Analysis::lu(n, &ptr, &rows, &Config::default()).unwrap();
        assert_eq!(p.kind(), Kind::Lu);
        let mut seen = vec![0; rows.len()];
        let mut upper_elements = 0usize;
        for node in &p.nodes {
            assert_eq!(node.uoffset as usize, upper_elements);
            upper_elements += node.nc as usize * (node.nr - node.nc) as usize;
            for e in &p.entries[node.entries as usize..(node.entries + node.nentries) as usize] {
                let i = e.position / node.nr;
                let j = e.position % node.nr;
                assert!(i < node.nr && j < node.nr);
                assert!(i < node.nc || j < node.nc);
                let old_i = p.perm[p.rows[(node.rows + i) as usize] as usize] as usize;
                let old_j = p.perm[p.rows[(node.rows + j) as usize] as usize] as usize;
                assert_eq!((old_i, old_j), original[e.source as usize]);
                seen[e.source as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&count| count == 1));
        assert_eq!(p.upper_elements, upper_elements);
    }

    #[test]
    fn lu_rejects_malformed_full_csc() {
        assert!(Analysis::lu(2, &[0, 2, 2], &[1, 0], &Config::default()).is_err());
        assert!(Analysis::lu(2, &[0, 2, 2], &[0, 0], &Config::default()).is_err());
        assert!(Analysis::lu(2, &[0, 1, 1], &[2], &Config::default()).is_err());
        // Upper-only entries are legal and must survive analysis.
        assert!(Analysis::lu(2, &[0, 0, 1], &[0], &Config::default()).is_ok());
    }
}

#[cfg(test)]
mod strip_tile_tests {
    use super::*;

    /// Every element `update_strip` writes, for one front at one panel step.
    ///
    /// Mirrors the kernel exactly: tile `(row, col)` owns rows
    /// `[row*strip_rows, +strip_rows)` and columns `[col*16, +16)`, writes the
    /// element `(i, j)` only when `j` is a fully summed column at or past the
    /// panel end and `j <= i`, and mirrors each write across the diagonal.
    fn written(
        tile: &UpdateTile,
        node: &Node,
        end: u32,
        config: &Config,
    ) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        if end >= node.nc || (tile.col + 1) * 16 <= end {
            return out;
        }
        for r in 0..config.strip_rows() {
            for c in 0..16 {
                let (i, j) = (tile.row * config.strip_rows() + r, tile.col * 16 + c);
                if j >= node.nc || j < end || i >= node.nr || j > i {
                    continue;
                }
                out.push((i, j));
                if i != j {
                    out.push((j, i));
                }
            }
        }
        out
    }

    #[test]
    /// The strip tiles of one front, at one step, write disjoint elements and
    /// cover every element the rank-k update owes.
    ///
    /// A duplicated element would be applied twice by racing threadgroups; a
    /// missing one would leave a stale value. Either shows up as a wrong factor
    /// with no error reported, so it is checked here rather than trusted.
    fn strip_tiles_partition_what_they_write() {
        let config = Config::default();
        // A 3D curl-curl grid, because only its fronts are wide enough to take
        // the panel path the strip tiles serve.
        let (n, ptr, rows, _) = sparta_matgen::curl_curl(&[10, 10, 10], 1.0, 0.1);
        let plan = Analysis::ldlt(n, &ptr, &rows, &config).unwrap();
        let dispatch = plan.dispatch_plan();
        let mut checked = 0;
        for level in &dispatch.levels {
            let tiles = &dispatch.tiles
                [level.tile_start as usize..(level.tile_start + level.tile_count) as usize];
            for node_index in 0..plan.nodes.len() {
                let node = &plan.nodes[node_index];
                let mine: Vec<&UpdateTile> =
                    tiles.iter().filter(|t| t.node as usize == node_index).collect();
                if mine.is_empty() {
                    continue;
                }
                // Walk the panel boundaries this front actually passes through.
                let mut end = config.panel;
                while end < node.nc {
                    let mut seen = std::collections::HashMap::new();
                    for tile in &mine {
                        for cell in written(tile, node, end, &config) {
                            if let Some(other) = seen.insert(cell, (tile.row, tile.col)) {
                                panic!(
                                    "front {node_index} ({}x{}) at end {end}: element {cell:?} \
                                     written by tiles {other:?} and {:?}",
                                    node.nr,
                                    node.nc,
                                    (tile.row, tile.col)
                                );
                            }
                        }
                    }
                    // Every fully summed element past the panel must be covered.
                    for i in 0..node.nr {
                        for j in end..node.nc.min(i + 1) {
                            assert!(
                                seen.contains_key(&(i, j)),
                                "front {node_index} ({}x{}) at end {end}: element ({i},{j}) \
                                 is written by no tile",
                                node.nr,
                                node.nc
                            );
                        }
                    }
                    checked += 1;
                    end += config.panel;
                }
            }
        }
        assert!(checked > 0, "no front exercised the strip tiles");
    }
}
