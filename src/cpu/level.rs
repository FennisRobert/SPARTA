//! One elimination level, assembled and factored across threads.
//!
//! Fronts in the same level are independent, so the level is the unit of
//! parallelism. Accelerate still threads inside a single large product, and the
//! two compose: the top levels are a handful of wide fronts whose own BLAS
//! calls fill the machine, and the bottom levels are thousands of narrow fronts
//! that fill it here instead.
use super::blocked::Scratch;
use super::dense::{self, Threading};
use rayon::prelude::*;
use super::profile::{Phase, Report};
use super::{factor::Front, C};
use crate::{Analysis, Error, Kind, Result};

/// Raw handles to the numerical storage a level's fronts write concurrently.
///
/// Within a level the regions any two nodes touch are disjoint.
/// `assign_front_offsets` never hands two fronts in one level the same
/// workspace slot, and `loffset`, `uoffset` and `first` are assigned per node,
/// so the factor, pivot and permutation stores partition by node as well. A
/// child's slot stays reserved until one level past its parent's, so reading a
/// child's Schur complement while that parent's siblings assemble is safe too.
/// The compiler cannot see any of that, so the storage crosses threads as
/// pointers and [`Shared::node`] carries the obligation.
pub(super) struct Shared {
    fronts: *mut C,
    factors: *mut C,
    upper: *mut C,
    diagonal: *mut C,
    subdiag: *mut C,
    perms: *mut usize,
    two: *mut bool,
}

// SAFETY: `Shared` is only ever used through `node`, which addresses one node's
// regions. The level invariant documented above makes those regions disjoint
// between any two nodes running at the same time, so no two threads reach the
// same element and nothing here is shared mutable state.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    /// Borrow the numerical storage of one solver for the length of a factorization.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        fronts: &mut [C],
        factors: &mut [C],
        upper: &mut [C],
        diagonal: &mut [C],
        subdiag: &mut [C],
        perms: &mut [usize],
        two: &mut [bool],
    ) -> Self {
        Self {
            fronts: fronts.as_mut_ptr(),
            factors: factors.as_mut_ptr(),
            upper: upper.as_mut_ptr(),
            diagonal: diagonal.as_mut_ptr(),
            subdiag: subdiag.as_mut_ptr(),
            perms: perms.as_mut_ptr(),
            two: two.as_mut_ptr(),
        }
    }

    /// Assemble node `id`, eliminate its fully summed columns, and copy the
    /// finished columns into permanent storage. Returns the pivots perturbed.
    ///
    /// # Safety
    ///
    /// No other thread may run `node` for a node outside `id`'s own level, and
    /// `id` must appear once per level. Both hold for a level-ordered walk that
    /// joins between levels.
    pub(super) unsafe fn node(
        &self,
        plan: &Analysis,
        id: u32,
        values: &[C],
        parallel: bool,
        scratch: &mut Scratch,
        report: &mut Report,
    ) -> Result<u64> {
        let node = plan.nodes[id as usize];
        let (m, n, off, first) = (
            node.nr as usize,
            node.nc as usize,
            node.offset as usize,
            node.first as usize,
        );
        let clock = Report::clock();
        let a = std::slice::from_raw_parts_mut(self.fronts.add(off), m * m);
        let zero = C::new(0.0, 0.0);
        if parallel {
            a.par_chunks_mut(BLANK).for_each(|chunk| chunk.fill(zero));
        } else {
            a.fill(zero);
        }
        report.stop(Phase::Blank, clock);
        let clock = Report::clock();
        for entry in &plan.entries[node.entries as usize..(node.entries + node.nentries) as usize] {
            // `locate` already reflected every LDL^T entry into the lower
            // triangle, which is the only one assembled.
            a[entry.position as usize] = values[entry.source as usize];
        }
        for child in &plan.children[node.children as usize..(node.children + node.count) as usize] {
            let cn = plan.nodes[child.node as usize];
            let (cm, nc) = (cn.nr as usize, cn.nc as usize);
            let tail = cm - nc;
            let sld = cn.sld as usize;
            let map = &plan.maps[child.map as usize..child.map as usize + tail];
            // `soffset`/`sld` locate the block whether it was compacted out of
            // the child's front -- whose slot this level has since reused -- or
            // left strided inside it.
            let schur =
                std::slice::from_raw_parts(self.fronts.add(cn.soffset as usize), (tail - 1) * sld + tail);
            // Indexed by the receiving row rather than the contributing one, so
            // the scatter can run as an ordinary parallel walk over the front's
            // rows: each row has at most one source row in this child.
            let source = scratch.inverse(m);
            for (i, &pi) in map.iter().enumerate() {
                source[pi as usize] = i as u32;
            }
            // Child maps ascend, so the child's lower triangle lands in the
            // parent's. On the LDL^T path that is the whole contribution: the
            // rest of this row of the parent is above its diagonal, which
            // nothing reads.
            let lower = plan.kind == Kind::Ldlt;
            let add = |pi: usize, out: &mut [C]| {
                let i = source[pi];
                if i == u32::MAX {
                    return;
                }
                let i = i as usize;
                let width = if lower { i + 1 } else { map.len() };
                let row = &schur[i * sld..i * sld + width];
                for (&value, &pj) in row.iter().zip(map.iter()) {
                    out[pj as usize] += value;
                }
            };
            if parallel {
                a.par_chunks_mut(m)
                    .enumerate()
                    .for_each(|(pi, out)| add(pi, out));
            } else {
                for (pi, out) in a.chunks_mut(m).enumerate() {
                    add(pi, out);
                }
            }
        }
        report.stop(Phase::Assemble, clock);

        let mut front = Front {
            a,
            m,
            n,
            perm: std::slice::from_raw_parts_mut(self.perms.add(first), n),
            d: std::slice::from_raw_parts_mut(self.diagonal.add(first), n),
            sub: std::slice::from_raw_parts_mut(self.subdiag.add(first), n),
            two: std::slice::from_raw_parts_mut(self.two.add(first), n),
            config: &plan.config,
            kind: plan.kind,
            bumped: 0,
            parallel,
        };
        let result = front
            .factor(scratch, report)
            .map_err(|(pivot, reason)| Error::Pivot {
                node: id as usize,
                pivot: pivot as u32 + 1,
                reason,
            });
        let bumped = front.bumped;
        result?;

        let clock = Report::clock();
        let a = std::slice::from_raw_parts(self.fronts.add(off), m * m);
        let l = std::slice::from_raw_parts_mut(self.factors.add(node.loffset as usize), n * m);
        // Transposed into column-major L, one contiguous output column at a time.
        let transpose = |j: usize, column: &mut [C]| {
            for (i, slot) in column.iter_mut().enumerate() {
                *slot = a[i * m + j];
            }
        };
        if parallel {
            l.par_chunks_mut(m)
                .enumerate()
                .for_each(|(j, column)| transpose(j, column));
        } else {
            l.chunks_mut(m)
                .enumerate()
                .for_each(|(j, column)| transpose(j, column));
        }
        if plan.kind == Kind::Lu {
            let u =
                std::slice::from_raw_parts_mut(self.upper.add(node.uoffset as usize), n * (m - n));
            if m > n {
                for (j, row) in u.chunks_mut(m - n).enumerate() {
                    row.copy_from_slice(&a[j * m + n..(j + 1) * m]);
                }
            }
        }
        report.stop(Phase::Store, clock);
        Ok(bumped)
    }

    /// Compact one finished front's trailing block into its own slot, so the
    /// `nr*nr` front can be recycled from the next level instead of being held
    /// until the parent assembles it.
    ///
    /// A pass of its own, and it has to stay one. The slot a node writes here
    /// may be storage that another node *in the same level* is still reading as
    /// a child's block -- the allocator hands a level's outgoing blocks the
    /// space its incoming ones occupied, which is what makes the compaction pay
    /// for itself. So every assembly in the level must be done first. The Metal
    /// path gets this from the encoder: assembly and `store_factors` are
    /// separate dispatches.
    ///
    /// # Safety
    /// Every node of the level must have returned from `node`, and `id` must
    /// appear once. Writes only this node's own slot.
    pub(super) unsafe fn compact(&self, plan: &Analysis, id: u32) {
        let node = plan.nodes[id as usize];
        let (m, n) = (node.nr as usize, node.nc as usize);
        // A fully summed front has no block, and one the analysis judged not
        // worth compacting already has its block in place inside the front.
        if m == n || node.soffset as usize == node.offset as usize + n * m + n {
            return;
        }
        let tail = m - n;
        let front = self.fronts.add(node.offset as usize);
        let block = self.fronts.add(node.soffset as usize);
        // Raw copies rather than two slices: both name the one `fronts`
        // allocation, and the disjointness that makes this sound is the
        // allocator's invariant, not something the borrow checker can see.
        for i in 0..tail {
            std::ptr::copy_nonoverlapping(front.add((n + i) * m + n), block.add(i * tail), tail);
        }
    }
}

/// Elements per task when clearing a front slot: enough that the parallel
/// region costs less than the memset it replaces.
const BLANK: usize = 1 << 16;

/// One level's nodes, widest first, and whether to spread them across threads.
pub(super) struct Level {
    pub nodes: Vec<u32>,
    pub spread: bool,
}

/// Plan how each level runs, once per solver.
///
/// A level is spread across the pool only when it holds at least one front per
/// thread. Below that, each front in turn gets the whole machine through
/// Accelerate's own threading, which beats handing one front per thread and
/// leaving most of the pool idle. Widest first, so Rayon's stealing does not
/// end a level waiting on one late large front.
///
/// Without `BLASSetThreading` -- macOS 13 and 14 -- a spread level would have
/// every front threading against every other, so only levels whose fronts are
/// too narrow to reach Accelerate at all are spread there.
pub(super) fn schedule(plan: &Analysis, threads: usize) -> Vec<Level> {
    let controllable = dense::set_threading(Threading::Machine);
    plan.levels
        .iter()
        .map(|level| {
            let mut nodes = level.clone();
            nodes.sort_unstable_by_key(|&id| std::cmp::Reverse(plan.nodes[id as usize].nr));
            let widest = nodes.first().map_or(0, |&id| plan.nodes[id as usize].nr) as usize;
            let spread = nodes.len() >= threads
                && (controllable || widest <= plan.config.small_front_rows as usize);
            Level { nodes, spread }
        })
        .collect()
}
