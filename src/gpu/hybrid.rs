//! The hybrid factorization: CPU subtrees alongside the GPU's share.
//!
//! [`Hybrid::factor`] commits the GPU's fronts below the cut, factors the CPU
//! subtrees in complex64 while those run (the `cpu_c` copy of the CPU backend),
//! copies the CPU results into the device buffers, and then commits the GPU's
//! fronts from the cut up. Every
//! buffer is shared storage, so the hand-over is a host copy with no transfer.
//!
//! The CPU writes only what the GPU reads from a finished front: its factor
//! columns, for LU its U12 rows, for LDL^T its `D` blocks and 2x2 flags, its
//! pivot permutation, its status word, and, for a subtree root, the
//! contribution block its GPU parent assembles. Both backends lay every one of
//! these out identically, for LU and LDL^T alike. All of it is copied after the first GPU range has finished, so no
//! command in flight touches it: the factor, pivot and status stores are
//! partitioned by node, and a root's contribution block slot is reserved from
//! the root's level to its parent's, which is at or above the cut.
use super::backend::Context;
use crate::cpu_c as cpu;
use crate::hybrid::Split;
use crate::{Analysis, Complex32, Kind, Result};
use rayon::prelude::*;
use std::time::Instant;

/// Smallest change of CPU share worth re-splitting for.
const REBALANCE: f64 = 0.03;

/// Device buffer indices, as `src/metal.m` binds them.
const FRONTS: u32 = 7;
const DIAGONAL: u32 = 8;
const SUBDIAG: u32 = 9;
const PERMS: u32 = 10;
const TWO: u32 = 11;
const STATUS: u32 = 12;
const FACTORS: u32 = 17;
const UPPER: u32 = 20;

pub(super) struct Hybrid {
    cpu: cpu::Solver,
    /// Every CPU node, and the CPU subtree roots.
    nodes: Vec<u32>,
    roots: Vec<u32>,
    cut: u32,
    levels: u32,
    /// Seconds of the latest factorization: CPU subtrees, the wait for the
    /// first GPU range after them, and the hand-over copy.
    pub cpu_seconds: f64,
    pub wait_seconds: f64,
    pub copy_seconds: f64,
    /// GPU device seconds below the cut and from the cut up.
    pub gpu_below: f64,
    pub gpu_above: f64,
    /// Fraction of the work the CPU subtrees carry, and of the GPU work from
    /// the cut up.
    pub share: f64,
    above: f64,
    /// The latest factorization was the first under its split, so its CPU
    /// time includes first touches of new storage and is not a rate sample.
    fresh: bool,
    /// The latest factorization's timings are a rate sample to rebalance on.
    sampled: bool,
}

impl Hybrid {
    /// Split `plan` for its `hybrid_cpu` share and install the GPU's part in
    /// `context`. `None` when the plan asks for no hybrid factorization or no
    /// subtree selection fits.
    pub(super) fn new(plan: &Analysis, context: &mut Context) -> Result<Option<Self>> {
        if plan.config.hybrid_cpu <= 0.0 {
            return Ok(None);
        }
        let Some(split) = plan.hybrid_split(plan.config.hybrid_cpu) else {
            return Ok(None);
        };
        let mut cpu_plan = plan.clone();
        cpu_plan.config.force_cpu = true;
        let mut hybrid = Self {
            cpu: cpu::Solver::new_lazy(cpu_plan)?,
            nodes: Vec::new(),
            roots: Vec::new(),
            cut: 0,
            levels: plan.levels.len() as u32,
            cpu_seconds: 0.0,
            wait_seconds: 0.0,
            copy_seconds: 0.0,
            gpu_below: 0.0,
            gpu_above: 0.0,
            share: 0.0,
            above: 0.0,
            fresh: true,
            sampled: false,
        };
        hybrid.install(split, context)?;
        Ok(Some(hybrid))
    }

    /// Make `split` the division of every factorization from now on.
    fn install(&mut self, split: Split, context: &mut Context) -> Result<()> {
        let plan = self.cpu.storage().plan;
        let gpu: Vec<bool> = split.cpu.iter().map(|&c| !c).collect();
        context.set_factor_plan(&plan.dispatch_plan_for(Some(&gpu)))?;
        self.nodes = (0..split.cpu.len() as u32).filter(|&i| split.cpu[i as usize]).collect();
        self.roots = split.roots;
        self.cut = split.cut as u32;
        self.share = split.share;
        self.above = split.above;
        self.fresh = true;
        self.cpu.restrict(&split.cpu);
        Ok(())
    }

    /// Re-split for the share at which the CPU subtrees and the GPU fronts
    /// below the cut would take equally long, at the rates the latest
    /// factorization measured. A change smaller than [`REBALANCE`] is ignored,
    /// so a balanced split stays put.
    fn rebalance(&mut self, context: &mut Context) -> Result<()> {
        let below = 1.0 - self.share - self.above;
        if self.cpu_seconds <= 0.0 || self.gpu_below <= 0.0 || below <= 0.0 {
            return Ok(());
        }
        // CPU and GPU rates in work per second, and the share at which the CPU
        // subtrees and the GPU fronts below the cut would finish together.
        let (cpu, gpu) = (self.share / self.cpu_seconds, below / self.gpu_below);
        let balanced = cpu * (1.0 - self.above) / (cpu + gpu);
        if (balanced - self.share).abs() < REBALANCE {
            return Ok(());
        }
        let plan = self.cpu.storage().plan;
        if let Some(split) = plan.hybrid_split(balanced) {
            if (split.share - self.share).abs() >= 0.5 * REBALANCE {
                self.install(split, context)?;
            }
        }
        Ok(())
    }

    /// Factor `values` across both devices. Returns the GPU's device seconds
    /// and writes every node's status word and the total perturbed pivots.
    pub(super) fn factor(
        &mut self,
        context: &mut Context,
        values: &[Complex32],
        status: &mut [u32],
        perturbed: &mut u64,
    ) -> Result<f64> {
        // Rebalance before rather than after, so the timings left behind
        // describe the split this factorization actually ran.
        if self.sampled {
            self.rebalance(context)?;
        }
        context.factor_start(Some(values), 0, self.cut)?;
        let started = Instant::now();
        let cpu = self.cpu.factor(values);
        self.cpu_seconds = started.elapsed().as_secs_f64();
        let waited = Instant::now();
        let mut gpu_perturbed = 0;
        self.gpu_below = context.factor_finish(status, &mut gpu_perturbed)?;
        self.wait_seconds = waited.elapsed().as_secs_f64();
        cpu?;
        let copied = Instant::now();
        // SAFETY: the first GPU range has finished and the second has not
        // started, so no command touches these buffers; each is the element
        // type `src/metal.m` allocates it with.
        unsafe { self.hand_over(context)? };
        self.copy_seconds = copied.elapsed().as_secs_f64();
        context.factor_start(None, self.cut, self.levels)?;
        self.gpu_above = context.factor_finish(status, &mut gpu_perturbed)?;
        *perturbed = gpu_perturbed + self.cpu.perturbed;
        self.sampled = !std::mem::replace(&mut self.fresh, false);
        Ok(self.gpu_below + self.gpu_above)
    }

    /// Copy the CPU nodes' results into the device buffers.
    ///
    /// # Safety
    /// No GPU command may be in flight.
    unsafe fn hand_over(&self, context: &mut Context) -> Result<()> {
        self.hand_over_factors(context)?;
        self.hand_over_blocks(context)
    }

    /// The factor columns, U12 for LU, `D` and the pivot data of every CPU node.
    unsafe fn hand_over_factors(&self, context: &mut Context) -> Result<()> {
        let store = self.cpu.storage();
        let plan = store.plan;
        let lu = plan.kind == Kind::Lu;
        let factors = Shared(context.buffer::<Complex32>(FACTORS)?.as_mut_ptr());
        // Bound only on the LU path; LDL^T has no U12 store.
        let upper = if lu { Some(Shared(context.buffer::<Complex32>(UPPER)?.as_mut_ptr())) } else { None };
        let diagonal = context.buffer::<Complex32>(DIAGONAL)?;
        let subdiag = context.buffer::<Complex32>(SUBDIAG)?;
        let perms = context.buffer::<u32>(PERMS)?;
        let two = context.buffer::<u32>(TWO)?;
        let status = context.buffer::<u32>(STATUS)?;
        for &id in &self.nodes {
            let node = plan.nodes[id as usize];
            let (first, nc) = (node.first as usize, node.nc as usize);
            for k in first..first + nc {
                diagonal[k] = store.diagonal[k];
                subdiag[k] = store.subdiag[k];
                perms[k] = store.perms[k] as u32;
                two[k] = u32::from(store.two[k]);
            }
            status[id as usize] = 0;
        }
        // The factor columns are the bulk of the copy, and every node's block
        // is its own, so the nodes are spread across the CPU solver's pool.
        let pool = self.cpu.pool().clone();
        pool.run(|| self.nodes.par_iter().for_each(|&id| {
            let node = plan.nodes[id as usize];
            let (start, len) = (node.loffset as usize, node.nr as usize * node.nc as usize);
            // SAFETY: this node's block, which no other task writes.
            unsafe {
                std::ptr::copy_nonoverlapping(store.factors[start..].as_ptr(), factors.at(start), len);
            }
            if let Some(upper) = upper {
                let (start, len) = (node.uoffset as usize, (node.nc * (node.nr - node.nc)) as usize);
                // SAFETY: as above, this node's own U12 rows.
                unsafe {
                    std::ptr::copy_nonoverlapping(store.upper[start..].as_ptr(), upper.at(start), len);
                }
            }
        }));
        Ok(())
    }

    /// Each CPU root's contribution block, where its GPU parent reads it.
    unsafe fn hand_over_blocks(&self, context: &mut Context) -> Result<()> {
        let store = self.cpu.storage();
        let plan = store.plan;
        // Each root's contribution block, where its GPU parent reads it. For
        // LDL^T the CPU forms only the lower triangle; the GPU's own fronts
        // carry both, so the block is mirrored on the way in. For LU both
        // sides form the whole square, which is copied row by row.
        let fronts = context.buffer::<Complex32>(FRONTS)?;
        for &root in &self.roots {
            let node = plan.nodes[root as usize];
            let (tail, ld, base) = ((node.nr - node.nc) as usize, node.sld as usize, node.soffset as usize);
            if plan.kind == Kind::Lu {
                for i in 0..tail {
                    let row = base + i * ld;
                    fronts[row..row + tail].copy_from_slice(&store.fronts[row..row + tail]);
                }
                continue;
            }
            for i in 0..tail {
                for j in 0..=i {
                    let v = store.fronts[base + i * ld + j];
                    fronts[base + i * ld + j] = v;
                    fronts[base + j * ld + i] = v;
                }
            }
        }
        Ok(())
    }
}

/// A device buffer address that crosses the Rayon pool. Each task writes only
/// its own node's factor block, which no other node shares.
#[derive(Clone, Copy)]
struct Shared(*mut Complex32);
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}
impl Shared {
    fn at(self, offset: usize) -> *mut Complex32 {
        // SAFETY: offsets come from the plan's factor layout, inside the buffer.
        unsafe { self.0.add(offset) }
    }
}
