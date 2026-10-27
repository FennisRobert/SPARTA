//! CPU multifrontal numerics using the same symbolic layout as the GPU.
//!
//! Compiled four times, once per value type: `cpu_z` (complex128), `cpu_c`
//! (complex64), `cpu_d` (float64) and `cpu_s` (float32). Each copy reads its
//! types from the `cpu_types!` defined just before it in `lib.rs`.
mod blocked;
mod blocked_lu;
mod dense;
mod factor;
mod level;
mod panel;
mod profile;
mod solve;
mod update;

use crate::{Analysis, Error, Kind, Result};

// `C`, the value type of this copy, and `R`, the real type of its precision.
cpu_types!();
pub(crate) use crate::scalar::Scalar;
use blocked::Scratch;
use dense::Threading;
use crate::threads::Pool;
use level::{Level, Shared};
use profile::Report;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Reusable state for one front at a time; see `Solver::workers`.
#[derive(Default)]
struct Worker {
    scratch: Scratch,
    report: Report,
}

/// Read access to a CPU solver's plan and factor storage; see [`Solver::storage`].
// Used by the hybrid factorization through the complex64 copy only.
#[allow(dead_code)]
pub(crate) struct Storage<'a> {
    pub plan: &'a Analysis,
    pub fronts: &'a [C],
    pub factors: &'a [C],
    /// LU only: U12 per node, row-major at `uoffset`.
    pub upper: &'a [C],
    pub diagonal: &'a [C],
    pub subdiag: &'a [C],
    pub perms: &'a [usize],
    pub two: &'a [bool],
}

pub(crate) struct Solver {
    plan: Analysis,
    fronts: Vec<C>,
    factors: Vec<C>,
    upper: Vec<C>,
    diagonal: Vec<C>,
    subdiag: Vec<C>,
    perms: Vec<usize>,
    two: Vec<bool>,
    /// Reusable per-front state, taken by whichever thread starts a front and
    /// returned when it finishes. Not one per thread: a thread waiting inside
    /// a front that splits its work may start another front of the level.
    workers: Mutex<Vec<Worker>>,
    schedule: Vec<Level>,
    /// The threads this solver factors and solves on; see `Config::threads`.
    pool: Pool,
    /// Whether each node's `nc` L columns are stored row-major, `nr` rows of
    /// `nc` values, rather than column-major. Row-major is what the CPU solve
    /// reads and what a front's rows copy into without a transpose; the hybrid
    /// hands its factors to the GPU, which reads them column-major.
    rowmajor: bool,
    valid: bool,
    destroyed: bool,
    pub(super) seconds: f64,
    pub(super) perturbed: u64,
}

impl Solver {
    /// Allocate buffers of this copy's value type for the immutable symbolic `plan`.
    pub(crate) fn new(plan: Analysis) -> Result<Self> {
        if plan.n == 0 {
            return Err(Error::Invalid("released analysis".into()));
        }
        Self::build(plan, true)
    }

    /// A solver for factoring only part of the tree, whose factors the GPU
    /// reads: column-major, and the pages of nodes it never factors are never
    /// committed.
    // `new_lazy`, `restrict` and `storage` serve the hybrid factorization,
    // which uses the complex64 copy of this module only.
    #[allow(dead_code)]
    pub(crate) fn new_lazy(plan: Analysis) -> Result<Self> {
        Self::build(plan, false)
    }

    fn build(plan: Analysis, rowmajor: bool) -> Result<Self> {
        plan.config.check().map_err(Error::Invalid)?;
        let pool = Pool::new(plan.config.threads)?;
        let threads = pool.threads();
        Ok(Self {
            workers: Mutex::new((0..threads).map(|_| Worker::default()).collect()),
            schedule: level::schedule(&plan),
            pool,
            rowmajor,
            fronts: allocate_lazy(plan.elements)?,
            factors: allocate_lazy(plan.factor_elements)?,
            upper: allocate_lazy(plan.upper_elements)?,
            diagonal: allocate_lazy(plan.n)?,
            subdiag: allocate_lazy(plan.n)?,
            perms: vec![0; plan.n],
            two: vec![false; plan.n],
            plan,
            valid: false,
            destroyed: false,
            seconds: 0.0,
            perturbed: 0,
        })
    }

    /// Factor only the nodes `keep` marks from now on, replacing any earlier
    /// restriction. They must be closed under taking children, so that no
    /// kept front waits on one left out.
    #[allow(dead_code)]
    pub(crate) fn restrict(&mut self, keep: &[bool]) {
        self.schedule = level::schedule(&self.plan);
        for level in &mut self.schedule {
            level.nodes.retain(|&id| keep[id as usize]);
            level.spread &= level.nodes.len() >= 2;
        }
        self.schedule.retain(|level| !level.nodes.is_empty());
    }

    /// The threads this solver runs on, for work done on its behalf.
    // Used by the hybrid factorization through the complex64 copy only.
    #[allow(dead_code)]
    pub(crate) fn pool(&self) -> &Pool {
        &self.pool
    }

    /// The plan and the numerical storage of the latest factorization: fronts,
    /// factor columns, `D`'s diagonal and subdiagonal, pivot permutations and
    /// the 2x2 flags, laid out exactly as the GPU buffers are.
    #[allow(dead_code)]
    pub(crate) fn storage(&self) -> Storage<'_> {
        Storage {
            plan: &self.plan,
            fronts: &self.fronts,
            factors: &self.factors,
            upper: &self.upper,
            diagonal: &self.diagonal,
            subdiag: &self.subdiag,
            perms: &self.perms,
            two: &self.two,
        }
    }

    /// Factor finite `values` in the analyzed CSC entry order.
    pub(crate) fn factor(&mut self, values: &[C]) -> Result<()> {
        self.valid = false;
        self.seconds = 0.0;
        self.perturbed = 0;
        if self.destroyed || values.len() != self.plan.nnz || values.iter().any(|&v| !finite(v)) {
            return Err(Error::Invalid(
                "finite values must match a live analyzed CSC pattern".into(),
            ));
        }
        let started = Instant::now();
        let mut report = Report::default();
        let pool = self.pool.clone();
        let result = pool.run(|| self.factor_all(values, &mut report));
        self.seconds = started.elapsed().as_secs_f64();
        report.print(self.seconds);
        self.valid = result.is_ok();
        result
    }

    /// Assemble and eliminate fronts leaves first, copying factors before slot reuse.
    ///
    /// Each level runs across the Rayon pool; the walk joins between levels,
    /// which is what makes the disjointness `Shared` relies on hold.
    fn factor_all(&mut self, values: &[C], report: &mut Report) -> Result<()> {
        let Self {
            plan,
            fronts,
            factors,
            upper,
            diagonal,
            subdiag,
            perms,
            two,
            workers,
            schedule,
            perturbed,
            rowmajor,
            pool,
            ..
        } = self;
        // A front alone on the machine may have the library thread its
        // products -- unless this solver is bounded to fewer threads, which
        // the library's own threading would not respect.
        let machine = !pool.bounded();
        let shared = Shared::new(fronts, factors, upper, diagonal, subdiag, perms, two, *rowmajor);
        // Rayon owns the parallelism: fronts across threads in wide levels, and
        // row blocks within one front in narrow ones. Accelerate threads only
        // inside `zsytrf`, which sets and clears this itself. The setting is
        // thread-local, so it has to reach every worker, not just this one.
        dense::set_threading(Threading::Caller);
        rayon::broadcast(|_| {
            dense::set_threading(Threading::Caller);
        });
        let workers: &Mutex<Vec<Worker>> = workers;
        for worker in workers.lock().unwrap().iter_mut() {
            worker.report = Report::default();
        }
        let take = || workers.lock().unwrap().pop().unwrap_or_default();
        let give = |worker: Worker| workers.lock().unwrap().push(worker);
        let bumped = AtomicU64::new(0);
        // SPARTA_CPU_DUMP=1 prints each level's wall time, work and root width.
        let dump = std::env::var_os("SPARTA_CPU_DUMP").is_some();
        let report_level = |depth: usize, how: &str, t0: Instant| {
            let level = &schedule[depth];
            let gflop = FLOPS * level.nodes.iter().map(|&id| work(plan, id)).sum::<f64>() / 1e9;
            eprintln!("LEVEL {depth} {how} {} {:.4} {gflop:.2} {}", level.nodes.len(),
                t0.elapsed().as_secs_f64(), plan.nodes[level.nodes[0] as usize].nr);
        };
        let result = schedule.iter().enumerate().try_for_each(|(depth, level)| {
            // OpenBLAS's control is process-wide, so it is set here, where the
            // previous level has joined and no thread is inside BLAS.
            dense::set_global_threading(if level.spread { Threading::Caller } else { Threading::Machine });
            let t0 = Instant::now();
            let clock = Report::clock();
            if !level.spread {
                let mut worker = take();
                for &id in &level.nodes {
                    // SAFETY: the loop is the only thing touching the storage,
                    // and every node appears once.
                    let count = unsafe {
                        shared.node(plan, id, values, true, machine, &mut worker.scratch, &mut worker.report)
                    };
                    let count = match count {
                        Ok(count) => count,
                        Err(e) => {
                            give(worker);
                            return Err(e);
                        }
                    };
                    bumped.fetch_add(count, Ordering::Relaxed);
                }
                // SAFETY: every node of this level has returned above.
                for &id in &level.nodes {
                    unsafe { shared.compact(plan, id) };
                }
                worker.report.level(false, clock);
                give(worker);
                if dump {
                    report_level(depth, "seq", t0);
                }
                return Ok(());
            }
            // A front with more than a thread's share of the level's work
            // would otherwise be the level's long pole on one thread; it splits
            // its own work instead, which the threads done with the small
            // fronts steal.
            let share = level.nodes.iter().map(|&id| work(plan, id)).sum::<f64>()
                / rayon::current_num_threads() as f64;
            level.nodes.par_iter().try_for_each(|&id| {
                let mut worker = take();
                let Worker { scratch, report } = &mut worker;
                let split = work(plan, id) > share;
                // SAFETY: every node in `level` is distinct and the iterator
                // joins before the next level starts, which is exactly the
                // contract `Shared::node` documents.
                let count = unsafe { shared.node(plan, id, values, split, false, scratch, report) };
                give(worker);
                bumped.fetch_add(count?, Ordering::Relaxed);
                Ok(())
            })?;
            // SAFETY: par_iter joined above, so every assembly in this level
            // is done; each node writes only its own slot.
            level
                .nodes
                .par_iter()
                .for_each(|&id| unsafe { shared.compact(plan, id) });
            let mut pool = workers.lock().unwrap();
            if pool.is_empty() {
                pool.push(Worker::default());
            }
            pool[0].report.level(true, clock);
            drop(pool);
            if dump {
                report_level(depth, "par", t0);
            }
            Ok(())
        });
        // Leave the pool as it was found, for whatever else in the process
        // calls Accelerate.
        dense::set_global_threading(Threading::Machine);
        dense::set_threading(Threading::Machine);
        rayon::broadcast(|_| {
            dense::set_threading(Threading::Machine);
        });
        *perturbed = bumped.load(Ordering::Relaxed);
        for worker in workers.lock().unwrap().iter() {
            report.merge(&worker.report);
        }
        result
    }

    /// Drop the numerical storage and give back the plan, for a solver of
    /// another value type to take over.
    pub(crate) fn into_plan(self) -> Analysis {
        self.plan
    }

    /// Release factors and symbolic storage; later numerical calls return errors.
    pub(crate) fn destroy(&mut self) {
        self.fronts = Vec::new();
        self.factors = Vec::new();
        self.upper = Vec::new();
        self.diagonal = Vec::new();
        self.subdiag = Vec::new();
        self.perms = Vec::new();
        self.two = Vec::new();
        self.workers = Mutex::new(Vec::new());
        self.schedule = Vec::new();
        self.plan.release();
        self.valid = false;
        self.destroyed = true;
        self.seconds = 0.0;
        self.perturbed = 0;
    }

    /// Return the pivot magnitude range after a successful factorization.
    ///
    /// LDLᵀ scales each 2x2 block pivot by its determinant. LU reads U's
    /// diagonal, which each front keeps inside its factor columns: pivot `k`
    /// of a node is `factors[loffset + k * nr + k]`.
    // `f64::from` is the identity in the double precision copies.
    #[allow(clippy::useless_conversion)]
    pub(crate) fn pivot_range(&self) -> Option<(f64, f64)> {
        if !self.valid {
            return None;
        }
        if self.plan.kind == Kind::Lu {
            let (mut lo, mut hi) = (R::INFINITY, 0.0 as R);
            for node in &self.plan.nodes {
                let (m, base) = (node.nr as usize, node.loffset as usize);
                for k in 0..node.nc as usize {
                    let v = self.factors[base + k * m + k].norm();
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            return (lo.is_finite() && hi > 0.0).then_some((f64::from(lo), f64::from(hi)));
        }
        let (mut lo, mut hi, mut k) = (R::INFINITY, 0.0 as R, 0);
        while k < self.plan.n {
            let v = if self.two[k] {
                let (_, _, _, s, det) =
                    block(self.diagonal[k], self.subdiag[k], self.diagonal[k + 1]);
                k += 2;
                det.norm() * s
            } else {
                k += 1;
                self.diagonal[k - 1].norm()
            };
            lo = lo.min(v);
            hi = hi.max(v);
        }
        Some((f64::from(lo), f64::from(hi)))
    }
}

/// Real flops per multiply-add of this copy's value type.
pub(crate) const FLOPS: f64 = if <C as Scalar>::KIND.is_complex() { 8.0 } else { 2.0 };

/// Relative arithmetic of node `id`: its leading block, off-diagonal solve and
/// Schur product, in multiply-adds.
fn work(plan: &Analysis, id: u32) -> f64 {
    let node = plan.nodes[id as usize];
    let (t, n) = ((node.nr - node.nc) as f64, node.nc as f64);
    t * t * n + t * n * n / 2.0 + n * n * n / 3.0
}

/// `count` zeros from the allocator's zeroed path. Large zeroed allocations
/// come back as fresh pages, which the kernel commits only when first written.
///
/// Every solver allocates this way. Nothing reads a zero it did not write --
/// each front clears the part of its slot it uses, and every factor element
/// is stored before a solve reads it -- so writing the zeros up front was
/// pure cost: several GiB on one thread, inside the first factorization,
/// where the pages are now committed by the parallel fronts that touch them.
fn allocate_lazy(count: usize) -> Result<Vec<C>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let layout = std::alloc::Layout::array::<C>(count)
        .map_err(|e| Error::Unsupported(format!("CPU allocation: {e}")))?;
    // SAFETY: the layout is nonzero and valid for `count` elements, and an
    // all-zero bit pattern is the number zero in each value type.
    unsafe {
        let data = std::alloc::alloc_zeroed(layout) as *mut C;
        if data.is_null() {
            return Err(Error::Unsupported("CPU allocation failed".into()));
        }
        Ok(Vec::from_raw_parts(data, count, count))
    }
}

/// Test every floating-point component of `v`.
fn finite(v: C) -> bool {
    v.finite()
}

/// Divide `a` by `b` using the GPU kernels' scaled component formula.
fn div(a: C, b: C) -> C {
    a.scaled_div(b)
}

/// Scale symmetric block entries `a`, `b`, `c` and return their determinant.
fn block(a: C, b: C, c: C) -> (C, C, C, R, C) {
    let scale = a.norm().max(b.norm()).max(c.norm());
    let (aa, bb, cc) = (a / scale, b / scale, c / scale);
    (aa, bb, cc, scale, aa * cc - bb * bb)
}
