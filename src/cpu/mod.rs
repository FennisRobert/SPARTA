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
mod sparse;
mod supernodal;
mod update;

use crate::{Analysis, Error, Kind, Result};
pub(crate) use sparse::Columns;

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
    sn: supernodal::Work,
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
    /// Present when this solver factors supernodally; see
    /// `Config::supernodal`.
    supernodal: Option<supernodal::Layout>,
    /// The real half of a supernodal complex `LDL^T`; see [`RealPart`].
    real: Option<Box<RealHalf>>,
    /// The nodes the latest factorization left in `real`'s store, by node;
    /// empty when it used no real half.
    mask: Vec<bool>,
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
        // Supernodal needs the row-major factor the CPU solve reads, so the
        // hybrid's column-major solvers stay multifrontal.
        let supernodal = (plan.config.supernodal && plan.kind == Kind::Ldlt && rowmajor)
            .then(|| supernodal::Layout::new(&plan));
        let real = (supernodal.is_some() && <C as Scalar>::KIND.is_complex())
            .then(|| RealHalf::new(&plan, threads).map(Box::new))
            .transpose()?;
        Ok(Self {
            real,
            mask: Vec::new(),
            workers: Mutex::new((0..threads).map(|_| Worker::default()).collect()),
            schedule: level::schedule(&plan),
            pool,
            rowmajor,
            // A supernodal solver never builds a front.
            fronts: allocate_lazy(if supernodal.is_some() { 0 } else { plan.elements })?,
            supernodal,
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
        self.schedule = restricted(&level::schedule(&self.plan), |id| keep[id as usize]);
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

    /// The matrix dimension.
    pub(crate) fn n(&self) -> usize {
        self.plan.n
    }

    /// Factor finite `values` in the analyzed CSC entry order.
    pub(crate) fn factor(&mut self, values: &[C]) -> Result<()> {
        self.valid = false;
        self.seconds = 0.0;
        self.perturbed = 0;
        if self.destroyed {
            return Err(Error::Invalid("this solver was destroyed".into()));
        }
        if values.len() != self.plan.nnz {
            return Err(Error::Invalid(format!(
                "expected {} matrix values for the analyzed pattern, got {}",
                self.plan.nnz,
                values.len()
            )));
        }
        if let Some(at) = values.iter().position(|&v| !finite(v)) {
            return Err(Error::Invalid(format!("matrix value {at} is not finite")));
        }
        let started = Instant::now();
        let mut report = Report::default();
        let pool = self.pool.clone();
        let result = pool.run(|| self.factor_all(values, &mut report));
        self.seconds = started.elapsed().as_secs_f64();
        report.print(self.seconds);
        self.valid = result.is_ok();
        result.map_err(|e| e.located(|node, pivot| unknown(&self.plan, &self.perms, node, pivot)))
    }

    /// Assemble and eliminate fronts leaves first, copying factors before slot reuse.
    ///
    /// Each level runs across the Rayon pool; the walk joins between levels,
    /// which is what makes the disjointness `Shared` relies on hold.
    fn factor_all(&mut self, values: &[C], report: &mut Report) -> Result<()> {
        // Static pivoting is relative to the whole matrix, not to a front,
        // whose entries may already carry the growth it is meant to stop.
        let tiny = if self.plan.kind == Kind::Ldlt {
            self.plan.config.static_pivot as R * values.iter().map(|v| v.norm()).fold(0.0, R::max)
        } else {
            0.0
        };
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
            supernodal,
            real,
            mask: last,
            ..
        } = self;
        last.clear();
        // A front alone on the machine may have the library thread its
        // products -- unless this solver is bounded to fewer threads, which
        // the library's own threading would not respect.
        let machine = !pool.bounded();
        let shared = Shared::new(fronts, factors, upper, diagonal, subdiag, perms, two, *rowmajor, tiny);
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
        let result = if let Some(layout) = supernodal {
            // The nodes no complex entry reaches go first, in real arithmetic.
            let mask = real.as_deref_mut().filter(|_| plan.config.real_split).and_then(|part| {
                let mask = plan.real_nodes(|s| values[s].im() == 0.0);
                mask.iter().any(|&r| r).then_some((part, mask))
            });
            match mask {
                None => supernodal::factor_all(
                    plan, layout, schedule, factors, diagonal, subdiag, perms, two, values, tiny, workers, machine,
                    report, None,
                ),
                Some((part, mask)) => {
                    let clock = Instant::now();
                    part.values(values.len()).par_iter_mut().zip(values.par_iter()).for_each(|(r, v)| *r = v.re());
                    let ours = part
                        .factor(plan, &mask, tiny, machine)
                        .map_err(|e| e.located(|node, pivot| unknown(plan, part.perms(), node, pivot)));
                    if dump {
                        eprintln!("REAL HALF {:.4} s", clock.elapsed().as_secs_f64());
                    }
                    ours.and_then(|count| {
                        bumped.fetch_add(count, Ordering::Relaxed);
                        let rest = restricted(schedule, |id| !mask[id as usize]);
                        let split = supernodal::Split {
                            real: &mask,
                            factors: part.factors(),
                            diagonal: part.diagonal(),
                            subdiag: part.subdiag(),
                            perms: part.perms(),
                            two: part.two(),
                        };
                        let result = supernodal::factor_all(
                            plan, layout, &rest, factors, diagonal, subdiag, perms, two, values, tiny, workers,
                            machine, report, Some(split),
                        );
                        *last = mask;
                        result
                    })
                }
            }
            .map(|count| {
                bumped.fetch_add(count, Ordering::Relaxed);
            })
        } else {
            schedule.iter().enumerate().try_for_each(|(depth, level)| {
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
                let Worker { scratch, report, .. } = &mut worker;
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
        })
        };
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
        self.real = None;
        self.mask = Vec::new();
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

/// The caller's index of the unknown at 1-based column `pivot` of `node`'s
/// front, after the symmetric interchanges recorded in `perms` (the node's
/// local pivot order). `None` when the column is outside the node.
///
/// LU interchanges rows only, so its column `pivot` is the unknown.
pub(crate) fn unknown(plan: &Analysis, perms: &[usize], node: usize, pivot: u32) -> Option<usize> {
    let node = plan.nodes.get(node)?;
    let (first, nc) = (node.first as usize, node.nc as usize);
    let k = (pivot as usize).checked_sub(1).filter(|&k| k < nc)?;
    let local = match plan.kind {
        Kind::Lu => k,
        Kind::Ldlt => perms.get(first + k).copied().filter(|&l| l < nc).unwrap_or(k),
    };
    plan.perm.get(first + local).map(|&p| p as usize)
}

/// `schedule` with only the nodes `keep` accepts, empty levels dropped.
fn restricted(schedule: &[Level], keep: impl Fn(u32) -> bool) -> Vec<Level> {
    schedule
        .iter()
        .filter_map(|level| {
            let nodes: Vec<u32> = level.nodes.iter().copied().filter(|&id| keep(id)).collect();
            (!nodes.is_empty()).then_some(Level { spread: level.spread && nodes.len() >= 2, nodes })
        })
        .collect()
}

/// The real-arithmetic half of a complex supernodal `LDL^T`.
///
/// In many complex systems the imaginary parts sit on a few unknowns, such as
/// the port and boundary unknowns of an FEM model. A node none of whose
/// descendants has a complex entry has a real factor
/// (see `Analysis::real_nodes`), and real arithmetic is about four times cheaper. The
/// complex copy of a precision keeps one of these, from the real copy of the
/// same precision, and runs it first: the supernodal factorization on the
/// real parts of every entry, restricted to the real nodes. What it leaves in
/// the other nodes' columns is their real parts plus every update the real
/// nodes sent them, which the complex half starts from. The ordering is not
/// touched, so neither is the fill.
pub(crate) struct RealPart {
    layout: supernodal::Layout,
    workers: Mutex<Vec<Worker>>,
    factors: Vec<C>,
    diagonal: Vec<C>,
    subdiag: Vec<C>,
    perms: Vec<usize>,
    two: Vec<bool>,
    values: Vec<C>,
}

// Only the complex copies construct one; the real copies just define it.
#[allow(dead_code)]
impl RealPart {
    /// Storage for the real half of `plan`, its nodes run by `threads` workers.
    pub(crate) fn new(plan: &Analysis, threads: usize) -> Result<Self> {
        Ok(Self {
            layout: supernodal::Layout::new(plan),
            workers: Mutex::new((0..threads).map(|_| Worker::default()).collect()),
            factors: allocate_lazy(plan.factor_elements)?,
            diagonal: allocate_lazy(plan.n)?,
            subdiag: allocate_lazy(plan.n)?,
            perms: vec![0; plan.n],
            two: vec![false; plan.n],
            values: Vec::new(),
        })
    }

    /// The buffer for the real parts of the entries, in analyzed CSC order.
    pub(crate) fn values(&mut self, nnz: usize) -> &mut [C] {
        self.values.resize(nnz, C::from_re(0.0));
        &mut self.values
    }

    /// Factor the nodes `real` marks from the entries [`values`] holds, with
    /// the complex matrix's static-pivoting threshold `tiny`. Returns the
    /// pivots perturbed. Runs on the calling pool.
    pub(crate) fn factor(&mut self, plan: &Analysis, real: &[bool], tiny: R, machine: bool) -> Result<u64> {
        let schedule = restricted(&level::schedule(plan), |id| real[id as usize]);
        let mut report = Report::default();
        let clock = Instant::now();
        let result = supernodal::factor_all(
            plan, &self.layout, &schedule, &mut self.factors, &mut self.diagonal, &mut self.subdiag,
            &mut self.perms, &mut self.two, &self.values, tiny, &self.workers, machine, &mut report, None,
        );
        for worker in self.workers.lock().unwrap().iter_mut() {
            report.merge(&worker.report);
            worker.report = Report::default();
        }
        report.print(clock.elapsed().as_secs_f64());
        result
    }

    /// The real factor store: the real nodes' `L` blocks, in the plan's layout.
    pub(crate) fn factors(&self) -> &[C] {
        &self.factors
    }
    /// `c[m][n] -= a[m][k] * b[k][n]`, row-major, in this copy's arithmetic;
    /// `lda`, `ldb` and `ldc` are the row strides.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_nn(m: usize, n: usize, k: usize, a: &[C], lda: usize, b: &[C], ldb: usize, c: &mut [C], ldc: usize) {
        dense::gemm_nn(m, n, k, a, lda, b, ldb, c, ldc);
    }
    /// `c[m][n] -= a[k][m]^T * b[k][n]`, row-major, in this copy's arithmetic;
    /// `lda`, `ldb` and `ldc` are the row strides.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_tn(m: usize, n: usize, k: usize, a: &[C], lda: usize, b: &[C], ldb: usize, c: &mut [C], ldc: usize) {
        dense::gemm_tn(m, n, k, a, lda, b, ldb, c, ldc);
    }
    /// `D`'s diagonal, by column; valid for the real nodes' columns.
    pub(crate) fn diagonal(&self) -> &[C] {
        &self.diagonal
    }
    /// `D`'s subdiagonal, by column; valid for the real nodes' columns.
    pub(crate) fn subdiag(&self) -> &[C] {
        &self.subdiag
    }
    /// Each real node's pivot order, by column.
    pub(crate) fn perms(&self) -> &[usize] {
        &self.perms
    }
    /// Whether each real node's column starts a 2x2 pivot.
    pub(crate) fn two(&self) -> &[bool] {
        &self.two
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
