//! Complex128 multifrontal numerics using the same symbolic layout as Metal.
mod blocked;
mod dense;
mod factor;
mod level;
mod panel;
mod profile;
mod solve;
mod update;

use crate::{Analysis, Complex64 as C, Error, Kind, Result};
use blocked::Scratch;
use dense::Threading;
use level::{Level, Shared};
use profile::Report;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// One Rayon thread's reusable state for the length of a factorization.
#[derive(Default)]
struct Worker {
    scratch: Scratch,
    report: Report,
}

/// CPU allocation; front slots are reused only after their factors are stored.
pub(crate) struct Solver {
    plan: Analysis,
    fronts: Vec<C>,
    factors: Vec<C>,
    upper: Vec<C>,
    diagonal: Vec<C>,
    subdiag: Vec<C>,
    perms: Vec<usize>,
    two: Vec<bool>,
    workers: Vec<Mutex<Worker>>,
    schedule: Vec<Level>,
    valid: bool,
    destroyed: bool,
    pub(super) seconds: f64,
    pub(super) perturbed: u64,
}

impl Solver {
    /// Allocate complex128 buffers for the immutable symbolic `plan`.
    pub(crate) fn new(plan: Analysis) -> Result<Self> {
        plan.config.check().map_err(Error::Invalid)?;
        if plan.n == 0 {
            return Err(Error::Invalid("released analysis".into()));
        }
        let threads = rayon::current_num_threads().max(1);
        Ok(Self {
            workers: (0..threads).map(|_| Mutex::new(Worker::default())).collect(),
            schedule: level::schedule(&plan, threads),
            fronts: allocate(plan.elements)?,
            factors: allocate(plan.factor_elements)?,
            upper: allocate(plan.upper_elements)?,
            diagonal: allocate(plan.n)?,
            subdiag: allocate(plan.n)?,
            perms: vec![0; plan.n],
            two: vec![false; plan.n],
            plan,
            valid: false,
            destroyed: false,
            seconds: 0.0,
            perturbed: 0,
        })
    }

    /// Factor finite complex128 `values` in the analyzed CSC entry order.
    pub(crate) fn factor(&mut self, values: &[C]) -> Result<()> {
        self.valid = false;
        self.seconds = 0.0;
        self.perturbed = 0;
        if self.destroyed || values.len() != self.plan.nnz || values.iter().any(|&v| !finite(v)) {
            return Err(Error::Invalid(
                "finite complex128 values must match a live analyzed CSC pattern".into(),
            ));
        }
        let started = Instant::now();
        let mut report = Report::default();
        let result = self.factor_all(values, &mut report);
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
            ..
        } = self;
        let shared = Shared::new(fronts, factors, upper, diagonal, subdiag, perms, two);
        // Rayon owns the parallelism: fronts across threads in wide levels, and
        // row blocks within one front in narrow ones. Accelerate threads only
        // inside `zsytrf`, which sets and clears this itself. The setting is
        // thread-local, so it has to reach every worker, not just this one.
        dense::set_threading(Threading::Caller);
        rayon::broadcast(|_| {
            dense::set_threading(Threading::Caller);
        });
        for worker in workers.iter_mut() {
            worker.get_mut().unwrap().report = Report::default();
        }
        let bumped = AtomicU64::new(0);
        let dump = std::env::var_os("SPARTA_CPU_DUMP").is_some();
        let result = schedule.iter().enumerate().try_for_each(|(depth, level)| {
            let t0 = std::time::Instant::now();
            let clock = Report::clock();
            if !level.spread {
                let worker = workers[0].get_mut().unwrap();
                for &id in &level.nodes {
                    // SAFETY: the loop is the only thing touching the storage,
                    // and every node appears once.
                    let count =
                        unsafe {
                            shared.node(
                                plan,
                                id,
                                values,
                                true,
                                &mut worker.scratch,
                                &mut worker.report,
                            )
                        }?;
                    bumped.fetch_add(count, Ordering::Relaxed);
                }
                // SAFETY: every node of this level has returned above.
                for &id in &level.nodes {
                    unsafe { shared.compact(plan, id) };
                }
                worker.report.level(false, clock);
                if dump {
                    let work: f64 = level.nodes.iter().map(|&i| {
                        let nd = plan.nodes[i as usize];
                        let (t, n) = ((nd.nr - nd.nc) as f64, nd.nc as f64);
                        8.0 * (t * t * n + t * n * n / 2.0 + n * n * n / 3.0)
                    }).sum();
                    eprintln!("LEVEL {depth} seq {} {:.4} {:.2} {}", level.nodes.len(),
                        t0.elapsed().as_secs_f64(), work / 1e9,
                        plan.nodes[level.nodes[0] as usize].nr);
                }
                return Ok(());
            }
            level.nodes.par_iter().try_for_each(|&id| {
                // Uncontended by construction: one worker per pool thread.
                let mut worker = workers[rayon::current_thread_index().unwrap_or(0)]
                    .lock()
                    .unwrap();
                let Worker { scratch, report } = &mut *worker;
                // SAFETY: every node in `level` is distinct and the iterator
                // joins before the next level starts, which is exactly the
                // contract `Shared::node` documents.
                let count = unsafe { shared.node(plan, id, values, false, scratch, report) };
                bumped.fetch_add(count?, Ordering::Relaxed);
                Ok(())
            })?;
            // SAFETY: par_iter joined above, so every assembly in this level
            // is done; each node writes only its own slot.
            level
                .nodes
                .par_iter()
                .for_each(|&id| unsafe { shared.compact(plan, id) });
            workers[0].lock().unwrap().report.level(true, clock);
            if dump {
                let work: f64 = level.nodes.iter().map(|&i| {
                    let nd = plan.nodes[i as usize];
                    let (t, n) = ((nd.nr - nd.nc) as f64, nd.nc as f64);
                    8.0 * (t * t * n + t * n * n / 2.0 + n * n * n / 3.0)
                }).sum();
                eprintln!("LEVEL {depth} par {} {:.4} {:.2} {}", level.nodes.len(),
                    t0.elapsed().as_secs_f64(), work / 1e9,
                    plan.nodes[level.nodes[0] as usize].nr);
            }
            Ok(())
        });
        // Leave the pool as it was found, for whatever else in the process
        // calls Accelerate.
        dense::set_threading(Threading::Machine);
        rayon::broadcast(|_| {
            dense::set_threading(Threading::Machine);
        });
        *perturbed = bumped.load(Ordering::Relaxed);
        for worker in workers.iter_mut() {
            report.merge(&worker.get_mut().unwrap().report);
        }
        result
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
        self.workers = Vec::new();
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
    pub(crate) fn pivot_range(&self) -> Option<(f64, f64)> {
        if !self.valid {
            return None;
        }
        if self.plan.kind == Kind::Lu {
            let (mut lo, mut hi) = (f64::INFINITY, 0.0_f64);
            for node in &self.plan.nodes {
                let (m, base) = (node.nr as usize, node.loffset as usize);
                for k in 0..node.nc as usize {
                    let v = self.factors[base + k * m + k].norm();
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            return (lo.is_finite() && hi > 0.0).then_some((lo, hi));
        }
        let (mut lo, mut hi, mut k) = (f64::INFINITY, 0.0_f64, 0);
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
        Some((lo, hi))
    }
}

/// Allocate `count` complex128 elements, reporting capacity failures as solver errors.
fn allocate(count: usize) -> Result<Vec<C>> {
    let mut data = Vec::new();
    data.try_reserve_exact(count)
        .map_err(|e| Error::Unsupported(format!("CPU allocation: {e}")))?;
    data.resize(count, C::new(0.0, 0.0));
    Ok(data)
}

/// Test both floating-point components of complex128 `v`.
fn finite(v: C) -> bool {
    v.re.is_finite() && v.im.is_finite()
}

/// Divide complex128 `a` by `b` using the Metal kernel's scaled component formula.
fn div(a: C, b: C) -> C {
    if b.re.abs() >= b.im.abs() {
        let r = b.im / b.re;
        let d = b.re + b.im * r;
        C::new((a.re + a.im * r) / d, (a.im - a.re * r) / d)
    } else {
        let r = b.re / b.im;
        let d = b.im + b.re * r;
        C::new((a.re * r + a.im) / d, (a.im * r - a.re) / d)
    }
}

/// Scale symmetric block entries `a`, `b`, `c` and return their determinant.
fn block(a: C, b: C, c: C) -> (C, C, C, f64, C) {
    let scale = a.norm().max(b.norm()).max(c.norm());
    let (aa, bb, cc) = (a / scale, b / scale, c / scale);
    (aa, bb, cc, scale, aa * cc - bb * bb)
}
