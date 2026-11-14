//! Supernodal `LDL^T` on the CPU, selected by `Config::supernodal`.
//!
//! The multifrontal route assembles every node into a dense `nr` by `nr`
//! front: the slot is cleared, the original entries and every child's
//! trailing block are added in, and the node's own trailing block is handed on
//! to its parent the same way. Here a node never has a front. Its `nr` by
//! `nc` columns are assembled and factored in place, in the node's block of
//! the factor store, and its update `L21 D L21^T` goes straight into the
//! columns of the ancestors that own those rows (a right-looking, fan-out
//! supernodal factorization).
//!
//! What that saves is the clearing of every front and the repeated hand-off of
//! trailing blocks up the tree; the only buffer cleared is the factor store
//! itself, once. The arithmetic and the pivoting are the multifrontal route's:
//! Bunch-Kaufman through `?sytrf` on the fully summed block, or the serial
//! complete search on narrow nodes, and no pivot leaves its node. So the factor
//! comes out in the multifrontal layout -- row-major `nr` by `nc` per node,
//! pivots and `D` per column -- and the solve is shared unchanged.
//!
//! Concurrency follows the level schedule. A node's own columns are complete
//! when its level starts, since every descendant is in an earlier level.
//! Nodes of one level may update the same ancestor at once, so ancestor rows
//! are written under striped locks; a node's own rows are never written by
//! anyone else while it runs.
use super::blocked::grow;
use super::dense::{self, Side, Threading, Tri};
use super::factor::Front;
use super::profile::{Phase, Report};
use super::{work, Scalar, Worker, C, R};
use crate::{Analysis, Error, PivotFailure, Result};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Lock stripes guarding ancestor rows.
const STRIPES: usize = 1 << 12;
/// An ancestor's rows share a stripe in runs of `1 << STRIPE_ROWS`.
const STRIPE_ROWS: u32 = 6;

/// What the supernodal route adds to a plan: which node owns each column, and
/// the locks its updates take.
pub(super) struct Layout {
    owner: Vec<u32>,
    locks: Vec<Mutex<()>>,
}

impl Layout {
    /// The owner of every column of `plan`, and a fresh set of locks.
    pub(super) fn new(plan: &Analysis) -> Self {
        let mut owner = vec![0u32; plan.n];
        for (id, node) in plan.nodes.iter().enumerate() {
            owner[node.first as usize..(node.first + node.nc) as usize].fill(id as u32);
        }
        Self {
            owner,
            locks: (0..STRIPES).map(|_| Mutex::new(())).collect(),
        }
    }

    /// The stripe guarding `row` (a position among node `node`'s rows) of
    /// node `node`'s block.
    fn lock(&self, node: u32, row: usize) -> usize {
        ((node as usize).wrapping_mul(0x9E37_79B9) ^ (row >> STRIPE_ROWS)) & (STRIPES - 1)
    }
}

/// One node's update targets: its trailing rows `j0..j1` are columns of
/// `node`, and `rel[at..]` places each trailing row from `j0` on in `node`'s
/// rows.
#[derive(Clone, Copy)]
struct Run {
    node: u32,
    j0: usize,
    j1: usize,
    at: usize,
}

/// Per-node buffers, grown to the largest node a worker has seen.
#[derive(Default)]
pub(super) struct Work {
    /// A dense front, for the nodes the serial kernel or the fallback factors.
    front: Vec<C>,
    runs: Vec<Run>,
    rel: Vec<u32>,
    /// Each trailing row's column within the node that owns it.
    cols: Vec<u32>,
}

thread_local! {
    /// The update block of one row block, per thread.
    static BLOCK: RefCell<Vec<C>> = const { RefCell::new(Vec::new()) };
}

/// The factor store and pivot metadata, as pointers the nodes of a level
/// write through concurrently.
#[derive(Clone, Copy)]
struct Store {
    factors: *mut C,
    diagonal: *mut C,
    subdiag: *mut C,
    perms: *mut usize,
    two: *mut bool,
    /// The absolute static-pivoting threshold; see `Front::tiny`.
    tiny: R,
}
// SAFETY: a node writes its own pivot metadata and its own factor block, which
// no other node of its level touches, and ancestors' factor rows only under
// the stripe lock of those rows.
unsafe impl Send for Store {}
unsafe impl Sync for Store {}

/// The real half of a complex factorization, finished before the complex
/// half starts; see `RealPart` in `mod.rs`.
///
/// `real[id]` marks the nodes it factored. Its store holds their factors, and
/// for every other node the real parts of its entries plus every update the
/// real nodes sent it, all in this store's layout.
#[derive(Clone, Copy)]
pub(super) struct Split<'a> {
    pub real: &'a [bool],
    pub factors: &'a [R],
    pub diagonal: &'a [R],
    pub subdiag: &'a [R],
    pub perms: &'a [usize],
    pub two: &'a [bool],
}

/// Factor every node of `plan` supernodally. Returns the pivots perturbed.
///
/// `layout` is the solver's [`Layout`] and `schedule` its level order. The
/// factor store and pivot metadata (`factors`, `diagonal`, `subdiag`,
/// `perms`, `two`) are laid out as for the multifrontal route; `values` are
/// the matrix entries in analyzed CSC order. Nodes borrow per-front buffers
/// from `workers`. `machine` lets a node alone in its level thread its BLAS
/// calls across the whole machine; `report` collects the profile.
///
/// With `split`, the nodes it marks real are not factored, nor is their block
/// of `factors` written: their factor stays in the split's store and only
/// their pivots are copied. Every other node starts from its store's real accumulation
/// plus the imaginary parts of its entries. `schedule` must then hold only
/// the nodes not marked.
#[allow(clippy::too_many_arguments)]
pub(super) fn factor_all(
    plan: &Analysis,
    layout: &Layout,
    schedule: &[super::level::Level],
    factors: &mut [C],
    diagonal: &mut [C],
    subdiag: &mut [C],
    perms: &mut [usize],
    two: &mut [bool],
    values: &[C],
    tiny: R,
    workers: &Mutex<Vec<Worker>>,
    machine: bool,
    report: &mut Report,
    split: Option<Split<'_>>,
) -> Result<u64> {
    let store = Store {
        tiny,
        factors: factors.as_mut_ptr(),
        diagonal: diagonal.as_mut_ptr(),
        subdiag: subdiag.as_mut_ptr(),
        perms: perms.as_mut_ptr(),
        two: two.as_mut_ptr(),
    };
    // Every node's columns, cleared and holding their original entries, before
    // any descendant adds to them.
    let clock = Report::clock();
    plan.nodes.par_iter().enumerate().try_for_each(|(id, node)| {
        // The whole `Store`, which is `Sync`, rather than its pointer field.
        let store = &store;
        let (m, n) = (node.nr as usize, node.nc as usize);
        let base = node.loffset as usize;
        // SAFETY: each node's block is its own.
        let panel = unsafe { std::slice::from_raw_parts_mut(store.factors.add(base), m * n) };
        let entries = &plan.entries[node.entries as usize..(node.entries + node.nentries) as usize];
        let Some(split) = split else {
            panel.fill(C::from_re(0.0));
            for entry in entries {
                let (r, c) = (entry.position as usize / m, entry.position as usize % m);
                if c >= n || r < c {
                    return Err(Error::Invalid("supernodal entry outside its node's columns".into()));
                }
                panel[r * n + c] = values[entry.source as usize];
            }
            return Ok(());
        };
        if split.real[id] {
            // The factor itself stays in the real store, where the solve
            // reads it; only the pivots, which every solve step reads by
            // column, are copied.
            let first = node.first as usize;
            // SAFETY: each node's pivot metadata is its own.
            unsafe {
                for k in first..first + n {
                    *store.diagonal.add(k) = C::from_re(split.diagonal[k]);
                    *store.subdiag.add(k) = C::from_re(split.subdiag[k]);
                    *store.perms.add(k) = split.perms[k];
                    *store.two.add(k) = split.two[k];
                }
            }
            return Ok(());
        }
        for (slot, &v) in panel.iter_mut().zip(&split.factors[base..base + m * n]) {
            *slot = C::from_re(v);
        }
        for entry in entries {
            let (r, c) = (entry.position as usize / m, entry.position as usize % m);
            if c >= n || r < c {
                return Err(Error::Invalid("supernodal entry outside its node's columns".into()));
            }
            let v = values[entry.source as usize];
            panel[r * n + c] += v - C::from_re(v.re());
        }
        Ok(())
    })?;
    report.stop(Phase::Blank, clock);

    let take = || workers.lock().unwrap().pop().unwrap_or_default();
    let give = |worker: Worker| workers.lock().unwrap().push(worker);
    let bumped = AtomicU64::new(0);
    for level in schedule {
        dense::set_global_threading(if level.spread { Threading::Caller } else { Threading::Machine });
        let clock = Report::clock();
        if !level.spread {
            let mut worker = take();
            for &id in &level.nodes {
                // SAFETY: the only node running.
                let count = unsafe { node(plan, layout, store, id, true, machine, &mut worker) };
                match count {
                    Ok(count) => bumped.fetch_add(count, Ordering::Relaxed),
                    Err(e) => {
                        give(worker);
                        return Err(e);
                    }
                };
            }
            worker.report.level(false, clock);
            give(worker);
            continue;
        }
        let share = level.nodes.iter().map(|&id| work(plan, id)).sum::<f64>() / rayon::current_num_threads() as f64;
        level.nodes.par_iter().try_for_each(|&id| {
            let mut worker = take();
            let split = work(plan, id) > share;
            // SAFETY: the nodes of a level are distinct and none is an ancestor
            // of another; the level joins before the next starts.
            let count = unsafe { node(plan, layout, store, id, split, false, &mut worker) };
            give(worker);
            bumped.fetch_add(count?, Ordering::Relaxed);
            Ok::<(), Error>(())
        })?;
        let mut pool = workers.lock().unwrap();
        if pool.is_empty() {
            pool.push(Worker::default());
        }
        pool[0].report.level(true, clock);
    }
    Ok(bumped.load(Ordering::Relaxed))
}

/// Factor node `id` in place and add its update to its ancestors. Returns
/// the pivots perturbed.
///
/// `store` points into the storage of `plan`, whose owners `layout` holds.
/// `parallel` splits the node's own work into Rayon tasks; `alone` lets its
/// BLAS calls thread across the machine. `worker` holds its buffers and
/// profile.
///
/// # Safety
/// Every descendant of `id` has finished, and no node running at the same time
/// is `id`, an ancestor of it, or a descendant of it.
unsafe fn node(
    plan: &Analysis,
    layout: &Layout,
    store: Store,
    id: u32,
    parallel: bool,
    alone: bool,
    worker: &mut Worker,
) -> Result<u64> {
    let node = plan.nodes[id as usize];
    let (m, n, first) = (node.nr as usize, node.nc as usize, node.first as usize);
    let tail = m - n;
    let panel = std::slice::from_raw_parts_mut(store.factors.add(node.loffset as usize), m * n);
    let perm = std::slice::from_raw_parts_mut(store.perms.add(first), n);
    let d = std::slice::from_raw_parts_mut(store.diagonal.add(first), n);
    let sub = std::slice::from_raw_parts_mut(store.subdiag.add(first), n);
    let two = std::slice::from_raw_parts_mut(store.two.add(first), n);
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }
    two.fill(false);
    sub.fill(C::from_re(0.0));
    let failed = |(pivot, reason): (usize, PivotFailure)| Error::Pivot {
        node: id as usize,
        pivot: pivot as u32 + 1,
        reason,
        unknown: None,
    };
    let nonfinite = || Error::Pivot {
        node: id as usize,
        pivot: n as u32,
        reason: PivotFailure::NonFinite,
        unknown: None,
    };
    let Worker { scratch, report, sn } = worker;
    let small = m <= plan.config.small_front_rows as usize || !dense::available();

    // The blocked route: `?sytrf` on the fully summed block, then the
    // off-diagonal solve, all inside the node's own columns.
    if !small {
        let clock = Report::clock();
        scratch.fit(m, n);
        let leading = &mut scratch.leading[..n * n];
        for j in 0..n {
            for i in j..n {
                leading[j * n + i] = panel[i * n + j];
            }
        }
        if alone {
            dense::set_threading(Threading::Machine);
        }
        let info = dense::sytrf(n, leading, n, &mut scratch.ipiv[..n], &mut scratch.work);
        if alone {
            dense::set_threading(Threading::Caller);
        }
        if info == 0 && !super::factor::sytrf_tiny(leading, &scratch.ipiv[..n], n, store.tiny) {
            let mut front = Front {
                a: &mut [],
                m,
                n,
                perm,
                d,
                sub,
                two,
                config: &plan.config,
                kind: plan.kind,
                bumped: 0,
                tiny: store.tiny,
                parallel,
                alone,
                l: Some(&mut *panel),
                stored: false,
                finite: true,
            };
            front.adopt(leading, &scratch.ipiv[..n], &mut scratch.twos);
            front.reciprocals(leading, &mut scratch.inv[..n], &mut scratch.off[..n]).map_err(failed)?;
            let mut finite = front.store_leading(leading);
            report.stop(Phase::Leading, clock);
            report.front(Phase::Leading, m, n);
            if tail == 0 {
                return if finite { Ok(0) } else { Err(nonfinite()) };
            }
            let clock = Report::clock();
            let rows = plan.config.cpu_block_rows.max(1);
            let packed = &mut scratch.packed[..tail * n];
            {
                // `A21`, its columns in pivot order.
                let (l, perm) = (front.l.as_deref().unwrap(), &*front.perm);
                let fill = |i0: usize, block: &mut [C]| {
                    for (i, out) in block.chunks_exact_mut(n).enumerate() {
                        let row = &l[(n + i0 + i) * n..(n + i0 + i + 1) * n];
                        for (slot, &p) in out.iter_mut().zip(perm.iter()) {
                            *slot = row[p];
                        }
                    }
                };
                let tri: &[C] = leading;
                let solve = |i0: usize, block: &mut [C]| {
                    fill(i0, block);
                    dense::trsm(Side::Right, Tri::UnitUpper, block.len() / n, n, tri, n, block, n);
                };
                if parallel {
                    packed.par_chunks_mut(rows * n).enumerate().for_each(|(b, block)| solve(b * rows, block));
                } else {
                    solve(0, packed);
                }
            }
            finite &= front.store_trailing(packed, &scratch.inv[..n], &scratch.off[..n], &scratch.twos);
            report.stop(Phase::Offdiag, clock);
            if !finite {
                return Err(nonfinite());
            }
            // `packed` is `L21 D`, and the node's block now holds `L21`.
            let l21 = &panel[n * n..];
            let packed = &scratch.packed[..tail * n];
            targets(plan, layout, &plan.rows[node.rows as usize..node.rows as usize + m], n, sn)?;
            let (gemm, scatter) = (AtomicU64::new(0), AtomicU64::new(0));
            let run = |b: usize| {
                let (i0, i1) = (b * rows, ((b + 1) * rows).min(tail));
                BLOCK.with(|cell| {
                    let mut w = cell.borrow_mut();
                    grow(&mut w, (i1 - i0) * i1);
                    let t0 = Instant::now();
                    dense::gemm_nt_set(i1 - i0, i1, n, &l21[i0 * n..], n, packed, n, &mut w, i1);
                    let t1 = Instant::now();
                    scatter_rows(plan, layout, store, sn, i0, i1, &w, i1);
                    gemm.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
                    scatter.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
                });
            };
            let blocks = tail.div_ceil(rows);
            if parallel {
                (0..blocks).into_par_iter().for_each(run);
            } else {
                (0..blocks).for_each(run);
            }
            report.add(Phase::Schur, gemm.load(Ordering::Relaxed) as f64 * 1e-9);
            report.add(Phase::Assemble, scatter.load(Ordering::Relaxed) as f64 * 1e-9);
            return Ok(0);
        }
        // An exactly singular pivot block: nothing outside the scratch has
        // been written, so the node is factored below as a dense front, whose
        // fallback rescues the pivot.
    }

    // Narrow nodes, and the fallback: the node as a dense front whose
    // trailing block starts at zero and so ends as minus the update.
    grow(&mut sn.front, m * m);
    let a = &mut sn.front[..m * m];
    for r in 0..m {
        a[r * m..r * m + r + 1].fill(C::from_re(0.0));
        let w = n.min(r + 1);
        a[r * m..r * m + w].copy_from_slice(&panel[r * n..r * n + w]);
    }
    let mut front = Front {
        a,
        m,
        n,
        perm,
        d,
        sub,
        two,
        config: &plan.config,
        kind: plan.kind,
        bumped: 0,
        tiny: store.tiny,
        parallel: false,
        alone,
        l: None,
        stored: false,
        finite: true,
    };
    front.factor(scratch, report).map_err(failed)?;
    let bumped = front.bumped;
    let clock = Report::clock();
    let a = &sn.front[..m * m];
    for (i, row) in panel.chunks_exact_mut(n).enumerate() {
        row.copy_from_slice(&a[i * m..i * m + n]);
        if i + 1 < n {
            row[i + 1..].fill(C::from_re(0.0));
        }
    }
    if !panel.iter().all(|v| v.finite()) {
        return Err(nonfinite());
    }
    report.stop(Phase::Store, clock);
    if tail > 0 {
        let clock = Report::clock();
        targets(plan, layout, &plan.rows[node.rows as usize..node.rows as usize + m], n, sn)?;
        let block = &sn.front[n * m + n..];
        let Work { runs, rel, cols, .. } = &*sn;
        scatter_with(plan, layout, store, runs, rel, cols, 0, tail, block, m);
        report.stop(Phase::Assemble, clock);
    }
    Ok(bumped)
}

/// Find where each of a node's trailing rows lands: the ancestor owning it as
/// a column, and its position among the rows of every ancestor it updates.
///
/// `rows` are the node's rows as permuted column indices, its own `n`
/// columns first; the result goes to `sn.runs`, `sn.rel` and `sn.cols`.
fn targets(plan: &Analysis, layout: &Layout, rows: &[u32], n: usize, sn: &mut Work) -> Result<()> {
    let tail = &rows[n..];
    sn.runs.clear();
    sn.rel.clear();
    sn.cols.clear();
    sn.cols.resize(tail.len(), 0);
    let mut j = 0;
    while j < tail.len() {
        let t = layout.owner[tail[j] as usize];
        let target = plan.nodes[t as usize];
        let end = target.first + target.nc;
        let mut j1 = j;
        while j1 < tail.len() && tail[j1] < end {
            sn.cols[j1] = tail[j1] - target.first;
            j1 += 1;
        }
        let them = &plan.rows[target.rows as usize..(target.rows + target.nr) as usize];
        let at = sn.rel.len();
        let mut lo = 0;
        for &g in &tail[j..] {
            let pos = lo + seek(&them[lo..], g);
            if pos >= them.len() || them[pos] != g {
                return Err(Error::Invalid("supernodal update row missing from its ancestor".into()));
            }
            sn.rel.push(pos as u32);
            lo = pos + 1;
        }
        sn.runs.push(Run { node: t, j0: j, j1, at });
        j = j1;
    }
    Ok(())
}

/// The first position in ascending `rows` not below the permuted index `g`,
/// searched outward from the start: the rows a node updates sit close
/// together.
fn seek(rows: &[u32], g: u32) -> usize {
    let mut hi = 1;
    while hi < rows.len() && rows[hi] < g {
        hi *= 2;
    }
    rows[..(hi + 1).min(rows.len())].partition_point(|&r| r < g)
}

/// [`scatter_with`] over the targets [`targets`] left in `sn`.
///
/// # Safety
/// As for [`scatter_with`].
#[allow(clippy::too_many_arguments)]
unsafe fn scatter_rows(plan: &Analysis, layout: &Layout, store: Store, sn: &Work, i0: usize, i1: usize, w: &[C], ldw: usize) {
    scatter_with(plan, layout, store, &sn.runs, &sn.rel, &sn.cols, i0, i1, w, ldw);
}

/// Add trailing rows `i0..i1` of a node's update -- row `i` at `w[(i - i0) *
/// ldw..]`, its lower triangle -- into the ancestors' columns.
///
/// `runs`, `rel` and `cols` are the node's targets from [`targets`]; `store`
/// points into the storage of `plan`, and `layout` holds its locks.
///
/// # Safety
/// The ancestors' blocks are live and not being factored.
#[allow(clippy::too_many_arguments)]
unsafe fn scatter_with(
    plan: &Analysis,
    layout: &Layout,
    store: Store,
    runs: &[Run],
    rel: &[u32],
    cols: &[u32],
    i0: usize,
    i1: usize,
    w: &[C],
    ldw: usize,
) {
    for run in runs {
        if run.j0 >= i1 {
            break;
        }
        let target = plan.nodes[run.node as usize];
        let width = target.nc as usize;
        let base = store.factors.add(target.loffset as usize);
        let mut i = i0.max(run.j0);
        while i < i1 {
            let stripe = layout.lock(run.node, rel[run.at + i - run.j0] as usize);
            let _guard = layout.locks[stripe].lock().unwrap();
            while i < i1 {
                let pos = rel[run.at + i - run.j0] as usize;
                if layout.lock(run.node, pos) != stripe {
                    break;
                }
                let row = std::slice::from_raw_parts_mut(base.add(pos * width), width);
                let source = &w[(i - i0) * ldw..];
                let end = run.j1.min(i + 1);
                for j in run.j0..end {
                    row[cols[j] as usize] += source[j];
                }
                i += 1;
            }
        }
    }
}
