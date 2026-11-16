//! Level-ordered multifrontal solves sharing Metal's global and local pivot mappings.
//!
//! The forward sweep is multifrontal, like the factorization: a node gathers
//! its front's right-hand side, adds its children's trailing updates through
//! the assembly maps, eliminates, and hands its own trailing update to its
//! parent. Nothing is scattered into rows another node owns, so the nodes of
//! a level run concurrently without synchronization. The backward sweep only
//! reads its ancestors' finished rows and writes its own, and is concurrent
//! the same way.
//!
//! Right-hand sides are held row-major, `nrhs` values per unknown, so every
//! factor column is read once for all of them, and every inner loop walks a
//! factor column contiguously.
use super::dense::{self, Side, Threading, Tri};
use super::Scalar;
use super::{block, div, finite, RealHalf, Solver, C, R};
use crate::{Error, Kind, Result};
use rayon::prelude::*;

/// The permuted right-hand sides, `nrhs` values per unknown, as a pointer the
/// nodes of one level write through concurrently.
#[derive(Clone, Copy)]
pub(super) struct Rows(pub(super) *mut C);
// SAFETY: `Rows` carries no access rights of its own. Within a level each node
// writes only its own pivot rows and reads its own and its ancestors' rows,
// none of which another node of the same level writes; see the callers.
unsafe impl Send for Rows {}
unsafe impl Sync for Rows {}

impl Rows {
    /// The `nrhs` values of unknown `g`, to read.
    ///
    /// # Safety
    /// `g` is in range, and no thread writes row `g` while this lives.
    unsafe fn read(&self, g: usize, nrhs: usize) -> &[C] {
        std::slice::from_raw_parts(self.0.add(g * nrhs), nrhs)
    }

    /// The `nrhs` values of unknown `g`, to write.
    ///
    /// # Safety
    /// `g` is in range, and no other thread reads or writes row `g` while this lives.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, g: usize, nrhs: usize) -> &mut [C] {
        std::slice::from_raw_parts_mut(self.0.add(g * nrhs), nrhs)
    }
}

/// Factor elements a node's trailing product must reach before its rows are
/// split across threads; below this a level's other nodes keep the pool busy.
const SPLIT: usize = 1 << 18;

/// Right-hand sides from which a node's trailing product goes to GEMM. Below
/// it each factor element meets too few values to repay packing them, and the
/// loops are bandwidth-bound anyway.
const GEMM_RHS: usize = 4;

/// Factor elements a trailing product must reach before it goes to GEMM.
const GEMM_MIN: usize = 4096;

/// Whether a trailing product over `elements` factor values goes to GEMM.
pub(super) fn gemm(r: usize, elements: usize) -> bool {
    r >= GEMM_RHS && elements >= GEMM_MIN
}

/// `t[i] -= l[i] * y` for every row `i`, each row holding `y.len()` values.
#[inline]
fn axpy(t: &mut [C], l: &[C], y: &[C]) {
    if let [y] = y {
        for (ti, &li) in t.iter_mut().zip(l) {
            *ti -= li * *y;
        }
    } else {
        for (row, &li) in t.chunks_exact_mut(y.len()).zip(l) {
            for (tc, &yc) in row.iter_mut().zip(y) {
                *tc -= li * yc;
            }
        }
    }
}

/// `acc -= sum_j l[j] * z[j]`, each `z[j]` holding `acc.len()` values.
#[inline]
fn dot(acc: &mut [C], l: &[C], z: &[C]) {
    if let [acc] = acc {
        let mut sum = C::from_re(0.0);
        for (&lj, &zj) in l.iter().zip(z) {
            sum += lj * zj;
        }
        *acc -= sum;
    } else {
        let r = acc.len();
        for (&lj, row) in l.iter().zip(z.chunks_exact(r)) {
            for (a, &zc) in acc.iter_mut().zip(row) {
                *a -= lj * zc;
            }
        }
    }
}

/// A node's `L` block: complex, or real when the factorization's real half
/// left it in its own store (`Solver::mask`).
#[derive(Clone, Copy)]
enum Block<'a> {
    Complex(&'a [C]),
    Real(&'a [R]),
}

/// Values of the real type per value of `C`: two in the complex copies.
const WIDE: usize = std::mem::size_of::<C>() / std::mem::size_of::<R>();

/// `values` as `WIDE` times as many real numbers: a row-major block of `r`
/// complex columns is the same block with `2 r` real ones, which a real
/// matrix multiplies from the left in real arithmetic.
fn reals(values: &[C]) -> &[R] {
    // SAFETY: `C` is `R` or `num_complex::Complex<R>`, which is `repr(C)`
    // with the real part first, so the memory is `WIDE` contiguous `R`s each.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() * WIDE) }
}

/// [`reals`], to write.
fn reals_mut(values: &mut [C]) -> &mut [R] {
    // SAFETY: as for `reals`.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len() * WIDE) }
}

/// [`dot`] with a real `l`.
#[inline]
fn dot_real(acc: &mut [C], l: &[R], z: &[C]) {
    let r = acc.len();
    for (&lj, row) in l.iter().zip(z.chunks_exact(r)) {
        for (a, &zc) in acc.iter_mut().zip(row) {
            *a -= zc * lj;
        }
    }
}

/// [`axpy`] with a real `l`.
#[inline]
fn axpy_real(t: &mut [C], l: &[R], y: &[C]) {
    for (row, &li) in t.chunks_exact_mut(y.len()).zip(l) {
        for (tc, &yc) in row.iter_mut().zip(y) {
            *tc -= yc * li;
        }
    }
}

/// Run `work` with this thread's dense calls spread over the machine when
/// `threads` is set, and back on the calling thread after.
fn with_threading<T>(threads: bool, work: impl FnOnce() -> T) -> T {
    if threads {
        dense::set_threading(Threading::Machine);
    }
    let out = work();
    if threads {
        dense::set_threading(Threading::Caller);
    }
    out
}

impl Solver {
    /// Call `solve` with a node's `n` by `n` diagonal block of its factor,
    /// row-major, as complex values (a real block is copied, which costs `n²`
    /// against the solve's `n² r`), and whether it may use every core: only
    /// a node alone in its level of a solve that set the pool to one thread
    /// per call (`with_caller_blas`).
    fn solve_diagonal(&self, block: Block<'_>, n: usize, r: usize, alone: bool, solve: impl FnOnce(&[C], bool)) {
        let threads = alone && gemm(r, usize::MAX);
        match block {
            Block::Complex(l) => solve(&l[..n * n], threads),
            Block::Real(l) => {
                let copy: Vec<C> = l[..n * n].iter().map(|&v| C::from_re(v)).collect();
                solve(&copy, threads)
            }
        }
    }

    /// Node `id`'s `nr` by `nc` block of `L`, row-major, wherever it is stored.
    fn block(&self, id: usize) -> Block<'_> {
        let node = self.plan.nodes[id];
        let range = node.loffset as usize..(node.loffset + node.nr * node.nc) as usize;
        match &self.real {
            Some(part) if self.mask.get(id) == Some(&true) => Block::Real(&part.factors()[range]),
            _ => Block::Complex(&self.factors[range]),
        }
    }

    /// Solve column-major `b` containing `nrhs` right-hand sides.
    pub(crate) fn solve_many(&self, b: &[C], nrhs: usize) -> Result<Vec<C>> {
        self.pool.run(|| self.solve_here(b, nrhs))
    }

    /// [`Solver::solve_many`] on the calling thread's pool.
    fn solve_here(&self, b: &[C], nrhs: usize) -> Result<Vec<C>> {
        if !self.valid {
            return Err(Error::Invalid(
                "successful factorization required before solve".into(),
            ));
        }
        let n = self.plan.n;
        if nrhs == 0 || n.checked_mul(nrhs) != Some(b.len()) || b.par_iter().any(|&v| !finite(v)) {
            return Err(Error::Invalid(
                "finite RHS columns of analyzed dimension required".into(),
            ));
        }
        let r = nrhs;
        let perm = &self.plan.perm;
        let mut x = vec![C::from_re(0.0); n * r];
        x.par_chunks_mut(r)
            .zip(perm.par_iter())
            .with_min_len(4096)
            .for_each(|(row, &old)| {
                for (c, slot) in row.iter_mut().enumerate() {
                    *slot = b[c * n + old as usize];
                }
            });
        self.with_caller_blas(gemm(r, usize::MAX), || {
            let rows = Rows(x.as_mut_ptr());
            // SAFETY: each sweep keeps to the per-level discipline `Rows` states.
            unsafe {
                self.forward(rows, r);
                self.finish_sweeps(rows, r);
            }
        });
        self.unpermute(&x, r)
    }

    /// Run `work` with the dense library single-threaded on each pool thread
    /// when `blas` is set: Rayon owns the parallelism, as in the
    /// factorization, so the GEMMs run one per task on the calling thread.
    pub(super) fn with_caller_blas<T>(&self, blas: bool, work: impl FnOnce() -> T) -> T {
        if blas {
            rayon::broadcast(|_| dense::set_threading(Threading::Caller));
        }
        let out = work();
        if blas {
            rayon::broadcast(|_| dense::set_threading(Threading::Machine));
        }
        out
    }

    /// Everything after the forward sweep: `D z = y` on LDL^T, then the
    /// backward sweep.
    ///
    /// # Safety
    /// As for [`Solver::forward`].
    pub(super) unsafe fn finish_sweeps(&self, x: Rows, r: usize) {
        if self.plan.kind == Kind::Ldlt {
            self.diagonal_solve(x, r);
        }
        self.backward(x, r);
    }

    /// The permuted row-major block `x`, `r` values per unknown, back in the
    /// caller's ordering as `r` columns end to end. Fails on a nonfinite value.
    pub(super) fn unpermute(&self, x: &[C], r: usize) -> Result<Vec<C>> {
        let (n, perm) = (self.plan.n, &self.plan.perm);
        let mut out = vec![C::from_re(0.0); n * r];
        out.par_chunks_mut(n).enumerate().for_each(|(c, column)| {
            for (row, &old) in x.chunks_exact(r).zip(perm.iter()) {
                column[old as usize] = row[c];
            }
        });
        if out.par_iter().any(|&v| !finite(v)) {
            return Err(Error::Invalid("nonfinite CPU solve result".into()));
        }
        Ok(out)
    }

    /// `L y = b`, leaves first. Each node's trailing update travels to its
    /// parent in `tails` rather than into rows the parent owns.
    ///
    /// # Safety
    /// `x` holds `n * r` values and nothing else accesses it meanwhile.
    unsafe fn forward(&self, x: Rows, r: usize) {
        let plan = &self.plan;
        let mut tails: Vec<Vec<C>> = vec![Vec::new(); plan.nodes.len()];
        for level in &plan.levels {
            let alone = level.len() == 1;
            let done: Vec<Vec<C>> = level
                .par_iter()
                .map(|&id| self.forward_node(id as usize, x, r, &tails, alone))
                .collect();
            for (&id, tail) in level.iter().zip(done) {
                let node = plan.nodes[id as usize];
                let children = &plan.children[node.children as usize..(node.children + node.count) as usize];
                for child in children {
                    tails[child.node as usize] = Vec::new();
                }
                tails[id as usize] = tail;
            }
        }
    }

    /// Eliminate node `id`'s pivots from the right-hand sides and return its
    /// trailing update, `nr - nc` rows of `r` values in front order.
    unsafe fn forward_node(&self, id: usize, x: Rows, r: usize, tails: &[Vec<C>], alone: bool) -> Vec<C> {
        let plan = &self.plan;
        let node = plan.nodes[id];
        let (m, n, first) = (node.nr as usize, node.nc as usize, node.first as usize);
        let rows = &plan.rows[node.rows as usize..node.rows as usize + m];
        let perm = &self.perms[first..first + n];
        let zero = C::from_re(0.0);
        // The front's right-hand side by front position: its own pivot rows,
        // which no other node of this level touches, plus every child's update.
        let mut w = vec![zero; m * r];
        for (p, slot) in w[..n * r].chunks_exact_mut(r).enumerate() {
            slot.copy_from_slice(x.read(rows[p] as usize, r));
        }
        for child in &plan.children[node.children as usize..(node.children + node.count) as usize] {
            let cn = plan.nodes[child.node as usize];
            let tail = (cn.nr - cn.nc) as usize;
            let map = &plan.maps[child.map as usize..child.map as usize + tail];
            for (&p, update) in map.iter().zip(tails[child.node as usize].chunks_exact(r)) {
                for (slot, &v) in w[p as usize * r..p as usize * r + r].iter_mut().zip(update) {
                    *slot += v;
                }
            }
        }
        let (y, t) = self.eliminate(id, w, r, alone);
        let lu = plan.kind == Kind::Lu;
        for (k, value) in y.chunks_exact(r).enumerate() {
            let g = if lu { rows[k] } else { rows[perm[k]] };
            x.row(g as usize, r).copy_from_slice(value);
        }
        t
    }

    /// Eliminate node `id`'s pivots from its gathered right-hand side and
    /// return the solved pivot rows and the trailing update.
    ///
    /// Arguments:
    /// - `id`: the node.
    /// - `w`: the front's right-hand side, `nr` rows of `r` values in front
    ///   order: its own pivot rows plus every child's update.
    /// - `r`: values per row.
    /// - `alone`: whether the node has its level to itself, so its triangular
    ///   solve may spread over the machine.
    ///
    /// Returns `y`, `nc` rows in pivot order, and the update, `nr - nc` rows
    /// in front order, for the parent.
    pub(super) fn eliminate(&self, id: usize, mut w: Vec<C>, r: usize, alone: bool) -> (Vec<C>, Vec<C>) {
        let node = self.plan.nodes[id];
        let (m, n, first) = (node.nr as usize, node.nc as usize, node.first as usize);
        let perm = &self.perms[first..first + n];
        let block = self.block(id);
        let mut y = vec![C::from_re(0.0); n * r];
        for (k, slot) in y.chunks_exact_mut(r).enumerate() {
            slot.copy_from_slice(&w[perm[k] * r..perm[k] * r + r]);
        }
        // L is row-major (`Solver::rowmajor`): row `i` holds `L[i][0..nc]`.
        if gemm(r, n * n) {
            // `y = L11^-1 y` by the blocked solve, whose work is mostly GEMM,
            // rather than row by row. With the same in `backward_node`, on
            // Zen 4 (8 cores, AVX-512, default refinement) 64 right-hand
            // sides went from 7.9 to 7.2 s on em_m4 and from 0.93 to 0.86 s
            // on em_m1. Without refinement, 16 and 64 right-hand sides now
            // take 0.7 to 0.8 times 0.3.6's time on em_m1 and em_m4 and
            // under half on em_m3 (LU).
            self.solve_diagonal(block, n, r, alone, |l11, threads| {
                with_threading(threads, || dense::trsm(Side::Left, Tri::UnitLower, n, r, l11, n, &mut y, r))
            });
        } else {
            for i in 1..n {
                let (before, at) = y.split_at_mut(i * r);
                match block {
                    Block::Complex(l) => dot(&mut at[..r], &l[i * n..i * n + i], before),
                    Block::Real(l) => dot_real(&mut at[..r], &l[i * n..i * n + i], before),
                }
            }
        }
        let mut t = w.split_off(n * r);
        let tail = m - n;
        if tail > 0 {
            // Rows of the update are independent: `t -= L21 y`, a block of
            // factor rows per task.
            let step = if n * tail >= SPLIT { (SPLIT / n).max(64) } else { tail };
            let update = |b: usize, part: &mut [C]| {
                let (i0, count) = (n + b * step, part.len() / r);
                match block {
                    Block::Complex(l) if gemm(r, n * count) => {
                        dense::gemm_nn(count, r, n, &l[i0 * n..], n, &y, r, part, r)
                    }
                    Block::Complex(l) => {
                        for (i, acc) in part.chunks_exact_mut(r).enumerate() {
                            dot(acc, &l[(i0 + i) * n..(i0 + i + 1) * n], &y);
                        }
                    }
                    // Real `L` times the complex block as `WIDE r` real columns.
                    Block::Real(l) if gemm(WIDE * r, n * count) => {
                        let w = WIDE * r;
                        RealHalf::gemm_nn(count, w, n, &l[i0 * n..], n, reals(&y), w, reals_mut(part), w)
                    }
                    Block::Real(l) => {
                        for (i, acc) in part.chunks_exact_mut(r).enumerate() {
                            dot_real(acc, &l[(i0 + i) * n..(i0 + i + 1) * n], &y);
                        }
                    }
                }
            };
            if step < tail {
                t.par_chunks_mut(step * r).enumerate().for_each(|(b, part)| update(b, part));
            } else {
                update(0, &mut t);
            }
        }
        (y, t)
    }

    /// `D z = y` for every LDL^T pivot, 1x1 and 2x2 blocks alike.
    ///
    /// # Safety
    /// As for [`Solver::forward`]. Each node writes only its own pivot rows.
    unsafe fn diagonal_solve(&self, x: Rows, r: usize) {
        self.plan.nodes.par_iter().with_min_len(64).for_each(|node| {
            let first = node.first as usize;
            let rows = &self.plan.rows[node.rows as usize..];
            let global = |k: usize| rows[self.perms[first + k]] as usize;
            let mut k = 0;
            while k < node.nc as usize {
                let xi = x.row(global(k), r);
                if self.two[first + k] {
                    let xj = x.row(global(k + 1), r);
                    let (a, b, c, s, det) = block(
                        self.diagonal[first + k],
                        self.subdiag[first + k],
                        self.diagonal[first + k + 1],
                    );
                    for (vi, vj) in xi.iter_mut().zip(xj.iter_mut()) {
                        let (u, v) = (*vi / s, *vj / s);
                        *vi = div(u * c - v * b, det);
                        *vj = div(v * a - u * b, det);
                    }
                    k += 2;
                } else {
                    let d = self.diagonal[first + k];
                    for v in xi.iter_mut() {
                        *v = div(*v, d);
                    }
                    k += 1;
                }
            }
        });
    }

    /// `L^T x = z` on LDL^T, `U x = y` on LU, root first.
    ///
    /// # Safety
    /// As for [`Solver::forward`]. A node reads its trailing rows, which belong
    /// to its ancestors and are final by the time its level runs, and writes
    /// only its own pivot rows.
    unsafe fn backward(&self, x: Rows, r: usize) {
        for level in self.plan.levels.iter().rev() {
            let alone = level.len() == 1;
            level.par_iter().for_each(|&id| self.backward_node(id as usize, x, r, alone));
        }
    }

    unsafe fn backward_node(&self, id: usize, x: Rows, r: usize, alone: bool) {
        let plan = &self.plan;
        let node = plan.nodes[id];
        let (m, n, first) = (node.nr as usize, node.nc as usize, node.first as usize);
        let rows = &plan.rows[node.rows as usize..node.rows as usize + m];
        let perm = &self.perms[first..first + n];
        let lu = plan.kind == Kind::Lu;
        let block = self.block(id);
        let tail = m - n;
        let own = |k: usize| (if lu { rows[k] } else { rows[perm[k]] }) as usize;
        // The node's pivots then its trailing rows, in front order, so one
        // contiguous factor column meets one contiguous run of values.
        let mut v = vec![C::from_re(0.0); m * r];
        for (k, slot) in v[..n * r].chunks_exact_mut(r).enumerate() {
            slot.copy_from_slice(x.read(own(k), r));
        }
        for (j, slot) in v[n * r..].chunks_exact_mut(r).enumerate() {
            slot.copy_from_slice(x.read(rows[n + j] as usize, r));
        }
        let (head, z) = v.split_at_mut(n * r);
        let z: &[C] = z;
        if tail > 0 {
            if lu {
                // `head -= U12 z`, U12 row-major: one row of U per pivot.
                let u = &self.upper[node.uoffset as usize..];
                let step = if n * tail >= SPLIT { (SPLIT / tail).max(16) } else { n };
                let update = |b: usize, part: &mut [C]| {
                    let (k0, count) = (b * step, part.len() / r);
                    if gemm(r, count * tail) {
                        dense::gemm_nn(count, r, tail, &u[k0 * tail..], tail, z, r, part, r);
                    } else {
                        for (i, acc) in part.chunks_exact_mut(r).enumerate() {
                            dot(acc, &u[(k0 + i) * tail..(k0 + i + 1) * tail], z);
                        }
                    }
                };
                if step < n {
                    head.par_chunks_mut(step * r).enumerate().for_each(|(b, part)| update(b, part));
                } else {
                    update(0, head);
                }
            } else if let Block::Real(l) = block {
                // `head -= L21^T z` with real `L21`: the complex blocks as
                // `WIDE r` real columns, and the transpose left to GEMM.
                let step = if n * tail >= SPLIT { (SPLIT / tail).max(64) } else { n };
                let w = WIDE * r;
                let update = |b: usize, part: &mut [C]| {
                    let (k0, count) = (b * step, part.len() / r);
                    if gemm(w, count * tail) {
                        RealHalf::gemm_tn(count, w, tail, &l[n * n + k0..], n, reals(z), w, reals_mut(part), w);
                    } else {
                        for j in 0..tail {
                            let row = &l[(n + j) * n + k0..(n + j) * n + k0 + count];
                            axpy_real(part, row, &z[j * r..(j + 1) * r]);
                        }
                    }
                };
                if step < n {
                    head.par_chunks_mut(step * r).enumerate().for_each(|(b, part)| update(b, part));
                } else {
                    update(0, head);
                }
            } else if let Block::Complex(l) = block {
                // `head -= L21^T z`: every trailing factor row scales into the
                // pivots, a block of pivot columns per task. GEMM wants the
                // right-hand sides as rows, `head^T -= z^T L21`.
                let step = if n * tail >= SPLIT { (SPLIT / tail).max(64) } else { n };
                let zt: Vec<C> = if gemm(r, tail * step.min(n)) {
                    (0..r * tail).map(|e| z[(e % tail) * r + e / tail]).collect()
                } else {
                    Vec::new()
                };
                let update = |b: usize, part: &mut [C]| {
                    let (k0, count) = (b * step, part.len() / r);
                    if gemm(r, count * tail) {
                        let mut ht: Vec<C> = (0..r * count).map(|e| part[(e % count) * r + e / count]).collect();
                        dense::gemm_nn(r, count, tail, &zt, tail, &l[n * n + k0..], n, &mut ht, count);
                        for (e, &v) in ht.iter().enumerate() {
                            part[(e % count) * r + e / count] = v;
                        }
                    } else {
                        for j in 0..tail {
                            let row = &l[(n + j) * n + k0..(n + j) * n + k0 + count];
                            axpy(part, row, &z[j * r..(j + 1) * r]);
                        }
                    }
                };
                if step < n {
                    head.par_chunks_mut(step * r).enumerate().for_each(|(b, part)| update(b, part));
                } else {
                    update(0, head);
                }
            }
        }
        if gemm(r, n * n) {
            // The blocked solve, whose work is mostly GEMM. LU: `head =
            // U11^-1 head`, U11 above L11's diagonal. LDL^T: `L11^T x = head`
            // is `x^T L11 = head^T`, solved on a transposed copy.
            self.solve_diagonal(block, n, r, alone, |l11, threads| {
                with_threading(threads, || {
                    if lu {
                        dense::trsm(Side::Left, Tri::Upper, n, r, l11, n, head, r);
                    } else {
                        let mut ht: Vec<C> = (0..r * n).map(|e| head[(e % n) * r + e / n]).collect();
                        dense::trsm(Side::Right, Tri::UnitLower, r, n, l11, n, &mut ht, n);
                        for (e, &v) in ht.iter().enumerate() {
                            head[(e % n) * r + e / n] = v;
                        }
                    }
                })
            });
        } else if lu {
            let Block::Complex(l) = block else {
                unreachable!("only LDL^T has a real half")
            };
            // U11 sits above L11's diagonal, row `k` of U in factor row `k`.
            for k in (0..n).rev() {
                let (upto, after) = head.split_at_mut((k + 1) * r);
                dot(&mut upto[k * r..], &l[k * n + k + 1..k * n + n], after);
                let d = l[k * n + k];
                for value in &mut upto[k * r..] {
                    *value = div(*value, d);
                }
            }
        } else {
            // `L11^T`: once pivot `j` is final, factor row `j` scales into
            // every earlier pivot.
            for j in (1..n).rev() {
                let (before, from) = head.split_at_mut(j * r);
                match block {
                    Block::Complex(l) => axpy(before, &l[j * n..j * n + j], &from[..r]),
                    Block::Real(l) => axpy_real(before, &l[j * n..j * n + j], &from[..r]),
                }
            }
        }
        for (k, value) in head.chunks_exact(r).enumerate() {
            x.row(own(k), r).copy_from_slice(value);
        }
    }
}
