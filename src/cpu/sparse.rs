//! Sparse right-hand sides on the `LDL^T` factor: the port matrix
//! `Z = B^T A^{-1} B`, and `A X = B` for a sparse `B`.
//!
//! With `P^T A P = L D L^T` (`P` the fill-reducing order composed with each
//! front's pivot order), `Z = Y^T D^{-1} Y` for `Y = L^{-1} P^T B`, plain
//! transpose throughout. A column of `Y` is zero outside the fronts on the
//! paths from its nonzeros' fronts to the root (its reach), so the forward
//! sweep visits only those fronts, and `Z` needs no backward sweep at all.
//! This is the classical sparse right-hand side argument (Gilbert and Peierls)
//! on the assembly tree.
//!
//! Columns are swept a block at a time, grouped by the first front they
//! reach. Each block keeps `Y` only on the fronts it reaches, in pivot order.
//! Large fronts go to the dense library (a triangular solve and a GEMM, as in
//! the full solve), and `Z` is formed block pair by block pair with one GEMM
//! per run of shared fronts. Only one triangle of `Z` is computed and it is
//! mirrored, so `Z` is exactly symmetric; like the other solves, its rounding
//! is not promised to be the same across thread counts or blockings.
use super::dense::{self, Side, Threading, Tri};
use super::solve::gemm;
use super::Scalar;
#[cfg(test)]
use super::R;
use super::{Solver, C};
use crate::sparse_rhs::{reach_of, Tree};
use crate::{Error, Kind, Result, SparseReach};
use rayon::prelude::*;
use std::cmp::Ordering;

/// Right-hand sides swept together through one pass over their reach.
pub(crate) const PORT_BLOCK: usize = 8;

/// Multiply-adds in a front's trailing product before its rows are split
/// across threads.
const SPLIT: usize = 1 << 16;

/// Factor elements of a front's trailing GEMM before its rows are split.
const SPLIT_GEMM: usize = 1 << 18;

/// A lane count `gemm` accepts, for asking whether the dense library is there.
const GEMM_RHS_MIN: usize = 4;

/// The columns of a sparse `B`: offsets, rows in the matrix's own numbering,
/// and values in this copy's type.
#[derive(Clone, Copy)]
pub(crate) struct Columns<'a> {
    pub ptr: &'a [usize],
    pub rows: &'a [usize],
    pub values: &'a [C],
}

impl Columns<'_> {
    fn ncols(&self) -> usize {
        self.ptr.len() - 1
    }
}

/// One block's forward sweep, kept only on the fronts it reaches.
struct Forward {
    /// The block's columns of `B`, one lane each.
    cols: Vec<usize>,
    /// Fronts reached, ascending, so children before parents.
    fronts: Vec<u32>,
    /// Where each front's pivots start in `y` and `v`; one past the end last.
    at: Vec<usize>,
    /// `Y` on the reach: each front's pivots in pivot order, lanes fastest.
    y: Vec<C>,
    /// `D^{-1} Y`, laid out as `y`. Formed for the port matrix only.
    v: Vec<C>,
    /// `Y` and `V` transposed, one row of pivots per lane, for GEMM products.
    yt: Vec<C>,
    vt: Vec<C>,
}

/// Run `op` with the dense library single-threaded on every thread of the
/// pool, as the solves run it: Rayon owns the parallelism.
fn caller_blas<R>(op: impl FnOnce() -> R) -> R {
    let blas = gemm(GEMM_RHS_MIN, usize::MAX);
    if blas {
        dense::set_global_threading(Threading::Caller);
        rayon::broadcast(|_| dense::set_threading(Threading::Caller));
    }
    let out = op();
    if blas {
        dense::set_global_threading(Threading::Machine);
        rayon::broadcast(|_| dense::set_threading(Threading::Machine));
    }
    out
}

/// `acc[c] -= l * z[c]` for every lane `c`.
#[inline]
fn subtract(acc: &mut [C], l: C, z: &[C]) {
    for (a, &zc) in acc.iter_mut().zip(z) {
        *a -= l * zc;
    }
}

impl Solver {
    /// [`crate::Solver::port_matrix`] on this copy's factors.
    pub(crate) fn port_matrix(&self, b: Columns<'_>) -> Result<Vec<C>> {
        self.pool
            .run(|| caller_blas(|| self.port_matrix_here(b, PORT_BLOCK)))
    }

    /// [`crate::Solver::solve_sparse`] on this copy's factors.
    pub(crate) fn solve_sparse(&self, b: Columns<'_>) -> Result<Vec<C>> {
        self.pool
            .run(|| caller_blas(|| self.solve_sparse_here(b, PORT_BLOCK)))
    }

    /// [`crate::Solver::sparse_reach`] on this copy's plan.
    pub(crate) fn sparse_reach(&self, rows: &[usize]) -> Result<SparseReach> {
        if self.destroyed {
            return Err(Error::Invalid("the solver was destroyed".into()));
        }
        reach_of(&self.plan, rows)
    }

    /// Refuse anything but valid row-major `LDL^T` factors whose 2x2 pivots
    /// all lie inside one front.
    fn check_sparse(&self, b: Columns<'_>) -> Result<()> {
        if !self.valid {
            return Err(Error::Invalid(
                "successful factorization required before solve".into(),
            ));
        }
        if self.plan.kind != Kind::Ldlt {
            return Err(Error::Unsupported(
                "sparse right-hand sides need an LDL^T factor".into(),
            ));
        }
        if !self.rowmajor {
            return Err(Error::Unsupported(
                "sparse right-hand sides need the CPU solve's factor layout".into(),
            ));
        }
        debug_assert!(b.rows.iter().all(|&row| row < self.plan.n));
        // A front pivots only inside its fully summed block and delays
        // nothing to its parent, so a 2x2 block cannot start at a front's
        // last column. The sweep below takes both rows of a block from one
        // front; were that ever untrue, `Z` would be wrong, so it is checked.
        for (id, node) in self.plan.nodes.iter().enumerate() {
            // The analysis builds no front without pivots; one would have
            // nothing to check.
            if node.nc == 0 {
                continue;
            }
            if self.two[(node.first + node.nc) as usize - 1] {
                return Err(Error::Unsupported(format!(
                    "front {id} has a 2x2 pivot at its last column, which sparse right-hand sides do not support"
                )));
            }
        }
        Ok(())
    }

    /// The nonempty columns of `b` in blocks of at most `width`, grouped by
    /// the first front each reaches. Empty columns are left out.
    fn blocks(&self, tree: &Tree, b: Columns<'_>, width: usize) -> Vec<Vec<usize>> {
        let mut keyed: Vec<(u32, usize)> = (0..b.ncols())
            .filter_map(|c| {
                b.rows[b.ptr[c]..b.ptr[c + 1]]
                    .iter()
                    .map(|&row| tree.owner[tree.iperm[row] as usize])
                    .min()
                    .map(|first| (first, c))
            })
            .collect();
        keyed.sort_unstable();
        keyed
            .chunks(width.max(1))
            .map(|chunk| chunk.iter().map(|&(_, c)| c).collect())
            .collect()
    }

    /// `Y = L^{-1} P^T B` for the columns `cols`, on their reach only.
    ///
    /// The multifrontal forward sweep of `Solver::forward_node`, restricted
    /// to the reach: each front gathers its own entries of `B` and the
    /// trailing updates of those children that are in the reach (the others
    /// are zero), eliminates its pivots and hands its update to its parent.
    fn forward_block(&self, tree: &Tree, b: Columns<'_>, cols: Vec<usize>) -> Forward {
        let plan = &self.plan;
        let r = cols.len();
        let zero = C::from_re(0.0);
        // (front, position in the front, lane, value), by front. The sort is
        // stable, so a row listed twice is summed in the order given.
        let mut entries: Vec<(u32, u32, u32, C)> = Vec::new();
        for (lane, &c) in cols.iter().enumerate() {
            for e in b.ptr[c]..b.ptr[c + 1] {
                let g = tree.iperm[b.rows[e]];
                let s = tree.owner[g as usize];
                entries.push((
                    s,
                    g - plan.nodes[s as usize].first,
                    lane as u32,
                    b.values[e],
                ));
            }
        }
        entries.sort_by_key(|e| e.0);
        let mut mark = vec![false; plan.nodes.len()];
        let fronts = tree.reach(entries.iter().map(|e| e.0), &mut mark);
        let mut at = Vec::with_capacity(fronts.len() + 1);
        let mut total = 0;
        for &s in &fronts {
            at.push(total);
            total += plan.nodes[s as usize].nc as usize * r;
        }
        at.push(total);
        let mut y = vec![zero; total];
        let mut tails: Vec<Vec<C>> = vec![Vec::new(); fronts.len()];
        let mut next = 0;
        for (i, &s) in fronts.iter().enumerate() {
            let node = plan.nodes[s as usize];
            let (m, n, first) = (node.nr as usize, node.nc as usize, node.first as usize);
            let perm = &self.perms[first..first + n];
            let l = &self.factors[node.loffset as usize..node.loffset as usize + n * m];
            let mut w = vec![zero; m * r];
            while next < entries.len() && entries[next].0 == s {
                let (_, p, lane, value) = entries[next];
                w[p as usize * r + lane as usize] += value;
                next += 1;
            }
            let children =
                &plan.children[node.children as usize..(node.children + node.count) as usize];
            for child in children {
                let Ok(at_child) = fronts[..i].binary_search(&child.node) else {
                    continue;
                };
                let tail = std::mem::take(&mut tails[at_child]);
                let cn = plan.nodes[child.node as usize];
                let map =
                    &plan.maps[child.map as usize..child.map as usize + (cn.nr - cn.nc) as usize];
                for (&p, update) in map.iter().zip(tail.chunks_exact(r)) {
                    let p = p as usize * r;
                    for (slot, &v) in w[p..p + r].iter_mut().zip(update) {
                        *slot += v;
                    }
                }
            }
            let ys = &mut y[at[i]..at[i + 1]];
            for (k, slot) in ys.chunks_exact_mut(r).enumerate() {
                slot.copy_from_slice(&w[perm[k] * r..perm[k] * r + r]);
            }
            // L is row-major (`Solver::rowmajor`): row `k` holds `L[k][0..nc]`,
            // unit lower triangular in its first `nc` rows. Large fronts go to
            // the dense library, as in `Solver::forward_node`.
            let blas = gemm(r, n * m);
            if blas {
                dense::trsm(Side::Left, Tri::UnitLower, n, r, l, n, ys, r);
            } else {
                for k in 1..n {
                    let (before, rest) = ys.split_at_mut(k * r);
                    let acc = &mut rest[..r];
                    for (j, &lj) in l[k * n..k * n + k].iter().enumerate() {
                        subtract(acc, lj, &before[j * r..j * r + r]);
                    }
                }
            }
            let tail = m - n;
            if tail == 0 {
                continue;
            }
            if blas {
                // `t -= L21 y`, a block of factor rows per task.
                let ys: &[C] = ys;
                let mut t = w.split_off(n * r);
                let step = if n * tail >= SPLIT_GEMM {
                    (SPLIT_GEMM / n).max(64)
                } else {
                    tail
                };
                let update = |k: usize, part: &mut [C]| {
                    let i0 = n + k * step;
                    dense::gemm_nn(part.len() / r, r, n, &l[i0 * n..], n, ys, r, part, r);
                };
                if step < tail {
                    t.par_chunks_mut(step * r)
                        .enumerate()
                        .for_each(|(k, part)| update(k, part));
                } else {
                    update(0, &mut t);
                }
                tails[i] = t;
                continue;
            }
            // `t -= L21 y`. Pivots whose lanes are all zero contribute
            // nothing, which on the first fronts of a reach is most of them.
            let ys: &[C] = ys;
            let live: Vec<usize> = (0..n)
                .filter(|&j| ys[j * r..j * r + r].iter().any(|&v| v != zero))
                .collect();
            let mut t = w.split_off(n * r);
            let update = |i0: usize, part: &mut [C]| {
                for (i, acc) in part.chunks_exact_mut(r).enumerate() {
                    let row = &l[(n + i0 + i) * n..(n + i0 + i + 1) * n];
                    for &j in &live {
                        subtract(acc, row[j], &ys[j * r..j * r + r]);
                    }
                }
            };
            // Rows are independent and each is computed the same way however
            // they are split, so splitting changes no result.
            let step = (SPLIT / (live.len() * r).max(1)).max(16);
            if step < tail {
                t.par_chunks_mut(step * r)
                    .enumerate()
                    .for_each(|(k, part)| update(k * step, part));
            } else {
                update(0, &mut t);
            }
            tails[i] = t;
        }
        Forward {
            cols,
            fronts,
            at,
            y,
            v: Vec::new(),
            yt: Vec::new(),
            vt: Vec::new(),
        }
    }

    /// `V = D^{-1} Y` on a block's reach. A 2x2 block lies inside one front
    /// (`check_sparse`), so both of its rows are in the reach.
    fn invert_block(&self, f: &mut Forward) {
        let r = f.cols.len();
        let mut v = f.y.clone();
        for (i, &s) in f.fronts.iter().enumerate() {
            let node = self.plan.nodes[s as usize];
            let (n, first) = (node.nc as usize, node.first as usize);
            let part = &mut v[f.at[i]..f.at[i + 1]];
            let mut k = 0;
            while k < n {
                let (xi, rest) = part[k * r..].split_at_mut(r);
                let xj = self.two[first + k].then(|| &mut rest[..r]);
                k += self.invert_pivot(first + k, xi, xj);
            }
        }
        f.v = v;
        if gemm(GEMM_RHS_MIN, usize::MAX) {
            let k = f.y.len() / r.max(1);
            let transpose = |x: &[C]| -> Vec<C> {
                let mut out = vec![C::from_re(0.0); x.len()];
                for (p, lanes) in x.chunks_exact(r).enumerate() {
                    for (c, &value) in lanes.iter().enumerate() {
                        out[c * k + p] = value;
                    }
                }
                out
            };
            f.yt = transpose(&f.y);
            f.vt = transpose(&f.v);
        }
    }

    /// `Z` between two blocks, `Y_a^T V_b` over the fronts both reach: entry
    /// `(i, j)` is `Z[a.cols[i], b.cols[j]]`, row-major. The other fronts hold
    /// a zero `Y` for one block or the other and add nothing. One GEMM per run
    /// of fronts that are consecutive in both blocks.
    fn product_gemm(&self, a: &Forward, b: &Forward) -> Vec<C> {
        let (ra, rb) = (a.cols.len(), b.cols.len());
        let (ka, kb) = (a.y.len() / ra, b.y.len() / rb);
        let mut acc = vec![C::from_re(0.0); ra * rb];
        let (mut p, mut q) = (0, 0);
        while p < a.fronts.len() && q < b.fronts.len() {
            match a.fronts[p].cmp(&b.fronts[q]) {
                Ordering::Less => p += 1,
                Ordering::Greater => q += 1,
                Ordering::Equal => {
                    let (p0, q0) = (p, q);
                    while p < a.fronts.len() && q < b.fronts.len() && a.fronts[p] == b.fronts[q] {
                        p += 1;
                        q += 1;
                    }
                    let (sa, sb) = (a.at[p0] / ra, b.at[q0] / rb);
                    let k = a.at[p] / ra - sa;
                    dense::gemm_nt(ra, rb, k, &a.yt[sa..], ka, &b.vt[sb..], kb, &mut acc, rb);
                }
            }
        }
        // `gemm_nt` subtracts.
        for v in &mut acc {
            *v = -*v;
        }
        acc
    }

    /// [`Solver::product_gemm`] in plain loops, for builds without the dense
    /// library.
    fn product(&self, a: &Forward, b: &Forward) -> Vec<C> {
        let (ra, rb) = (a.cols.len(), b.cols.len());
        let mut acc = vec![C::from_re(0.0); ra * rb];
        let (mut p, mut q) = (0, 0);
        while p < a.fronts.len() && q < b.fronts.len() {
            match a.fronts[p].cmp(&b.fronts[q]) {
                Ordering::Less => p += 1,
                Ordering::Greater => q += 1,
                Ordering::Equal => {
                    let ya = &a.y[a.at[p]..a.at[p + 1]];
                    let vb = &b.v[b.at[q]..b.at[q + 1]];
                    for (yak, vbk) in ya.chunks_exact(ra).zip(vb.chunks_exact(rb)) {
                        for (row, &yi) in acc.chunks_exact_mut(rb).zip(yak) {
                            for (z, &vj) in row.iter_mut().zip(vbk) {
                                *z += yi * vj;
                            }
                        }
                    }
                    p += 1;
                    q += 1;
                }
            }
        }
        acc
    }

    /// The port matrix with columns swept `width` at a time.
    pub(super) fn port_matrix_here(&self, b: Columns<'_>, width: usize) -> Result<Vec<C>> {
        self.check_sparse(b)?;
        let p = b.ncols();
        let tree = Tree::new(&self.plan);
        let blocks = self.blocks(&tree, b, width);
        let forwards: Vec<Forward> = blocks
            .into_par_iter()
            .map(|cols| {
                let mut f = self.forward_block(&tree, b, cols);
                self.invert_block(&mut f);
                f
            })
            .collect();
        let count = forwards.len();
        let pairs: Vec<(usize, usize)> = (0..count)
            .flat_map(|i| (i..count).map(move |j| (i, j)))
            .collect();
        let gemm_product = gemm(GEMM_RHS_MIN, usize::MAX);
        let products: Vec<Vec<C>> = pairs
            .par_iter()
            .map(|&(i, j)| {
                if gemm_product {
                    self.product_gemm(&forwards[i], &forwards[j])
                } else {
                    self.product(&forwards[i], &forwards[j])
                }
            })
            .collect();
        let mut z = vec![C::from_re(0.0); p * p];
        for (&(i, j), acc) in pairs.iter().zip(&products) {
            let rb = forwards[j].cols.len();
            for (x, &ca) in forwards[i].cols.iter().enumerate() {
                for (y, &cb) in forwards[j].cols.iter().enumerate() {
                    // One triangle of a block with itself, mirrored.
                    if i == j && y < x {
                        continue;
                    }
                    let value = acc[x * rb + y];
                    z[ca + cb * p] = value;
                    z[cb + ca * p] = value;
                }
            }
        }
        if z.par_iter().any(|v| !v.finite()) {
            return Err(Error::Invalid("nonfinite port matrix".into()));
        }
        Ok(z)
    }

    /// `A X = B` with the forward sweep limited to the columns' reach, then
    /// the full diagonal and backward sweeps.
    pub(super) fn solve_sparse_here(&self, b: Columns<'_>, width: usize) -> Result<Vec<C>> {
        self.check_sparse(b)?;
        let (n, p) = (self.plan.n, b.ncols());
        if p == 0 {
            return Ok(Vec::new());
        }
        let tree = Tree::new(&self.plan);
        let forwards: Vec<Forward> = self
            .blocks(&tree, b, width)
            .into_par_iter()
            .map(|cols| self.forward_block(&tree, b, cols))
            .collect();
        // Permuted, `p` values per unknown, as the sweeps hold them; zero
        // outside each column's reach.
        let mut x = vec![C::from_re(0.0); n * p];
        for f in &forwards {
            let r = f.cols.len();
            for (i, &s) in f.fronts.iter().enumerate() {
                let node = self.plan.nodes[s as usize];
                let (first, rows) = (node.first as usize, &self.plan.rows[node.rows as usize..]);
                let y = &f.y[f.at[i]..f.at[i + 1]];
                for (k, lanes) in y.chunks_exact(r).enumerate() {
                    let g = rows[self.perms[first + k]] as usize;
                    for (&c, &value) in f.cols.iter().zip(lanes) {
                        x[g * p + c] = value;
                    }
                }
            }
        }
        self.sweeps(&mut x, p, false);
        self.unpermute(&x, p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Analysis, Config};

    fn assert_symmetric(z: &[C], p: usize) {
        for i in 0..p {
            for j in 0..p {
                assert!(z[i + j * p] == z[j + i * p], "Z[{i},{j}] is not its mirror");
            }
        }
    }

    /// Agreement to rounding in this copy's precision.
    fn tol() -> R {
        1e3 * R::EPSILON
    }

    /// A complex symmetric 2D grid Laplacian shifted to be indefinite, as CSC
    /// lower triangle, and a CPU solver factored on it.
    fn grid(side: usize, config: &Config) -> (Solver, usize) {
        let n = side * side;
        let (mut ptr, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
        for j in 0..n {
            rows.push(j);
            values.push(C::from_c64(num_complex::Complex64::new(3.1, 0.05)));
            for i in [j + 1, j + side] {
                if i < n && (i != j + 1 || i % side != 0) {
                    rows.push(i);
                    values.push(C::from_c64(num_complex::Complex64::new(-1.0, 0.0)));
                }
            }
            ptr.push(rows.len());
        }
        let plan = Analysis::ldlt(n, &ptr, &rows, config).unwrap();
        let mut s = Solver::new(plan).unwrap();
        s.factor(&values).unwrap();
        (s, n)
    }

    #[test]
    /// The reach holds every nonzero of the full forward sweep, and the
    /// compact sweep equals it to rounding.
    fn reach_holds_the_forward_sweep() {
        let config = Config {
            force_cpu: true,
            nemin: 2,
            small_front_rows: 1,
            ..Config::default()
        };
        let (s, n) = grid(14, &config);
        let tree = Tree::new(&s.plan);
        for column in [vec![0usize], vec![n / 2, n / 2 + 1], vec![3, n - 1, 77]] {
            let values: Vec<C> = column.iter().map(|&i| C::from_re(1.0 + i as R)).collect();
            let ptr = [0, column.len()];
            let b = Columns {
                ptr: &ptr,
                rows: &column,
                values: &values,
            };
            let f = s.forward_block(&tree, b, vec![0]);
            // The full sweep, as `solve_many` runs it.
            let mut x = vec![C::from_re(0.0); n];
            for (&row, &v) in column.iter().zip(&values) {
                x[tree.iperm[row] as usize] += v;
            }
            s.forward_all(&mut x, 1);
            let mut seen = vec![false; n];
            for (i, &id) in f.fronts.iter().enumerate() {
                let node = s.plan.nodes[id as usize];
                let rows = &s.plan.rows[node.rows as usize..];
                for k in 0..node.nc as usize {
                    let g = rows[s.perms[node.first as usize + k]] as usize;
                    seen[g] = true;
                    let (got, want) = (f.y[f.at[i] + k], x[g]);
                    assert!((got - want).norm() <= tol() * (1.0 + want.norm()));
                }
            }
            for g in 0..n {
                assert!(
                    seen[g] || x[g] == C::from_re(0.0),
                    "row {g} outside the reach is nonzero"
                );
            }
        }
    }

    #[test]
    /// Every block width gives the same `Z` and `X` to rounding, and `Z` is
    /// exactly symmetric.
    fn blocking_does_not_change_z() {
        let config = Config {
            force_cpu: true,
            nemin: 4,
            ..Config::default()
        };
        let (s, n) = grid(20, &config);
        let p = 23;
        let (mut ptr, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
        for c in 0..p {
            for e in 0..3 {
                rows.push((c * 97 + e * 31) % n);
                values.push(C::from_re(1.0 + (c + e) as R / 7.0));
            }
            ptr.push(rows.len());
        }
        let b = Columns {
            ptr: &ptr,
            rows: &rows,
            values: &values,
        };
        let reference = s.port_matrix_here(b, 1).unwrap();
        let scale = reference.iter().map(|v| v.norm()).fold(0.0, R::max);
        for width in [1, 2, 3, 8, 64] {
            let z = s.port_matrix_here(b, width).unwrap();
            assert_symmetric(&z, p);
            assert!(
                z.iter()
                    .zip(&reference)
                    .all(|(a, b)| (*a - *b).norm() <= tol() * scale),
                "width {width}"
            );
            let x = s.solve_sparse_here(b, width).unwrap();
            let y = s.solve_sparse_here(b, 1).unwrap();
            assert!(x
                .iter()
                .zip(&y)
                .all(|(a, b)| (*a - *b).norm() <= tol() * (1.0 + b.norm())));
        }
    }

    #[test]
    /// Pairs with a small diagonal and a large coupling force 2x2 pivots,
    /// which the port matrix takes whole; a 2x2 pivot at a front's last
    /// column is refused, not misread.
    fn two_by_two_pivots() {
        let pairs = 40;
        let n = 2 * pairs;
        let (mut ptr, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
        for j in 0..n {
            rows.push(j);
            values.push(C::from_re(0.01));
            if j % 2 == 0 {
                rows.push(j + 1);
                values.push(C::from_re(2.0 + (j % 7) as R));
            }
            if j + 2 < n {
                rows.push(j + 2);
                values.push(C::from_re(0.25));
            }
            ptr.push(rows.len());
        }
        // In one front every pair pivots as a 2x2 block. Over many fronts the
        // ordering can separate a pair, whose rows then pivot 1x1 in their
        // own fronts (no pivot is delayed to a parent).
        for (nemin, whole) in [(n, true), (4, false)] {
            let config = Config {
                force_cpu: true,
                nemin,
                small_front_rows: 1,
                ..Config::default()
            };
            let plan = Analysis::ldlt(n, &ptr, &rows, &config).unwrap();
            let mut s = Solver::new(plan).unwrap();
            s.factor(&values).unwrap();
            assert!(s.two.iter().any(|&t| t), "no 2x2 pivot");
            assert_eq!(s.perturbed, 0);
            assert_eq!(s.plan.nodes.len() == 1, whole);
            // Every unit column: Z is the whole inverse, so each 2x2 block is
            // met with ports on one of its rows and on both.
            let cols: Vec<usize> = (0..n).collect();
            let ptr_b: Vec<usize> = (0..=n).collect();
            let ones = vec![C::from_re(1.0); n];
            let b = Columns {
                ptr: &ptr_b,
                rows: &cols,
                values: &ones,
            };
            let z = s.port_matrix_here(b, PORT_BLOCK).unwrap();
            let mut identity = vec![C::from_re(0.0); n * n];
            for i in 0..n {
                identity[i * n + i] = C::from_re(1.0);
            }
            let x = s.solve_many(&identity, n).unwrap();
            let scale = x.iter().map(|v| v.norm()).fold(0.0, R::max);
            for (got, want) in z.iter().zip(&x) {
                assert!((*got - *want).norm() <= tol() * scale);
            }
            let last = s
                .plan
                .nodes
                .iter()
                .map(|node| (node.first + node.nc) as usize - 1)
                .next()
                .unwrap();
            s.two[last] = true;
            assert!(matches!(
                s.port_matrix_here(b, PORT_BLOCK),
                Err(Error::Unsupported(_))
            ));
            assert!(matches!(
                s.solve_sparse_here(b, PORT_BLOCK),
                Err(Error::Unsupported(_))
            ));
        }
    }

    #[test]
    /// One factorization gives the same `Z` and `X` to rounding on pools of
    /// 1, 2, 4 and 8 threads, and `Z` stays exactly symmetric.
    fn thread_counts_agree() {
        let config = Config {
            force_cpu: true,
            nemin: 8,
            ..Config::default()
        };
        let (s, n) = grid(48, &config);
        let p = 45;
        let (mut ptr, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
        for c in 0..p {
            for e in 0..6 {
                rows.push((c * 389 + e * 13) % n);
                values.push(C::from_re(0.5 + (c * e % 11) as R / 3.0));
            }
            ptr.push(rows.len());
        }
        let b = Columns {
            ptr: &ptr,
            rows: &rows,
            values: &values,
        };
        let runs: Vec<(Vec<C>, Vec<C>)> = [1, 2, 4, 8]
            .iter()
            .map(|&t| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap();
                pool.install(|| (s.port_matrix(b).unwrap(), s.solve_sparse(b).unwrap()))
            })
            .collect();
        let scale = runs[0].0.iter().map(|v| v.norm()).fold(0.0, R::max);
        for (z, x) in &runs[1..] {
            assert_symmetric(z, p);
            assert!(z
                .iter()
                .zip(&runs[0].0)
                .all(|(a, b)| (*a - *b).norm() <= tol() * scale));
            assert!(x
                .iter()
                .zip(&runs[0].1)
                .all(|(a, b)| (*a - *b).norm() <= tol() * (1.0 + b.norm())));
        }
    }
}
