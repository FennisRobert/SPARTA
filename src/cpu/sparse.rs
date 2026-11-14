//! Solves with sparse right-hand sides, and projections `W^T A^-1 B`.
//!
//! With `P^T A P = L D L^T`, the forward sweep `L y = P^T b` of a right-hand
//! side whose nonzeros sit in a few unknowns only ever reaches the fronts on
//! the paths from those unknowns' fronts to the root: everywhere else `y` is
//! zero. So a sparse right-hand side is swept over that reach alone, with its
//! values held for the reach's pivot rows only.
//!
//! A projection needs nothing more. `W^T A^-1 B = (L^-1 P^T W)^T D^-1 (L^-1
//! P^T B)` for the plain transpose, since `A` is complex symmetric: two
//! restricted forward sweeps, no backward sweep, and no `n`-row block at all.
//! Each front's share of the product is one small matrix product over the
//! fronts both reaches share.
//!
//! The elimination itself is [`Solver::eliminate`], the full solve's own, so
//! real fronts of the real split, 2x2 pivots and the dense library are
//! handled exactly as there. `LDL^T` only: LU solves take the dense route in
//! `crate::Solver`.
use super::solve::{gemm, Rows};
use super::{block, div, finite, Scalar, Solver, C};
use crate::{Error, Kind, Result};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Right-hand sides swept together. Each column's reach is a few paths up
/// the tree, so columns are grouped by where they start, and a group shares
/// most of its reach; eight keeps a group's rows wide enough for the dense
/// library without letting unrelated columns widen each other's reach.
const GROUP: usize = 8;

/// Sparse columns: column `j` has rows `rows[ptr[j]..ptr[j + 1]]` of the
/// caller's numbering, with `values` alongside. Rows may repeat, and repeats
/// add up. Checked by `crate::SparseColumns`.
#[derive(Clone, Copy)]
pub(crate) struct Columns<'a> {
    pub ptr: &'a [usize],
    pub rows: &'a [usize],
    pub values: &'a [C],
}

impl Columns<'_> {
    fn count(&self) -> usize {
        self.ptr.len() - 1
    }
}

/// Where each permuted unknown is a pivot: its node and its front position.
struct Owners {
    node: Vec<u32>,
    position: Vec<u32>,
    /// The caller's index to the permuted one.
    inverse: Vec<u32>,
    /// Each node's parent, or `u32::MAX` at a root.
    parent: Vec<u32>,
    /// Each node's level, the order the forward sweep visits them in.
    level: Vec<u32>,
}

/// One group of columns swept over its reach.
struct Swept {
    /// The caller's indices of the group's columns.
    columns: Vec<usize>,
    /// The reach in node order, with each node's first row in `y`.
    reach: Vec<(u32, usize)>,
    /// The reach's pivot rows, `columns.len()` values each, by node and then
    /// by front position: the slice of `L^-1 P^T B` that is not zero.
    y: Vec<C>,
}

impl Swept {
    fn width(&self) -> usize {
        self.columns.len()
    }
}

impl Solver {
    /// `A^-1 B`, its `n` rows for each column of `b`, end to end.
    ///
    /// Arguments:
    /// - `b`: the right-hand sides, checked by the caller.
    pub(crate) fn solve_sparse(&self, b: Columns<'_>) -> Result<Vec<C>> {
        self.pool.run(|| {
            self.check_sparse()?;
            let (n, p) = (self.plan.n, b.count());
            let owners = self.owners();
            let groups = self.with_caller_blas(true, || self.sweep_all(&owners, b));
            // Scatter the reaches into the permuted block, then finish the
            // solve as `solve_many` does: everything off the reach is zero.
            let mut x = vec![C::from_re(0.0); n * p];
            for group in &groups {
                let width = group.width();
                for &(id, at) in &group.reach {
                    let node = self.plan.nodes[id as usize];
                    let rows = &self.plan.rows[node.rows as usize..node.rows as usize + node.nc as usize];
                    for (s, &g) in rows.iter().enumerate() {
                        let g = g as usize;
                        for (j, &c) in group.columns.iter().enumerate() {
                            x[g * p + c] = group.y[(at + s) * width + j];
                        }
                    }
                }
            }
            self.with_caller_blas(gemm(p, usize::MAX), || {
                // SAFETY: the sweeps keep to the discipline `Rows` states, and
                // `x` holds `n * p` values that nothing else touches.
                unsafe { self.finish_sweeps(Rows(x.as_mut_ptr()), p) }
            });
            self.unpermute(&x, p)
        })
    }

    /// `W^T A^-1 B`, `w.count()` rows by `b.count()` columns, column-major.
    ///
    /// Arguments:
    /// - `b`: the right-hand sides.
    /// - `w`: the columns to project on, or `None` for `b` itself. With
    ///   `None` the result is exactly symmetric: one triangle is formed and
    ///   mirrored.
    pub(crate) fn project(&self, b: Columns<'_>, w: Option<Columns<'_>>) -> Result<Vec<C>> {
        self.pool.run(|| {
            self.check_sparse()?;
            let owners = self.owners();
            let (right, left) = self.with_caller_blas(true, || {
                let right = self.sweep_all(&owners, b);
                let left = w.map(|w| self.sweep_all(&owners, w));
                (right, left)
            });
            // `D^-1` on the right-hand reaches; the left ones stay `L^-1 W`.
            let scaled: Vec<Vec<C>> = right.par_iter().map(|g| self.invert(g)).collect();
            let symmetric = left.is_none();
            let left = left.as_ref().unwrap_or(&right);
            let (q, p) = (w.map_or(b.count(), |w| w.count()), b.count());
            let pairs: Vec<(usize, usize)> = (0..left.len())
                .flat_map(|i| (0..right.len()).map(move |j| (i, j)))
                .filter(|&(i, j)| !symmetric || i >= j)
                .collect();
            let blocks: Vec<Vec<C>> = self.with_caller_blas(true, || {
                pairs.par_iter().map(|&(i, j)| self.product(&left[i], &right[j], &scaled[j])).collect()
            });
            let mut z = vec![C::from_re(0.0); q * p];
            for (&(i, j), block) in pairs.iter().zip(&blocks) {
                let (rows, cols) = (&left[i].columns, &right[j].columns);
                for (b, &c) in cols.iter().enumerate() {
                    for (a, &r) in rows.iter().enumerate() {
                        let v = block[a * cols.len() + b];
                        z[c * q + r] = v;
                        // One triangle of block pairs is formed; each value is
                        // written to both places, so `Z` is exactly symmetric.
                        if symmetric {
                            z[r * q + c] = v;
                        }
                    }
                }
            }
            if z.par_iter().any(|&v| !finite(v)) {
                return Err(Error::Invalid("nonfinite CPU projection result".into()));
            }
            Ok(z)
        })
    }

    /// Refuse anything but valid `LDL^T` factors.
    fn check_sparse(&self) -> Result<()> {
        if !self.valid {
            return Err(Error::Invalid("successful factorization required before solve".into()));
        }
        if self.plan.kind != Kind::Ldlt {
            return Err(Error::Unsupported("sparse right-hand sides take the dense route on LU".into()));
        }
        Ok(())
    }

    fn owners(&self) -> Owners {
        let plan = &self.plan;
        let n = plan.n;
        let (mut node, mut position) = (vec![u32::MAX; n], vec![0u32; n]);
        let mut parent = vec![u32::MAX; plan.nodes.len()];
        for (id, d) in plan.nodes.iter().enumerate() {
            for p in 0..d.nc as usize {
                let g = plan.rows[d.rows as usize + p] as usize;
                node[g] = id as u32;
                position[g] = p as u32;
            }
            for child in &plan.children[d.children as usize..(d.children + d.count) as usize] {
                parent[child.node as usize] = id as u32;
            }
        }
        let mut inverse = vec![0u32; n];
        for (k, &old) in plan.perm.iter().enumerate() {
            inverse[old as usize] = k as u32;
        }
        let mut level = vec![0u32; plan.nodes.len()];
        for (l, nodes) in plan.levels.iter().enumerate() {
            for &id in nodes {
                level[id as usize] = l as u32;
            }
        }
        Owners { node, position, inverse, parent, level }
    }

    /// Group `b`'s columns by where their reaches start and sweep every group.
    fn sweep_all(&self, owners: &Owners, b: Columns<'_>) -> Vec<Swept> {
        // A column's first permuted unknown places it in the tree: nested
        // dissection numbers each subtree contiguously, so neighbours in this
        // order share most of their paths to the root.
        let mut order: Vec<(u32, usize)> = (0..b.count())
            .map(|c| {
                let start = b.rows[b.ptr[c]..b.ptr[c + 1]]
                    .iter()
                    .map(|&i| owners.inverse[i])
                    .min()
                    .unwrap_or(u32::MAX);
                (start, c)
            })
            .collect();
        order.sort_unstable();
        let groups: Vec<Vec<usize>> = order.chunks(GROUP).map(|g| g.iter().map(|&(_, c)| c).collect()).collect();
        groups.into_par_iter().map(|columns| self.sweep(owners, b, columns)).collect()
    }

    /// The forward sweep of `columns` of `b` over their reach.
    fn sweep(&self, owners: &Owners, b: Columns<'_>, columns: Vec<usize>) -> Swept {
        let plan = &self.plan;
        let r = columns.len();
        // The reach: every node holding a nonzero, and all its ancestors.
        let mut seen: HashSet<u32> = HashSet::new();
        for &c in &columns {
            for &i in &b.rows[b.ptr[c]..b.ptr[c + 1]] {
                let mut id = owners.node[owners.inverse[i] as usize];
                while id != u32::MAX && seen.insert(id) {
                    id = owners.parent[id as usize];
                }
            }
        }
        let mut reach: Vec<u32> = seen.into_iter().collect();
        // Children before parents, as the full sweep's levels.
        reach.sort_unstable_by_key(|&id| (owners.level[id as usize], id));
        let mut at = HashMap::with_capacity(reach.len());
        let mut rows = 0;
        for &id in &reach {
            at.insert(id, rows);
            rows += plan.nodes[id as usize].nc as usize;
        }
        // Each pivot row starts as the matching entry of `B`.
        let zero = C::from_re(0.0);
        let mut y = vec![zero; rows * r];
        for (j, &c) in columns.iter().enumerate() {
            for (&i, &v) in b.rows[b.ptr[c]..b.ptr[c + 1]].iter().zip(&b.values[b.ptr[c]..b.ptr[c + 1]]) {
                let g = owners.inverse[i] as usize;
                let row = at[&owners.node[g]] + owners.position[g] as usize;
                y[row * r + j] += v;
            }
        }
        // The full sweep's `forward_node`, on the reach: a node gathers its
        // own rows and its children's updates, and hands its own on. A child
        // outside the reach has no update, since its right-hand side is zero.
        let mut tails: HashMap<u32, Vec<C>> = HashMap::new();
        for &id in &reach {
            let node = plan.nodes[id as usize];
            let (m, n) = (node.nr as usize, node.nc as usize);
            let own = at[&id];
            let mut w = vec![zero; m * r];
            w[..n * r].copy_from_slice(&y[own * r..(own + n) * r]);
            for child in &plan.children[node.children as usize..(node.children + node.count) as usize] {
                let Some(tail) = tails.remove(&child.node) else { continue };
                let cn = plan.nodes[child.node as usize];
                let map = &plan.maps[child.map as usize..child.map as usize + (cn.nr - cn.nc) as usize];
                for (&p, update) in map.iter().zip(tail.chunks_exact(r)) {
                    for (slot, &v) in w[p as usize * r..p as usize * r + r].iter_mut().zip(update) {
                        *slot += v;
                    }
                }
            }
            let (solved, tail) = self.eliminate(id as usize, w, r);
            let perm = &self.perms[node.first as usize..node.first as usize + n];
            for (k, value) in solved.chunks_exact(r).enumerate() {
                let row = own + perm[k];
                y[row * r..row * r + r].copy_from_slice(value);
            }
            if m > n {
                tails.insert(id, tail);
            }
        }
        let mut reach: Vec<(u32, usize)> = reach.iter().map(|&id| (id, at[&id])).collect();
        reach.sort_unstable();
        Swept { columns, reach, y }
    }

    /// `D^-1` applied to a group's reach, 1x1 and 2x2 pivots alike, as
    /// `diagonal_solve` applies it to a whole solve.
    fn invert(&self, group: &Swept) -> Vec<C> {
        let r = group.width();
        let mut z = group.y.clone();
        for &(id, at) in &group.reach {
            let node = self.plan.nodes[id as usize];
            let first = node.first as usize;
            let slot = |k: usize| (at + self.perms[first + k]) * r;
            let mut k = 0;
            while k < node.nc as usize {
                if self.two[first + k] {
                    let (a, b, c, s, det) = block(
                        self.diagonal[first + k],
                        self.subdiag[first + k],
                        self.diagonal[first + k + 1],
                    );
                    let (i, j) = (slot(k), slot(k + 1));
                    for e in 0..r {
                        let (u, v) = (z[i + e] / s, z[j + e] / s);
                        z[i + e] = div(u * c - v * b, det);
                        z[j + e] = div(v * a - u * b, det);
                    }
                    k += 2;
                } else {
                    let (i, d) = (slot(k), self.diagonal[first + k]);
                    for value in &mut z[i..i + r] {
                        *value = div(*value, d);
                    }
                    k += 1;
                }
            }
        }
        z
    }

    /// `Y_left^T (D^-1 Y_right)` over the nodes both reaches share,
    /// `left.width()` by `right.width()`, row-major.
    fn product(&self, left: &Swept, right: &Swept, scaled: &[C]) -> Vec<C> {
        let (a, b) = (left.width(), right.width());
        // Accumulated negated: the dense library's product subtracts.
        let mut acc = vec![C::from_re(0.0); a * b];
        let (mut i, mut j) = (0, 0);
        while i < left.reach.len() && j < right.reach.len() {
            let ((li, la), (ri, ra)) = (left.reach[i], right.reach[j]);
            if li != ri {
                if li < ri { i += 1 } else { j += 1 }
                continue;
            }
            let n = self.plan.nodes[li as usize].nc as usize;
            let (yl, yr) = (&left.y[la * a..(la + n) * a], &scaled[ra * b..(ra + n) * b]);
            if gemm(a.min(b), n * a.max(b)) {
                super::dense::gemm_tn(a, b, n, yl, a, yr, b, &mut acc, b);
            } else {
                for s in 0..n {
                    for (u, &l) in yl[s * a..(s + 1) * a].iter().enumerate() {
                        for (v, &x) in yr[s * b..(s + 1) * b].iter().enumerate() {
                            acc[u * b + v] -= l * x;
                        }
                    }
                }
            }
            i += 1;
            j += 1;
        }
        acc.iter_mut().for_each(|v| *v = -*v);
        acc
    }
}
