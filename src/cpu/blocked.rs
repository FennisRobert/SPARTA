//! Partial `LDL^T` of one front through Accelerate's blocked kernels.
//!
//! The fully summed block goes to LAPACK's Bunch-Kaufman `zsytrf`, and the
//! whole Schur complement is then one rank-`nc` product. Rank is what decides
//! the speed of the update: measured on an M4 Pro, `zgemm` runs at 170 GFLOP/s
//! at rank 32 and 740 GFLOP/s at rank 800, so deferring the update until the
//! entire fully summed block is eliminated is worth roughly four times the
//! throughput of the per-panel rank-`cpu_panel` update in `panel.rs`.
//!
//! Pivoting is unchanged in reach: `choice` in `factor.rs` already confines the
//! Bunch-Kaufman search to the fully summed rows, which is exactly the block
//! LAPACK sees. When LAPACK reports an exactly singular pivot block the front
//! is handed back untouched and `panel.rs` factors it with its rescue floor.
use super::dense::{self, Side, Threading, Tri};
use rayon::prelude::*;
use super::profile::{Phase, Report};
use super::Scalar;
use super::{block, div, factor::Front, finite, C};
use crate::PivotFailure;

type Failure = (usize, PivotFailure);

/// A front's storage as a bare pointer, for the one kernel whose tiles are not
/// contiguous. Every use is documented at the call site.
#[derive(Clone, Copy)]
struct Cells(*mut C);
// SAFETY: `Cells` carries no access rights of its own; the kernels that
// dereference it argue their own disjointness.
unsafe impl Send for Cells {}
unsafe impl Sync for Cells {}

/// Per-front dense workspace, grown to the largest front a thread has seen.
///
/// One of these belongs to each worker rather than to the solver, so the
/// buffers are sized by the fronts that thread happened to take. The wide
/// fronts live in the top levels, where there are few enough nodes that only a
/// few workers ever grow to the largest shape.
#[derive(Default)]
pub(crate) struct Scratch {
    /// Column-major copy of the fully summed block, order `nc`.
    pub(super) leading: Vec<C>,
    /// LAPACK interchanges for the fully summed block.
    pub(super) ipiv: Vec<i32>,
    /// LAPACK's preferred `zsytrf` workspace, and the order it was sized for.
    work: Vec<C>,
    fitted: usize,
    /// Row-major `(nr - nc)` by `nc` block, holding `A21` then `L21 * D`.
    packed: Vec<C>,
    /// Reciprocal pivots, and the off-diagonal coefficient of a 2x2 block.
    inv: Vec<C>,
    off: Vec<C>,
    /// Fully summed columns that begin a 2x2 pivot block.
    twos: Vec<u32>,
    /// Child row owning each row of the front being assembled, or `u32::MAX`.
    inverse: Vec<u32>,
}

impl Scratch {
    /// Grow every buffer to hold one `rows` by `columns` front.
    fn fit(&mut self, rows: usize, columns: usize) {
        grow(&mut self.leading, columns * columns);
        grow(&mut self.packed, (rows - columns) * columns);
        grow(&mut self.inv, columns);
        grow(&mut self.off, columns);
        if self.ipiv.len() < columns {
            self.ipiv.resize(columns, 0);
        }
        // LAPACK's preferred length is asked for once per new high-water order,
        // not once per front. A short workspace would still factor correctly,
        // on the unblocked path, so this is purely about speed.
        if columns > self.fitted {
            self.fitted = columns;
            grow(&mut self.work, dense::sytrf_workspace(columns));
        }
    }

    /// Borrow a cleared `rows`-long inverse map for one child scatter.
    pub(super) fn inverse(&mut self, rows: usize) -> &mut [u32] {
        if self.inverse.len() < rows {
            self.inverse.resize(rows, u32::MAX);
        }
        let map = &mut self.inverse[..rows];
        map.fill(u32::MAX);
        map
    }

}

/// Extend `buffer` with zeros until it holds at least `want` elements.
pub(super) fn grow(buffer: &mut Vec<C>, want: usize) {
    if buffer.len() < want {
        buffer.resize(want, C::from_re(0.0));
    }
}

impl Front<'_> {
    /// Eliminate every fully summed column with LAPACK and subtract the whole
    /// rank-`nc` Schur complement in one product.
    ///
    /// Returns `false`, having changed nothing, when LAPACK reports an exactly
    /// singular pivot block; the caller then falls back to `factor_panels`.
    pub(super) fn factor_blocked(
        &mut self,
        s: &mut Scratch,
        report: &mut Report,
    ) -> Result<bool, Failure> {
        let (m, n) = (self.m, self.n);
        if n == 0 {
            return Ok(true);
        }
        let clock = Report::clock();
        s.fit(m, n);
        let leading = &mut s.leading[..n * n];
        for j in 0..n {
            for i in j..n {
                leading[j * n + i] = self.a[i * m + j];
            }
        }
        let ipiv = &mut s.ipiv[..n];
        // A panelled factorization has a serial chain down the diagonal that
        // row blocks cannot split from the outside, so this is left to the
        // library. Accelerate (macOS 26) runs `?sytrf` and `?getrf` on one
        // thread either way, legacy and `$NEWLAPACK` alike; the setting is
        // for libraries that do thread them.
        dense::set_threading(Threading::Machine);
        let info = dense::sytrf(n, leading, n, ipiv, &mut s.work);
        dense::set_threading(Threading::Caller);
        // Nothing outside `leading` and `ipiv` has been written yet, so a
        // refusal here leaves the front exactly as it was assembled.
        if info != 0 {
            return Ok(false);
        }
        self.adopt(leading, ipiv, &mut s.twos);
        self.reciprocals(leading, &mut s.inv[..n], &mut s.off[..n])?;
        self.store_leading(leading);
        report.stop(Phase::Leading, clock);
        report.front(Phase::Leading, m, n);
        let tail = m - n;
        if tail > 0 {
            let clock = Report::clock();
            let packed = &mut s.packed[..tail * n];
            self.gather(packed);
            let tri: &[C] = leading;
            if self.parallel {
                // A right-side solve is independent per row of the unknown.
                packed
                    .par_chunks_mut(n * self.config.cpu_block_rows.max(1))
                    .for_each(|block| {
                        dense::trsm(Side::Right, Tri::UnitUpper, block.len() / n, n, tri, n, block, n);
                    });
            } else {
                dense::trsm(Side::Right, Tri::UnitUpper, tail, n, tri, n, packed, n);
            }
            self.store_trailing(packed, &s.inv[..n], &s.off[..n], &s.twos);
            report.stop(Phase::Offdiag, clock);
            let clock = Report::clock();
            // SAFETY-adjacent invariant checked by `gemm_nt_front`: the left
            // operand is columns `0..n` and the target columns `n..m` of the
            // same rows, so the two rectangles never overlap.
            self.schur(packed);
            report.stop(Phase::Schur, clock);
        }
        // Non-finite values are caught when the factor columns are stored;
        // see `Shared::node`.
        Ok(true)
    }

    /// Replay LAPACK's interchanges into `perm`, `two` and the stored factor.
    ///
    /// `zsytrf` interleaves each interchange with its elimination step and
    /// leaves earlier columns unpermuted. Applying every interchange to the
    /// columns already written converts the factor to the single-permutation
    /// form `P A P^T = L D L^T`, which is the form the solve and the child
    /// scatter maps assume.
    fn adopt(&mut self, leading: &mut [C], ipiv: &[i32], twos: &mut Vec<u32>) {
        let n = self.n;
        twos.clear();
        let mut k = 0;
        while k < n {
            let wide = ipiv[k] < 0;
            // A 2x2 block interchanges its second row, not its first.
            let (row, pivot) = if wide {
                (k + 1, (-ipiv[k]) as usize - 1)
            } else {
                (k, ipiv[k] as usize - 1)
            };
            if pivot != row {
                for j in 0..k {
                    leading.swap(j * n + row, j * n + pivot);
                }
                self.perm.swap(row, pivot);
            }
            if wide {
                self.two[k] = true;
                twos.push(k as u32);
                k += 2;
            } else {
                k += 1;
            }
        }
    }

    /// Split the block diagonal out of `leading` and invert it columnwise.
    ///
    /// Writes the pivots to `d`/`sub` and clears the slot LAPACK shares between
    /// `D(k+1,k)` and the transposed factor, which must read as zero. `inv` and
    /// `off` hold the coefficients that turn `L21 * D` into `L21` with
    /// multiplications only; see [`Front::store_trailing`].
    fn reciprocals(
        &mut self,
        leading: &mut [C],
        inv: &mut [C],
        off: &mut [C],
    ) -> Result<(), Failure> {
        let (n, one) = (self.n, C::from_re(1.0));
        let mut k = 0;
        while k < n {
            self.d[k] = leading[k * n + k];
            if !self.two[k] {
                inv[k] = div(one, self.d[k]);
                off[k] = C::from_re(0.0);
                if !finite(inv[k]) {
                    return Err((k, PivotFailure::NonFinite));
                }
                k += 1;
                continue;
            }
            self.sub[k] = leading[k * n + k + 1];
            self.d[k + 1] = leading[(k + 1) * n + k + 1];
            leading[k * n + k + 1] = C::from_re(0.0);
            let (aa, bb, cc, scale, det) = block(self.d[k], self.sub[k], self.d[k + 1]);
            let r = div(one, det * scale);
            inv[k] = cc * r;
            inv[k + 1] = aa * r;
            off[k] = bb * r;
            off[k + 1] = C::from_re(0.0);
            if !finite(inv[k]) || !finite(inv[k + 1]) || !finite(off[k]) {
                return Err((k, PivotFailure::NonFinite));
            }
            k += 2;
        }
        Ok(())
    }

    /// Copy `L11` and the pivots back into the front's lower triangle, matching
    /// the layout `panel.rs` writes: pivots on the diagonal, a zero in the
    /// subdiagonal slot of a 2x2 block, unit multipliers below.
    fn store_leading(&mut self, leading: &[C]) {
        let (m, n) = (self.m, self.n);
        for j in 0..n {
            self.a[j * m + j] = self.d[j];
            for i in j + 1..n {
                self.a[i * m + j] = leading[j * n + i];
            }
        }
    }

    /// Pack `A21` into row-major `packed`, permuting its columns to the pivot order.
    fn gather(&self, packed: &mut [C]) {
        let (m, n, a, perm) = (self.m, self.n, &*self.a, &*self.perm);
        let fill = |i: usize, out: &mut [C]| {
            let row = &a[(n + i) * m..(n + i) * m + n];
            for (slot, &p) in out.iter_mut().zip(perm.iter()) {
                *slot = row[p];
            }
        };
        if self.parallel {
            packed
                .par_chunks_mut(n * self.stripe())
                .enumerate()
                .for_each(|(b, block)| {
                    let base = b * self.stripe();
                    block
                        .chunks_mut(n)
                        .enumerate()
                        .for_each(|(i, out)| fill(base + i, out));
                });
        } else {
            packed
                .chunks_mut(n)
                .enumerate()
                .for_each(|(i, out)| fill(i, out));
        }
    }

    /// Rows per block in the trailing-row loops.
    fn stripe(&self) -> usize {
        self.config.cpu_block_rows.max(1)
    }

    /// Subtract `L21 * (L21 D)^T` from the trailing block, lower triangle only.
    ///
    /// Row blocks are the unit of both the blocking and the parallelism: block
    /// `[i0, i1)` needs columns `0..i1` and nothing beyond, so each block is one
    /// rectangular product into a contiguous row range of the front. Half the
    /// square is never multiplied, which is the whole point -- the block is
    /// symmetric, and the parent's extend-add reads only its lower triangle.
    fn schur(&mut self, packed: &[C]) {
        let (m, n, rows) = (self.m, self.n, self.stripe());
        let tail = m - n;
        let run = |i0: usize, chunk: &mut [C]| {
            let r = chunk.len() / m;
            let i1 = (i0 + r).min(tail);
            dense::gemm_nt_front(r, i1, n, chunk, 0, n, m, &packed[..i1 * n], n);
        };
        let trailing = &mut self.a[n * m..];
        if self.parallel {
            trailing
                .par_chunks_mut(rows * m)
                .enumerate()
                .for_each(|(b, chunk)| run(b * rows, chunk));
        } else {
            trailing
                .chunks_mut(rows * m)
                .enumerate()
                .for_each(|(b, chunk)| run(b * rows, chunk));
        }
    }

    /// Reflect the front's lower triangle into its upper triangle.
    ///
    /// Assembly fills only the triangle this module reads, so the kernels that
    /// want a full square -- the serial narrow-front kernel and the `panel.rs`
    /// fallback, both of which permute rows and columns in place -- restore it
    /// here first. Tiled for the strided side of the copy, and parallel over
    /// destination row blocks: a task writes only rows it owns and reads only
    /// below the diagonal, which no task writes.
    pub(super) fn symmetrize(&mut self) {
        let (m, tile) = (self.m, self.stripe());
        if m < 2 {
            return;
        }
        let base = Cells(self.a.as_mut_ptr());
        let reflect = |b: usize| {
            // Bound through the wrapper so the closure captures `Cells`, which
            // is `Send`, rather than the bare pointer field, which is not.
            let cells = base;
            let (j0, j1) = (b * tile, ((b + 1) * tile).min(m));
            for i0 in (j0..m).step_by(tile) {
                let i1 = (i0 + tile).min(m);
                for j in j0..j1 {
                    for i in i0.max(j + 1)..i1 {
                        // SAFETY: both indices are inside the front, and the
                        // only references alive over it are these raw pointers.
                        // Element `(i, j)` is below the diagonal, so no task
                        // writes it; element `(j, i)` is in row `j`, which only
                        // this task writes.
                        unsafe {
                            let v = *cells.0.add(i * m + j);
                            *cells.0.add(j * m + i) = v;
                        }
                    }
                }
            }
        };
        let blocks = m.div_ceil(tile);
        if self.parallel {
            (0..blocks).into_par_iter().for_each(reflect);
        } else {
            (0..blocks).for_each(reflect);
        }
    }

    /// Write `L21 = packed * D^-1` into the front, leaving `packed` as `L21 * D`.
    ///
    /// The scaling runs as one contiguous multiply per row, then a short fixup
    /// over the 2x2 columns only, so the common all-1x1 front never branches
    /// inside the row loop.
    fn store_trailing(&mut self, packed: &[C], inv: &[C], off: &[C], twos: &[u32]) {
        let (m, n, rows) = (self.m, self.n, self.stripe());
        let scale = |out: &mut [C], row: &[C]| {
            for (slot, (&v, &r)) in out.iter_mut().zip(row.iter().zip(inv.iter())) {
                *slot = v * r;
            }
            for &k in twos {
                let k = k as usize;
                out[k] -= row[k + 1] * off[k];
                out[k + 1] -= row[k] * off[k];
            }
        };
        let trailing = &mut self.a[n * m..];
        if self.parallel {
            trailing
                .par_chunks_mut(rows * m)
                .zip(packed.par_chunks(rows * n))
                .for_each(|(block, source)| {
                    for (out, row) in block.chunks_mut(m).zip(source.chunks(n)) {
                        scale(&mut out[..n], row);
                    }
                });
        } else {
            for (out, row) in trailing.chunks_mut(m).zip(packed.chunks(n)) {
                scale(&mut out[..n], row);
            }
        }
    }
}
