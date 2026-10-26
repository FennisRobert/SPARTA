//! Partial LU of one front through Accelerate's blocked kernels.
//!
//! The fully summed block goes to LAPACK's `?getrf`, and the rest is three
//! level-3 calls: `L11 U12 = A12` and `L21 U11 = A21` as triangular solves,
//! then the whole rank-`nc` Schur complement `A22 -= L21 U12` as one product.
//! This replaces `panel.rs`'s LU route, which pivots and solves for U12 one
//! scalar loop at a time and updates the trailing block at rank `cpu_panel`.
//!
//! Pivoting is unchanged in reach: `getrf` on the fully summed block searches
//! the same rows `lu_pivot` does, the fully summed ones. When LAPACK reports an
//! exactly zero pivot, or `lu_diag_threshold` asks for a diagonal preference
//! LAPACK cannot express, the front is handed back untouched and `panel.rs`
//! factors it with its rescue floor.
use super::blocked::{grow, Scratch};
use super::dense::{self, Side, Threading, Tri};
use super::profile::{Phase, Report};
use super::factor::Front;
use crate::PivotFailure;

type Failure = (usize, PivotFailure);

impl Front<'_> {
    /// Eliminate every fully summed column with LAPACK and subtract the whole
    /// rank-`nc` Schur complement in one product.
    ///
    /// Returns `false`, having changed nothing, when LAPACK meets an exactly
    /// zero pivot or a diagonal preference is configured; the caller then falls
    /// back to `factor_panels`.
    pub(super) fn factor_blocked_lu(&mut self, s: &mut Scratch, report: &mut Report) -> Result<bool, Failure> {
        let (m, n) = (self.m, self.n);
        if n == 0 {
            return Ok(true);
        }
        if self.config.lu_diag_threshold > 0.0 {
            return Ok(false);
        }
        let clock = Report::clock();
        grow(&mut s.leading, n * n);
        if s.ipiv.len() < n {
            s.ipiv.resize(n, 0);
        }
        let (leading, ipiv) = (&mut s.leading[..n * n], &mut s.ipiv[..n]);
        for j in 0..n {
            for i in 0..n {
                leading[j * n + i] = self.a[i * m + j];
            }
        }
        // As for `sytrf` in `blocked.rs`: left to the library to thread.
        dense::set_threading(Threading::Machine);
        let info = dense::getrf(n, leading, n, ipiv);
        dense::set_threading(Threading::Caller);
        // Nothing outside `leading` and `ipiv` has been written yet.
        if info != 0 {
            return Ok(false);
        }
        // `getrf` has applied every interchange across the whole block, so
        // the same sequence applied to the rest of each fully summed row, and
        // to `perm`, gives the one-sided `P A = L U` form `panel.rs` writes.
        for (k, &p) in ipiv.iter().enumerate() {
            let p = p as usize - 1;
            if p != k {
                for j in n..m {
                    self.a.swap(k * m + j, p * m + j);
                }
                self.perm.swap(k, p);
            }
        }
        for i in 0..n {
            for j in 0..n {
                self.a[i * m + j] = leading[j * n + i];
            }
        }
        report.stop(Phase::Leading, clock);
        report.front(Phase::Leading, m, n);
        let tail = m - n;
        if tail > 0 {
            let clock = Report::clock();
            // U12: rows 0..n, columns n..m, against L11 at the origin; then
            // L21: rows n..m, columns 0..n, against U11. Accelerate does not
            // thread a triangular solve, so a front alone on the machine
            // spreads its solves over the pool instead.
            let solve = if self.parallel { dense::trsm_front_spread } else { dense::trsm_front };
            solve(Side::Left, Tri::UnitLower, n, tail, self.a, 0, n, m);
            solve(Side::Right, Tri::Upper, tail, n, self.a, 0, n * m, m);
            report.stop(Phase::Offdiag, clock);
            let clock = Report::clock();
            // The product, unlike the solves, Accelerate threads well -- when
            // the front has the machine to itself. One sharing its level with
            // others splits the product into row blocks for the pool instead.
            if self.alone {
                dense::set_threading(Threading::Machine);
                dense::gemm_nn_front(tail, tail, n, self.a, n * m, n, n * m + n, m);
                dense::set_threading(Threading::Caller);
            } else if self.parallel {
                dense::gemm_nn_front_spread(tail, tail, n, self.a, n * m, n, n * m + n, m);
            } else {
                dense::gemm_nn_front(tail, tail, n, self.a, n * m, n, n * m + n, m);
            }
            report.stop(Phase::Schur, clock);
        }
        // Non-finite values are caught when the factor columns are stored;
        // see `Shared::node`.
        Ok(true)
    }
}
