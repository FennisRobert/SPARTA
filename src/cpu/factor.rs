//! Restricted pivoting and tiled trailing updates, without conjugation.
use super::profile::{Phase, Report};
use super::Scalar;
use super::{block, blocked::Scratch, dense, div, finite, C, R};
use crate::{Config, Kind, PivotFailure};

type Failure = (usize, PivotFailure);

/// Borrowed row-major front and its permanent pivot metadata.
pub(super) struct Front<'a> {
    pub a: &'a mut [C],
    pub m: usize,
    pub n: usize,
    pub perm: &'a mut [usize],
    pub d: &'a mut [C],
    pub sub: &'a mut [C],
    pub two: &'a mut [bool],
    pub config: &'a Config,
    pub kind: Kind,
    pub bumped: u64,
    /// Whether this front is alone enough on the machine to split its own work.
    pub parallel: bool,
}
impl Front<'_> {
    /// Eliminate fully summed columns, then finish the deferred Schur complement.
    ///
    /// Wide fronts go to Accelerate through `blocked.rs` (LDL^T) or
    /// `blocked_lu.rs` (LU), which hand them back untouched when LAPACK meets
    /// an exactly singular pivot block; `panel.rs`
    /// then factors them with its rescue floor. Narrow fronts stay on the
    /// serial kernel, whose complete pivot search matches Metal's short-front
    /// path and whose fronts are too small to pay a BLAS call.
    pub fn factor(&mut self, scratch: &mut Scratch, report: &mut Report) -> Result<(), Failure> {
        for (i, p) in self.perm.iter_mut().enumerate() {
            *p = i;
        }
        self.two.fill(false);
        self.sub.fill(C::from_re(0.0));
        let small = self.m <= self.config.small_front_rows as usize;
        if !small {
            if dense::available() {
                let done = if self.kind == Kind::Ldlt {
                    self.factor_blocked(scratch, report)?
                } else {
                    self.factor_blocked_lu(scratch, report)?
                };
                if done {
                    return Ok(());
                }
            }
            let clock = Report::clock();
            if self.kind == Kind::Ldlt {
                self.symmetrize();
            }
            let result = self.factor_panels();
            report.stop(Phase::Panels, clock);
            report.front(Phase::Panels, self.m, self.n);
            return result;
        }
        let clock = Report::clock();
        report.front(Phase::Serial, self.m, self.n);
        if self.kind == Kind::Ldlt {
            self.symmetrize();
        }
        let mut k = 0;
        let mut floor = None;
        while k < self.n {
            let step = if self.kind == Kind::Lu {
                self.lu_pivot(k, true, &mut floor)?
            } else {
                self.ldlt_pivot(k, true, &mut floor)?
            };
            self.update(k, step, true);
            k += step;
        }
        report.stop(Phase::Serial, clock);
        if self.a.iter().any(|&v| !finite(v)) {
            return Err((k.saturating_sub(1), PivotFailure::NonFinite));
        }
        Ok(())
    }

    /// Swap local fully summed indices `p` and `q`; LDLᵀ swaps both axes.
    pub(super) fn swap(&mut self, p: usize, q: usize) {
        if p == q {
            return;
        }
        for j in 0..self.m {
            self.a.swap(p * self.m + j, q * self.m + j);
        }
        if self.kind == Kind::Ldlt {
            for i in 0..self.m {
                self.a.swap(i * self.m + p, i * self.m + q);
            }
        }
        self.perm.swap(p, q);
    }

    /// Rescue a dead pivot at `k`, using a front-relative sqrt(epsilon_f64) floor.
    /// `small` selects the serial kernel's cached whole-front scale.
    pub(super) fn rescue(
        &self,
        k: usize,
        small: bool,
        cached: &mut Option<R>,
        reason: PivotFailure,
    ) -> Result<C, Failure> {
        let floor = if small {
            *cached.get_or_insert_with(|| {
                self.a.iter().map(|v| v.norm()).fold(0.0, R::max) * R::EPSILON.sqrt()
            })
        } else {
            self.a
                .chunks(self.m)
                .flat_map(|row| &row[..self.n])
                .map(|v| v.norm())
                .fold(0.0, R::max)
                * R::EPSILON.sqrt()
        };
        if !floor.is_finite() {
            return Err((k, PivotFailure::NonFinite));
        }
        if floor <= 0.0 {
            return Err((k, reason));
        }
        Ok(C::from_re(floor))
    }

    /// Partial row pivoting at column `k`, confined to fully summed rows.
    fn lu_pivot(
        &mut self,
        k: usize,
        small: bool,
        floor: &mut Option<R>,
    ) -> Result<usize, Failure> {
        let (mut best, mut pick) = (0.0, k);
        for i in k..self.n {
            let v = self.a[i * self.m + k].norm();
            if !v.is_finite() {
                return Err((k, PivotFailure::NonFinite));
            }
            if v > best {
                best = v;
                pick = i;
            }
        }
        let d = self.a[k * self.m + k].norm();
        let threshold = self.config.lu_diag_threshold as R;
        if threshold > 0.0 && d > 0.0 && d >= threshold * best {
            pick = k;
        }
        self.swap(k, pick);
        let mut d = self.a[k * self.m + k];
        if d.norm() == 0.0 {
            d = self.rescue(k, small, floor, PivotFailure::ZeroColumn)?;
            self.a[k * self.m + k] = d;
            self.bumped += 1;
        }
        for i in k + 1..self.m {
            self.a[i * self.m + k] = div(self.a[i * self.m + k], d);
        }
        Ok(1)
    }

    /// Select the serial complete-search or cooperative Bunch–Kaufman pivot at `k`.
    fn choice(&self, k: usize, small: bool) -> Result<(usize, usize, usize), Failure> {
        let m = self.m;
        // Metal's short-front kernel uses a complete diagonal/off-diagonal
        // search; its cooperative kernels use the Bunch–Kaufman column/row test.
        let alpha = (1.0 + (17.0 as R).sqrt()) / 8.0;
        let checked = |z: C| {
            if finite(z) {
                Ok(z.norm())
            } else {
                Err((k, PivotFailure::NonFinite))
            }
        };
        if small {
            let (mut md, mut mo, mut pd, mut po, mut qo) = (0.0, 0.0, k, k, k);
            for i in k..self.n {
                let v = checked(self.a[i * m + i])?;
                if v > md {
                    md = v;
                    pd = i;
                }
                for j in k..i {
                    let v = checked(self.a[i * m + j])?;
                    if v > mo {
                        mo = v;
                        po = j;
                        qo = i;
                    }
                }
            }
            return Ok(if md >= alpha * mo || k + 1 == self.n {
                (1, pd, k)
            } else {
                (2, po, qo)
            });
        }
        let (mut cm, mut imax) = (0.0, k);
        for i in k + 1..self.n {
            let v = checked(self.a[i * m + k])?;
            if v > cm {
                cm = v;
                imax = i;
            }
        }
        let akk = checked(self.a[k * m + k])?;
        if akk >= alpha * cm {
            return Ok((1, k, k));
        }
        let mut rm = 0.0 as R;
        for j in k..self.n {
            if j != imax {
                rm = rm.max(checked(self.a[imax * m + j])?);
            }
        }
        let aii = checked(self.a[imax * m + imax])?;
        Ok(if akk >= alpha * cm * (cm / rm) {
            (1, k, k)
        } else if aii >= alpha * rm {
            (1, imax, k)
        } else {
            (2, k, imax)
        })
    }

    /// Form one 1×1 or scaled 2×2 LDLᵀ pivot at `k` and its multipliers.
    fn ldlt_pivot(
        &mut self,
        k: usize,
        small: bool,
        floor: &mut Option<R>,
    ) -> Result<usize, Failure> {
        let (mut step, p, q) = self.choice(k, small)?;
        self.swap(k, p);
        if step == 2 {
            self.swap(k + 1, q);
        }
        let m = self.m;
        if step == 2 {
            let (da, db, dc) = (
                self.a[k * m + k],
                self.a[(k + 1) * m + k],
                self.a[(k + 1) * m + k + 1],
            );
            let (aa, bb, cc, scale, det) = block(da, db, dc);
            if !finite(det) {
                return Err((k, PivotFailure::NonFinite));
            }
            if det.norm() <= R::EPSILON {
                step = 1;
                if da.norm() == 0.0 {
                    self.a[k * m + k] = self.rescue(k, small, floor, PivotFailure::TinyTwoByTwo)?;
                }
                self.bumped += 1;
            } else {
                self.d[k] = da;
                self.d[k + 1] = dc;
                self.sub[k] = db;
                self.two[k] = true;
                for i in k + 2..m {
                    let (x, y) = (self.a[i * m + k] / scale, self.a[i * m + k + 1] / scale);
                    self.a[i * m + k] = div(x * cc - y * bb, det);
                    self.a[i * m + k + 1] = div(y * aa - x * bb, det);
                }
                self.a[(k + 1) * m + k] = C::from_re(0.0);
            }
        }
        if step == 1 {
            let mut d = self.a[k * m + k];
            if d.norm() == 0.0 {
                d = self.rescue(k, small, floor, PivotFailure::ZeroColumn)?;
                self.a[k * m + k] = d;
                self.bumped += 1;
            }
            self.d[k] = d;
            for i in k + 1..m {
                self.a[i * m + k] = div(self.a[i * m + k], d);
            }
        }
        Ok(step)
    }

    /// Apply pivot `k` of width `step`; defer tail-by-tail entries unless `small`.
    fn update(&mut self, k: usize, step: usize, small: bool) {
        let (m, n) = (self.m, self.n);
        let lu = self.kind == Kind::Lu;
        let left: Vec<_> = (0..m).map(|i| self.a[i * m + k]).collect();
        let right: Vec<_> = (0..m)
            .map(|j| {
                if lu {
                    self.a[k * m + j]
                } else {
                    self.a[j * m + k]
                }
            })
            .collect();
        let second: Vec<_> = (0..m)
            .map(|i| {
                if step == 2 {
                    self.a[i * m + k + 1]
                } else {
                    C::from_re(0.0)
                }
            })
            .collect();
        let (da, db, dc) = (
            self.d[k],
            self.sub[k],
            if step == 2 {
                self.d[k + 1]
            } else {
                C::from_re(0.0)
            },
        );
        let run = |base: usize, chunk: &mut [C]| {
            for (r, row) in chunk.chunks_mut(m).enumerate() {
                let i = base + r;
                if i < k + step {
                    continue;
                }
                let end = if small || i < n { m } else { n };
                let end = if lu { end } else { end.min(i + 1) };
                for j in k + step..end {
                    if lu {
                        row[j] -= left[i] * right[j];
                    } else if step == 1 {
                        row[j] -= (left[i] * da) * right[j];
                    } else {
                        row[j] -= (left[i] * da + second[i] * db) * right[j];
                        row[j] -= (left[i] * db + second[i] * dc) * second[j];
                    }
                }
            }
        };
        run(0, self.a);
        if !lu {
            let end = if small { m } else { n };
            for j in k + step..end {
                for i in j + 1..m {
                    self.a[j * m + i] = self.a[i * m + j];
                }
            }
        }
    }
}
