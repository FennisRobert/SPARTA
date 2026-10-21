//! LAPACK-shaped panels: stale LDLᵀ fronts and right-looking LU panels.
use super::Scalar;
use super::{block, div, factor::Front, finite, update, C, R};
use crate::{Kind, PivotFailure};

type Failure = (usize, PivotFailure);
impl Front<'_> {
    /// Factor panels and apply each completed rank-k update to the remaining front.
    pub(super) fn factor_panels(&mut self) -> Result<(), Failure> {
        let mut start = 0;
        while start < self.n {
            let end = if self.kind == Kind::Ldlt {
                self.ldlt_panel(start)?
            } else {
                self.lu_panel(start)?
            };
            update::trailing(self, start, end);
            start = end;
        }
        if self.a.iter().any(|&v| !finite(v)) {
            return Err((self.n - 1, PivotFailure::NonFinite));
        }
        Ok(())
    }

    /// Form column `col` below `k`, subtracting `count` pending pivots from `start`.
    /// `w` is the row-major panel with row stride `width`, containing L columns.
    fn column(
        &self,
        col: usize,
        k: usize,
        start: usize,
        count: usize,
        w: &[C],
        width: usize,
    ) -> Vec<C> {
        let mut coefficient = vec![C::from_re(0.0); count];
        let mut t = 0;
        while t < count {
            let p = start + t;
            if self.two[p] {
                coefficient[t] =
                    w[col * width + t] * self.d[p] + w[col * width + t + 1] * self.sub[p];
                coefficient[t + 1] =
                    w[col * width + t] * self.sub[p] + w[col * width + t + 1] * self.d[p + 1];
                t += 2;
            } else {
                coefficient[t] = w[col * width + t] * self.d[p];
                t += 1;
            }
        }
        let mut out = vec![C::from_re(0.0); self.m];
        for i in k..self.m {
            let mut v = self.a[i * self.m + col];
            for t in 0..count {
                v -= w[i * width + t] * coefficient[t];
            }
            out[i] = v;
        }
        out
    }

    /// Factor one stale-front LDLᵀ panel beginning at fully summed column `start`.
    fn ldlt_panel(&mut self, start: usize) -> Result<usize, Failure> {
        let capacity = self.config.cpu_panel.min(self.n - start);
        let width = capacity + 1;
        let mut w = vec![C::from_re(0.0); self.m * width];
        let mut k = start;
        let alpha = (1.0 + (17.0 as R).sqrt()) / 8.0;
        while k < self.n && k - start < capacity {
            let t = k - start;
            let mut col = self.column(k, k, start, t, &w, width);
            if col[k..self.n].iter().any(|&v| !finite(v)) {
                return Err((k, PivotFailure::NonFinite));
            }
            let (mut cm, mut imax) = (0.0, k);
            for (i, v) in col.iter().enumerate().take(self.n).skip(k + 1) {
                if v.norm() > cm {
                    cm = v.norm();
                    imax = i;
                }
            }
            let akk = col[k].norm();
            let (mut step, mut pick) = (1, k);
            if akk < alpha * cm {
                let row = self.column(imax, k, start, t, &w, width);
                if row[k..self.n].iter().any(|&v| !finite(v)) {
                    return Err((k, PivotFailure::NonFinite));
                }
                let rm = (k..self.n)
                    .filter(|&j| j != imax)
                    .map(|j| row[j].norm())
                    .fold(0.0, R::max);
                if akk >= alpha * cm * (cm / rm) {
                    pick = k;
                } else if row[imax].norm() >= alpha * rm {
                    pick = imax;
                } else {
                    step = 2;
                }
            }
            if step == 1 && pick != k {
                self.swap(k, pick);
                swap_rows(&mut w, width, t, k, pick);
                col = self.column(k, k, start, t, &w, width);
            }
            if step == 2 {
                self.swap(k + 1, imax);
                swap_rows(&mut w, width, t, k + 1, imax);
                col.swap(k + 1, imax);
                let partner = self.column(k + 1, k, start, t, &w, width);
                let (da, db, dc) = (col[k], col[k + 1], partner[k + 1]);
                let (aa, bb, cc, scale, det) = block(da, db, dc);
                if !finite(det) {
                    return Err((k, PivotFailure::NonFinite));
                }
                if det.norm() <= R::EPSILON {
                    step = 1;
                    if da.norm() == 0.0 {
                        col[k] = self.rescue(k, false, &mut None, PivotFailure::TinyTwoByTwo)?;
                    }
                    self.bumped += 1;
                } else {
                    self.d[k] = da;
                    self.d[k + 1] = dc;
                    self.sub[k] = db;
                    self.two[k] = true;
                    w[k * width + t] = da;
                    w[(k + 1) * width + t + 1] = dc;
                    for i in k + 2..self.m {
                        let (x, y) = (col[i] / scale, partner[i] / scale);
                        w[i * width + t] = div(x * cc - y * bb, det);
                        w[i * width + t + 1] = div(y * aa - x * bb, det);
                    }
                }
            }
            if step == 1 {
                let mut d = col[k];
                if d.norm() == 0.0 {
                    d = self.rescue(k, false, &mut None, PivotFailure::ZeroColumn)?;
                    self.bumped += 1;
                }
                self.d[k] = d;
                w[k * width + t] = d;
                for i in k + 1..self.m {
                    w[i * width + t] = div(col[i], d);
                }
            }
            if w.iter().any(|&v| !finite(v)) {
                return Err((k, PivotFailure::NonFinite));
            }
            k += step;
        }
        for col in start..k {
            for i in col..self.m {
                self.a[i * self.m + col] = w[i * width + col - start];
            }
        }
        Ok(k)
    }

    /// Factor a restricted partial-pivot LU panel at `start`, then solve for U12.
    fn lu_panel(&mut self, start: usize) -> Result<usize, Failure> {
        let end = (start + self.config.cpu_panel.min(self.n - start)).min(self.n);
        let m = self.m;
        for k in start..end {
            let (mut best, mut pick) = (0.0, k);
            for i in k..self.n {
                let v = self.a[i * m + k];
                if !finite(v) {
                    return Err((k, PivotFailure::NonFinite));
                }
                if v.norm() > best {
                    best = v.norm();
                    pick = i;
                }
            }
            let d = self.a[k * m + k].norm();
            let threshold = self.config.lu_diag_threshold as R;
            if threshold > 0.0 && d > 0.0 && d >= threshold * best {
                pick = k;
            }
            self.swap(k, pick);
            let mut d = self.a[k * m + k];
            if d.norm() == 0.0 {
                d = self.rescue(k, false, &mut None, PivotFailure::ZeroColumn)?;
                self.a[k * m + k] = d;
                self.bumped += 1;
            }
            for i in k + 1..m {
                self.a[i * m + k] = div(self.a[i * m + k], d);
            }
            for i in k + 1..m {
                for j in k + 1..end {
                    let delta = self.a[i * m + k] * self.a[k * m + j];
                    self.a[i * m + j] -= delta;
                }
            }
        }
        // L11 U12 = A12. Finished panel rows are final; later rows remain stale.
        for k in start..end {
            for i in k + 1..end {
                let l = self.a[i * m + k];
                for j in end..m {
                    let delta = l * self.a[k * m + j];
                    self.a[i * m + j] -= delta;
                }
            }
        }
        Ok(end)
    }
}

/// Interchange rows `p` and `q` in the `count` formed columns of a `width`-stride panel.
fn swap_rows(w: &mut [C], width: usize, count: usize, p: usize, q: usize) {
    for t in 0..count {
        w.swap(p * width + t, q * width + t);
    }
}
