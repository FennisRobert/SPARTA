//! The systems the optimizer times, and how one measurement is taken.

use crate::analysis::{Analysis, Kind};
use crate::config::Config;
use crate::{Complex32, Complex64, Error, Result, Solver};
use std::time::Instant;

/// A measurement repeats until it has at least this many samples...
const MIN_REPEATS: usize = 3;
/// ...and this much timed work, but never more than `MAX_REPEATS` samples.
const MIN_SECONDS: f64 = 0.3;
const MAX_REPEATS: usize = 50;

/// One system to tune on: a matrix as its analysis takes it, and right-hand sides.
#[derive(Clone, Debug)]
pub struct Problem {
    kind: Kind,
    n: usize,
    ptr: Vec<usize>,
    rows: Vec<usize>,
    values: Vec<Complex64>,
    rhs: Vec<Complex64>,
    nrhs: usize,
}

impl Problem {
    /// A complex-symmetric system for `LDL^T`: the lower triangle in CSC, as
    /// [`Analysis::ldlt`] takes it, with its values in the same order.
    pub fn ldlt(n: usize, ptr: Vec<usize>, rows: Vec<usize>, values: Vec<Complex64>) -> Result<Self> {
        Self::new(Kind::Ldlt, n, ptr, rows, values)
    }

    /// A general system for `LU`: the whole matrix in CSC, as [`Analysis::lu`]
    /// takes it, with its values in the same order.
    pub fn lu(n: usize, ptr: Vec<usize>, rows: Vec<usize>, values: Vec<Complex64>) -> Result<Self> {
        Self::new(Kind::Lu, n, ptr, rows, values)
    }

    fn new(kind: Kind, n: usize, ptr: Vec<usize>, rows: Vec<usize>, values: Vec<Complex64>) -> Result<Self> {
        if n == 0 || ptr.len() != n + 1 || ptr[n] != rows.len() || rows.len() != values.len() {
            return Err(Error::Invalid(
                "a problem needs n + 1 column offsets and one value per stored row".into(),
            ));
        }
        // A fixed pseudo-random right-hand side, the same on every run.
        let mut state = n as u64 ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let rhs = (0..n).map(|_| Complex64::new(next(), next())).collect();
        Ok(Self { kind, n, ptr, rows, values, rhs, nrhs: 1 })
    }

    /// Replaces the right-hand side with `nrhs` columns packed end to end.
    pub fn with_rhs(mut self, rhs: Vec<Complex64>, nrhs: usize) -> Result<Self> {
        if nrhs == 0 || rhs.len() != self.n * nrhs {
            return Err(Error::Invalid(format!(
                "a right-hand side block for order {} needs {} x nrhs values",
                self.n, self.n
            )));
        }
        self.rhs = rhs;
        self.nrhs = nrhs;
        Ok(self)
    }

    /// The matrix order.
    pub fn order(&self) -> usize {
        self.n
    }

    /// Stored entries, one triangle for `LDL^T`.
    pub fn stored(&self) -> usize {
        self.rows.len()
    }

    /// A complex-symmetric Q1 finite-element Helmholtz system on a cube.
    ///
    /// Trilinear hexahedra on a `k` by `k` by `k` grid of nodes: 27 entries
    /// per row, indefinite, with a little loss, like a lossy cavity about a
    /// wavelength across. It stands in for an electromagnetic or acoustic FEM
    /// assembly when there is none to hand.
    pub fn helmholtz(k: usize) -> Self {
        assert!(k >= 2, "the grid needs at least two nodes per side");
        let h = 1.0 / (k - 1) as f64;
        // 1D element matrices of linear elements, entry for offset -1, 0, +1.
        let stiff = |d: isize| if d == 0 { 2.0 / h } else { -1.0 / h };
        let mass = |d: isize| if d == 0 { 4.0 * h / 6.0 } else { h / 6.0 };
        let wave = 0.35 * std::f64::consts::PI * (k - 1) as f64 / 4.0;
        let shift = Complex64::new(wave * wave, -0.02 * wave * wave);
        let at = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
        let n = k * k * k;
        let (mut ptr, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
        for z in 0..k {
            for y in 0..k {
                for x in 0..k {
                    let j = at(x, y, z);
                    // Neighbours in increasing index order: z, then y, then x.
                    for dz in -1isize..=1 {
                        for dy in -1isize..=1 {
                            for dx in -1isize..=1 {
                                let (nx, ny, nz) = (x as isize + dx, y as isize + dy, z as isize + dz);
                                let inside = |v: isize| (0..k as isize).contains(&v);
                                if !(inside(nx) && inside(ny) && inside(nz)) {
                                    continue;
                                }
                                let i = at(nx as usize, ny as usize, nz as usize);
                                if i < j {
                                    continue;
                                }
                                let k3 = stiff(dx) * mass(dy) * mass(dz)
                                    + mass(dx) * stiff(dy) * mass(dz)
                                    + mass(dx) * mass(dy) * stiff(dz);
                                let m3 = mass(dx) * mass(dy) * mass(dz);
                                rows.push(i);
                                values.push(Complex64::new(k3, 0.0) - shift * m3);
                            }
                        }
                    }
                    ptr.push(rows.len());
                }
            }
        }
        Self::ldlt(n, ptr, rows, values).expect("the generated pattern is valid")
    }

    /// The problems [`optimize`] tunes on when given none: two Helmholtz
    /// cubes, of about 14 and 47 thousand unknowns. They suit the hardware in
    /// general and no application in particular; tune on your own matrices
    /// where you can.
    pub fn standard() -> Vec<Problem> {
        vec![Problem::helmholtz(24), Problem::helmholtz(36)]
    }

    /// Median seconds per factor-and-solve under `config`, the worst relative
    /// residual over the right-hand sides, and the timing noise: the
    /// interquartile range of the samples as a fraction of their median.
    pub(super) fn measure(&self, config: &Config, analysis: bool) -> Result<(f64, f64, f64)> {
        let begin = Instant::now();
        let plan = match self.kind {
            Kind::Ldlt => Analysis::ldlt(self.n, &self.ptr, &self.rows, config)?,
            Kind::Lu => Analysis::lu(self.n, &self.ptr, &self.rows, config)?,
        };
        let analyzed = begin.elapsed().as_secs_f64();
        let mut solver = Solver::new(plan)?;
        let narrow = |v: &[Complex64]| -> Vec<Complex32> {
            v.iter().map(|z| Complex32::new(z.re as f32, z.im as f32)).collect()
        };
        let (values32, rhs32) = (narrow(&self.values), narrow(&self.rhs));
        let run = |solver: &mut Solver| -> Result<Vec<Complex64>> {
            if config.force_cpu {
                solver.factor_c128(&self.values)?;
                solver.solve_many_c128(&self.rhs, self.nrhs)
            } else {
                solver.factor(&values32)?;
                let x = solver.solve_many(&rhs32, self.nrhs)?;
                Ok(x.iter().map(|z| Complex64::new(z.re.into(), z.im.into())).collect())
            }
        };
        let result = (|| {
            run(&mut solver)?;
            let (mut samples, mut x) = (Vec::new(), Vec::new());
            while samples.len() < MIN_REPEATS
                || (samples.iter().sum::<f64>() < MIN_SECONDS && samples.len() < MAX_REPEATS)
            {
                let begin = Instant::now();
                x = run(&mut solver)?;
                samples.push(begin.elapsed().as_secs_f64());
            }
            samples.sort_by(f64::total_cmp);
            let count = samples.len();
            let median = samples[count / 2];
            let spread = (samples[3 * count / 4] - samples[count / 4]) / median;
            Ok((median + if analysis { analyzed } else { 0.0 }, self.residual(&x), spread))
        })();
        solver.destroy();
        result
    }

    /// The largest `||b - A x|| / ||b||` over the right-hand sides.
    pub(super) fn residual(&self, x: &[Complex64]) -> f64 {
        let n = self.n;
        (0..self.nrhs)
            .map(|c| {
                let (xc, bc) = (&x[c * n..(c + 1) * n], &self.rhs[c * n..(c + 1) * n]);
                let mut r: Vec<Complex64> = bc.to_vec();
                for j in 0..n {
                    for k in self.ptr[j]..self.ptr[j + 1] {
                        let (i, v) = (self.rows[k], self.values[k]);
                        r[i] -= v * xc[j];
                        if self.kind == Kind::Ldlt && i != j {
                            r[j] -= v * xc[i];
                        }
                    }
                }
                let norm = |v: &[Complex64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                norm(&r) / norm(bc).max(f64::MIN_POSITIVE)
            })
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// The Helmholtz generator makes a valid lower triangle whose symmetric
    /// residual of the zero vector is exactly one.
    fn helmholtz_is_a_valid_lower_triangle() {
        let p = Problem::helmholtz(4);
        assert_eq!(p.order(), 64);
        for j in 0..p.n {
            let col = &p.rows[p.ptr[j]..p.ptr[j + 1]];
            assert_eq!(col[0], j, "diagonal first");
            assert!(col.windows(2).all(|w| w[0] < w[1]));
        }
        assert!((p.residual(&vec![Complex64::new(0.0, 0.0); 64]) - 1.0).abs() < 1e-12);
    }
}
