//! Level-ordered solves sharing Metal's global and local pivot mappings.
use super::Scalar;
use super::{block, div, finite, Solver, C};
use crate::{Error, Kind, Result};
use rayon::prelude::*;

impl Solver {
    /// Solve column-major complex128 `b` containing `nrhs` right-hand sides.
    pub(crate) fn solve_many(&self, b: &[C], nrhs: usize) -> Result<Vec<C>> {
        if !self.valid {
            return Err(Error::Invalid(
                "successful factorization required before solve".into(),
            ));
        }
        let n = self.plan.n;
        if nrhs == 0 || n.checked_mul(nrhs) != Some(b.len()) || b.iter().any(|&v| !finite(v)) {
            return Err(Error::Invalid(
                "finite RHS columns of analyzed dimension required".into(),
            ));
        }
        let mut out = vec![C::from_re(0.0); b.len()];
        out.par_chunks_mut(n)
            .zip(b.par_chunks(n))
            .for_each(|(out, rhs)| self.solve_column(rhs, out));
        if out.iter().any(|&v| !finite(v)) {
            return Err(Error::Invalid("nonfinite CPU solve result".into()));
        }
        Ok(out)
    }

    /// Solve one complex128 `b` and scatter into original-order `out`.
    fn solve_column(&self, b: &[C], out: &mut [C]) {
        let mut x: Vec<_> = self.plan.perm.iter().map(|&i| b[i as usize]).collect();
        let lu = self.plan.kind == Kind::Lu;
        for level in &self.plan.levels {
            for &id in level {
                let node = self.plan.nodes[id as usize];
                let (m, n, first) = (node.nr as usize, node.nc as usize, node.first as usize);
                let rows = &self.plan.rows[node.rows as usize..node.rows as usize + m];
                let global =
                    |i: usize| rows[if i < n { self.perms[first + i] } else { i }] as usize;
                let l = &self.factors[node.loffset as usize..];
                let mut y: Vec<_> = (0..n).map(|k| x[global(k)]).collect();
                for k in 0..n {
                    let v = y[k];
                    for i in k + 1..n {
                        y[i] -= l[k * m + i] * v;
                    }
                }
                for i in n..m {
                    let mut acc = C::from_re(0.0);
                    for k in 0..n {
                        acc += l[k * m + i] * y[k];
                    }
                    x[rows[i] as usize] -= acc;
                }
                for k in 0..n {
                    x[if lu { rows[k] as usize } else { global(k) }] = y[k];
                }
            }
        }
        if !lu {
            for node in &self.plan.nodes {
                let first = node.first as usize;
                let rows = &self.plan.rows[node.rows as usize..];
                let global = |k: usize| rows[self.perms[first + k]] as usize;
                let mut k = 0;
                while k < node.nc as usize {
                    let i = global(k);
                    if self.two[first + k] {
                        let j = global(k + 1);
                        let (a, b, c, s, det) = block(
                            self.diagonal[first + k],
                            self.subdiag[first + k],
                            self.diagonal[first + k + 1],
                        );
                        let (u, v) = (x[i] / s, x[j] / s);
                        x[i] = div(u * c - v * b, det);
                        x[j] = div(v * a - u * b, det);
                        k += 2;
                    } else {
                        x[i] = div(x[i], self.diagonal[first + k]);
                        k += 1;
                    }
                }
            }
        }
        for level in self.plan.levels.iter().rev() {
            for &id in level {
                let node = self.plan.nodes[id as usize];
                let (m, n, first) = (node.nr as usize, node.nc as usize, node.first as usize);
                let rows = &self.plan.rows[node.rows as usize..node.rows as usize + m];
                let global = |i: usize| {
                    rows[if !lu && i < n {
                        self.perms[first + i]
                    } else {
                        i
                    }] as usize
                };
                let l = &self.factors[node.loffset as usize..];
                let u = &self.upper[node.uoffset as usize..];
                for k in (0..n).rev() {
                    let mut v = x[global(k)];
                    for j in k + 1..m {
                        let a = if !lu {
                            l[k * m + j]
                        } else if j < n {
                            l[j * m + k]
                        } else {
                            u[k * (m - n) + j - n]
                        };
                        v -= a * x[global(j)];
                    }
                    x[global(k)] = if lu { div(v, l[k * m + k]) } else { v };
                }
            }
        }
        for (i, &old) in self.plan.perm.iter().enumerate() {
            out[old as usize] = x[i];
        }
    }
}
