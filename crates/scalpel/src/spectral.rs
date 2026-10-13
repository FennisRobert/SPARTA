//! Spectral initial partitions. The lowest nontrivial eigenvectors of a mesh
//! graph's Laplacian vary smoothly along its principal directions, so sorting
//! vertices by one gives a cut roughly perpendicular to that direction, where
//! BFS level sets from a peripheral vertex give diagonal cuts of larger area.

use crate::graph::Graph;
use crate::Options;

/// Number of eigenvectors whose span is searched for a cut direction.
const MODES: usize = 3;

/// Returns separator labels from cuts along several directions in the span of
/// the lowest `MODES` eigenvectors of `g`'s vertex-weighted Laplacian, the
/// best `keep` of them under `o`. Empty when `g` is too small to have them.
pub(crate) fn candidates(g: &Graph, o: &Options, keep: usize) -> Vec<Vec<u8>> {
    let n = g.len();
    if n < 4 * MODES {
        return Vec::new();
    }
    let modes = lowest_modes(g, (n - 1).min(64));
    let mut found: Vec<((bool, u64, usize), Vec<u8>)> = Vec::new();
    for d in directions() {
        let f: Vec<f64> = (0..n)
            .map(|v| (0..MODES).map(|k| d[k] * modes[k][v]).sum())
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| f[a].total_cmp(&f[b]).then(a.cmp(&b)));
        for q in [0.35, 0.42, 0.5, 0.58, 0.65] {
            let labels = split(g, &order, q);
            found.push((crate::separator::score(g, &labels, o), labels));
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    found.dedup_by(|a, b| a.1 == b.1);
    found.into_iter().take(keep).map(|(_, s)| s).collect()
}

/// Labels the lightest `q` fraction of weight in `order` as child 0 and the
/// rest as child 1, then moves every child-1 vertex that touches child 0 into
/// the separator.
fn split(g: &Graph, order: &[usize], q: f64) -> Vec<u8> {
    let total: usize = g.weight.iter().sum();
    let mut s = vec![1u8; g.len()];
    let mut acc = 0;
    for &v in order {
        if (acc as f64) >= q * total as f64 {
            break;
        }
        s[v] = 0;
        acc += g.weight[v];
    }
    let boundary: Vec<usize> = (0..g.len())
        .filter(|&v| s[v] == 1 && g.neighbors(v).iter().any(|&u| s[u as usize] == 0))
        .collect();
    for v in boundary {
        s[v] = 2;
    }
    s
}

/// Fixed unit directions in a 3-dimensional mode space: the axes, face
/// diagonals and body diagonals of a cube. When the eigenvalues are nearly
/// equal, as on a cube, the computed modes are an arbitrary rotation of the
/// axis modes, so no single mode is guaranteed to give an axis-aligned cut.
fn directions() -> Vec<[f64; MODES]> {
    let mut out = Vec::new();
    for x in -1i32..=1 {
        for y in -1i32..=1 {
            for z in -1i32..=1 {
                // One of each +/- pair: the first nonzero coordinate is positive.
                let first = [x, y, z].into_iter().find(|&c| c != 0);
                if first != Some(1) {
                    continue;
                }
                let norm = ((x * x + y * y + z * z) as f64).sqrt();
                out.push([x as f64 / norm, y as f64 / norm, z as f64 / norm]);
            }
        }
    }
    out
}

/// Approximates the `MODES` lowest nontrivial eigenvectors of the generalized
/// problem `L x = lambda W x`, with `L` the edge-weighted Laplacian of `g` and
/// `W` its vertex weights, by `steps` Lanczos steps with full
/// reorthogonalization on `S = W^-1/2 L W^-1/2`. Each vector has `g.len()`
/// entries.
fn lowest_modes(g: &Graph, steps: usize) -> Vec<Vec<f64>> {
    let n = g.len();
    let root: Vec<f64> = g.weight.iter().map(|&w| (w as f64).sqrt()).collect();
    let norm = root.iter().map(|x| x * x).sum::<f64>().sqrt();
    let trivial: Vec<f64> = root.iter().map(|x| x / norm).collect();
    let apply = |x: &[f64], y: &mut [f64]| {
        for v in 0..n {
            let mut acc = 0.0;
            let xv = x[v] / root[v];
            for k in g.ptr[v]..g.ptr[v + 1] {
                let u = g.adj[k] as usize;
                acc += g.edge[k] as f64 * (xv - x[u] / root[u]);
            }
            y[v] = acc / root[v];
        }
    };
    let project = |x: &mut [f64], basis: &[Vec<f64>]| {
        for b in std::iter::once(&trivial).chain(basis.iter()) {
            let dot: f64 = x.iter().zip(b).map(|(a, c)| a * c).sum();
            x.iter_mut().zip(b).for_each(|(a, c)| *a -= dot * c);
        }
    };
    // Deterministic start vector with no symmetry to be orthogonal to modes.
    let mut q: Vec<f64> = (0..n)
        .map(|v| ((v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
        .collect();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let (mut alpha, mut beta) = (Vec::new(), Vec::new());
    let mut w = vec![0.0; n];
    for _ in 0..steps {
        project(&mut q, &basis);
        project(&mut q, &basis);
        let len = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        if len < 1e-10 {
            break;
        }
        if !basis.is_empty() {
            beta.push(len);
        }
        q.iter_mut().for_each(|x| *x /= len);
        apply(&q, &mut w);
        alpha.push(q.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>());
        basis.push(std::mem::replace(&mut q, w.clone()));
    }
    let m = basis.len();
    let mut t = vec![vec![0.0; m]; m];
    for i in 0..m {
        t[i][i] = alpha[i];
        if i + 1 < m {
            t[i][i + 1] = beta[i];
            t[i + 1][i] = beta[i];
        }
    }
    let (values, vectors) = jacobi(t);
    let mut rank: Vec<usize> = (0..m).collect();
    rank.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    rank.iter()
        .take(MODES)
        .map(|&k| {
            (0..n)
                .map(|v| (0..m).map(|i| vectors[i][k] * basis[i][v]).sum::<f64>() / root[v])
                .collect()
        })
        .chain(std::iter::repeat_with(|| vec![0.0; n]))
        .take(MODES)
        .collect()
}

/// Cyclic Jacobi eigendecomposition of symmetric `a`. Returns eigenvalues and
/// a matrix whose column `k` is the eigenvector of eigenvalue `k`.
fn jacobi(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let m = a.len();
    let mut v = vec![vec![0.0; m]; m];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..50 {
        let off: f64 = (0..m)
            .flat_map(|i| (0..m).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..m {
            for r in p + 1..m {
                if a[p][r].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[r][r] - a[p][p]) / (2.0 * a[p][r]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..m {
                    let (akp, akr) = (a[k][p], a[k][r]);
                    a[k][p] = c * akp - s * akr;
                    a[k][r] = s * akp + c * akr;
                }
                for k in 0..m {
                    let (apk, ark) = (a[p][k], a[r][k]);
                    a[p][k] = c * apk - s * ark;
                    a[r][k] = s * apk + c * ark;
                }
                for row in v.iter_mut() {
                    let (vp, vr) = (row[p], row[r]);
                    row[p] = c * vp - s * vr;
                    row[r] = s * vp + c * vr;
                }
            }
        }
    }
    ((0..m).map(|i| a[i][i]).collect(), v)
}
