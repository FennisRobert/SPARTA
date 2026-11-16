"""Pathology bench: small matrices built to break a sparse direct solver.

Every case says what a correct solver must do with it, and the bench checks
that SPARTA does exactly that on every route (CPU, GPU, multifrontal, hybrid):

    reject    bad input. Must raise ValueError, TypeError or
              NotImplementedError before any numerical work is trusted.
    solve     well-posed, even if hard to pivot. Must return x with a small
              relative residual.
    flag      singular or numerically hopeless. May raise RuntimeError, or
              return a result, but then it must say so: perturbed pivots,
              refinement that did not converge, a 'degraded' or 'unusable'
              quality verdict, or a warning.
    either    well-posed but outside what the precision can represent (for
              example entries near the f32 range on the GPU). Solving
              accurately passes, and so does a clear error or a flag.
    limit     well-posed, but outside what SPARTA's pivoting can handle (a
              known limitation, named in the case). Solving passes; otherwise
              it must warn with sparta.AccuracyWarning (or its subclass
              UnsupportedMatrixWarning) or raise, and the result is reported
              as KNOWN rather than a failure.

Anything else is a failure. The one that matters most is SILENT: a result
with a large residual and no warning of any kind.

Each case runs in its own subprocess, so a crash or a hang is reported as
such instead of ending the run.

Usage:
    python tools/pathology_bench.py              # every case
    python tools/pathology_bench.py -k scaled    # cases whose name contains "scaled"
    python tools/pathology_bench.py --list
Exit status is 1 when any case fails.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import sparse

# Relative residual a passing solve must reach. Both routes refine in complex128
# to refinement_rtol (1e-6 by default), which is SPARTA's own promise; the GPU
# gets a little slack because it stops on its complex64 correction.
TOL = {'cpu': 1e-6, 'gpu': 1e-5}
TIMEOUT = 120

ROUTES = {
    'cpu': dict(force_cpu=True),
    'cpu-mf': dict(force_cpu=True, supernodal=False),
    'gpu': dict(force_cpu=False),
    'gpu-mf': dict(force_cpu=False, supernodal=False),
    'gpu-hyb': dict(force_cpu=False, hybrid=0.3),
}
INPUT_ERRORS = ('ValueError', 'TypeError', 'NotImplementedError')

RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------
# Matrix builders
# --------------------------------------------------------------------------

def lap3d(k: int, shift: complex = 0.0) -> sparse.csc_matrix:
    """7-point Laplacian on a k^3 grid, minus `shift` on the diagonal."""
    one = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k))
    eye = sparse.identity(k)
    a = (sparse.kron(sparse.kron(one, eye), eye) + sparse.kron(sparse.kron(eye, one), eye)
         + sparse.kron(sparse.kron(eye, eye), one))
    return (a - shift * sparse.identity(k ** 3)).astype(np.complex128).tocsc()


def neumann2d(k: int) -> sparse.csc_matrix:
    """Pure Neumann 2D Laplacian: constant null space, exactly singular."""
    one = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k)).tolil()
    one[0, 0] = one[k - 1, k - 1] = 1.0
    one = one.tocsr()
    eye = sparse.identity(k)
    return (sparse.kron(one, eye) + sparse.kron(eye, one)).astype(np.complex128).tocsc()


def curl2d(k: int) -> sparse.csr_matrix:
    """Discrete curl from the edges of a k x k cell grid to its cells."""
    nh = k * (k + 1)                     # horizontal edges, (k+1) rows of k
    h = lambda i, j: i * k + j           # edge below/above cell row i, column j
    v = lambda i, j: nh + i * (k + 1) + j
    rows, cols, vals = [], [], []
    for i in range(k):
        for j in range(k):
            c = i * k + j
            for e, s in ((h(i, j), 1), (v(i, j + 1), 1), (h(i + 1, j), -1), (v(i, j), -1)):
                rows.append(c)
                cols.append(e)
                vals.append(s)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(k * k, nh + k * (k + 1)))


def curlcurl2d(k: int, k0sq: complex) -> sparse.csc_matrix:
    """Edge-element style C^T C - k0^2 I. Singular at k0 = 0 (gradients)."""
    c = curl2d(k)
    a = (c.T @ c).astype(np.complex128)
    return (a - k0sq * sparse.identity(a.shape[0])).tocsc()


def kkt(n: int, m: int) -> sparse.csc_matrix:
    """Saddle point [[A, B^T], [B, 0]]: zero diagonal block, needs 2x2 pivots."""
    side = round(n ** (1 / 3))
    a = lap3d(side).real
    b = sparse.random(m, a.shape[0], density=0.05, random_state=3, format='csr')
    b = b + sparse.eye(m, a.shape[0])
    return sparse.bmat([[a, b.T], [b, None]]).astype(np.complex128).tocsc()


def blockdiag(blocks) -> sparse.csc_matrix:
    return sparse.block_diag(blocks, format='csc').astype(np.complex128)


def patched(k: int) -> sparse.csc_matrix:
    """Indefinite real grid with complex entries only on one face (ports)."""
    a = lap3d(k, shift=0.9).tolil()
    face = range(k * k)
    for i in face:
        a[i, i] += 0.5j
    return a.tocsc()


def sym(a) -> sparse.csc_matrix:
    a = sparse.csc_matrix(a)
    return ((a + a.T) * 0.5).tocsc()


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    expect: str                  # reject / solve / flag / either
    build: Callable              # () -> (matrix, rhs) or a custom runner
    note: str
    analyze: dict = field(default_factory=dict)
    custom: bool = False         # build is a runner(route_kwargs) -> outcome
    routes: tuple = tuple(ROUTES)


CASES: list[Case] = []


def case(name, expect, note, analyze=None, custom=False, routes=tuple(ROUTES)):
    def register(fn):
        CASES.append(Case(name, expect, fn, note, analyze or {}, custom, routes))
        return fn
    return register


def with_rhs(a, x=None):
    """The matrix and a right-hand side with a known solution where possible."""
    n = a.shape[0]
    if x is None:
        x = np.exp(1j * np.arange(n) * 0.37) * (1 + (np.arange(n) % 5))
    return a, a @ x


# Input validation ---------------------------------------------------------

@case('empty_0x0', 'reject', 'zero-size matrix')
def _():
    return sparse.csc_matrix((0, 0), dtype=np.complex128), np.zeros(0)


@case('non_square', 'reject', '3 x 4 matrix')
def _():
    return sparse.random(3, 4, density=1.0, format='csc', random_state=1), np.ones(3)


@case('dense_ndarray', 'reject', 'numpy array instead of a SciPy sparse matrix')
def _():
    return np.eye(4, dtype=np.complex128), np.ones(4)


@case('nan_entry', 'reject', 'NaN stored in the matrix')
def _():
    a = lap3d(4).tolil()
    a[5, 5] = np.nan
    return with_rhs(a.tocsc(), np.ones(64))[0], np.ones(64)


@case('inf_entry', 'reject', 'inf stored off the diagonal')
def _():
    a = lap3d(4).tolil()
    a[3, 2] = a[2, 3] = np.inf
    return a.tocsc(), np.ones(64)


@case('integer_dtype', 'either', 'int64 matrix: may convert or refuse, never misread')
def _():
    a = (lap3d(4).real * 2).astype(np.int64)
    return with_rhs(a.tocsc(), np.arange(64.0))


# NumPy's long double is plain float64 on macOS and Windows, which SPARTA
# solves, and 80- or 128-bit on Linux x86, which it must refuse rather than
# round silently.
@case('longdouble_dtype', 'reject' if np.dtype(np.longdouble).itemsize > 8 else 'solve',
      'longdouble matrix: refused when wider than float64')
def _():
    a = lap3d(4, 0.3).real.astype(np.longdouble)
    return a.tocsc(), np.ones(64)


@case('nan_imaginary', 'reject', 'NaN only in the imaginary part of one entry')
def _():
    a = lap3d(4).tolil()
    a[5, 5] = complex(2.0, np.nan)
    return a.tocsc(), np.ones(64)


@case('negative_zero_imag', 'solve', 'real values stored as x - 0j (negative zero imaginary)')
def _():
    a = lap3d(5, 0.4)
    a.data = a.data.real - 0j * 1.0
    a.data.imag = -0.0
    return with_rhs(a)


@case('one_entry_1e300', 'either', 'one entry 1e300: finite in f64, inf in complex64')
def _():
    a = lap3d(5, 0.4 + 0.1j).tolil()
    a[60, 60] = 1e300
    return with_rhs(a.tocsc())


@case('one_entry_subnormal', 'solve', 'one off-diagonal pair 1e-310 (flushes to 0 in f32)')
def _():
    a = lap3d(5, 0.4 + 0.1j).tolil()
    a[60, 61] = a[61, 60] = 1e-310
    return with_rhs(a.tocsc())


@case('rhs_1e300', 'either', 'right-hand side of size 1e300')
def _():
    a = lap3d(5, 0.4 + 0.1j)
    return a, np.full(125, 1e300 + 0j)


@case('rhs_tiny', 'solve', 'right-hand side of size 1e-300')
def _():
    a, b = with_rhs(lap3d(5, 0.4 + 0.1j))
    return a, b * 1e-300


@case('long_chain', 'solve', 'tridiagonal, n = 200000: a deep elimination tree')
def _():
    n = 200_000
    a = sparse.diags([np.full(n - 1, -1.0), np.full(n, 2.5 + 0.1j), np.full(n - 1, -1.0)],
                     [-1, 0, 1], format='csc')
    return with_rhs(a)


@case('rhs_wrong_length', 'reject', 'RHS one entry short')
def _():
    a = lap3d(4)
    return a, np.ones(63)


@case('rhs_nan', 'reject', 'NaN in the right-hand side')
def _():
    b = np.ones(64, dtype=np.complex128)
    b[7] = np.nan
    return lap3d(4), b


@case('declared_symmetric_but_not', 'reject', 'mtype=SYMMETRIC on an unsymmetric matrix',
      analyze={'mtype': 'complex_symmetric'})
def _():
    a = lap3d(4).tolil()
    a[1, 0] += 0.5
    return a.tocsc(), np.ones(64)


@case('hermitian_full', 'solve', 'Hermitian (not complex symmetric): must take LU and solve')
def _():
    a = lap3d(5).tolil()
    for i in range(a.shape[0] - 1):
        a[i, i + 1] += 0.3j
        a[i + 1, i] -= 0.3j
    return with_rhs(a.tocsc())


@case('lower_triangle_input', 'solve', "triangle='lower' given the lower triangle",
      analyze={'triangle': 'lower'})
def _():
    a = lap3d(5, shift=0.5 + 0.1j)
    x = np.exp(1j * np.arange(a.shape[0]))
    return sparse.tril(a).tocsc(), a @ x


@case('lower_given_upper', 'flag', "triangle='lower' handed only the upper triangle",
      analyze={'triangle': 'lower'})
def _():
    a = lap3d(5, shift=0.5 + 0.1j)
    return sparse.triu(a, 1).tocsc() + sparse.diags(a.diagonal()).tocsc(), np.ones(a.shape[0])


@case('duplicates_unsorted', 'solve', 'COO with duplicate entries, unsorted CSC indices')
def _():
    a = lap3d(5, shift=0.3).tocoo()
    r = np.concatenate([a.row, a.row]); c = np.concatenate([a.col, a.col])
    v = np.concatenate([a.data * 0.5, a.data * 0.5])
    perm = RNG.permutation(len(r))
    dup = sparse.coo_matrix((v[perm], (r[perm], c[perm])), shape=a.shape)
    csc = dup.tocsc(); csc.has_sorted_indices = False
    x = np.exp(1j * np.arange(a.shape[0]))
    return csc, a.tocsc() @ x


@case('near_symmetric_1e-8', 'solve', 'asymmetry above EPS: must route to LU and solve')
def _():
    a = lap3d(5, shift=0.4).tolil()
    a[3, 2] += 1e-8
    return with_rhs(a.tocsc())


# Tiny and degenerate shapes -----------------------------------------------

@case('one_by_one', 'solve', '1 x 1 complex matrix')
def _():
    return with_rhs(sparse.csc_matrix(np.array([[2 - 3j]])))


@case('two_by_two_zero_diag', 'solve', '[[0,1],[1,0]]: only a 2x2 pivot works')
def _():
    return with_rhs(sparse.csc_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128)))


@case('diagonal_only', 'solve', 'diagonal matrix, widely varying entries')
def _():
    d = np.logspace(-6, 6, 300) * np.exp(1j * np.arange(300))
    return with_rhs(sparse.diags(d).tocsc())


@case('many_components', 'solve', '200 disconnected 2x2 and 3x3 blocks plus one grid')
def _():
    blocks = [np.array([[1, 2], [2, 1]]) if i % 2 else np.array([[0, 1, 0], [1, 0, 1], [0, 1, 2]])
              for i in range(200)]
    return with_rhs(blockdiag([sparse.csc_matrix(b) for b in blocks] + [lap3d(5, 0.7)]))


@case('arrow', 'solve', 'one dense row and column: a single front of size n')
def _():
    n = 600
    a = sparse.lil_matrix((n, n), dtype=np.complex128)
    a.setdiag(4 + np.arange(n) % 3)
    a[0, :] = 1.0; a[:, 0] = 1.0; a[0, 0] = n
    return with_rhs(a.tocsc())


@case('dense_small', 'solve', 'fully dense 150 x 150 complex symmetric')
def _():
    m = RNG.standard_normal((150, 150)) + 1j * RNG.standard_normal((150, 150))
    return with_rhs(sparse.csc_matrix(m + m.T + 30 * np.eye(150)))


# Pivoting -----------------------------------------------------------------

@case('kkt_saddle', 'solve', 'saddle point with a zero (2,2) block')
def _():
    return with_rhs(kkt(343, 60))


@case('zero_diagonal_sym', 'limit', '[[0, B], [B^T, 0]], cond ~50: 2x2 partners in other fronts (no delayed pivots)')
def _():
    b = lap3d(5, 1.0 + 0.2j) + 0.3 * sparse.eye(125, k=3)
    return with_rhs(sparse.bmat([[None, b], [b.T, None]]).tocsc())


@case('tiny_pivots', 'limit', 'diagonal 1e-30 under O(1) couplings: effectively a zero diagonal')
def _():
    a = lap3d(5, 0).tolil()
    a.setdiag(1e-30)
    return with_rhs(a.tocsc() + 0.1j * sparse.eye(125, k=7) + 0.1j * sparse.eye(125, k=-7))


@case('helmholtz_indefinite', 'solve', '3D Laplacian shifted deep into its spectrum')
def _():
    return with_rhs(lap3d(8, shift=3.1 + 0.01j))


@case('imaginary_diagonal', 'solve', 'purely imaginary diagonal, real couplings')
def _():
    a = lap3d(6).tolil()
    a.setdiag(1j * (1 + np.arange(216) % 4))
    return with_rhs(a.tocsc())


@case('lu_zero_diagonal', 'limit', 'unsymmetric zero diagonal, cond ~11: needs row exchanges across fronts')
def _():
    n = 200
    a = sparse.random(n, n, density=0.03, random_state=5, format='lil') * (1 + 1j)
    a.setdiag(0)
    a = a + sparse.eye(n, k=1) * 3 + sparse.eye(n, k=-(n - 1)) * 3
    return with_rhs(a.tocsc())


@case('real_patched_ports', 'solve', 'real grid with complex entries on one face (real_split)')
def _():
    return with_rhs(patched(8))


@case('real_float64', 'solve', 'real float64 indefinite matrix')
def _():
    a = lap3d(6, shift=2.2).real
    return a.tocsc(), a @ np.arange(216.0)


# Singular and near-singular ------------------------------------------------

@case('all_zero_values', 'flag', 'nonempty pattern, every value zero')
def _():
    a = lap3d(4); a.data[:] = 0
    return a, np.ones(64)


@case('empty_column', 'flag', 'one row and column with no entries at all')
def _():
    a = lap3d(5, 0.2).tolil()
    a[40, :] = 0; a[:, 40] = 0
    a = a.tocsc(); a.eliminate_zeros()
    return a, np.ones(125)


@case('duplicate_rows', 'flag', 'two identical rows and columns: rank n-1')
def _():
    a = lap3d(4, 0.3).toarray()
    a[10, :] = a[11, :]; a[:, 10] = a[:, 11]
    return sparse.csc_matrix(a), np.ones(64)


@case('neumann_laplacian', 'flag', 'constant null space, random RHS (inconsistent)')
def _():
    return neumann2d(20), RNG.standard_normal(400) + 0j


@case('curlcurl_k0_zero', 'flag', 'C^T C: gradient null space of dimension ~nodes')
def _():
    a = curlcurl2d(12, 0.0)
    return a, RNG.standard_normal(a.shape[0]) + 0j


@case('curlcurl_resonance', 'flag', 'C^T C - k0^2 I with k0^2 an exact eigenvalue')
def _():
    a = curlcurl2d(8, 0.0)
    ev = np.linalg.eigvalsh(a.toarray().real)
    k0sq = ev[ev > 1e-8][3]
    return curlcurl2d(8, k0sq), RNG.standard_normal(a.shape[0]) + 0j


@case('curlcurl_low_freq', 'either', 'C^T C - (1e-6)^2 I: nonsingular but cond ~1e12')
def _():
    return with_rhs(curlcurl2d(12, 1e-12))


@case('hilbert_dense', 'either', 'dense Hilbert-like, cond ~1e17')
def _():
    i = np.arange(14)
    return with_rhs(sparse.csc_matrix(1.0 / (i[:, None] + i[None, :] + 1) + 0j))


@case('graded_laplacian', 'either', 'diffusion coefficient spanning 1e-8..1e8')
def _():
    a = lap3d(8, 0.0)
    s = sparse.diags(np.logspace(-4, 4, 512))
    return with_rhs((s @ a @ s + 1e-3j * sparse.eye(512)).tocsc())


# Scaling against the precision ----------------------------------------------

@case('huge_entries', 'either', 'every entry ~1e30 (f32 products overflow)')
def _():
    return with_rhs(lap3d(6, 0.5 + 0.1j) * 1e30)


@case('tiny_entries', 'either', 'every entry ~1e-30 (f32 products underflow)')
def _():
    return with_rhs(lap3d(6, 0.5 + 0.1j) * 1e-30)


@case('rows_scaled_1e20', 'either', 'symmetric diagonal scaling 1e-10..1e10')
def _():
    s = sparse.diags(np.logspace(-10, 10, 216))
    return with_rhs((s @ lap3d(6, 0.5 + 0.1j) @ s).tocsc())


@case('stored_zero_diagonal', 'solve', 'explicit stored zeros on part of the diagonal')
def _():
    a = kkt(125, 30)
    a = (a + sparse.diags(np.zeros(a.shape[0]))).tocsc()  # keeps zeros stored
    return with_rhs(a)


# Lifecycle and misuse ----------------------------------------------------

def outcome_of(fn):
    """Run fn() and turn its result or exception into an outcome record."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = fn()
        own = [w for w in caught if not issubclass(w.category, RuntimeWarning)]
        return dict(status='ok', warnings=[f'{w.category.__name__}: {str(w.message)[:300]}' for w in own],
                    **(result or {}))
    except Exception as e:  # noqa: BLE001 - the type is the point
        return dict(status='error', etype=type(e).__name__, message=str(e)[:200])


@case('solve_before_factorize', 'reject', 'solve() on an analyzed, unfactored solver', custom=True)
def _(route):
    import sparta
    s = sparta.analyze(lap3d(4), **route)
    return outcome_of(lambda: s.solve(np.ones(64)) and None)


@case('solve_after_destroy', 'reject', 'solve() after destroy()', custom=True)
def _(route):
    import sparta
    s = sparta.analyze(lap3d(4), **route); s.factorize()
    s.destroy()
    return outcome_of(lambda: s.solve(np.ones(64)) and None)


@case('refactor_good_singular_good', 'solve',
      'good, then singular (must fail, no solve), then good again (must recover)', custom=True)
def _(route):
    import sparta
    a = lap3d(5, 0.3 + 0.05j)
    bad = a.copy(); bad.data[:] = 0
    x = np.exp(1j * np.arange(125)); b = a @ x
    s = sparse_solver = sparta.analyze(a, **route)
    s.factorize(a)
    first = outcome_of(lambda: s.factorize(bad) and None)
    stale = outcome_of(lambda: s.solve(b) and None)
    s.factorize(a)
    y = s.solve(b)
    r = float(np.linalg.norm(a @ y - b) / np.linalg.norm(b))
    problems = []
    if first['status'] == 'ok' and not first.get('flagged'):
        try:
            if sparse_solver.perturbed_pivots == 0:
                problems.append('singular refactor accepted without a flag')
        except Exception:  # noqa: BLE001
            pass
    if stale['status'] == 'ok' and first['status'] == 'error':
        problems.append('solve allowed after a failed factorization')
    return dict(status='ok', residual=r, note='; '.join(problems) or
                f"singular step: {first.get('etype', 'accepted')}, stale solve: {stale.get('etype', 'allowed')}",
                problems=problems)


@case('pattern_change', 'solve', 'factorize() with a different sparsity pattern', custom=True)
def _(route):
    import sparta
    a = lap3d(5, 0.3)
    s = sparta.analyze(a, **route); s.factorize(a)
    b2 = lap3d(5, 0.3) + 0.1 * sparse.eye(125, k=11) + 0.1 * sparse.eye(125, k=-11)
    def run():
        s.factorize(b2.tocsc())
        x = np.exp(1j * np.arange(125)); y = s.solve(b2 @ x)
        return dict(residual=float(np.linalg.norm(b2 @ y - b2 @ x) / np.linalg.norm(b2 @ x)))
    return outcome_of(run)


@case('shape_change', 'solve', 'factorize() with a matrix of another size: re-analyzes', custom=True)
def _(route):
    import sparta
    s = sparta.analyze(lap3d(5), **route); s.factorize()
    a = lap3d(4, 0.2)
    def run():
        s.factorize(a)
        b = a @ np.ones(64); y = s.solve(b)
        return dict(residual=float(np.linalg.norm(a @ y - b) / np.linalg.norm(b)))
    return outcome_of(run)


@case('multi_rhs_one_nan', 'reject', 'solve_many with one NaN column', custom=True)
def _(route):
    import sparta
    a = lap3d(4, 0.2); s = sparta.analyze(a, **route); s.factorize()
    b = np.ones((64, 3), dtype=np.complex128); b[5, 1] = np.nan
    return outcome_of(lambda: s.solve_many(b) and None)


@case('zero_rhs', 'solve', 'b = 0 must give x = 0 exactly', custom=True)
def _(route):
    import sparta
    a = lap3d(5, 0.3 + 0.1j); s = sparta.analyze(a, **route); s.factorize()
    def run():
        y = s.solve(np.zeros(125, dtype=np.complex128))
        return dict(residual=float(np.abs(y).max()))
    return outcome_of(run)


# --------------------------------------------------------------------------
# Running one case (in the worker subprocess)
# --------------------------------------------------------------------------

def run_route(c: Case, route: str) -> dict:
    import sparta
    kwargs = dict(ROUTES[route])
    if c.custom:
        out = c.build(kwargs)
        out.setdefault('warnings', [])
        return out
    try:
        a, b = c.build()
    except Exception as e:  # noqa: BLE001
        return dict(status='builderror', etype=type(e).__name__, message=str(e)[:200])
    analyze = dict(c.analyze)
    if 'mtype' in analyze:
        analyze['mtype'] = sparta.MatrixType(analyze['mtype'])
    info: dict = {}

    def run():
        s = sparta.analyze(a, **kwargs, **analyze)
        s.factorize()
        x = s.solve(b)
        x = np.asarray(x)
        if not np.all(np.isfinite(x)):
            return dict(residual=float('inf'), nonfinite=True, **signals(s))
        aa = sparse.csc_matrix(a).astype(np.complex128)
        if analyze.get('triangle') == 'lower':
            aa = aa + sparse.tril(aa, -1).T
        r = relative_residual(aa, x, b)
        return dict(residual=r, **signals(s))

    out = outcome_of(run)
    out.update(info)
    return out


def relative_residual(a, x, b) -> float:
    """||A x - b|| / ||b||, scaled so entries near 1e300 do not overflow."""
    s = float(np.abs(b).max())
    if not s:
        return float(np.abs(x).max())
    sx = max(float(np.abs(x).max()), 1e-300)
    r = (a @ (x / sx)) * (sx / s) - b / s
    return float(np.linalg.norm(r) / np.linalg.norm(b / s))


def signals(s) -> dict:
    """Everything the solver offers to say 'do not trust this'."""
    out = {}
    try:
        out['perturbed'] = int(s.perturbed_pivots)
    except Exception:  # noqa: BLE001
        out['perturbed'] = -1
    try:
        out['verdict'] = s.quality.verdict
    except Exception:  # noqa: BLE001
        out['verdict'] = '?'
    try:
        reps = s.refinement_reports
        out['refine'] = ','.join(sorted({r.status.value for r in reps})) or 'off'
    except Exception:  # noqa: BLE001
        out['refine'] = '?'
    out['cpu'] = s.compute_dtype == np.complex128 if hasattr(s, 'compute_dtype') else None
    return out


def flagged(o: dict) -> bool:
    return bool(o.get('perturbed', 0) > 0 or o.get('verdict') in ('degraded', 'unusable')
                or o.get('refine') not in (None, 'converged', 'off', 'unrefined')
                or o.get('warnings') or o.get('nonfinite'))


def judge(c: Case, route: str, o: dict) -> tuple[str, str]:
    """(verdict, detail) for one outcome. Verdicts: PASS, FAIL, SILENT, CRASH."""
    tol = TOL['cpu' if route.startswith('cpu') else 'gpu']
    if o['status'] in ('crash', 'timeout', 'builderror'):
        return 'CRASH', o.get('message', o['status'])
    err = o['status'] == 'error'
    et = o.get('etype', '')
    if o.get('problems'):
        return 'FAIL', '; '.join(o['problems'])
    if c.expect == 'reject':
        if err and et in INPUT_ERRORS:
            return 'PASS', f'{et}: {o["message"][:70]}'
        if err:
            return 'FAIL', f'wrong error type {et}: {o["message"][:70]}'
        return 'FAIL', 'accepted bad input'
    if err:
        if c.expect == 'limit' and et in ('RuntimeError', 'ValueError'):
            return 'KNOWN', f'{et}: {o["message"][:70]}'
        if c.expect in ('flag', 'either') and et in ('RuntimeError', 'ValueError'):
            return 'PASS', f'{et}: {o["message"][:70]}'
        return 'FAIL', f'{et}: {o["message"][:70]}'
    r = o.get('residual', float('nan'))
    good = np.isfinite(r) and r <= tol
    sig = flagged(o)
    tag = f'res {r:.1e} p{o.get("perturbed", 0)} {o.get("verdict", "")} {o.get("refine", "")}'
    if c.expect == 'solve':
        return ('PASS' if good else ('FAIL' if sig else 'SILENT')), tag
    if c.expect == 'limit':
        warned = any(w.startswith(('AccuracyWarning', 'UnsupportedMatrixWarning')) for w in o.get('warnings', []))
        return ('PASS' if good else 'KNOWN' if warned else 'SILENT'), tag
    if good:
        return 'PASS', tag
    return ('PASS' if sig else 'SILENT'), tag


def worker(name: str) -> None:
    c = next(c for c in CASES if c.name == name)
    for route in c.routes:
        if route.startswith('gpu'):
            import sparta
            if not sparta.gpu_available():
                continue
        t = time.perf_counter()
        o = run_route(c, route)
        o['seconds'] = time.perf_counter() - t
        print(json.dumps(dict(route=route, **o), default=str), flush=True)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-k', default='', help='run cases whose name contains this')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--worker', help=argparse.SUPPRESS)
    ap.add_argument('-v', action='store_true', help='print every route, not only failures')
    args = ap.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    cases = [c for c in CASES if args.k in c.name]
    if args.list:
        for c in cases:
            print(f'{c.name:30} {c.expect:7} {c.note}')
        return 0
    counts: dict[str, int] = {}
    failures = []
    for c in cases:
        try:
            proc = subprocess.run([sys.executable, __file__, '--worker', c.name], capture_output=True,
                                  text=True, timeout=TIMEOUT)
            lines = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith('{')]
            done = {o['route'] for o in lines}
            if proc.returncode != 0:
                tail = (proc.stderr.strip().splitlines() or ['?'])[-1][:150]
                for route in c.routes:
                    if route not in done:
                        lines.append(dict(route=route, status='crash',
                                          message=f'exit {proc.returncode}: {tail}'))
                        break
        except subprocess.TimeoutExpired:
            lines = [dict(route='?', status='timeout', message=f'over {TIMEOUT}s')]
        verdicts = []
        for o in lines:
            v, detail = judge(c, o['route'], o)
            counts[v] = counts.get(v, 0) + 1
            verdicts.append((o['route'], v, detail))
            if v not in ('PASS', 'KNOWN'):
                failures.append((c.name, o['route'], v, detail))
        worst = next((v for v in ('CRASH', 'SILENT', 'FAIL', 'KNOWN') if any(x[1] == v for x in verdicts)),
                     'PASS')
        print(f'{worst:6} {c.name:30} [{c.expect}] {c.note}')
        for route, v, detail in verdicts:
            if args.v or v != 'PASS':
                print(f'         {route:8} {v:6} {detail}')
                if args.v:
                    for w in next(o for o in lines if o['route'] == route).get('warnings', []):
                        print(f'                         warns {w}')
        sys.stdout.flush()
    print('\n' + '  '.join(f'{k}: {v}' for k, v in sorted(counts.items())))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
