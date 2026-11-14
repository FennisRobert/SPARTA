"""Sparse right-hand sides and projections: solve_many(B) with a sparse B, and
solve_many(B, project=W) for W^T A^-1 B, against the dense route."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

import sparta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import pathology_bench as bench  # noqa: E402


def ports(n: int, count: int, seed: int = 1, complex_: bool = True) -> sparse.csc_matrix:
    """Port-like columns: six unknowns each, column 2 left empty."""
    rng = np.random.default_rng(seed)
    columns = []
    for j in range(count):
        c = np.zeros(n, dtype=complex if complex_ else float)
        if j != 2:
            idx = rng.choice(n, 6, replace=False)
            c[idx] = rng.standard_normal(6) + (1j * rng.standard_normal(6) if complex_ else 0)
        columns.append(c)
    return sparse.csc_matrix(np.column_stack(columns))


def gap(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.abs(actual - expected).max() / np.abs(expected).max())


ROUTES = [dict(force_cpu=True), dict(force_cpu=True, equilibrate=True),
          dict(force_cpu=True, supernodal=False), dict(force_cpu=True, real_split=False),
          dict(force_cpu=True, refinement_steps=0), dict(force_cpu=True, path=sparta.Path.LU)]
if sparta.gpu_available():
    ROUTES += [dict(force_cpu=False), dict(force_cpu=False, hybrid=0.3)]


class ProjectionTests(unittest.TestCase):
    def test_every_route_matches_the_dense_one(self) -> None:
        # Ports on a real body (the real split), on every route, fast or not.
        a = bench.patched(8)
        n = a.shape[0]
        b, w = ports(n, 9), ports(n, 4, seed=5) * (0.3 - 1j)
        for route in ROUTES:
            with self.subTest(**{k: str(v) for k, v in route.items()}):
                solver = sparta.analyze(a, **route).factorize()
                tol = 1e-12 if route.get('force_cpu') else 1e-5
                x = solver.solve_many(b.toarray())
                self.assertLess(gap(solver.solve_many(b), x), tol)
                z = solver.solve_many(b, project=b)
                route_taken = solver.diagnostics()['solve_route']
                self.assertLess(gap(z, b.T @ x), tol)
                if route.get('path') != sparta.Path.LU:
                    self.assertTrue(np.array_equal(z, z.T), 'B^T A^-1 B must be exactly symmetric')
                self.assertTrue(np.all(z[2] == 0) and np.all(z[:, 2] == 0), 'empty port')
                self.assertLess(gap(solver.solve_many(b, project=w), w.T @ x), tol)
                # One column through solve, sparse or dense, projected or not.
                self.assertLess(gap(solver.solve(b[:, [0]]), x[:, 0]), tol)
                self.assertLess(gap(solver.solve(b[:, [0]], project=w), w.T @ x[:, 0]), tol)
                self.assertLess(gap(solver.solve(x[:, 0] * 0 + b[:, [0]].toarray()[:, 0], project=w),
                                    w.T @ x[:, 0]), tol)
                fast = route.get('force_cpu') and route.get('path') != sparta.Path.LU
                self.assertEqual(route_taken, 'sparse' if fast else 'dense')

    def test_real_matrix_with_complex_ports(self) -> None:
        a = bench.lap3d(7, 2.1).real.tocsc()
        n = a.shape[0]
        b, w = ports(n, 5), ports(n, 3, seed=9, complex_=False)
        solver = sparta.analyze(a, force_cpu=True).factorize()
        x = solver.solve_many(b.toarray())
        self.assertLess(gap(solver.solve_many(b, project=b), b.T @ x), 1e-12)
        self.assertLess(gap(solver.solve_many(b, project=w), w.T @ x), 1e-12)
        real = ports(n, 5, complex_=False)
        z = solver.solve_many(real, project=real)
        self.assertFalse(np.iscomplexobj(z), 'all-real inputs give a real projection')

    def test_sparse_solve_keeps_the_refinement_contract(self) -> None:
        # A sparse solve on the fast route is checked like a refined one: it
        # reports convergence, and falls back to refinement when it must.
        a = bench.patched(6)
        solver = sparta.analyze(a, force_cpu=True).factorize()
        solver.solve_many(ports(a.shape[0], 4))
        reports = solver.refinement_reports
        self.assertEqual(len(reports), 4)
        self.assertTrue(all(r.converged and r.corrections == 0 for r in reports))

    def test_bad_projection_is_rejected(self) -> None:
        a = bench.lap3d(4, 0.3)
        solver = sparta.analyze(a, force_cpu=True).factorize()
        b = ports(64, 2)
        with self.assertRaisesRegex(ValueError, 'n rows'):
            solver.solve_many(b, project=np.ones((63, 2)))
        bad = np.ones((64, 1))
        bad[3] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            solver.solve_many(b, project=bad)
        with self.assertRaisesRegex(ValueError, 'shape'):
            solver.solve(b)  # two columns through solve()

    def test_multi_column_refinement_with_mixed_convergence(self) -> None:
        # Regression: when some columns met refinement_rtol after the first
        # solve and others did not, the first correction carried every
        # column's residual and failed on a shape mismatch.
        a = bench.patched(9)
        rng = np.random.default_rng(1)
        b = rng.standard_normal((a.shape[0], 12)) + 0j
        route = dict(force_cpu=False) if sparta.gpu_available() else dict(force_cpu=True, refinement_rtol=1e-15)
        solver = sparta.analyze(a, **route).factorize()
        x = solver.solve_many(b)
        self.assertLess(np.linalg.norm(a @ x - b) / np.linalg.norm(b), 1e-5)


if __name__ == '__main__':
    unittest.main()
