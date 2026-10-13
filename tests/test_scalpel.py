"""SCALPEL routing and numerical validation through the public Python API."""
import unittest
import numpy as np
from scipy import sparse
import sparta


class ScalpelTests(unittest.TestCase):
    """Exercise both analysis paths and preserve the SCALPEL default."""

    def test_ordering_selection(self) -> None:
        """Both named backends reach Rust for LDLT and LU on a recursive mesh."""
        k = 20
        line = sparse.diags([-np.ones(k-1), np.full(k, 3), -np.ones(k-1)], [-1, 0, 1])
        a = (sparse.kronsum(line, line) + 0.2j*sparse.eye(k*k)).tocsc()
        for ordering in (sparta.Ordering.SCALPEL,):
            for path in (sparta.Path.LDLT, sparta.Path.LU):
                plan = sparta.analyze(a, ordering=ordering, path=path)
                self.assertEqual(plan.n, k*k)
                self.assertGreater(plan.info['supernodes'], 1)
                self.assertEqual(plan._options.ordering, ordering)
        self.assertEqual(sparta.analyze(a)._options.ordering, sparta.Ordering.SCALPEL)

    @unittest.skipUnless(sparta.gpu_available(), 'requires Apple Silicon Metal')
    def test_solve_and_refactor(self) -> None:
        """Both backends solve and refactor complex symmetric and unsymmetric meshes."""
        k = 20
        line = sparse.diags([-np.ones(k-1), np.full(k, 3), -np.ones(k-1)], [-1, 0, 1])
        symmetric = (sparse.kronsum(line, line) + 0.2j*sparse.eye(k*k)).tocsc()
        for ordering in (sparta.Ordering.SCALPEL,):
            for lu in (False, True):
                a = symmetric.copy()
                if lu:
                    a = (a + 0.1j*sparse.triu(a, 1)).tocsc()
                exact = np.sin(np.arange(k*k)) + 0.3j
                solver = sparta.analyze(a, ordering=ordering, refinement_steps=0).factorize()
                for scale in (1.0, 1.3):
                    current = a*scale
                    if scale != 1.0:
                        solver.factorize(current)
                    rhs = current @ exact
                    x = solver.solve(rhs)
                    self.assertLess(np.linalg.norm(current @ x-rhs)/np.linalg.norm(rhs), 2e-5)
                    np.testing.assert_allclose(x, exact, rtol=2e-4, atol=2e-4)
