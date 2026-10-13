"""Equilibration contracts independent of the Metal device."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
import sparta
import sparta._solver as solver_impl
from sparta._scaling import _equilibrate as equilibrate


class ScalingTests(unittest.TestCase):
    """Check transformations and API lifecycle with a dense test double."""

    def test_transform(self):
        """Symmetric scaling balances both halves without conjugating entries."""
        full = np.array([[1e-12+1e-12j, 1j, 0], [1j, 1e12, 0], [0,0,0]], complex)
        for symmetric in (False, True):
            a = sparse.csc_matrix(np.tril(full) if symmetric else full)
            data, left, right = equilibrate(a, a.data, symmetric)
            scaled = sparse.csc_matrix((data, a.indices, a.indptr), shape=a.shape).toarray()
            np.testing.assert_allclose(scaled, left[:,None]*a.toarray()*right[None,:])
            self.assertEqual(left[2], 1)
            self.assertEqual(right[2], 1)
            if symmetric:
                np.testing.assert_array_equal(left, right)
            self.assertLess(np.max(np.abs(scaled)), 2)

    def test_factor_refactor(self):
        """API solves original equations and replaces scaling on value changes."""
        for path in ('lu', 'ldlt'):
            dense = np.array([[2+1j, 1j], [1j, 3]], complex)
            dense = np.diag([1e-4, 1e4])@dense@np.diag([1e-4, 1e4])
            if path == 'lu':
                dense[0,1] *= 2
            a = sparse.csc_matrix(dense)
            class Plan:
                """Dense stand-in isolates Python scaling from Metal numerics."""
                def __init__(self, n, ptr, rows, limit, lu, *args, **kwargs):
                    """Retain canonical CSC pattern and factorization kind."""
                    self.ptr, self.rows, self.lu = ptr, rows, lu
                def factor(self, values):
                    """Store transformed values as a dense test matrix."""
                    self.refactor(values)
                    return self
                def refactor(self, values):
                    """Replace values with the same CSC pattern."""
                    mat = sparse.csc_matrix((values, self.rows, self.ptr), shape=(2,2))
                    self.mat = (mat if self.lu else mat+sparse.tril(mat,-1).T).toarray()
                def solve_many(self, block):
                    """Solve each row of a (nrhs, n) block, as the extension does."""
                    return np.stack([self.solve(row) for row in block])
                def solve(self, b):
                    """Solve transformed test equations in single precision."""
                    return np.linalg.solve(self.mat,b).astype(np.complex64)
            with patch.object(solver_impl, '_Analysis', Plan):
                plan = sparta.analyze(a, path=path, triangle='full', equilibrate=True, refinement_steps=8)
                factor = plan.factorize(a*2)
                b = np.array([1+1j, 2j], complex)
                np.testing.assert_allclose(factor.solve(b), np.linalg.solve(dense*2,b), rtol=1e-7)
                factor.factorize(a*3)
                np.testing.assert_allclose(factor.solve(b), np.linalg.solve(dense*3,b), rtol=1e-7)
                self.assertTrue(factor.refinement_reports[0].converged)

    def test_options(self):
        """Equilibration is optional and accepts only booleans."""
        self.assertFalse(sparta.analyze(sparse.eye(2, dtype=complex))._options.equilibrate)
        with self.assertRaises(ValueError):
            sparta.analyze(sparse.eye(2, dtype=complex), equilibrate='yes')
