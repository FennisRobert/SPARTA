"""CPU contract tests for residual correction; fake solves model imperfect factors."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

import sparta
from sparta._refinement import _refine as refine, RefinementStatus as Status
from sparta._refinement import _Operator, _residual_product

# The benchmark container reader lives beside the tools, not in the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from sparta_io import System  # noqa: E402


class RefinementTests(unittest.TestCase):
    """Exercise convergence and failure handling without requiring Metal."""

    def test_convergence_and_tiny_rhs(self):
        """Normalized corrections retain RHS values below float32 range."""
        a = sparse.eye(3, dtype=np.complex128, format="csc")
        for scale in (1., 1e-200, 1e200):
            b = np.array([1+2j, -3j, 2], dtype=np.complex128) * scale
            x, report = refine(a, b, lambda r: r * np.float32(.99), 10, 1e-10)
            self.assertEqual(report.status, Status.CONVERGED)
            self.assertLess(np.linalg.norm((x-b)/scale), 1e-9)
            self.assertGreater(report.corrections, 0)

    def test_stops_and_keeps_best(self):
        """Reject worsening and nonfinite corrections instead of corrupting x."""
        a = sparse.eye(2, dtype=np.complex128, format="csc")
        b = np.ones(2, dtype=np.complex128)
        for bad, status in [(0., Status.STAGNATED), (np.nan, Status.NONFINITE)]:
            calls = []
            def solve(rhs):
                """Return an inaccurate initial solve followed by a bad correction."""
                calls.append(1)
                return rhs * (.5 if len(calls) == 1 else bad)
            x, report = refine(a, b, solve, 4, 1e-10)
            np.testing.assert_array_equal(x, b*.5)
            self.assertEqual(report.status, status)
            self.assertEqual(report.corrections, 1)
        _, report = refine(a, b, lambda r: r*.5, 1, 1e-10)
        self.assertEqual(report.status, Status.ITERATION_LIMIT)

    def test_zero_and_invalid_initial(self):
        """Zero RHS still invokes the solver, and invalid initial output fails."""
        a = sparse.eye(2, dtype=np.complex128, format="csc")
        calls = []
        def solve(rhs):
            """Record native validity check on a zero vector."""
            calls.append(1)
            return rhs
        _, report = refine(a, np.zeros(2, complex), solve, 5, 1e-10)
        self.assertEqual(len(calls), 1)
        self.assertTrue(report.converged)
        with self.assertRaises(FloatingPointError):
            refine(a, np.ones(2, complex), lambda r: r*np.nan, 5, 1e-10)

    def test_factor_snapshot_and_refactor(self):
        """Original values survive caller mutation and failed refactors invalidate solves."""
        from unittest.mock import patch
        import sparta._solver as impl
        a = sparse.csc_matrix(np.array([[2+1j, 1+2j], [1+2j, 3]], complex))
        class Plan:
            """Dense test double with canonical lower storage."""
            def __init__(self, n, ptr, rows, limit, lu, *args, **kwargs):
                self.ptr, self.rows = ptr, rows
            def factor(self, values):
                self.refactor(values)
                return self
            def refactor(self, values):
                if not np.any(values):
                    raise RuntimeError('failed factor')
                lower = sparse.csc_matrix((values, self.rows, self.ptr), shape=(2,2))
                self.matrix = (lower+sparse.tril(lower,-1).T).toarray()
            def solve_many(self, block):
                """Solve each row of a (nrhs, n) block, as the extension does."""
                return np.stack([self.solve(row) for row in block])
            def solve(self, rhs):
                return np.linalg.solve(self.matrix,rhs).astype(np.complex64)*np.float32(.999)
        with patch.object(impl, '_Analysis', Plan):
            f = sparta.analyze(a, refinement_steps=8).factorize(a*2)
            expected = a.toarray()*2
            a.data[:] = 0
            b = np.array([[1+2j,2],[3j,1]],complex)
            np.testing.assert_allclose(expected@f.solve_many(b),b,atol=1e-9)
            f.factorize(sparse.csc_matrix(expected*1.5))
            np.testing.assert_allclose(expected*1.5@f.solve(b[:,0]),b[:,0],atol=1e-9)
            self.assertEqual(f.solve_many(b[:,:0]).shape,(2,0))
            with self.assertRaises(RuntimeError): f.factorize(a)
            with self.assertRaises(ValueError): f.solve(b[:,0])

    def test_settings(self):
        """Reject invalid iteration counts and dimensionless tolerances."""
        # -1 is legal now: it means as many corrections as rtol needs.
        for value in (-2, True, 1.5):
            with self.assertRaises(ValueError):
                sparta.analyze(sparse.eye(2,dtype=complex), refinement_steps=value)
        for value in (0, 1, np.nan, np.inf, -1):
            with self.assertRaises(ValueError):
                sparta.analyze(sparse.eye(2,dtype=complex), refinement_rtol=value)

    @unittest.skipUnless(sparta.gpu_available(), "Metal unavailable")
    def test_gpu_original_precision(self):
        """LU and LDLT correct rounded factors against original complex128 input."""
        rng = np.random.default_rng(419)
        base = rng.normal(size=(32,32)) + 1j*rng.normal(size=(32,32))
        b = (rng.normal(size=(32,2)) + 1j*rng.normal(size=(32,2))).astype(complex)
        for path in ("lu", "ldlt"):
            dense = (base if path == "lu" else base + base.T) + 40*np.eye(32)
            a = sparse.csc_matrix(dense)
            raw = sparta.analyze(a, path=path, triangle="full",
                                 refinement_steps=0).factorize().solve_many(b)
            factor = sparta.analyze(a, path=path, triangle="full", refinement_steps=8,
                                    refinement_rtol=1e-12).factorize()
            x = factor.solve_many(b)
            self.assertTrue(all(r.converged for r in factor.refinement_reports))
            self.assertLess(np.linalg.norm(a@x-b), np.linalg.norm(a@raw-b)*.01)
            np.testing.assert_allclose(x, np.linalg.solve(dense,b), rtol=1e-9, atol=1e-11)

    def test_minimized_complex_step(self):
        """Optimal complex step corrects a mis-scaled inverse in one iteration."""
        a = sparse.eye(3, dtype=np.complex128, format="csc")
        b = np.array([1+2j, 3-1j, -2j], complex)
        approximate = lambda rhs: rhs * np.complex64(.3+.4j)
        _, plain = refine(a, b, approximate, 1, 1e-6)
        x, minimized = refine(a, b, approximate, 1, 1e-6, True)
        self.assertFalse(plain.converged)
        self.assertTrue(minimized.converged)
        np.testing.assert_allclose(x, b, atol=1e-6)


class ResidualProductTests(unittest.TestCase):
    """The threaded complex128 product refinement measures its residual with."""

    def _matrix(self, n: int = 500) -> sparse.csr_matrix:
        """Return a reproducible complex128 CSR matrix with a usable diagonal."""
        rng = np.random.default_rng(11)
        a = sparse.random(n, n, density=0.01, format='csr', random_state=3)
        a = a.astype(np.complex128)
        a.data = a.data + 1j*rng.standard_normal(a.nnz)
        return (a + sparse.eye(n, format='csr', dtype=np.complex128)*n).tocsr()

    def test_matches_scipy_for_every_block_width(self) -> None:
        """The threaded product agrees with SciPy's to complex128 rounding."""
        a = self._matrix()
        rng = np.random.default_rng(12)
        operator = _Operator(a)
        for k in (1, 2, 5, 16):
            x = rng.standard_normal((a.shape[0], k)) + 1j*rng.standard_normal((a.shape[0], k))
            want = np.column_stack([a @ x[:, j] for j in range(k)])
            got = operator._multiply(x)
            self.assertEqual(got.shape, want.shape)
            scale = np.abs(want).max()
            self.assertLess(np.abs(got - want).max(), 1e-12*scale, f'block width {k}')

    def test_vector_form_matches(self) -> None:
        """`operator @ vector` keeps the one-dimensional shape its callers expect."""
        a = self._matrix()
        x = np.random.default_rng(13).standard_normal(a.shape[0]).astype(np.complex128)
        got = _Operator(a) @ x
        self.assertEqual(got.shape, (a.shape[0],))
        self.assertLess(np.abs(got - a @ x).max(), 1e-12*np.abs(a @ x).max())

    def test_rejects_a_mismatched_row_pointer(self) -> None:
        """A pattern that does not describe the block is refused, not read past."""
        a = self._matrix(64)
        x = np.zeros((63, 1), dtype=np.complex128)
        with self.assertRaises(ValueError):
            _residual_product(np.ascontiguousarray(a.indptr, dtype=np.int64),
                              np.ascontiguousarray(a.indices, dtype=np.int64),
                              np.ascontiguousarray(a.data, dtype=np.complex128), x)


@unittest.skipUnless(sparta.gpu_available(), 'Metal GPU required')
class ErrorEstimateTests(unittest.TestCase):
    """The reported estimate of how wrong the returned solution still is."""

    def test_estimate_bounds_the_true_error(self) -> None:
        """On an ill-conditioned system the estimate exceeds the real error.

        The residual does not: em_m2_sym1 reaches a 4e-7 residual with a
        solution error near 8e-5, so an estimate that merely tracked the
        residual would be useless. This checks the estimate is both above the
        true error and within a couple of orders of it, which is what makes it
        worth printing.
        """
        path = Path(__file__).resolve().parents[1]/'test_matrices/em_m2_sym1.sparta'
        if not path.exists():
            self.skipTest('fixture not present')
        system = System(str(path))
        a = system.matrix().astype(np.complex128).tocsc()
        b = system.rhs[:, 0]
        reference = splu(a).solve(b)
        solver = sparta.analyze(a, triangle='full', refinement_steps=50,
                                refinement_rtol=1e-6).factorize()
        x = solver.solve(b)
        report = solver.refinement_reports[0]
        solver.destroy()
        truth = np.linalg.norm(x - reference)/np.linalg.norm(reference)
        estimate = report.estimated_relative_error
        self.assertGreater(estimate, truth, 'the estimate must not understate the error')
        self.assertLess(estimate, 100*truth, 'the estimate must stay informative')
        self.assertLess(report.relative_residual, truth,
                        'this fixture is only interesting while its residual flatters it')