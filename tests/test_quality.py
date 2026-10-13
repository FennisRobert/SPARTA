"""Factor-quality reporting: a verdict that costs no matvec."""
import math
import unittest

import numpy as np
from scipy import sparse

import sparta
from sparta import FactorQuality
from sparta._quality import DEGRADED_RATIO, LU_UNUSABLE_RATIO, UNUSABLE_RATIO


def _well_conditioned(n: int = 400) -> sparse.csc_matrix:
    """A complex-symmetric matrix with a dominant diagonal, lower triangle."""
    rng = np.random.default_rng(5)
    a = sparse.random(n, n, density=0.01, format='csc', random_state=7)
    a = a.astype(np.complex128)
    a.data = a.data + 1j*rng.standard_normal(a.nnz)
    a = a + a.T + sparse.eye(n, format='csc', dtype=np.complex128)*n
    return sparse.tril(a).tocsc()


class VerdictTests(unittest.TestCase):
    """The verdict rests on measured thresholds, so pin them to the data."""

    def test_healthy_pivot_spread_reads_good(self) -> None:
        q = FactorQuality(0, 1.0, 1e4, math.nan)
        self.assertEqual(q.verdict, 'good')
        self.assertTrue(q.usable)

    def test_pivots_below_float32_resolution_read_degraded(self) -> None:
        """em_m2_sym1 measures 3.4e-8 and still solves to 1.6e-3."""
        q = FactorQuality(0, 5.4e-4, 1.6e4, math.nan)
        self.assertLess(q.pivot_ratio, DEGRADED_RATIO)
        self.assertEqual(q.verdict, 'degraded')
        self.assertTrue(q.usable)

    def test_collapsed_pivot_spread_reads_unusable(self) -> None:
        """em_bandcond_sym1 measures 9.2e-19; refinement cannot move it."""
        q = FactorQuality(6, 2.86e-13, 3.11e5, math.nan)
        self.assertLess(q.pivot_ratio, UNUSABLE_RATIO)
        self.assertEqual(q.verdict, 'unusable')
        self.assertFalse(q.usable)

    def test_a_perturbed_pivot_alone_downgrades(self) -> None:
        self.assertEqual(FactorQuality(1, 1.0, 1e4, math.nan).verdict, 'degraded')

    def test_a_bad_residual_overrides_a_healthy_spread(self) -> None:
        """The residual is the stronger evidence wherever both exist."""
        self.assertEqual(FactorQuality(0, 1.0, 1e4, 0.5).verdict, 'unusable')

    def test_lu_failure_above_the_ldlt_cutoff_reads_unusable(self) -> None:
        """em_bandcond_sym1 on LU measures 2.9e-11 and solves to 5.8."""
        q = FactorQuality(3, 2.9e-11, 1.0, math.nan, lu=True)
        self.assertGreater(q.pivot_ratio, UNUSABLE_RATIO)
        self.assertLess(q.pivot_ratio, LU_UNUSABLE_RATIO)
        self.assertEqual(q.verdict, 'unusable')

    def test_lu_worst_usable_fixture_reads_degraded(self) -> None:
        """em_m2_sym1 on LU measures 4.4e-8 and still solves to 2.7e-3."""
        q = FactorQuality(0, 4.4e-8, 1.0, math.nan, lu=True)
        self.assertEqual(q.verdict, 'degraded')
        self.assertTrue(q.usable)

    def test_unknown_pivot_range_without_a_residual(self) -> None:
        """With no pivot range, absence of perturbation proves nothing."""
        q = FactorQuality(0, math.nan, math.nan, math.nan)
        self.assertTrue(math.isnan(q.pivot_ratio))
        self.assertEqual(q.verdict, 'unknown')
        self.assertTrue(q.usable)


@unittest.skipUnless(sparta.gpu_available(), 'requires Metal')
class SolverQualityTests(unittest.TestCase):
    """The property is reachable on a real factorization, before any solve."""

    def test_available_after_factorize_without_solving(self) -> None:
        solver = sparta.analyze(_well_conditioned(), refinement_steps=0, triangle='lower')
        solver.factorize()
        q = solver.quality
        self.assertEqual(q.verdict, 'good')
        self.assertGreater(q.largest_pivot, 0.0)
        self.assertGreater(q.pivot_ratio, UNUSABLE_RATIO)
        # No solve has run, so no residual has been paid for.
        self.assertTrue(math.isnan(q.relative_residual))
        solver.destroy()

    def test_refinement_fills_in_the_residual_it_already_measured(self) -> None:
        a = _well_conditioned()
        solver = sparta.analyze(a, refinement_steps=3, triangle='lower')
        solver.factorize()
        self.assertTrue(math.isnan(solver.quality.relative_residual))
        rng = np.random.default_rng(9)
        solver.solve(rng.standard_normal(a.shape[0]).astype(np.complex128))
        self.assertTrue(math.isfinite(solver.quality.relative_residual))
        solver.destroy()

    def test_lu_factors_report_a_pivot_range(self) -> None:
        lower = _well_conditioned()
        full = (lower + sparse.tril(lower, -1).T).tocsc()
        solver = sparta.analyze(full, refinement_steps=0, path=sparta.Path.LU)
        solver.factorize()
        q = solver.quality
        self.assertTrue(q.lu)
        self.assertGreater(q.largest_pivot, 0.0)
        self.assertEqual(q.verdict, 'good')
        solver.destroy()

    def test_quality_needs_a_factorization(self) -> None:
        with self.assertRaises(ValueError):
            sparta.analyze(_well_conditioned()).quality
