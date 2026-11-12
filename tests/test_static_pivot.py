"""Static pivoting (`static_pivot`) on matrices with zero diagonal entries.

SPARTA pivots only inside each front and never delays a column to its parent.
When small fronts split a zero-diagonal pair, the front holding one member
meets a pivot that is zero or tiny but nonzero. Before `static_pivot` only an
exact zero was replaced, so a tiny pivot was divided by as found and the
growth it caused made the factor useless while factorize() still succeeded.
"""
import os
import unittest
import numpy as np
import scipy.sparse as sparse
import sparta

if os.environ.get('SPARTA_REQUIRE_METAL') == '1' and not sparta.gpu_available():
    raise RuntimeError('Metal required: run in normal macOS Terminal')


def zero_diagonal_pairs(pairs):
    """Pairs (2i, 2i+1) with a zero diagonal, coupled weakly to the next pair.

    Nonsingular and well conditioned (condition number about 3), with the
    zero diagonal stored explicitly, as an assembled matrix would carry it.
    """
    n = 2 * pairs
    a = np.zeros((n, n), complex)
    for i in range(pairs):
        j, k = 2 * i, 2 * i + 1
        a[k, j] = a[j, k] = 2.0 + i % 5 + 0.7j
        if i + 1 < pairs:
            a[j + 2, j] = a[j, j + 2] = 0.3
            a[k + 2, j] = a[j, k + 2] = -0.2
    rows, cols = np.nonzero(a)
    values = np.r_[a[rows, cols], np.zeros(n)]
    rows, cols = np.r_[rows, np.arange(n)], np.r_[cols, np.arange(n)]
    return a, sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsc()


class StaticPivotTests(unittest.TestCase):
    def test_default_and_validation(self):
        self.assertEqual(sparta.Config.defaults().static_pivot, 1e-12)
        _, m = zero_diagonal_pairs(4)
        for bad in (-1e-12, 1.0, 'small'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sparta.analyze(m, force_cpu=True, static_pivot=bad)

    def test_split_pairs_solve_accurately(self):
        """Small fronts split the pairs; the solve must still be accurate.

        Without static pivoting the CPU error was about 50 and the GPU's 1e11
        here, even after refinement.
        """
        a, m = zero_diagonal_pairs(60)
        expected = np.sin(np.arange(a.shape[0])) + 0.3j
        b = a @ expected
        backends = [True] + ([False] if sparta.gpu_available() else [])
        for force_cpu in backends:
            for supernodal in (True, False):
                with self.subTest(force_cpu=force_cpu, supernodal=supernodal):
                    solver = sparta.analyze(m, force_cpu=force_cpu, nemin=4,
                                            supernodal=supernodal)
                    # Accepted and ignored, like supernodal.
                    solver.factorize(m, static_pivot=0.0)
                    self.assertGreater(solver.perturbed_pivots, 0)
                    error = np.abs(solver.solve(b) - expected).max()
                    self.assertLess(error, 1e-12 if force_cpu else 1e-4)

    def test_zero_keeps_the_exact_zero_rule(self):
        """static_pivot=0 restores the old rule, which this matrix defeats."""
        a, m = zero_diagonal_pairs(60)
        expected = np.sin(np.arange(a.shape[0])) + 0.3j
        solver = sparta.analyze(m, force_cpu=True, nemin=4, static_pivot=0.0,
                                refinement_steps=0)
        solver.factorize(m)
        self.assertGreater(np.abs(solver.solve(a @ expected) - expected).max(), 1.0)

    def test_healthy_matrix_is_untouched(self):
        """A well-scaled indefinite matrix perturbs nothing either way."""
        rng = np.random.default_rng(3)
        n = 300
        a = sparse.random(n, n, density=0.02, random_state=rng, format='csc')
        a = (a + a.T) + sparse.diags(rng.normal(size=n) + 0.5j)
        a = a.tocsc()
        expected = rng.normal(size=n) + 1j * rng.normal(size=n)
        for static_pivot in (0.0, 1e-12):
            with self.subTest(static_pivot=static_pivot):
                solver = sparta.analyze(a, force_cpu=True, nemin=4, static_pivot=static_pivot)
                solver.factorize(a)
                self.assertEqual(solver.perturbed_pivots, 0)
                self.assertLess(np.abs(solver.solve(a @ expected) - expected).max(), 1e-9)


if __name__ == '__main__':
    unittest.main()
