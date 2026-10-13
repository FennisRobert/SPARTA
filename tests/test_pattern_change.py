"""factorize() with a new sparsity pattern repeats the symbolic analysis."""
import unittest
import numpy as np
from scipy import sparse
import sparta

METAL = sparta.gpu_available()


def banded(n: int, width: int, seed: int, symmetric: bool = True) -> sparse.csc_matrix:
    """Return a diagonally dominant complex n x n band matrix with `width` off-diagonals.

    symmetric=True gives A == A.T, so AUTO selects LDLT; False gives LU.
    """
    rng = np.random.default_rng(seed)
    a = sparse.diags([np.full(n, 4.0 * width + 2)] + [np.ones(n - k) for k in range(1, width + 1)],
                     list(range(0, width + 1)), shape=(n, n), format="csc", dtype=np.complex128)
    a.data *= 1 + 0.1 * (rng.normal(size=a.nnz) + 1j * rng.normal(size=a.nnz))
    lower = sparse.tril(a, -1).tocsc()
    if symmetric:
        return (sparse.triu(a) + lower.T).tocsc() + sparse.triu(a, 1).T.tocsc()
    other = lower.copy()
    other.data = 1 + 0.1 * (rng.normal(size=other.nnz) + 1j * rng.normal(size=other.nnz))
    return (sparse.triu(a) + other).tocsc()


class PatternChangeTests(unittest.TestCase):
    """New patterns, sizes and storage orders on the CPU backend, so they run anywhere."""

    def check(self, f: sparta.Solver, a: sparse.spmatrix) -> None:
        """Assert f solves a to the default refinement tolerance."""
        rng = np.random.default_rng(5)
        b = rng.normal(size=a.shape[0]) + 1j * rng.normal(size=a.shape[0])
        self.assertLess(np.linalg.norm(a @ f.solve(b) - b) / np.linalg.norm(b), 1e-6)

    def test_new_pattern_is_analyzed_again(self):
        """Both safe_mode settings, both paths: a wider band factors correctly."""
        for symmetric in (True, False):
            for safe in (False, True):
                with self.subTest(symmetric=symmetric, safe_mode=safe):
                    first, second = banded(40, 1, 1, symmetric), banded(40, 3, 2, symmetric)
                    f = sparta.analyze(first, force_cpu=True, safe_mode=safe).factorize()
                    self.assertEqual(f.reanalyses, 0)
                    f.factorize(second)
                    self.assertEqual(f.reanalyses, 1)
                    self.check(f, second)
                    # Values on the new pattern reuse the new analysis.
                    f.factorize(second * 2)
                    self.assertEqual(f.reanalyses, 1)
                    self.check(f, second * 2)
                    # And the old pattern is simply a new one again.
                    f.factorize(first)
                    self.assertEqual(f.reanalyses, 2)
                    self.check(f, first)
                    self.assertEqual(f.diagnostics()['reanalyses'], 2)

    def test_new_size(self):
        """A matrix of a different size replaces n."""
        f = sparta.analyze(banded(30, 2, 3), force_cpu=True).factorize()
        bigger = banded(45, 2, 4)
        f.factorize(bigger)
        self.assertEqual(f.n, 45)
        self.assertEqual(f.info['n'], 45)
        self.check(f, bigger)

    def test_lower_triangle_input(self):
        """triangle='lower' keeps reading only the lower triangle of the new pattern."""
        first, second = banded(30, 1, 5), banded(30, 4, 6)
        f = sparta.analyze(sparse.tril(first, format="csc"), triangle="lower", force_cpu=True).factorize()
        f.factorize(sparse.tril(second, format="csc"))
        self.assertEqual(f.reanalyses, 1)
        self.check(f, second)

    def test_other_storage_order_is_not_a_new_pattern(self):
        """CSR, or unsorted CSC, of the analyzed pattern skips the gather but not the plan."""
        a = banded(30, 2, 7)
        f = sparta.analyze(a, force_cpu=True).factorize()
        f.factorize((a * 3).tocsr())
        shuffled = (a * 5).tocsc()
        column = slice(shuffled.indptr[4], shuffled.indptr[5])
        shuffled.indices[column] = shuffled.indices[column][::-1]
        shuffled.data[column] = shuffled.data[column][::-1]
        shuffled.has_sorted_indices = False
        f.factorize(shuffled)
        self.assertEqual(f.reanalyses, 0)
        self.check(f, a * 5)

    def test_explicit_zero_is_part_of_the_pattern(self):
        """A stored zero that comes and goes changes the pattern, as documented."""
        a = banded(20, 1, 8)
        f = sparta.analyze(a, force_cpu=True).factorize()
        coo = a.tocoo()
        padded = sparse.csc_matrix((np.r_[coo.data, 0, 0], (np.r_[coo.row, 0, 5], np.r_[coo.col, 5, 0])),
                                   shape=a.shape)
        self.assertGreater(padded.nnz, a.nnz)
        f.factorize(padded)
        self.assertEqual(f.reanalyses, 1)
        self.check(f, a)

    def test_failed_analysis_keeps_the_old_factors(self):
        """A new pattern that fails analysis leaves the previous factorization usable."""
        a = banded(30, 1, 9)
        largest = sparta.analyze(a, force_cpu=True).info['max_front']
        f = sparta.analyze(a, force_cpu=True, max_front=largest).factorize()
        dense = sparse.csc_matrix(np.eye(30) * 60 + np.ones((30, 30)), dtype=np.complex128)
        with self.assertRaisesRegex(RuntimeError, "front bound"):
            f.factorize(dense)
        self.assertEqual(f.reanalyses, 0)
        self.check(f, a)
        f.factorize(a * 2)
        self.check(f, a * 2)

    def test_path_still_applies(self):
        """An unsymmetric matrix cannot replace an LDLT analysis."""
        f = sparta.analyze(banded(20, 1, 10), force_cpu=True).factorize()
        with self.assertRaises(ValueError):
            f.factorize(banded(20, 2, 11, symmetric=False))
        self.check(f, banded(20, 1, 10))


@unittest.skipUnless(METAL, "requires Apple Silicon Metal")
class MetalPatternChangeTests(unittest.TestCase):
    """The same change on the GPU, where the factor buffers are rebuilt too."""

    def test_new_pattern_on_the_gpu(self):
        rng = np.random.default_rng(12)
        for safe in (False, True):
            with self.subTest(safe_mode=safe):
                first, second = banded(200, 1, 13), banded(200, 6, 14)
                f = sparta.analyze(first, safe_mode=safe).factorize()
                f.factorize(second)
                self.assertEqual(f.reanalyses, 1)
                b = rng.normal(size=200) + 1j * rng.normal(size=200)
                x = f.solve(b)
                self.assertLess(np.linalg.norm(second @ x - b) / np.linalg.norm(b), 1e-6)


if __name__ == "__main__":
    unittest.main()
