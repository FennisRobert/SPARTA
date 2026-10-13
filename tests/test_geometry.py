"""Frontal geometry is chosen per analysis and compiled into the Metal kernels.

A wrong value here does not raise: the plan and the shader would simply
disagree about tile shapes and the factorization would return quiet garbage.
These tests check that each setting still solves, so the two sides stay tied
to one source of truth.
"""
import os
import unittest
import numpy as np
from scipy import sparse
import sparta

METAL = sparta.gpu_available()
if os.environ.get("SPARTA_REQUIRE_METAL") == "1" and not METAL:
    raise RuntimeError("Metal required: run in normal macOS Terminal")


def poisson(k: int) -> sparse.csc_matrix:
    """Return the complex-symmetric 7-point Laplacian on a k by k by k grid."""
    n = k * k * k
    at = lambda x, y, z: (z * k + y) * k + x
    rows, cols, vals = [], [], []
    for z in range(k):
        for y in range(k):
            for x in range(k):
                j = at(x, y, z)
                rows.append(j); cols.append(j); vals.append(6.0 + 0.5j)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    if x + dx < k and y + dy < k and z + dz < k:
                        i = at(x + dx, y + dy, z + dz)
                        rows += [i, j]; cols += [j, i]; vals += [-1.0, -1.0]
    return sparse.csc_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.complex128)


@unittest.skipUnless(METAL, "Metal device required")
class GeometryTests(unittest.TestCase):
    """Every supported geometry must factor and solve the same system."""

    @classmethod
    def setUpClass(cls):
        """Build one grid big enough to produce wide fronts and tiled updates."""
        cls.a = poisson(14)
        cls.b = np.ones(cls.a.shape[0], dtype=np.complex128)

    def solve_with(self, **settings) -> float:
        """Analyze, factor and solve with `settings`; return relative residual."""
        f = sparta.analyze(self.a, ordering=sparta.Ordering.SCALPEL, **settings)
        f.factorize(matrix=self.a)
        x = f.solve(self.b)
        return float(np.linalg.norm(self.a @ x - self.b) / np.linalg.norm(self.b))

    def test_non_default_geometry_still_solves(self):
        """Each geometry setting solves to the same accuracy as the default."""
        baseline = self.solve_with()
        self.assertLess(baseline, 1e-4, "default geometry must solve accurately")
        for settings in [
            dict(schur_block=2), dict(schur_block=8),
            dict(strip_block=2), dict(strip_block=8),
            dict(pivot_threads=64), dict(pivot_threads=256),
            dict(schur_chunk=8), dict(schur_chunk=32),
            dict(narrow_cols=16), dict(narrow_cols=64),
            # Two at once: the shader takes them as one macro set, so a
            # combination has to work, not just each setting alone.
            dict(schur_block=2, strip_block=8, schur_chunk=16),
        ]:
            with self.subTest(**settings):
                try:
                    residual = self.solve_with(**settings)
                except ValueError as error:
                    # A geometry past this device's shared memory is refused at
                    # analysis, which is the behaviour Config::check promises.
                    if 'shared memory' not in str(error):
                        raise
                    self.skipTest(f'beyond this device: {error}')
                self.assertLess(residual, 1e-4)

    def test_invalid_geometry_is_rejected(self):
        """Out-of-range geometry raises instead of producing a broken plan."""
        for settings in [
            dict(schur_block=0), dict(schur_block=9),
            dict(strip_block=0), dict(narrow_cols=0),
            dict(pivot_threads=24),          # not a multiple of 32
            dict(schur_chunk=0),
            dict(schur_block=8, schur_chunk=64),   # update_schur over 32 KiB
            dict(strip_block=8, schur_chunk=32),   # update_strip over 32 KiB
        ]:
            with self.subTest(**settings):
                with self.assertRaises(ValueError):
                    sparta.analyze(self.a, **settings)


if __name__ == "__main__":
    unittest.main()
