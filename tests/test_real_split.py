"""The `real_split` setting: complex LDLT with real subtrees in real arithmetic."""
from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

import sparta


def patched_grid(side: int = 12) -> sparse.csc_matrix:
    """3D 7-point grid, indefinite, with imaginary parts on one face only."""
    n = side ** 3
    i = np.arange(n)
    diag = np.where(i % 3 == 0, 0.01, -2.5 + (i % 5)).astype(np.complex128)
    face = i < side * side
    diag[face] += 0.3j
    rows, cols, vals = [i], [i], [diag]
    for step in (1, side, side * side):
        j = i + step
        ok = j < n
        if step == 1:
            ok &= (i // side) == (j // side)
        if step == side:
            ok &= (i // (side * side)) == (j // (side * side))
        a, b = i[ok], j[ok]
        v = (-1.0 + 0.1 * (a % 7)).astype(np.complex128)
        v[face[a] & face[b % n]] += 0.1j
        rows += [a, b]
        cols += [b, a]
        vals += [v, v]
    return sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, n))


class RealSplitTests(unittest.TestCase):
    def test_default_is_on(self) -> None:
        self.assertTrue(sparta.Config.defaults().real_split)

    def test_same_answer_on_and_off(self) -> None:
        a = patched_grid()
        expected = np.exp(1j * np.arange(a.shape[0]) * 0.37)
        b = a @ expected
        for supernodal in (True, False):
            for real_split in (True, False):
                with self.subTest(supernodal=supernodal, real_split=real_split):
                    solver = sparta.analyze(a, force_cpu=True, nemin=4, supernodal=supernodal,
                                            real_split=real_split, refinement_steps=0)
                    x = solver.factorize().solve(b)
                    self.assertLess(np.linalg.norm(x - expected) / np.linalg.norm(expected), 1e-9)

    @unittest.skipUnless(sparta.gpu_available(), 'Metal unavailable')
    def test_gpu_refactorizations(self) -> None:
        # Four factorizations: the GPU times the split against no split on the
        # second and third and keeps one, and every one must solve.
        a = patched_grid()
        expected = np.exp(1j * np.arange(a.shape[0]) * 0.37)
        b = a @ expected
        for real_split in (True, False):
            with self.subTest(real_split=real_split):
                solver = sparta.analyze(a, real_split=real_split)
                for _ in range(4):
                    x = solver.factorize(a).solve(b)
                    self.assertLess(np.linalg.norm(x - expected) / np.linalg.norm(expected), 1e-6)

    def test_config_round_trip(self) -> None:
        config = sparta.Config.defaults().with_(real_split=False)
        self.assertFalse(config.real_split)
        solver = sparta.analyze(patched_grid(6), force_cpu=True, config=config)
        solver.factorize()

    def test_bad_value(self) -> None:
        with self.assertRaises(ValueError):
            sparta.analyze(patched_grid(6), force_cpu=True, real_split=1)


if __name__ == '__main__':
    unittest.main()
