"""The hybrid factorization, CPU and GPU together, through the Python API.

hybrid=True must give the answers the GPU alone gives, for LDLT and LU, real
and complex, and must be a no-op where there is no hybrid path.
"""
import os
import unittest
import numpy as np
from scipy import sparse
import sparta

from sparta._sparta import device

METAL = sparta.gpu_available() and device() == 'metal'
if os.environ.get("SPARTA_REQUIRE_METAL") == "1" and not sparta.gpu_available():
    raise RuntimeError("Metal required: run in normal macOS Terminal")


def laplacian(side: int, shift: complex, skew: float = 0.0) -> sparse.csc_matrix:
    """The 7-point 3D Laplacian on a side^3 grid plus shift on the diagonal.

    skew adds an antisymmetric convection term along x, which makes it
    nonsymmetric.
    """
    one = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(side, side))
    conv = sparse.diags([-skew, skew], [-1, 1], shape=(side, side))
    eye = sparse.eye(side)
    a = (sparse.kron(sparse.kron(eye, eye), one + conv) + sparse.kron(sparse.kron(eye, one), eye)
         + sparse.kron(sparse.kron(one, eye), eye))
    return sparse.csc_matrix(a + shift * sparse.eye(side ** 3))


@unittest.skipUnless(METAL, "the hybrid path needs the Metal backend")
class HybridTests(unittest.TestCase):
    def solve(self, a: sparse.csc_matrix, path: sparta.Path, **options) -> tuple[np.ndarray, dict | None]:
        """Factor a three times, as a sweep would, and solve the last factors."""
        x = np.sin(np.arange(a.shape[0]))
        f = sparta.analyze(a, path=path, refinement_steps=0, **options)
        for _ in range(3):
            f.factorize(a)
        out = f.solve(a @ x)
        timing = f.hybrid_timing
        f.destroy()
        return out, timing

    def check(self, a: sparse.csc_matrix, path: sparta.Path) -> None:
        """hybrid=True matches the GPU alone and reports a CPU share."""
        alone, none = self.solve(a, path)
        both, timing = self.solve(a, path, hybrid=True)
        self.assertIsNone(none)
        self.assertIsNotNone(timing)
        self.assertGreater(timing['share'], 0)
        scale = np.abs(alone).max()
        self.assertLess(np.abs(both - alone).max(), 1e-3 * scale)

    def test_ldlt_complex(self) -> None:
        self.check(laplacian(28, 0.3j - 0.5), sparta.Path.LDLT)

    def test_lu_complex(self) -> None:
        self.check(laplacian(28, 0.3j - 0.5, skew=0.2), sparta.Path.LU)

    def test_real(self) -> None:
        self.check(laplacian(28, -0.5).real.astype(np.float64), sparta.Path.LDLT)

    def test_switch_at_factorize(self) -> None:
        """factorize(hybrid=...) turns it on and off on an existing solver."""
        a = laplacian(24, 0.2j)
        f = sparta.analyze(a, refinement_steps=0).factorize()
        self.assertIsNone(f.hybrid_timing)
        f.factorize(hybrid=True)
        self.assertIsNotNone(f.hybrid_timing)
        self.assertIsNotNone(f.diagnostics()['hybrid'])
        f.factorize(hybrid=0.2)
        self.assertIsNotNone(f.hybrid_timing)
        f.factorize(hybrid=False)
        self.assertIsNone(f.hybrid_timing)
        f.destroy()

    def test_config(self) -> None:
        """Config carries the share, and hybrid= overrides it."""
        a = laplacian(20, 0.2j)
        config = sparta.Config.defaults().with_(hybrid_cpu=0.25)
        f = sparta.analyze(a, config=config, refinement_steps=0).factorize()
        self.assertIsNotNone(f.hybrid_timing)
        f.destroy()
        f = sparta.analyze(a, config=config, hybrid=False, refinement_steps=0).factorize()
        self.assertIsNone(f.hybrid_timing)
        f.destroy()


class HybridOptionTests(unittest.TestCase):
    def test_ignored_on_cpu(self) -> None:
        """The CPU backend has no hybrid path; the setting does nothing there."""
        a = laplacian(10, 0.2j)
        f = sparta.analyze(a, force_cpu=True, hybrid=True, refinement_steps=0).factorize()
        self.assertIsNone(f.hybrid_timing)
        x = np.ones(a.shape[0])
        np.testing.assert_allclose(f.solve(a @ x), x, atol=1e-10)

    def test_rejects_bad_share(self) -> None:
        a = laplacian(6, 0.2j)
        for bad in (1.0, -0.1, 'yes'):
            with self.subTest(hybrid=bad), self.assertRaises(ValueError):
                sparta.analyze(a, hybrid=bad)


if __name__ == "__main__":
    unittest.main()
