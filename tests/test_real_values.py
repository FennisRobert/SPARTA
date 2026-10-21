"""Real-valued matrices through the Python API: dtype detection end to end.

A float32 or float64 matrix is factored with real arithmetic, float32 on the
GPU and float64 with force_cpu, and a real right-hand side comes back real.
"""
import os
import unittest
import numpy as np
from scipy import sparse
import sparta

METAL = sparta.gpu_available()
if os.environ.get("SPARTA_REQUIRE_METAL") == "1" and not METAL:
    raise RuntimeError("Metal required: run in normal macOS Terminal")


def grid(side: int, shift: float, skew: float, dtype: type) -> sparse.csc_matrix:
    """The 5-point Laplacian on a side x side grid plus shift on the diagonal.

    skew adds an antisymmetric convection term, which makes it nonsymmetric.
    """
    one = sparse.diags([-1 - skew, 0, -1 + skew], [-1, 0, 1], shape=(side, side))
    eye = sparse.eye(side)
    a = sparse.kron(eye, one) + sparse.kron(one.T if skew else one, eye)
    a = a + sparse.diags(np.full(side * side, 4.0 + shift))
    return sparse.csc_matrix(a, dtype=dtype)


class RealValueTests(unittest.TestCase):
    def check(self, a: sparse.csc_matrix, force_cpu: bool, tolerance: float) -> None:
        """Solve real and complex right-hand sides against real a."""
        n = a.shape[0]
        x = np.sin(0.37 * np.arange(n))
        z = x + 1j * np.cos(0.11 * np.arange(n))
        reference = sparse.csc_matrix(a, dtype=np.float64)
        f = sparta.analyze(a, force_cpu=force_cpu, refinement_steps=0).factorize()
        self.assertEqual(f.compute_dtype, np.float64 if force_cpu else np.float32)
        out = f.solve((reference @ x).astype(a.dtype))
        self.assertEqual(out.dtype.kind, 'f')
        np.testing.assert_allclose(out, x, atol=tolerance)
        many = f.solve_many(np.column_stack([reference @ x, 2 * (reference @ x)]).astype(a.dtype))
        self.assertEqual(many.dtype.kind, 'f')
        np.testing.assert_allclose(many, np.column_stack([x, 2 * x]), atol=2 * tolerance)
        out = f.solve(reference @ z)
        self.assertEqual(out.dtype.kind, 'c')
        np.testing.assert_allclose(out, z, atol=tolerance)
        f.destroy()

    def test_cpu_float64(self) -> None:
        """LDLT on an indefinite matrix and LU on a nonsymmetric one, in float64."""
        self.check(grid(16, -2.5, 0.0, np.float64), True, 1e-11)
        self.check(grid(16, 0.5, 0.3, np.float64), True, 1e-11)

    @unittest.skipUnless(METAL, "no GPU")
    def test_gpu_float32(self) -> None:
        """The same systems on the GPU in float32, from float32 and float64 input."""
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                self.check(grid(16, -2.5, 0.0, dtype), False, 1e-3)
                self.check(grid(16, 0.5, 0.3, dtype), False, 1e-3)

    @unittest.skipUnless(METAL, "no GPU")
    def test_gpu_refinement_returns_float64(self) -> None:
        """Refinement against a real matrix stays real and reaches double precision."""
        a = grid(16, -2.5, 0.0, np.float64)
        x = np.sin(0.37 * np.arange(a.shape[0]))
        f = sparta.analyze(a, refinement_steps=-1, refinement_rtol=1e-12).factorize()
        out = f.solve(a @ x)
        self.assertEqual(out.dtype, np.float64)
        np.testing.assert_allclose(out, x, atol=1e-10)
        self.assertTrue(f.refinement_reports[0].converged)

    def test_switch_between_real_and_complex(self) -> None:
        """Refactoring the same pattern with complex values switches the arithmetic."""
        a = grid(8, -2.5, 0.0, np.float64)
        x = np.cos(np.arange(a.shape[0]))
        f = sparta.analyze(a, force_cpu=True, refinement_steps=0).factorize()
        self.assertEqual(f.compute_dtype, np.float64)
        b = a.astype(np.complex128) * (1 + 0.5j)
        f.factorize(b)
        self.assertEqual(f.compute_dtype, np.complex128)
        np.testing.assert_allclose(f.solve(b @ x), x, atol=1e-11)
        f.factorize(a)
        self.assertEqual(f.compute_dtype, np.float64)
        np.testing.assert_allclose(f.solve(a @ x), x, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
