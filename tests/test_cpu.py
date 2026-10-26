"""Complex128 CPU regression tests, independent of Metal availability."""
import unittest
import numpy as np
from scipy import sparse
import sparta


class CpuTests(unittest.TestCase):
    """Check CPU routing, precision, pivoting and reusable factor lifetimes."""

    def test_dense_reference_and_panel_boundaries(self) -> None:
        """Compare both factorizations with NumPy across CPU panel/update routes."""
        rng = np.random.default_rng(630)
        for path in (sparta.Path.LDLT, sparta.Path.LU):
            n = 97
            raw = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
            a = raw + raw.T if path == sparta.Path.LDLT else raw
            b = rng.normal(size=(n, 3)) + 1j*rng.normal(size=(n, 3))
            expected = np.linalg.solve(a, b)
            for panel, threshold in ((1, 1), (3, 10000), (16, 1), (32, 64)):
                with self.subTest(path=path, panel=panel, threshold=threshold):
                    f = sparta.analyze(sparse.csc_matrix(a), path=path, force_cpu=True,
                                      small_front_rows=1, cpu_panel=panel,
                                      cpu_blas_min_rows=threshold, refinement_steps=0).factorize()
                    np.testing.assert_allclose(f.solve_many(b), expected, rtol=2e-10, atol=2e-10)
                    self.assertEqual(f.perturbed_pivots, 0)
                    self.assertEqual(f.diagnostics()['backend'], 'cpu')
                    self.assertEqual(f.diagnostics()['compute_dtype'], 'complex128')
                    self.assertEqual(f.info['gpu_bytes'], 0)
                    self.assertGreater(f.factor_seconds, 0)
                    self.assertEqual(f.factor_device_seconds, 0)
                    f.destroy()

    def test_default_refinement_keeps_double_precision(self) -> None:
        """Refinement, on by default, must hand the double factors unrounded
        right-hand sides; narrowing them to single precision left a residual
        near 1e-8 that refinement then accepted as converged."""
        rng = np.random.default_rng(631)
        n = 200
        for complex_ in (True, False):
            raw = rng.normal(size=(n, n)) + (1j*rng.normal(size=(n, n)) if complex_ else 0)
            a = sparse.csc_matrix(raw + raw.T + n*np.eye(n))
            b = rng.normal(size=(n, 2)) + (1j*rng.normal(size=(n, 2)) if complex_ else 0)
            for path in (sparta.Path.LDLT, sparta.Path.LU):
                with self.subTest(complex_=complex_, path=path):
                    f = sparta.analyze(a, path=path, force_cpu=True).factorize()
                    for x, rhs in ((f.solve(b[:, 0]), b[:, 0]), (f.solve_many(b), b)):
                        residual = np.linalg.norm(a @ x - rhs)/np.linalg.norm(rhs)
                        self.assertLess(residual, 1e-13)
                    f.destroy()

    def test_ill_conditioned_original_values(self) -> None:
        """A difference lost by complex64 must survive factorization without refinement."""
        a = np.array([[1., 1.], [1., 1.+1e-10]], dtype=np.complex128)
        expected = np.array([1., -1.], dtype=np.complex128)
        b = a @ expected
        self.assertEqual(np.linalg.matrix_rank(a.astype(np.complex64)), 1)
        for path in (sparta.Path.LDLT, sparta.Path.LU):
            f = sparta.analyze(sparse.csc_matrix(a), force_cpu=True,
                              path=path, refinement_steps=0).factorize()
            np.testing.assert_allclose(f.solve(b), expected, rtol=2e-6, atol=2e-6)
            self.assertEqual(f.perturbed_pivots, 0)

    def test_dtype_scaling_and_refactor(self) -> None:
        """CPU inputs and outputs retain complex128 through scaling and repeated values."""
        for equilibrate in (False, True):
            a = sparse.diags(np.array([1e-100, 2., 1e100], dtype=np.complex128), format='csc')
            f = sparta.analyze(a, force_cpu=True, equilibrate=equilibrate,
                              refinement_steps=0).factorize()
            x = np.array([1+2j, -3+4j, 0.7-0.2j])
            native = f._native
            for scale in (1., 1.2, -0.9):
                f.factorize(a*scale, safe_mode=True)
                self.assertIs(f._native, native)
                out = f.solve(a @ x * scale)
                self.assertEqual(out.dtype, np.complex128)
                np.testing.assert_allclose(out, x, rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(f.solve_many((a @ x * scale)[:, None]), x[:, None], rtol=1e-13, atol=1e-13)
            f.destroy()
            with self.assertRaises(ValueError):
                f.factorize()
        f = sparta.analyze(sparse.eye(3, dtype=np.float32, format='csc'),
                          force_cpu=True, refinement_steps=0).factorize()
        # A real matrix is factored in float64 on the CPU: a real RHS stays
        # real, a complex one comes back complex128.
        self.assertEqual(f.compute_dtype, np.float64)
        self.assertEqual(f.solve(np.ones(3, dtype=np.float32)).dtype, np.float64)
        self.assertEqual(f.solve(np.ones(3, dtype=np.complex64)).dtype, np.complex128)
        self.assertEqual(f.solve_many(np.empty((3, 0), dtype=np.float32)).dtype, np.float64)

    def test_select_cpu_at_factorize(self) -> None:
        """Late CPU selection reuses analysis, preserves c128 values, and updates byte estimates."""
        a = sparse.csc_matrix([[1., 1.], [1., 1.+1e-10]], dtype=np.complex128)
        x = np.array([1., -1.], dtype=np.complex128)
        f = sparta.analyze(a, refinement_steps=0)
        analysis = f._analysis
        old_bytes = f.info['factor_bytes']
        f.factorize(force_cpu=True)
        self.assertIs(f._analysis, analysis)
        self.assertEqual(f.info['factor_bytes'], old_bytes*2)
        self.assertEqual(f.diagnostics()['backend'], 'cpu')
        np.testing.assert_allclose(f.solve(a @ x), x, rtol=2e-6, atol=2e-6)
        native = f._native
        f.factorize(a*2)
        self.assertIs(f._native, native)
        self.assertEqual(f.diagnostics()['backend'], 'cpu')
        with self.assertRaises(ValueError):
            f.factorize(force_cpu='yes')
        np.testing.assert_allclose(f.solve(a @ x*2), x, rtol=2e-6, atol=2e-6)
        if not sparta.gpu_available():
            with self.assertRaises(RuntimeError):
                f.factorize(force_cpu=False)
            with self.assertRaises(ValueError):
                f.solve(a @ x)
            f.factorize()  # Failed backend selection did not change the successful settings.
            np.testing.assert_allclose(f.solve(a @ x*2), x, rtol=2e-6, atol=2e-6)

    def test_quality_uses_cpu_precision(self) -> None:
        """A small pivot ratio must not apply Metal's unusable cutoff to CPU factors."""
        a = sparse.diags([1., 1e-14], format='csc')
        f = sparta.analyze(a, force_cpu=True, refinement_steps=0).factorize()
        self.assertEqual(f.quality.verdict, 'unknown')
        self.assertTrue(f.quality.usable)
        np.testing.assert_allclose(f.solve(np.array([1., 1e-14])), 1., rtol=1e-14)

    def test_config_and_validation(self) -> None:
        """Config selects CPU, explicit false overrides it, and invalid tuning fails early."""
        config = sparta.Config.defaults().with_(force_cpu=True, cpu_panel=7, cpu_block_rows=3)
        a = sparse.eye(3, dtype=complex, format='csc')
        f = sparta.analyze(a, config=config).factorize()
        np.testing.assert_allclose(f.solve(np.ones(3)), 1)
        self.assertEqual(f.diagnostics()['backend'], 'cpu')
        self.assertIn(sparta.analyze(a, config=config, force_cpu=False).diagnostics()['backend'], ('metal', 'vulkan'))
        for settings in (dict(force_cpu=1), dict(cpu_panel=0), dict(cpu_block_rows=0),
                         dict(cpu_blas_min_rows=0), dict(cpu_panel=-1)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                sparta.analyze(a, **settings)

    def test_sparse_tree_against_scipy(self) -> None:
        """A many-front unsymmetric grid matches an independent sparse LU solve."""
        from scipy.sparse.linalg import spsolve
        n = 400
        a = sparse.diags([np.full(n-1, -0.8+0.1j), np.full(n, 4+0.5j),
                          np.full(n-1, -0.3-0.2j), np.full(n-20, 0.4j)],
                         [-1, 0, 1, 20], format='csc')
        b = np.column_stack([np.ones(n), np.arange(n)/n+0.3j])
        f = sparta.analyze(a, force_cpu=True, path=sparta.Path.LU, nemin=2,
                          small_front_rows=1, cpu_panel=3, cpu_blas_min_rows=1,
                          refinement_steps=0).factorize()
        self.assertGreater(f.info['levels'], 2)
        np.testing.assert_allclose(f.solve_many(b), spsolve(a, b), rtol=1e-12, atol=1e-12)

    @unittest.skipUnless(sparta.gpu_available(), 'Metal unavailable')
    def test_gpu_compatibility(self) -> None:
        """The explicit switch preserves the equations solved by the default Metal path."""
        rng = np.random.default_rng(19)
        for path in (sparta.Path.LDLT, sparta.Path.LU):
            a = rng.normal(size=(80, 80)) + 1j*rng.normal(size=(80, 80))
            if path == sparta.Path.LDLT:
                a += a.T.copy()
            a += np.eye(80)*100
            a = sparse.csc_matrix(a)
            b = rng.normal(size=(80, 2)) + 1j*rng.normal(size=(80, 2))
            cpu = sparta.analyze(a, path=path, force_cpu=True, refinement_steps=0).factorize()
            gpu = sparta.analyze(a, path=path, force_cpu=False, refinement_steps=0).factorize()
            np.testing.assert_allclose(cpu.solve_many(b), gpu.solve_many(b), rtol=2e-5, atol=2e-7)
            analysis = gpu._analysis
            gpu.factorize(force_cpu=True)
            self.assertIs(gpu._analysis, analysis)
            np.testing.assert_allclose(gpu.solve_many(b), cpu.solve_many(b), rtol=1e-12, atol=1e-12)
            gpu.factorize(force_cpu=False)
            self.assertIs(gpu._analysis, analysis)
            self.assertIn(gpu.diagnostics()['backend'], ('metal', 'vulkan'))
            np.testing.assert_allclose(cpu.solve_many(b), gpu.solve_many(b), rtol=2e-5, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
