"""Port matrices and sparse right-hand sides on CPU LDLT factors."""
import unittest
import numpy as np
from scipy import sparse
import sparta


def _grid(m: int, k2: complex) -> sparse.csc_matrix:
    """The 2D 5-point operator 4 - k2 on an m x m grid, complex symmetric."""
    t = sparse.diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(m, m))
    a = sparse.kron(sparse.eye(m), t) + sparse.kron(sparse.diags([-1.0, -1.0], [-1, 1], shape=(m, m)), sparse.eye(m))
    return (a - k2 * sparse.eye(m * m)).tocsc()


def _ports(n: int, p: int, k: int, rng: np.random.Generator, complex_: bool) -> sparse.csc_matrix:
    """p columns of k random rows each, with random values."""
    rows = rng.integers(0, n, size=p * k)
    cols = np.repeat(np.arange(p), k)
    values = rng.normal(size=p * k) + (1j * rng.normal(size=p * k) if complex_ else 0)
    return sparse.csc_matrix((values, (rows, cols)), shape=(n, p))


class PortMatrixTests(unittest.TestCase):
    """Check port_matrix and solve_sparse against solve_many on the dense block."""

    def test_against_dense_solves(self) -> None:
        """Real and complex matrices and blocks, with and without equilibration."""
        rng = np.random.default_rng(700)
        n = 400
        for complex_matrix in (True, False):
            raw = sparse.random(n, n, density=0.01, random_state=701, format='csc')
            if complex_matrix:
                raw = raw + 1j * sparse.random(n, n, density=0.01, random_state=702, format='csc')
            a = (raw + raw.T + sparse.eye(n) * 0.5).tocsc()
            for complex_block in (True, False):
                for equilibrate in (False, True):
                    with self.subTest(complex_matrix=complex_matrix, complex_block=complex_block,
                                      equilibrate=equilibrate):
                        f = sparta.analyze(a, path=sparta.Path.LDLT, force_cpu=True,
                                           refinement_steps=0, equilibrate=equilibrate).factorize()
                        b = _ports(n, 23, 4, rng, complex_block)
                        x = f.solve_many(b.toarray())
                        z = f.port_matrix(b)
                        self.assertEqual(z.shape, (23, 23))
                        np.testing.assert_array_equal(z, z.T)
                        expected = b.T @ x
                        scale = np.abs(expected).max()
                        np.testing.assert_allclose(z, expected, rtol=0, atol=1e-10 * scale)
                        np.testing.assert_allclose(f.solve_sparse(b), x, rtol=0, atol=1e-10 * np.abs(x).max())
                        self.assertEqual(z.dtype, x.dtype)
                        f.destroy()

    def test_helmholtz_grid_and_reach(self) -> None:
        """A complex Helmholtz grid; a port reaches part of the factor only."""
        a = _grid(40, 1.3 + 0.05j)
        f = sparta.analyze(a, force_cpu=True, refinement_steps=0).factorize()
        b = _ports(a.shape[0], 30, 6, np.random.default_rng(703), True)
        np.testing.assert_allclose(f.port_matrix(b), b.T @ f.solve_many(b.toarray()), rtol=1e-10, atol=1e-12)
        reach = f.sparse_reach(b[:, 0].indices)
        self.assertGreater(reach['fronts'], 0)
        self.assertLess(reach['factor_entries'], reach['total_factor_entries'])
        every = f.sparse_reach(np.arange(a.shape[0]))
        self.assertEqual(every['fronts'], every['total_fronts'])
        f.destroy()

    def test_empty_and_duplicate_columns(self) -> None:
        """Empty columns give zero rows and columns; duplicate rows are summed."""
        a = _grid(20, 0.9 + 0.1j)
        n = a.shape[0]
        f = sparta.analyze(a, force_cpu=True, refinement_steps=0).factorize()
        rows = np.array([3, 3, 50, 7])
        cols = np.array([0, 0, 0, 2])
        b = sparse.csc_matrix((np.array([1.0, 1.0, 0.5, 1.0]), (rows, cols)), shape=(n, 3))
        z = f.port_matrix(b)
        np.testing.assert_array_equal(z[1], 0)
        np.testing.assert_array_equal(z[:, 1], 0)
        np.testing.assert_allclose(z, b.T @ f.solve_many(b.toarray()), rtol=1e-10, atol=1e-14)
        self.assertEqual(f.port_matrix(sparse.csc_matrix((n, 0))).shape, (0, 0))
        f.destroy()

    def test_refusals_and_input_errors(self) -> None:
        """GPU and LU factors are refused; malformed blocks raise ValueError."""
        a = _grid(12, 0.5 + 0.1j)
        n = a.shape[0]
        b = _ports(n, 3, 2, np.random.default_rng(704), True)
        f = sparta.analyze(a, force_cpu=True, refinement_steps=0)
        with self.assertRaises(ValueError):
            f.port_matrix(b)
        f.factorize()
        with self.assertRaises(ValueError):
            f.port_matrix(b.toarray())
        with self.assertRaises(ValueError):
            f.port_matrix(sparse.csc_matrix((n + 1, 2)))
        bad = b.copy()
        bad.data[0] = np.nan
        with self.assertRaises(ValueError):
            f.port_matrix(bad)
        with self.assertRaises(ValueError):
            f.sparse_reach(np.array([n]))
        f.destroy()
        lu = sparta.analyze(a, path=sparta.Path.LU, force_cpu=True, refinement_steps=0).factorize()
        with self.assertRaises(RuntimeError):
            lu.port_matrix(b)
        with self.assertRaises(RuntimeError):
            lu.solve_sparse(b)
        lu.destroy()
        if sparta.gpu_available():
            g = sparta.analyze(a, refinement_steps=0).factorize()
            with self.assertRaises(RuntimeError):
                g.port_matrix(b)
            with self.assertRaises(RuntimeError):
                g.solve_sparse(b)
            g.destroy()


if __name__ == '__main__':
    unittest.main()
