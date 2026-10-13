"""LU validation against complex128 references and manufactured solutions.

Included in validate.sh. SPARTA_REQUIRE_METAL=1 forbids hardware skips.
Real fixtures are tested in test_lu_real.py, separately from these small tests.
"""
import os
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sparse
import sparta

METAL = sparta.gpu_available()
if os.environ.get("SPARTA_REQUIRE_METAL") == "1" and not METAL:
    raise RuntimeError("Metal required: run in normal macOS Terminal")


def dense_problem(n, pivot=False):
    rng = np.random.default_rng(417 + n)
    a = (rng.uniform(-0.05, 0.05, (n, n)) +
         1j * rng.uniform(-0.05, 0.05, (n, n))) / max(1, n)
    a[np.diag_indices(n)] += 3 + 0.4j
    if pivot and n > 1:
        # A cyclic row permutation preserves conditioning and forces exchanges,
        # including for columns beyond a panel boundary. Pattern remains dense.
        a = np.roll(a, 1, axis=0)
    return sparse.csc_matrix(a)


def tree_problem():
    """Two dense children coupled through a separator; unequal Aij and Aji.

    The 24-column child exercises a narrow front; the larger branch and root
    exercise panels. Cyclic row permutations within each dense block force
    local exchanges in children and the root. Tests inspect the actual analyzed shapes, not just sizes.
    """
    sizes = (24, 96, 48)
    rng = np.random.default_rng(82)
    def edge(m, n):
        return sparse.csc_matrix(rng.uniform(-0.03, 0.03, (m, n)) +
                                 1j * rng.uniform(-0.03, 0.03, (m, n)))
    a, b, r = sizes
    return sparse.bmat([[dense_problem(a, pivot=True), None, edge(a, r)],
                        [None, dense_problem(b, pivot=True), edge(b, r)],
                        [edge(r, a), edge(r, b), dense_problem(r, pivot=True)]], format="csc")


def expected_vectors(n):
    rng = np.random.default_rng(625)
    return rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))


class LuSymbolicTests(unittest.TestCase):
    def test_selection_and_full_pattern_contract(self):
        a = tree_problem()
        np.testing.assert_array_equal((a != 0).toarray(), (a.T != 0).toarray())
        self.assertGreater(np.linalg.norm((a - a.T).data), 0)
        sym = sparta.analyze(a, path="auto", triangle="full")
        self.assertEqual(sym.path, "lu")
        self.assertEqual(sym._a.nnz, a.nnz)
        self.assertGreater(sym.info["upper_bytes"], 0)
        self.assertGreater(sym.info["levels"], 1)
        shapes = sym._analysis.cooperative_front_shapes()
        self.assertTrue(any(nc <= 32 < nr for nr, nc in shapes), shapes)
        self.assertTrue(any(nc > 32 for nr, nc in shapes), shapes)
        with self.assertRaises(ValueError):
            sparta.analyze(a, path="lu", triangle="lower")
        with self.assertRaises(ValueError):
            sparta.analyze(a, path="ldlt", triangle="full")
        symmetric = (a + a.T).tocsc()
        self.assertEqual(sparta.analyze(symmetric, triangle="full").path, "ldlt")
        self.assertEqual(sparta.analyze(symmetric, path="lu", triangle="full").path, "lu")
        changed = a.copy()
        changed[0, 0] += 0.3j
        values, new_pattern = sym._validated(changed)
        np.testing.assert_array_equal(values.data, changed.data)
        self.assertFalse(new_pattern)
        # Remove an UPPER entry: lower-only pattern checking would miss this.
        changed = a.copy()
        changed[0, 1] = 0
        changed.eliminate_zeros()
        self.assertTrue(sym._validated(changed)[1])

    def test_nonstructurally_symmetric_input_is_preserved(self):
        a = sparse.csc_matrix(np.array([[4, 1+2j, 0], [0, 5, 2j],
                                        [0.5, 0, 6]], dtype=np.complex128))
        sym = sparta.analyze(a, path="lu", triangle="full")
        np.testing.assert_array_equal(sym._validated(None)[0].data, a.data)
        self.assertEqual(sym._a.nnz, a.nnz)


@unittest.skipUnless(METAL, "requires Apple Silicon Metal")
class LuHardwareTests(unittest.TestCase):
    def verify(self, a, f, expected=None):
        if expected is None:
            expected = expected_vectors(a.shape[0])
        b = a @ expected
        result = f.solve_many(b)
        self.assertEqual(result.dtype, np.dtype(np.complex128))
        self.assertTrue(np.isfinite(result).all())
        for port in range(b.shape[1]):
            with self.subTest(port=port):
                residual = np.linalg.norm(a @ result[:, port] - b[:, port]) / np.linalg.norm(b[:, port])
                error = np.linalg.norm(result[:, port] - expected[:, port]) / np.linalg.norm(expected[:, port])
                self.assertLess(residual, 3e-5)
                self.assertLess(error, 3e-4)
        return result

    def test_dense_pivoting_and_partial_tiles(self):
        for n in (1, 2, 17, 32, 33, 65, 129, 166, 257, 286):
            with self.subTest(n=n):
                a = dense_problem(n, pivot=True)
                original = a.data.copy()
                f = sparta.analyze(a, path=sparta.Path.LU).factorize()
                self.assertEqual(f.perturbed_pivots, 0)
                x = self.verify(a, f)
                # Independent numerical reference as well as manufactured x.
                b = a @ expected_vectors(n)
                ref = np.linalg.solve(a.toarray(), b)
                self.assertLess(np.linalg.norm(x-ref)/np.linalg.norm(ref), 3e-4)
                np.testing.assert_array_equal(a.data, original)
                del f

    def test_large_fronts(self):
        for n in (511, 1024, 2048):
            with self.subTest(n=n):
                a = dense_problem(n, pivot=True)
                f = sparta.analyze(a, path=sparta.Path.LU).factorize()
                self.verify(a, f, expected_vectors(n)[:, :1])
                self.assertEqual(f.perturbed_pivots, 0)
                del f, a

    def test_tree_routes_and_drain_cadence(self):
        a = tree_problem()
        symbolic = sparta.analyze(a, path="lu", triangle="full")
        shapes = symbolic._analysis.cooperative_front_shapes()
        self.assertTrue(any(nc <= 32 < nr for nr, nc in shapes), shapes)
        self.assertTrue(any(nr > nc > 32 for nr, nc in shapes), shapes)
        self.assertGreater(symbolic.info["levels"], 1)
        results = []
        for cadence in (1, 2, 3, 5, 0):
            with self.subTest(cadence=cadence), patch.dict(os.environ, SPARTA_SCHUR_BATCH=str(cadence)):
                f = symbolic.factorize()
                results.append(self.verify(a, f))
                self.assertEqual(f.perturbed_pivots, 0)
                del f
        for result in results[1:]:
            self.assertLess(np.linalg.norm(result-results[0])/np.linalg.norm(results[0]), 3e-5)

    def test_symmetric_and_one_sided_patterns(self):
        a = tree_problem()
        for a in ((a+a.T+12*sparse.eye(a.shape[0])).tocsc(),
                  sparse.triu(dense_problem(65), format="csc")):
            with self.subTest(nnz=a.nnz):
                f = sparta.analyze(a, path=sparta.Path.LU).factorize()
                self.verify(a, f)
                del f

    def test_refactor_invalidation_recovery_and_input_dtypes(self):
        a = tree_problem()
        # refinement_steps=0: the dtype assertion below is about output precision
        # following the inputs, which only holds when refinement is off.
        f = sparta.analyze(a, path=sparta.Path.LU, refinement_steps=0).factorize()
        b = (a @ expected_vectors(a.shape[0]))[:, 0]
        saved = b.copy()
        f.solve(b)
        np.testing.assert_array_equal(b, saved)
        changed = a * (1.15 + 0.2j)
        f.factorize(changed)
        self.verify(changed, f)
        f.factorize(a)
        self.verify(a, f)
        f.factorize(a.astype(np.complex64))
        self.assertEqual(f.solve(b.astype(np.complex64)).dtype, np.dtype(np.complex64))
        f.factorize(a)
        dead = a.copy(); dead.data[:] = 0
        with self.assertRaises(RuntimeError):
            f.factorize(dead)
        with self.assertRaises(ValueError):
            f.solve(b)
        f.factorize(a)
        self.verify(a, f)

    def test_exact_dead_pivot_reports_rescue(self):
        # Dense stored pattern keeps the zero row inside a nonzero front.
        a = dense_problem(65)
        dense = a.toarray()
        dense[-1, :] = 0
        dense[:, -1] = 0
        a.data[:] = dense.ravel(order="F")
        f = sparta.analyze(a, path=sparta.Path.LU).factorize()
        self.assertGreater(f.perturbed_pivots, 0)
        b = np.ones(65, dtype=np.complex128)
        self.assertTrue(np.isfinite(f.solve(b)).all())
        # No accuracy claim for the perturbed singular system.
        f.factorize(dense_problem(65))
        self.assertEqual(f.perturbed_pivots, 0)
        self.verify(dense_problem(65), f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
