"""All six EMerge exports, explicitly factored with LU, every supplied port.

Missing fixtures skip on source-only checkouts; SPARTA_REQUIRE_REAL_MATRICES=1
makes that a failure. SPARTA_MATRIX_DIR overrides test_matrices/. These are
complex64 screening limits, not a claim of double-precision/application accuracy.
"""
import importlib.util
import os
from pathlib import Path
import unittest
import numpy as np
import sparta

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sparta_test_io", ROOT / "tools/sparta_io.py")
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)
#: A supplied reference whose own relative residual exceeds this is not
#: accurate enough to measure a forward error against; see
#: `LuRealMatrixTests.trustworthy_reference`.
REFERENCE_RESIDUAL = 1e-6

METAL = sparta.gpu_available()
if os.environ.get("SPARTA_REQUIRE_METAL") == "1" and not METAL:
    raise RuntimeError("Metal required: run in normal macOS Terminal")


class LuRealFixtureTests(unittest.TestCase):
    def test_exports_have_references_and_reusable_patterns(self):
        directory = Path(os.environ.get("SPARTA_MATRIX_DIR", ROOT / "test_matrices"))
        if not directory.is_dir():
            if os.environ.get("SPARTA_REQUIRE_REAL_MATRICES") == "1":
                self.fail("Missing real LU fixtures")
            self.skipTest("real matrices not installed")
        for model, kind in ((1, "sym"), (2, "sym"), (3, "asym")):
            previous = None
            for frequency in (1, 2):
                path = directory/f"em_m{model}_{kind}{frequency}.sparta"
                with self.subTest(file=path.name):
                    self.assertTrue(path.is_file(), str(path))
                    system = io.System(path)
                    a = system.matrix().tocsc()
                    a.sum_duplicates(); a.sort_indices()
                    self.assertEqual(system.rhs.shape, (system.n, system.nrhs))
                    self.assertIsNotNone(system.solution)
                    self.assertEqual(system.solution.shape, system.rhs.shape)
                    self.assertTrue(np.isfinite(a.data).all())
                    self.assertTrue(np.isfinite(system.rhs).all())
                    self.assertTrue(np.isfinite(system.solution).all())
                    if kind == "asym":
                        self.assertTrue(system.full)
                        pattern = a.copy(); pattern.data[:] = 1
                        self.assertEqual((pattern-pattern.T).nnz, 0)
                        self.assertGreater((a-a.T).nnz, 0)
                    if previous is not None:
                        np.testing.assert_array_equal(a.indptr, previous.indptr)
                        np.testing.assert_array_equal(a.indices, previous.indices)
                    previous = a


@unittest.skipUnless(METAL, "requires Apple Silicon Metal")
class LuRealMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(os.environ.get("SPARTA_MATRIX_DIR", ROOT / "test_matrices"))
        cls.names = [f"em_m{model}_{kind}{freq}.sparta"
                     for model, kind in ((1, "sym"), (2, "sym"), (3, "asym"))
                     for freq in (1, 2)]
        missing = [n for n in cls.names if not (cls.directory/n).is_file()]
        if missing:
            message = "Missing real LU fixtures: " + ", ".join(missing)
            if os.environ.get("SPARTA_REQUIRE_REAL_MATRICES") == "1":
                raise AssertionError(message)
            raise unittest.SkipTest(message)

    def trustworthy_reference(self, system, a):
        """Return a reference solution accurate enough to measure against, and its source.

        The supplied solutions are application exports, and some were produced
        at lower precision than SPARTA now reaches: `em_m2_sym1`'s own residual
        is 1.6e-3, which is no better than the *unrefined* complex64 answer.
        Measuring a forward error against a reference less accurate than the
        result measures the reference. When one is that weak, recompute it with
        the complex128 CPU backend -- an independent implementation of the same
        multifrontal algorithm, which reaches 5e-12 on this system.
        """
        reference, b = system.solution, system.rhs
        worst = max(
            float(np.linalg.norm(a @ reference[:, p] - b[:, p]) / np.linalg.norm(b[:, p]))
            for p in range(system.nrhs)
        )
        if worst <= REFERENCE_RESIDUAL:
            return reference, "supplied", worst
        cpu = sparta.analyze(a, path="lu", triangle="full", force_cpu=True,
                             refinement_steps=0).factorize()
        return cpu.solve_many(b), "complex128 CPU", worst

    def check_solution(self, system, a, factor):
        self.assertIsNotNone(system.solution, "export must contain a reference solution")
        self.assertTrue(np.isfinite(system.solution).all())
        reference, source, supplied_residual = self.trustworthy_reference(system, a)
        self.assertTrue(np.isfinite(reference).all())
        if source != "supplied":
            print(f"LU {Path(system.path).name}: supplied reference residual "
                  f"{supplied_residual:.3g} exceeds {REFERENCE_RESIDUAL:g}; "
                  f"measuring against the {source} solution instead", flush=True)
        x = factor.solve_many(system.rhs)
        self.assertTrue(np.isfinite(x).all())
        a_norm = float(np.asarray(abs(a).sum(axis=1)).max())
        for port in range(system.nrhs):
            with self.subTest(file=Path(system.path).name, port=port):
                b, ref, actual = system.rhs[:, port], reference[:, port], x[:, port]
                r = a @ actual - b
                bnorm = np.linalg.norm(b)
                self.assertGreater(bnorm, 0)
                self.assertGreater(np.linalg.norm(ref), 0)
                residual = float(np.linalg.norm(r)/bnorm)
                reference_residual = float(np.linalg.norm(a @ ref-b)/bnorm)
                error = float(np.linalg.norm(actual-ref)/np.linalg.norm(ref))
                denominator = a_norm*np.linalg.norm(actual, np.inf)+np.linalg.norm(b, np.inf)
                backward_error = float(np.linalg.norm(r, np.inf)/denominator)
                print(f"LU {Path(system.path).name} port={port} ref={source} residual={residual:.6g} "
                      f"ref_residual={reference_residual:.6g} error={error:.6g} "
                      f"backward={backward_error:.6g} rescued={factor.perturbed_pivots} "
                      f"device={factor.factor_device_seconds:.6f}s", flush=True)
                # `trustworthy_reference` has already replaced any reference
                # too weak to measure against, so the forward error is a
                # statement about the solver and the residual floor is real.
                self.assertLess(residual, max(5e-3, 5*reference_residual))
                self.assertLess(error, 1e-2, f"complex64 LU differs by >=1% from the {source} reference")
                self.assertLess(backward_error, 1e-5)

    def test_all_models_all_ports_and_frequency_refactor(self):
        for model, kind in ((1, "sym"), (2, "sym"), (3, "asym")):
            with self.subTest(model=model):
                symbolic = factor = previous = None
                for frequency in (1, 2):
                    system = io.System(self.directory/f"em_m{model}_{kind}{frequency}.sparta")
                    a = system.matrix().tocsc()
                    a.sum_duplicates(); a.sort_indices()
                    if kind == "asym":
                        self.assertTrue(system.full)
                        pattern = a.copy(); pattern.data[:] = 1
                        self.assertEqual((pattern-pattern.T).nnz, 0, "expected structurally symmetric export")
                        self.assertGreater((a-a.T).nnz, 0, "expected numerical asymmetry")
                        self.assertEqual(sparta.analyze(a, triangle="full").path, "lu")
                    if factor is None:
                        symbolic = sparta.analyze(a, path="lu", triangle="full")
                        self.assertEqual(symbolic.path, "lu")
                        factor = symbolic.factorize()
                    else:
                        np.testing.assert_array_equal(previous.indptr, a.indptr)
                        np.testing.assert_array_equal(previous.indices, a.indices)
                        factor.factorize(a)
                    self.check_solution(system, a, factor)
                    previous = a
                del factor, symbolic, previous


if __name__ == "__main__":
    unittest.main(verbosity=2)
