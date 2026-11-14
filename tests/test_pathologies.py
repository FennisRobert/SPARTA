"""Pathological inputs: SPARTA must reject them, solve them, or say clearly
why it could not. The full catalogue is tools/pathology_bench.py; this runs it
and checks the individual messages users rely on."""
from __future__ import annotations

import subprocess
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse

import sparta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import pathology_bench as bench  # noqa: E402


class PathologyTests(unittest.TestCase):
    def test_bench_has_no_failures(self) -> None:
        # Every case on every route: no crash, no silent wrong answer, no
        # wrong error type. Known limitations must warn to pass.
        run = subprocess.run([sys.executable, str(ROOT / 'tools' / 'pathology_bench.py')],
                             capture_output=True, text=True, timeout=1800)
        bad = [line for line in run.stdout.splitlines() if not line.startswith('PASS')]
        self.assertEqual(run.returncode, 0, '\n'.join(bad)[-6000:] + run.stderr[-2000:])

    def test_lower_triangle_rejects_upper_entries(self) -> None:
        a = bench.lap3d(4, 0.3)
        with self.assertRaisesRegex(ValueError, "above the diagonal"):
            sparta.analyze(sparse.triu(a).tocsc(), triangle='lower', force_cpu=True)
        sparta.analyze(sparse.tril(a).tocsc(), triangle='lower', force_cpu=True).factorize()

    def test_singular_error_names_the_unknown(self) -> None:
        a = bench.lap3d(5, 0.2).tolil()
        a[40, :] = 0
        a[:, 40] = 0
        a = a.tocsc()
        a.eliminate_zeros()
        routes = [True] + ([False] if sparta.gpu_available() else [])
        for force_cpu in routes:
            with self.subTest(force_cpu=force_cpu):
                with self.assertRaisesRegex(RuntimeError, r"unknown 40 .*singular"):
                    sparta.analyze(a, force_cpu=force_cpu).factorize()

    def test_inaccurate_solve_warns_with_cause(self) -> None:
        # Zero diagonal whose 2x2 partners land in other fronts: SPARTA cannot
        # pivot across fronts, so the solve must say so rather than look fine.
        b = bench.lap3d(5, 1.0 + 0.2j) + 0.3 * sparse.eye(125, k=3)
        a = sparse.bmat([[None, b], [b.T, None]]).tocsc()
        rhs = a @ np.ones(250)
        solver = sparta.analyze(a, force_cpu=False) if sparta.gpu_available() else None
        if solver is None:
            self.skipTest('needs the GPU, where this case does not converge')
        solver.factorize()
        with self.assertWarnsRegex(sparta.AccuracyWarning, "zero or negligible diagonal entries"):
            solver.solve(rhs)

    def test_zero_diagonal_warns_before_solving(self) -> None:
        # Unsupported yet, so factorize() says so up front, on every route.
        b = bench.lap3d(4, 1.0 + 0.2j)
        a = sparse.bmat([[None, b], [b.T, None]]).tocsc()
        routes = [True] + ([False] if sparta.gpu_available() else [])
        for force_cpu in routes:
            with self.subTest(force_cpu=force_cpu):
                solver = sparta.analyze(a, force_cpu=force_cpu)
                with self.assertWarnsRegex(sparta.UnsupportedMatrixWarning, "128 zero or negligible"):
                    solver.factorize()

    def test_healthy_solve_does_not_warn(self) -> None:
        a = bench.lap3d(6, 0.5 + 0.1j)
        with warnings.catch_warnings():
            warnings.simplefilter('error', sparta.AccuracyWarning)
            sparta.analyze(a).factorize().solve(a @ np.ones(216))

    def test_complex64_overflow_message(self) -> None:
        if not sparta.gpu_available():
            self.skipTest('Metal unavailable')
        a = bench.lap3d(4, 0.3).tolil()
        a[5, 5] = 1e300
        with self.assertRaisesRegex(ValueError, "complex64"):
            sparta.analyze(a.tocsc(), force_cpu=False).factorize()


if __name__ == '__main__':
    unittest.main()
