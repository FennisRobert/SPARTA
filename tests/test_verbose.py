"""Concise per-call logging without logging every refinement correction."""
import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
import sparta
import sparta._solver as implementation


class VerboseTests(unittest.TestCase):
    """Test silence by default, residuals, toggling and bounded multi-RHS output."""

    def test_logging(self):
        """One solve_many summary is bounded and includes independently computed residuals."""
        class Plan:
            """Diagonal native test double with timing and pivot metadata."""
            factor_device_seconds = 0.01
            factor_seconds = 0.01
            perturbed_pivots = 0

            def __init__(self, *args, **kwargs):
                """Accept symbolic constructor arguments."""
                pass

            def factor(self, values, hybrid_cpu=0.0):
                """Store complex diagonal values."""
                self.values = values
                return self

            def refactor(self, values):
                """Replace diagonal values."""
                self.values = values

            def solve_many(self, block):
                """Solve each row of a (nrhs, n) block, as the extension does."""
                return np.stack([self.solve(row) for row in block])
            def solve(self, rhs):
                """Solve the diagonal test system."""
                return rhs/self.values

            def _print_stats(self):
                """Stand in for four Rust stderr lines."""
                import sys
                for line in ('SPARTA LDLT SCALPEL', 'matrix', 'fill', 'storage'):
                    print(line, file=sys.stderr)

        with patch.object(implementation, '_Analysis', Plan):
            f = sparta.analyze(sparse.eye(2,dtype=complex)).factorize()
            quiet = io.StringIO()
            with contextlib.redirect_stderr(quiet):
                f.solve(np.ones(2))
            self.assertEqual(quiet.getvalue(), '')
            f.factorize(verbose=True)
            out = io.StringIO()
            with contextlib.redirect_stderr(out):
                f.solve_many(np.ones((2,20)))
            self.assertEqual(len(out.getvalue().splitlines()),16)
            self.assertEqual(out.getvalue().count('SPARTA'),1)
            self.assertIn('relative residual=0.000e+00',out.getvalue())
            self.assertIn('12 more RHS omitted',out.getvalue())
            f.factorize(verbose=False)
            with contextlib.redirect_stderr(quiet):
                f.solve(np.ones(2))
            self.assertEqual(quiet.getvalue(),'')

    def test_boolean(self):
        """Reject ambiguous verbosity flags."""
        with self.assertRaises(ValueError):
            sparta.analyze(sparse.eye(2,dtype=complex),verbose='yes')
