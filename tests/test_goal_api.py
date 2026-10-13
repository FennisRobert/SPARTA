"""Public enum and reusable factorization contracts without a GPU."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
import sparta
import sparta._solver as solver_impl


class GoalApiTests(unittest.TestCase):
    """Validate routing, future choices and factor lifecycle."""

    def test_enum_errors(self):
        """Known future options fail explicitly rather than silently falling back."""
        a = sparse.eye(2, dtype=complex, format='csc')
        for ordering in sparta.Ordering:
            if ordering != sparta.Ordering.SCALPEL:
                with self.assertRaises(NotImplementedError):
                    sparta.analyze(a, ordering=ordering)
        for path in sparta.Path:
            if path not in (sparta.Path.AUTO, sparta.Path.LU, sparta.Path.LDLT):
                with self.assertRaises(NotImplementedError):
                    sparta.analyze(a, path=path)
        with self.assertRaises(NotImplementedError):
            sparta.analyze(a, mtype=sparta.MatrixType.HERMITIAN_POSITIVE_DEFINITE)

    def test_types(self):
        """Real inputs and Hermitian inputs use mathematically compatible paths."""
        self.assertEqual(sparta.analyze(sparse.eye(2, format='csc')).path, sparta.Path.LDLT)
        a = sparse.csc_matrix([[2, 1j], [-1j, 3]])
        self.assertEqual(sparta.analyze(a, mtype=sparta.MatrixType.HERMITIAN).path, sparta.Path.LU)
        with self.assertRaises(ValueError):
            sparta.analyze(a, mtype=sparta.MatrixType.COMPLEX_SYMMETRIC)
        with self.assertRaises(ValueError):
            sparta.analyze(a, mtype=sparta.MatrixType.HERMITIAN, path=sparta.Path.LDLT)

    def test_lifecycle(self):
        """Factorize keeps buffers for repeated values and refuses premature solves."""
        a = sparse.eye(2, dtype=complex, format='csc')
        class Native:
            """Diagonal numerical stand-in for Metal."""
            def __init__(self, values):
                """Store diagonal complex values."""
                self.values = values
            def refactor(self, values):
                """Replace diagonal values."""
                self.values = values
            def solve_many(self, block):
                """Solve each row of a (nrhs, n) block, as the extension does."""
                return np.stack([self.solve(row) for row in block])
            def solve(self, b):
                """Solve diagonal equations for vector b."""
                return b/self.values
        class Plan:
            """Symbolic test double."""
            def __init__(self, *args, **kwargs):
                """Accept native symbolic metadata."""
                pass
            def factor(self, values):
                """Allocate one numerical factor."""
                return Native(values)
        with patch.object(solver_impl, '_Analysis', Plan):
            settings = dict(ordering=sparta.Ordering.SCALPEL, mtype=sparta.MatrixType.COMPLEX_SYMMETRIC, path=sparta.Path.LDLT)
            f = sparta.analyze(a, **settings)
            with self.assertRaises(ValueError):
                f.solve(np.ones(2))
            self.assertIs(f.factorize(**settings), f)
            np.testing.assert_allclose(f.solve(np.ones(2)), 1)
            native = f._native
            f.factorize(matrix=a*2)
            self.assertIs(f._native, native)
            np.testing.assert_allclose(f.solve_many(np.ones((2,3))), .5)
            f.factorize(matrix=a*3, refinement_steps=3)
            f.factorize(matrix=a*4)
            self.assertEqual(f._options.refinement_steps, 3)
            np.testing.assert_allclose(f.solve(np.ones(2)), .25)

    def test_documented_signatures(self):
        """Public operations explain every named argument and accept no catch-all kwargs."""
        import inspect
        for function in (sparta.analyze, sparta.Solver.factorize, sparta.Solver.solve, sparta.Solver.solve_many):
            for name, parameter in inspect.signature(function).parameters.items():
                self.assertNotEqual(parameter.kind, inspect.Parameter.VAR_KEYWORD)
                if name != 'self':
                    self.assertIn(name, function.__doc__)
        for name in ('Settings', 'Analysis', 'Factor', 'lu', 'ldlt', 'solve'):
            self.assertFalse(hasattr(sparta,name))

    def test_same_settings_and_latest_values(self):
        """A declared non-diagonal symmetric matrix can repeat settings and latest values."""
        import sparta._solver as impl
        a=sparse.csc_matrix(np.array([[3.,1.],[1.,4.]],complex))
        class Plan:
            def __init__(self,n,ptr,rows,limit,lu,*args,**kwargs):
                """Capture lower CSC indices for the numerical stand-in."""
                self.ptr,self.rows=ptr,rows
            def factor(self,values):
                """Allocate the test factor."""
                self.refactor(values)
                return self
            def refactor(self,values):
                """Set the lower test matrix values."""
                lower=sparse.csc_matrix((values,self.rows,self.ptr),shape=(2,2))
                self.a=(lower+sparse.tril(lower,-1).T).toarray()
            def solve_many(self, block):
                """Solve each row of a (nrhs, n) block, as the extension does."""
                return np.stack([self.solve(row) for row in block])
            def solve(self,b):
                """Solve test RHS b."""
                return np.linalg.solve(self.a,b)
        settings=dict(mtype=sparta.MatrixType.COMPLEX_SYMMETRIC,path=sparta.Path.LDLT)
        with patch.object(impl,'_Analysis',Plan):
            f=sparta.analyze(a,**settings).factorize(**settings)
            f.factorize(a*2).factorize(refinement_steps=3)
            np.testing.assert_allclose(a*2@f.solve(np.ones(2)),np.ones(2))
            with self.assertRaises(TypeError): f.factorize(a.data)
            np.testing.assert_allclose(a*2@f.solve(np.ones(2)),np.ones(2))

    @unittest.skipUnless(sparta.gpu_available(), "Metal unavailable")
    def test_destroy_releases_and_disables(self):
        """destroy() hands storage back at once and leaves the solver spent."""
        a = sparse.csc_matrix(np.diag([4.0, 3.0, 2.0]).astype(np.complex128))
        b = np.ones(3, dtype=np.complex128)
        f = sparta.analyze(a).factorize()
        np.testing.assert_allclose(a @ f.solve(b), b, atol=1e-6)
        f.destroy()
        # Size and path survive; anything needing released storage does not.
        self.assertEqual(f.n, 3)
        self.assertEqual(f.path, sparta.Path.LDLT)
        for call in (lambda: f.solve(b), lambda: f.factorize(), lambda: f.info):
            with self.assertRaises(ValueError):
                call()
        f.destroy()

    @unittest.skipUnless(sparta.gpu_available(), "Metal unavailable")
    def test_solver_is_a_context_manager(self):
        """Leaving the block releases the solver even when the body raises."""
        a = sparse.csc_matrix(np.diag([2.0, 5.0]).astype(np.complex128))
        with sparta.analyze(a) as f:
            f.factorize()
            np.testing.assert_allclose(a @ f.solve(np.ones(2)), np.ones(2), atol=1e-6)
        with self.assertRaises(ValueError):
            f.info
        outer = sparta.analyze(a)
        with self.assertRaises(RuntimeError):
            with outer:
                raise RuntimeError('body failed')
        with self.assertRaises(ValueError):
            outer.info

    def test_internal_functions_are_typed_and_private(self):
        """Keep helper names and annotations explicit as the implementation evolves."""
        import ast
        from pathlib import Path
        package = Path(sparta.__file__).parent
        public = {'analyze', 'solve', 'solve_many', 'factorize', 'diagnostics', 'destroy',
                  'n', 'info', 'refinement_reports', 'perturbed_pivots', 'reanalyses',
                  'factor_device_seconds', 'factor_seconds', 'converged', 'estimated_relative_error',
                  # FactorQuality, reachable as Solver.quality.
                  'quality', 'pivot_ratio', 'verdict', 'usable',
                  # Config and its constructors, which mirror sparta::Config.
                  'presets', 'defaults', 'preset', 'with_', 'to_dict',
                  # Saved profiles and the optimizer that makes them.
                  'profiles', 'save', 'load', 'delete', 'store', 'resolve', 'optimize'}
        for source in package.glob('*.py'):
            for node in ast.walk(ast.parse(source.read_text())):
                if not isinstance(node, ast.FunctionDef):
                    continue
                self.assertTrue(node.name.startswith('_') or node.name in public, node.name)
                self.assertIsNotNone(node.returns, (source.name, node.name))
                for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                    if argument.arg not in ('self', 'cls'):
                        self.assertIsNotNone(argument.annotation, (node.name, argument.arg))
