"""Run with unittest; SPARTA_REQUIRE_METAL=1 forbids hardware skips."""
import os
import unittest
import numpy as np
import scipy.sparse as sparse
import sparta

if os.environ.get('SPARTA_REQUIRE_METAL') == '1' and not sparta.gpu_available():
    raise RuntimeError('Metal required: run in normal macOS Terminal')

class SymbolicTests(unittest.TestCase):
    def test_tree_and_precision_contract(self):
        a = sparse.diags([np.ones(31), np.full(32, 4), np.ones(31)], [-1,0,1], dtype=np.complex64, format='csc')
        info = sparta.analyze(a).info
        self.assertEqual(info['n'],32)
        self.assertGreater(info['supernodes'],1)
        self.assertGreater(info['levels'],1)
        self.assertEqual(sparta.analyze(a.astype(np.complex128)).n, 32)
        with self.assertRaises(ValueError): sparta.analyze(sparse.tril(a), path="ldlt", triangle="full")
        with self.assertRaises(ValueError): sparta.analyze(a, max_front=0)

    def test_emerge_api_and_options(self):
        """Only the new explicit public surface is exported."""
        a=sparse.eye(4,dtype=complex,format='csc')
        f=sparta.analyze(a,ordering=sparta.Ordering.SCALPEL,max_front=64)
        self.assertEqual(f.n,4)
        for name in ('Settings','Analysis','Factor','ldlt','lu','solve'):
            self.assertFalse(hasattr(sparta,name))
        with self.assertRaises(TypeError): sparta.analyze(a,unknown_option=True)
        with self.assertRaises(ValueError): sparta.analyze(a,path='lu',triangle='lower')
        with self.assertRaises(NotImplementedError): sparta.analyze(a,ordering=sparta.Ordering.AMD)

    def test_conversion_boundary_without_gpu(self):
        from unittest.mock import patch
        class Native:
            def refactor(self, values):
                self.values = values.copy()
            def solve_many(self, block):
                """Solve each row of a (nrhs, n) block, as the extension does."""
                return np.stack([self.solve(row) for row in block])
            def solve(self, rhs):
                self.rhs = rhs.copy()
                return rhs.copy()  # Boundary test only, not a numerical solver.
        a=sparse.eye(3,dtype=np.complex128,format="csc")
        analysis=sparta.analyze(a)
        native=Native()
        import sparta._solver as impl
        with patch.object(impl, '_Analysis', return_value=type('Plan', (), {'factor': lambda self, values, hybrid_cpu=0.0: (native.refactor(values), native)[1]})()):
            # Output precision follows the input dtypes only when refinement is
            # off; refinement accumulates in complex128 whatever it was given.
            f=sparta.analyze(a,refinement_steps=0).factorize()
        self.assertEqual(native.values.dtype,np.complex64)
        self.assertEqual(a.dtype,np.complex128)
        b=np.array([1+2j,3-4j,5],dtype=np.complex128)
        saved=b.copy()
        x=f.solve(b)
        self.assertEqual(x.dtype,np.complex128)
        self.assertEqual(native.rhs.dtype,np.complex64)
        np.testing.assert_array_equal(saved,b)
        self.assertEqual(f.solve_many(b[:,None]).dtype,np.complex128)
        self.assertEqual(f.solve(np.ones(3,dtype=np.complex64)).dtype,np.complex128)
        f.factorize(a.astype(np.complex64))
        self.assertEqual(f.solve(np.ones(3,dtype=np.complex64)).dtype,np.complex64)
        f.factorize(a)
        self.assertEqual(native.values.dtype,np.complex64)
        bad=a.copy();bad.data[0]=1e300
        with self.assertRaisesRegex(ValueError,"overflows"): f.factorize(bad,safe_mode=True)
        # Without safe_mode the value checks are skipped: the overflow reaches
        # the factorization as an infinity instead of raising here.
        f.factorize(bad)
        f.factorize(a)
        with self.assertRaisesRegex(ValueError,"overflows"): f.solve(b*1e300)
        with self.assertRaises(ValueError): f.solve(b*np.nan)

    def test_limits_and_explicit_unavailability(self):
        a = sparse.csc_matrix(np.ones((8,8),dtype=np.complex64)+np.eye(8,dtype=np.complex64))
        with self.assertRaises(RuntimeError): sparta.analyze(a,max_front=4)
        if not sparta.gpu_available():
            with self.assertRaisesRegex(RuntimeError,'Metal'): sparta.analyze(a,path=sparta.Path.LDLT).factorize()

@unittest.skipUnless(sparta.gpu_available(), 'requires Apple Silicon Metal')
class HardwareTests(unittest.TestCase):
    def check(self, dense):
        a=sparse.csc_matrix(np.asarray(dense,dtype=np.complex64))
        analysis=sparta.analyze(a)
        f=analysis.factorize()
        rng=np.random.default_rng(431)
        b=(rng.normal(size=a.shape[0])+1j*rng.normal(size=a.shape[0])).astype(np.complex64)
        def verify(matrix):
            x=f.solve(b)
            residual=np.linalg.norm(matrix@x-b)/np.linalg.norm(b)
            reference=np.linalg.solve(matrix.toarray().astype(np.complex128),b.astype(np.complex128))
            self.assertLess(residual,3e-5)
            self.assertLess(np.linalg.norm(x-reference)/np.linalg.norm(reference),3e-4)
        verify(a)
        self.assertEqual(f.n,a.shape[0])
        self.assertIn(f.diagnostics()["backend"],("metal","vulkan"))
        np.testing.assert_allclose(f.solve_many(np.column_stack([b,b*2])),
                                   np.column_stack([f.solve(b),f.solve(b*2)]),rtol=3e-5,atol=3e-5)
        f.factorize(a)
        verify(a)
        changed=a*np.complex64(1.25+0.15j)
        f.factorize(changed)
        verify(changed)
        self.assertGreater(f.factor_device_seconds,0)
        # Same pattern but singular values invalidate factors. A later successful
        # refactor recovers; stale factors must never be used after GPU failure.
        broken=changed.copy();broken.data[:]=0
        with self.assertRaises(RuntimeError): f.factorize(broken)
        with self.assertRaises(ValueError): f.solve(b)
        f.factorize(a);verify(a)
        with self.assertRaises(ValueError): f.solve(b[:-1])
        return analysis.info

    def test_complex128_emerge_workflow(self):
        a=sparse.csc_matrix(np.array([[4+0.1j,1j],[1j,3-0.2j]],dtype=np.complex128))
        b=np.array([1+2j,3-1j],dtype=np.complex128)
        f=sparta.analyze(a,path=sparta.Path.LDLT).factorize()
        x=f.solve(b)
        self.assertEqual(x.dtype,np.complex128)
        self.assertLess(np.linalg.norm(a@x-b)/np.linalg.norm(b),3e-5)
        a_next=a*(1.1+0.2j)
        f.factorize(a_next)
        self.assertLess(np.linalg.norm(a_next@f.solve(b)-b)/np.linalg.norm(b),3e-5)
        np.testing.assert_allclose(sparta.analyze(a).factorize().solve(b),x,rtol=3e-5,atol=3e-5)

    def test_two_by_two_and_permutation(self):
        self.check([[0,1+0.5j],[1+0.5j,0]])
        self.check([[0,0.2j,2+1j],[0.2j,4+0.3j,1],[2+1j,1,0]])

    def test_dense_random_complex_symmetric(self):
        rng=np.random.default_rng(93)
        for n in [1,3,8,17,32]:
            x=(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))).astype(np.complex64)
            a=x+x.T+np.eye(n,dtype=np.complex64)*(n+3)
            self.check(a)

    def test_complete_subtrees(self):
        rng=np.random.default_rng(12)
        n=63
        a=np.eye(n,dtype=np.complex64)*8
        # Binary-tree graph: many fronts and genuine child Schur contributions.
        for i in range(1,n):
            parent=(i-1)//2
            a[parent,i]=a[i,parent]=np.complex64(0.2+0.1j)
        info=self.check(a)
        self.assertGreater(info['supernodes'],1)
        self.assertGreater(info['levels'],1)
        # Irregular symmetric sparsity and independent components.
        for n in [12,27]:
            x=rng.uniform(-0.2,0.2,(n,n)).astype(np.float32)
            x*=rng.random((n,n))<0.18
            self.check((x+x.T)*(1+0.2j)+np.eye(n)*6)

if __name__ == '__main__': unittest.main(verbosity=2)
