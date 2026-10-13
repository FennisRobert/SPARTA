"""Real Metal validation for cooperative pivots and tiled large-front updates."""
import os
import time
import unittest
import numpy as np
import scipy.sparse as sparse
import sparta

if os.environ.get('SPARTA_REQUIRE_METAL') == '1' and not sparta.gpu_available():
    raise RuntimeError('Metal required: run in normal macOS Terminal')

@unittest.skipUnless(sparta.gpu_available(), 'requires Apple Silicon Metal')
class LargeFrontTests(unittest.TestCase):
    def verify(self, a, factor, expected):
        b=a@expected
        x=factor.solve(b)
        residual=float(np.linalg.norm(a@x-b)/np.linalg.norm(b))
        error=float(np.linalg.norm(x-expected)/np.linalg.norm(expected))
        self.assertLess(residual,3e-5)
        self.assertLess(error,3e-4)
        return residual

    def test_boundaries_through_4096(self):
        for n in [129,166,257,286,511,1024,2048,4096]:
            with self.subTest(n=n):
                a=np.full((n,n),(0.005+0.003j)/n,dtype=np.complex128)
                a[np.diag_indices(n)]+=4+np.linspace(0,1,n)+0.2j
                a=sparse.csc_matrix(a)
                expected=(np.sin(np.arange(n)*0.17)+1j*np.cos(np.arange(n)*0.31)).astype(np.complex128)
                symbolic=sparta.analyze(a,ordering='scalpel')
                self.assertEqual(symbolic.info['max_front'],n)
                self.assertEqual(symbolic.info['large_fronts'],1)
                start=time.perf_counter();f=symbolic.factorize();wall=time.perf_counter()-start
                residual=self.verify(a,f,expected)
                self.assertGreater(f.factor_device_seconds,0)
                print(f'large front={n}, factor wall={wall:.6f}s, device={f.factor_device_seconds:.6f}s, residual={residual:.3g}',flush=True)
                if n in (286,1024):
                    changed=a*(1.1+0.2j);f.factorize(changed);self.verify(changed,f,expected)
                del f,symbolic,a

    def test_large_two_by_two_permutations_failure_and_recovery(self):
        n=286;rng=np.random.default_rng(922)
        a=np.full((n,n),1e-6*(1+0.1j),dtype=np.complex128)
        a[np.diag_indices(n)]=0
        pairs=rng.permutation(n).reshape(-1,2)
        for i,j in pairs:a[i,j]=a[j,i]=2+0.7j
        a=sparse.csc_matrix(a);symbolic=sparta.analyze(a)
        self.assertGreater(symbolic.info['large_fronts'],0)
        f=symbolic.factorize();expected=rng.normal(size=n)+1j*rng.normal(size=n)
        self.verify(a,f,expected)
        broken=a.copy();broken.data[:]=0
        with self.assertRaises(RuntimeError):f.factorize(broken)
        with self.assertRaises(ValueError):f.solve(np.ones(n,dtype=np.complex128))
        f.factorize(a);self.verify(a,f,expected)

    def test_late_singular_pivot_invalidates_and_recovers(self):
        n=257
        singular=np.full((n,n),1/1024,dtype=np.complex128)
        singular[np.arange(n-2),np.arange(n-2)]+=4
        good=singular.copy();good[-2,-2]+=4;good[-1,-1]+=4
        good=sparse.csc_matrix(good);singular=sparse.csc_matrix(singular)
        f=sparta.analyze(good, path=sparta.Path.LDLT).factorize()
        self.assertEqual(f.perturbed_pivots,0)
        # A rank-deficient refactor no longer aborts the front: the dead pivot is
        # floored against the front's own scale so the factorization completes
        # and stays usable as a preconditioner. The count is how the caller
        # learns the factors are of a perturbed matrix.
        f.factorize(singular)
        self.assertGreater(f.perturbed_pivots,0)
        f.factorize(good)
        self.assertEqual(f.perturbed_pivots,0)
        self.verify(good,f,np.ones(n,dtype=np.complex128))

    def test_parallel_fronts_and_child_contributions(self):
        rng=np.random.default_rng(77)
        def block(n):
            x=rng.uniform(-0.01,0.01,(n,n))+1j*rng.uniform(-0.01,0.01,(n,n))
            return sparse.csc_matrix(x+x.T+np.eye(n)*8)
        left,right,root=160,192,160
        c1=sparse.csc_matrix(rng.uniform(-0.01,0.01,(left,root))*(1+0.1j))
        c2=sparse.csc_matrix(rng.uniform(-0.01,0.01,(right,root))*(1+0.1j))
        one=sparse.bmat([[block(left),None,c1],[None,block(right),c2],[c1.T,c2.T,block(root)]],format='csc')
        # An independent small tree exercises mixed small/large level dispatch.
        small=sparse.diags([np.ones(31),np.full(32,4),np.ones(31)],[-1,0,1],dtype=np.complex128,format='csc')
        a=sparse.block_diag([one,one*(1.1+0.1j),small],format='csc')
        symbolic=sparta.analyze(a)
        self.assertGreaterEqual(symbolic.info['large_fronts'],2)
        self.assertGreater(symbolic.info['levels'],1)
        self.assertGreater(symbolic.info['max_front'],256)
        f=symbolic.factorize();expected=rng.normal(size=a.shape[0])+1j*rng.normal(size=a.shape[0])
        self.verify(a,f,expected)
        changed=a*(0.9-0.1j);f.factorize(changed);self.verify(changed,f,expected)

if __name__=='__main__':unittest.main(verbosity=2)
