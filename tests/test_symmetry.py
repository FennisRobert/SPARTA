"""Global numerical tolerance applies consistently to every symmetry gate."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
import sparta


class SymmetryTests(unittest.TestCase):
    """Test small complex defects, tunability and exact structural contracts."""

    def test_roundoff_and_override(self):
        """AUTO, declared types and explicit LDLT accept roundoff but reject larger defects."""
        for delta in (1e-16, -1e-16, 1e-16j):
            a=sparse.csc_matrix([[2+0j,delta],[0,3]])
            for kwargs in ({},dict(path=sparta.Path.LDLT),dict(mtype=sparta.MatrixType.COMPLEX_SYMMETRIC)):
                f=sparta.analyze(a,**kwargs)
                self.assertEqual(f.path,sparta.Path.LDLT)
                f._validated(a)
            with patch.object(sparta,'EPS',0):
                self.assertEqual(sparta.analyze(a).path,sparta.Path.LU)
                with self.assertRaises(ValueError): sparta.analyze(a,path=sparta.Path.LDLT)
        # A defect above the default EPS selects LU; raising EPS past it
        # selects LDLT. The two bounds have to straddle the defect, so both
        # move together if the default ever changes.
        self.assertLess(1e-11, sparta.EPS, 'defect below must stay under the default')
        defect=1e-9
        self.assertGreater(defect, sparta.EPS, 'defect above must stay over the default')
        a=sparse.csc_matrix([[2+0j,defect],[0,3]])
        self.assertEqual(sparta.analyze(a).path,sparta.Path.LU)
        with patch.object(sparta,'EPS',defect*10):
            self.assertEqual(sparta.analyze(a).path,sparta.Path.LDLT)

    def test_hermitian_and_structure(self):
        """Hermitian tolerance includes imaginary diagonal defects; pattern stays exact."""
        a=sparse.csc_matrix([[2+1e-16j,1j],[-1j,3]])
        self.assertEqual(sparta.analyze(a,mtype=sparta.MatrixType.HERMITIAN).path,sparta.Path.LU)
        a=sparse.csc_matrix([[2+0j,1e-16],[0,3]])
        with self.assertRaises(ValueError):
            sparta.analyze(a,mtype=sparta.MatrixType.COMPLEX_STRUCTURALLY_SYMMETRIC)
        for value in (-1,float('nan'),float('inf')):
            with patch.object(sparta,'EPS',value), self.assertRaises(ValueError):
                sparta.analyze(a)
