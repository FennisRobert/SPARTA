"""Named ordering, factorization, backend and input-matrix choices."""
from enum import Enum


class Ordering(str, Enum):
    """Available fill-reducing orderings."""
    SCALPEL = 'scalpel'
    NATURAL = 'natural'
    AMD = 'amd'
    COLAMD = 'colamd'
    SYMAMD = 'symamd'
    MMD_AT_PLUS_A = 'mmd_at_plus_a'
    MMD_ATA = 'mmd_ata'
    SCOTCH = 'scotch'
    NESDIS = 'nesdis'
    USER = 'user' 


class Backend(str, Enum):
    """Where factorization and solves run: the GPU, or the CPU in complex128."""
    GPU = 'gpu'
    CPU = 'cpu'


class Path(str, Enum):
    """AUTO chooses from matrix type or numerical transpose symmetry."""
    AUTO = 'auto'
    LU = 'lu'
    LDLT = 'ldlt'
    LDLH = 'ldlh'
    CHOLESKY = 'cholesky'
    QR = 'qr'
    LQ = 'lq'
    SVD = 'svd' 


class MatrixType(str, Enum):
    """Input contracts; structural symmetry alone does not permit LDLT."""
    AUTO = 'auto'
    REAL_SYMMETRIC = 'real_symmetric'
    REAL_UNSYMMETRIC = 'real_unsymmetric'
    REAL_STRUCTURALLY_SYMMETRIC = 'real_structurally_symmetric'
    COMPLEX_SYMMETRIC = 'complex_symmetric'
    COMPLEX_UNSYMMETRIC = 'complex_unsymmetric'
    COMPLEX_STRUCTURALLY_SYMMETRIC = 'complex_structurally_symmetric'
    HERMITIAN = 'hermitian'
    REAL_SYMMETRIC_POSITIVE_DEFINITE = 'real_symmetric_positive_definite'
    REAL_SYMMETRIC_POSITIVE_SEMIDEFINITE = 'real_symmetric_positive_semidefinite'
    REAL_SYMMETRIC_INDEFINITE = 'real_symmetric_indefinite'
    COMPLEX_SYMMETRIC_INDEFINITE = 'complex_symmetric_indefinite'
    HERMITIAN_POSITIVE_DEFINITE = 'hermitian_positive_definite'
    HERMITIAN_POSITIVE_SEMIDEFINITE = 'hermitian_positive_semidefinite'
    HERMITIAN_INDEFINITE = 'hermitian_indefinite'
    REAL_SKEW_SYMMETRIC = 'real_skew_symmetric'
    COMPLEX_SKEW_SYMMETRIC = 'complex_skew_symmetric'
    SKEW_HERMITIAN = 'skew_hermitian'
    REAL_RECTANGULAR = 'real_rectangular'
    COMPLEX_RECTANGULAR = 'complex_rectangular' 

