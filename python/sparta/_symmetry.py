"""Shared tolerance for numerical transpose and conjugate-transpose checks."""
import numpy as np
from scipy import sparse


def _is_symmetric(matrix: sparse.spmatrix, hermitian: bool = False) -> bool:
    """Check sparse matrix symmetry using the current public sparta.EPS.

    EPS is an absolute entrywise tolerance in matrix-value units. hermitian
    selects conjugate transpose. Structural symmetry is a separate exact check.
    """
    from . import EPS
    if not isinstance(EPS, (int, float)) or not np.isfinite(EPS) or EPS < 0:
        raise ValueError('sparta.EPS must be finite and nonnegative')
    a = matrix.tocsc()
    if a.shape[0] != a.shape[1]:
        return False
    difference = a - (a.conjugate().T if hermitian else a.T)
    return bool(np.all(np.isfinite(difference.data)) and
                np.all(np.abs(difference.data) <= EPS))
