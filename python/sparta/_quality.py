"""Numerical quality of a finished factorization, measured without a matvec."""
from dataclasses import dataclass
import math
from typing import Literal

# float32 resolution. A pivot this far below the largest one carries no
# significant bits relative to it.
EPSILON = 1.1920929e-7

# Measured pivot ratios, every fixture factored with refinement off, next to
# the relative residual of the bare solve they produced:
#
#   em_m2_sym2          4.7e-05     2.9e-05
#   em_m1_sym2          2.3e-06     7.6e-05
#   em_m1_sym1          2.1e-06     6.5e-05
#   em_m2_sym1          3.4e-08     1.6e-03   cond ~2.4e9, still usable
#   em_bandcond_sym1    9.2e-19     5.4e-01   refinement cannot move it
#
# The ratio tracks the residual across four orders of magnitude of it, and the
# system refinement cannot rescue sits eleven orders below the worst usable
# one. Nothing here is delicate; the thresholds sit in that gap.
DEGRADED_RATIO = EPSILON
UNUSABLE_RATIO = 1e-12

# The LU path reports U's diagonal instead. Partial pivoting keeps it larger
# than an LDL^T block diagonal on the same system, so a failed factorization
# sits much closer to the usable ones. Measured on Metal (M4 Pro), refinement
# off, pivot ratio next to the bare-solve residual:
#
#   em_m3_asym2         1.5e-04     2.1e-05
#   em_m3_asym1         1.0e-04     2.8e-05
#   em_m2_sym2          7.3e-05     4.8e-05   forced to LU
#   em_m1_sym2          4.7e-06     1.3e-04   forced to LU
#   em_m1_sym1          4.3e-06     1.1e-04   forced to LU
#   em_m2_sym1          4.4e-08     2.7e-03   forced to LU, still usable
#   em_bandcond_sym1    2.9e-11     5.8e+00   forced to LU
#   em_bandcond_sym1    8.5e-16     5.1e-01   forced to LU, lu_diag_threshold 0.1
#
# The LDL^T cutoff of 1e-12 would call the first bandcond run merely degraded.
# The LU cutoff sits at the geometric middle of the gap, about 1.6 decades from
# each side -- a far thinner margin than LDL^T's, resting on one failed system.
# EPSILON still separates the healthy systems from em_m2_sym1.
LU_DEGRADED_RATIO = EPSILON
LU_UNUSABLE_RATIO = 1e-9


@dataclass(frozen=True)
class FactorQuality:
    """What the factors themselves say about how much to trust a solve.

    Built from data already resident after `factorize`: the perturbed-pivot
    count the factorization returned, and a host scan of the block diagonal
    sitting in shared memory. Reading it costs a pass over 8n bytes and no GPU
    work, so a caller can check every factorization without paying for a
    residual, which would cost a matvec against the original matrix.

    `relative_residual` is the exception and is nan unless iterative refinement
    ran, in which case it is that first measurement, already paid for.
    `single_precision` is True for Metal complex64, False for CPU complex128.
    CPU factors do not use the empirically calibrated Metal pivot-ratio cutoffs;
    without a residual or perturbation their verdict is 'unknown'.
    """
    perturbed_pivots: int
    smallest_pivot: float
    largest_pivot: float
    relative_residual: float
    # The empirical pivot-ratio cutoffs below were measured only for Metal.
    single_precision: bool = True
    # True for LU factors, whose pivots are U's diagonal and have their own
    # cutoffs; False for LDL^T block pivots.
    lu: bool = False

    @property
    def pivot_ratio(self) -> float:
        """Smallest pivot over largest, or nan when the range is unknown.

        Bounded above by the reciprocal of the factorization's condition
        number, so a tiny value means the elimination came near breakdown.
        On the LU path the pivots are U's diagonal.
        """
        if not math.isfinite(self.smallest_pivot) or not self.largest_pivot > 0:
            return math.nan
        return self.smallest_pivot / self.largest_pivot

    @property
    def verdict(self) -> Literal['good','degraded','unusable','unknown']:
        """One of 'good', 'degraded', 'unusable' or 'unknown'.

        'degraded' means the solve is worth having but its accuracy is well
        short of what the residual alone suggests: drive it with refinement and
        read estimated_relative_error rather than the residual. 'unusable'
        means the factors have stopped approximating the matrix, and no number
        of corrections will fix that, because the iteration does not contract.
        """
        from ._refinement import SUSPECT_RESIDUAL
        residual = self.relative_residual
        if math.isfinite(residual) and residual > SUSPECT_RESIDUAL:
            return 'unusable'
        ratio = self.pivot_ratio
        if not self.single_precision:
            # No empirical complex128 cutoff has been calibrated. Perturbations
            # and a measured residual remain evidence; a ratio alone is not an
            # accuracy estimate for the CPU factors.
            if self.perturbed_pivots:
                return 'degraded'
            return 'good' if math.isfinite(residual) else 'unknown'
        if math.isnan(ratio):
            # No pivot range. A perturbed pivot is still evidence, but its
            # absence proves nothing: the badly conditioned fixture factored
            # on LU with lu_diag_threshold=0.1 perturbed nothing at all and
            # still stagnated at 3.9e-1.
            if math.isfinite(residual):
                return 'good'
            return 'degraded' if self.perturbed_pivots else 'unknown'
        unusable, degraded = ((LU_UNUSABLE_RATIO, LU_DEGRADED_RATIO) if self.lu
                              else (UNUSABLE_RATIO, DEGRADED_RATIO))
        if ratio < unusable:
            return 'unusable'
        if ratio < degraded or self.perturbed_pivots:
            return 'degraded'
        return 'good'

    @property
    def usable(self) -> bool:
        """False only when the factors are known to have failed."""
        return self.verdict != 'unusable'

    def __str__(self) -> str:
        parts = [f'factors {self.verdict}']
        ratio = self.pivot_ratio
        if not math.isnan(ratio):
            parts.append(f'pivot ratio {ratio:.2e} '
                         f'(min {self.smallest_pivot:.3e}, max {self.largest_pivot:.3e})')
        if self.perturbed_pivots:
            parts.append(f'{self.perturbed_pivots} perturbed pivots')
        if math.isfinite(self.relative_residual):
            parts.append(f'relative residual {self.relative_residual:.3e}')
        return ', '.join(parts)
