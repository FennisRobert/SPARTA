"""Internal immutable solver options."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._types import MatrixType, Ordering


@dataclass(frozen=True)
class _Options:
    """One solver's validated current settings, with no defaults of its own.

    analyze() supplies every field, using its public argument defaults. Solver
    stores this record as _options. factorize() uses dataclasses.replace to copy
    it with explicitly supplied overrides, validates the copy, and adopts that
    copy only after successful factorization. Existing records never mutate.

    safe_mode selects the checked input pipeline on every factorize.
    nemin through lu_diag_threshold are solver tuning, all optional: None
    keeps the compiled-in build profile. They are fixed at analysis time.
    ordering/mtype/triangle/max_front describe fixed analysis choices.
    equilibrate and refinement_* control value preparation and solve accuracy.
    refinement_steps is a correction cap, 0 to disable, or -1 for as many as the
    requested refinement_rtol needs.
    No GPU storage or matrix data lives in this record.
    """
    verbose: bool
    safe_mode: bool
    ordering: Ordering
    mtype: MatrixType
    max_front: int | None  # Optional front-size cap in rows.
    triangle: str
    equilibrate: bool
    refinement_minimize: bool
    refinement_steps: int
    refinement_rtol: float
    # None means "use whatever the build profile compiled in". See profiles/.
    nemin: int | None
    small_front_rows: int | None
    imbalance: float | None
    wide_cycles: int | None
    fm_passes: int | None
    separator_trials: int | None
    schur_batch: int | None
    panel_threads: int | None
    solve_threads: int | None
    lu_diag_threshold: float | None
    panel: int | None
    schur_block: int | None
    strip_block: int | None
    narrow_cols: int | None
    pivot_threads: int | None
    schur_chunk: int | None
    force_cpu: bool
    cpu_panel: int | None
    cpu_block_rows: int | None
    cpu_blas_min_rows: int | None

    def __post_init__(self) -> None:
        """Validate enum choices, positive limits and dimensionless tolerances."""
        
        if type(self.force_cpu) is not bool:
            raise ValueError("force_cpu must be a boolean")
        
        if type(self.verbose) is not bool:
            raise ValueError("verbose must be a boolean")
        
        if type(self.safe_mode) is not bool:
            raise ValueError("safe_mode must be a boolean")
        
        object.__setattr__(self, "ordering", Ordering(self.ordering))
        
        object.__setattr__(self, "mtype", MatrixType(self.mtype))
        
        if type(self.refinement_minimize) is not bool:
            raise ValueError("refinement_minimize must be a boolean")
        
        if type(self.equilibrate) is not bool:
            raise ValueError("equilibrate must be a boolean")
        
        if type(self.refinement_steps) is not int or self.refinement_steps < -1:
            raise ValueError("refinement_steps must be -1, 0, or a positive integer")
        
        if not isinstance(self.refinement_rtol, (int, float)) or not np.isfinite(self.refinement_rtol) or not 0 < self.refinement_rtol < 1:
            raise ValueError("refinement_rtol must be finite and between 0 and 1")
        
        if self.ordering != Ordering.SCALPEL:
            raise NotImplementedError(f"Ordering {self.ordering.value} is not implemented")
        
        if self.max_front is not None and (type(self.max_front) is not int or self.max_front < 1):
            raise ValueError("max_front must be None (unlimited) or an integer of at least 1")
        
        if self.triangle not in ("lower", "full"):
            raise ValueError("triangle must be lower or full")
        
        # Ranges are enforced by Rust's Config::check; only reject wrong types
        # here, where the message can name the Python argument.
        
        for name in ("nemin", "small_front_rows", "wide_cycles", "fm_passes",
                     "separator_trials", "schur_batch", "panel_threads", "solve_threads",
                     "panel", "schur_block", "strip_block", "narrow_cols",
                     "pivot_threads", "schur_chunk", "cpu_panel", "cpu_block_rows", "cpu_blas_min_rows"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be None or a non-negative integer")
        
        for name in ("imbalance", "lu_diag_threshold"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be None or a number")


