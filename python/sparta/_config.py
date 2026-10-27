"""Solver settings, mirroring ``sparta::Config`` in the Rust library.

Every field here has the same name and meaning as the Rust field and as the
keyword argument of :func:`sparta.analyze`, so one vocabulary covers the
library and the Python API.

Defaults are not written here. They are read from the compiled extension, so
Rust stays the single source of truth and the two cannot drift apart.
"""
from __future__ import annotations
from dataclasses import dataclass, fields, replace
import warnings

from ._sparta import (_presets, _settings, _profile_save, _profile_load,
                      _profile_delete, _profile_list, _profile_path)


@dataclass(frozen=True)
class Config:
    """One complete set of solver settings.

    Build one from :meth:`defaults`, :meth:`preset`, :meth:`load` (a profile
    saved by :func:`sparta.optimize` or :meth:`save`), then use
    ``dataclasses.replace`` or :meth:`with_` to change individual fields::

        config = sparta.Config.preset('sweep').with_(nemin=32)
        solver = sparta.analyze(a, config=config)

    Attributes
    ----------
    force_cpu : bool
        Use the Rust CPU numerics in double precision instead of the GPU.
    hybrid_cpu : float
        Share of the factorization work, from 0 to below 1, that the CPU does
        at the same time as the GPU does the rest. 0 is the GPU alone. The
        share is rebalanced after every factorization, so this sets the first
        one. Only the Metal backend has a hybrid path; elsewhere it is ignored.
    cpu_panel : int
        Pivots per CPU panel; a 2x2 pivot can extend it by one column. Used by
        the LU path and by fronts LAPACK cannot factor, not by ordinary LDLT.
    cpu_block_rows : int
        Rows per block in the CPU trailing update, and the unit of work when a
        single front is spread across threads.
    cpu_blas_min_rows : int
        Minimum trailing rows before CPU updates use Accelerate.
    threads : int
        CPU threads for SPARTA's own work: the analysis, CPU factorization and
        solves, and the hybrid's CPU share. 0 uses the process-wide pool, which
        RAYON_NUM_THREADS sizes; a positive count, up to 4096, gives the solver
        a pool of its own and keeps Accelerate from threading past it. The
        SPARTA_NUM_THREADS environment variable, when set to a positive count,
        overrides this.
    ordering : str
        Nested-dissection backend; 'scalpel' is the only one.
    max_front : int or None
        Largest front allowed, in rows. None is unlimited.
    nemin : int
        Fewest eliminated columns before two supernodes merge.
    small_front_rows : int
        Front height at which the GPU leaves its serial kernel.
    imbalance : float
        Largest share of weight allowed in one half of a split, 0.5 to 0.9.
    wide_cycles : int
        Separator refinement cycles per split. 0 orders faster and fills more.
    fm_passes : int
        Fiduccia-Mattheyses passes. 0 is much faster and much worse.
    separator_trials : int
        Starting separators tried per split, best kept.
    schur_batch : int
        Pivot panels between Schur drains. 0 drains once per level.
    panel_threads : int
        Threads per threadgroup in the panel factorization.
    solve_threads : int
        Threads per front in the solves. 0 picks per front width.
    lu_diag_threshold : float
        LU only. 0 is strict partial pivoting. Above 0 trades stability for
        speed. This one changes results, not just timing.
    panel, schur_block, strip_block, narrow_cols, pivot_threads, schur_chunk : int
        Frontal geometry, compiled into the Metal kernels. A combination a
        process has not run before costs about 300 ms of shader compilation
        once. Leave these alone unless tuning for different hardware.
    """

    ordering: str
    max_front: int | None
    nemin: int
    small_front_rows: int
    imbalance: float
    wide_cycles: int
    fm_passes: int
    separator_trials: int
    schur_batch: int
    panel_threads: int
    solve_threads: int
    lu_diag_threshold: float
    panel: int
    schur_block: int
    strip_block: int
    narrow_cols: int
    pivot_threads: int
    schur_chunk: int
    force_cpu: bool
    hybrid_cpu: float
    cpu_panel: int
    cpu_block_rows: int
    cpu_blas_min_rows: int
    threads: int

    @classmethod
    def defaults(cls) -> Config:
        """Return the measured default settings."""
        return cls(**_settings(preset='balanced'))

    @classmethod
    def preset(cls, name: str) -> Config:
        """Return a named starting point; see :func:`presets` for the list."""
        return cls(**_settings(preset=name))

    def with_(self, **changes) -> Config:
        """Return a copy with the named fields replaced."""
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown setting(s): {', '.join(sorted(unknown))}")
        return replace(self, **changes)

    def to_dict(self) -> dict:
        """Return the settings as keyword arguments for :func:`sparta.analyze`."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def save(self, name: str, *, note: str = '') -> None:
        """Store these settings as the profile `name`, replacing any before it.

        :func:`sparta.optimize` saves its result this way, with what it
        measured; a Config tuned by hand can be kept the same way. `note` is
        shown by :func:`profiles`. Built-in preset names are refused.
        """
        _profile_save(name, self.to_dict(), note)

    @classmethod
    def load(cls, name: str) -> Config:
        """Return the profile saved as `name`; KeyError if there is none.

        Settings missing from an older profile take today's defaults, and ones
        this version no longer has are dropped, so a profile survives an
        upgrade. A profile tuned on another chip loads with a warning: it is
        valid there, but its timings were measured on different hardware.
        """
        settings, chip, foreign = _profile_load(name)
        if foreign:
            warnings.warn(f"profile '{name}' was tuned on {chip}, not this computer; "
                          f"run sparta.optimize here to retune it", stacklevel=2)
        return cls(**settings)

    @staticmethod
    def delete(name: str) -> None:
        """Remove the saved profile `name`; KeyError if there is none."""
        _profile_delete(name)

    @staticmethod
    def store() -> str:
        """Return the path of the file profiles are saved in.

        ``configs.json`` in the SPARTA directory: ``~/Library/SPARTA`` on
        macOS, ``~/.sparta`` elsewhere, or the file ``SPARTA_CONFIG`` names.
        """
        return _profile_path()

    @classmethod
    def resolve(cls, config: Config | str) -> Config:
        """Return `config` itself, or the saved profile or preset it names."""
        if isinstance(config, Config):
            return config
        if isinstance(config, str):
            if config in dict(_profile_list()):
                return cls.load(config)
            return cls.preset(config)
        raise TypeError("config must be a Config, a saved profile name or a preset name")


def presets() -> dict[str, str]:
    """Return every built-in preset name with a one-line summary of its use."""
    return dict(_presets())


def profiles() -> dict[str, str]:
    """Return every saved profile name with a one-line summary of it.

    Profiles live in :meth:`Config.store`; see :meth:`Config.save`,
    :meth:`Config.load`, :meth:`Config.delete` and :func:`sparta.optimize`.
    """
    return dict(_profile_list())
