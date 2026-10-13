"""sparta.optimize: tune a Config to this computer and to your matrices.

The search runs in the Rust core (``sparta::optimize``); this module prepares
the matrices the way :func:`sparta.analyze` does, relays progress and wraps the
result. See the Rust module documentation for how the search works.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse

from ._config import Config
from ._matrix import _prepare_matrix
from ._sparta import _optimize, _profile_check
from ._types import Backend, MatrixType, Path


@dataclass(frozen=True)
class Progress:
    """One progress report, as passed to a `progress` callback."""
    line: str                 # The report as one terminal line.
    done: int                 # Measurements finished...
    total: int                # ...out of the total now expected.
    elapsed: float            # Seconds since the start.
    remaining: float | None   # Estimated seconds left; None until known.


class Optimization:
    """What :func:`optimize` found, and how much faster it is.

    ``print(result)`` gives the summary: per-backend timings under the default
    and the tuned settings, the headline gain and every setting that moved.

    Attributes
    ----------
    config : Config
        The tuned settings, with ``force_cpu`` selecting the faster backend.
        Both backends keep their own tuned settings in it.
    name : str or None
        The profile it was saved under, if any.
    backend : Backend
        Where the tuned settings run: Backend.GPU or Backend.CPU.
    speedup : float
        How many times faster than the default configuration, each on the
        backend it selects.
    seconds : dict
        Per Backend, ``(default, tuned)`` summed seconds over the problems,
        measured back to back at the end.
    changes : dict
        Every setting that moved, as ``name: (default, tuned)``.
    trials : list
        Every candidate: ``(Backend, setting, value, seconds or None, note)``;
        seconds is None when the candidate failed or lost accuracy, and note
        says why.
    iterations : tuple of int
        ``(run, asked)``: fewer ran when the settings settled or the time
        limit ran out.
    elapsed : float
        Seconds the optimization took.
    """

    def __init__(self, outcome: object, name: str | None) -> None:
        self._outcome = outcome
        self.name = name
        self.config = Config(**outcome.config())
        self.backend = Backend(outcome.backend)
        self.speedup = outcome.speedup
        self.seconds = {Backend(b): (start, tuned) for b, start, tuned in outcome.seconds}
        self.changes = {n: (a, b) for n, a, b in outcome.changes()}
        self.trials = [(Backend(b), *rest) for b, *rest in outcome.trials()]
        self.iterations = outcome.iterations
        self.elapsed = outcome.elapsed

    def save(self, name: str, *, note: str = '') -> None:
        """Save the result, with its measurements, as the profile `name`."""
        self._outcome.save(name, note)
        self.name = name

    def __str__(self) -> str:
        text = str(self._outcome)
        if self.name:
            text += (f"\n  Saved as '{self.name}' in {Config.store()}; "
                     f"use sparta.analyze(A, config='{self.name}')")
        return text

    def __repr__(self) -> str:
        return (f"Optimization(backend={self.backend.value!r}, speedup={self.speedup:.3f}, "
                f"changes={self.changes!r}, name={self.name!r})")


def optimize(problems: object = None, *, name: str | None = None, gpu: bool = True, cpu: bool = True,
             start: Config | str | None = None, analysis: bool = False,
             time_limit: float | None = None, iterations: int = 1, min_gain: float = 0.03,
             verbose: bool = True,
             progress: Callable[[Progress], None] | None = None,
             mtype: MatrixType = MatrixType.AUTO, path: Path = Path.AUTO,
             triangle: str = 'full', note: str = '') -> Optimization:
    """Find the fastest settings for these problems on this computer.

    Every candidate is timed on the problems (factorize plus solve, median of
    repeats), compared against the current setting in the same pass, and kept
    only when it is clearly faster and no less accurate. Progress is printed
    as it goes, with the time left, and a summary at the end reports the gain
    over the default configuration.

    Parameters
    ----------
    problems : matrix, (matrix, rhs), or a list of either; optional
        The systems to tune on: SciPy sparse matrices, each optionally paired
        with a right-hand side vector or ``(n, k)`` block. Use matrices like
        the ones you will factor -- your own assemblies at representative
        sizes -- because the best settings follow the matrices. A missing
        right-hand side is a fixed random vector. None tunes on two standard
        finite-element Helmholtz problems (14k and 47k unknowns), which suit
        the hardware in general and no application in particular.
    name : str, optional
        Save the result under this profile name, replacing any profile of that
        name; then ``sparta.analyze(A, config=name)`` uses it. Built-in preset
        names are refused.
    gpu, cpu : bool, default True
        Which backends to tune. With both, the result selects the faster, and
        ``factorize(force_cpu=...)`` can still switch to the other with its
        own tuned settings. The GPU is left out when none is available.
    start : Config or str, optional
        Where the search starts, and what the summary compares against: a
        Config, or a profile or preset name. Config.defaults() when omitted.
    analysis : bool, default False
        Count analysis time as well. Leave False when a pattern is factored
        many times; set True when each pattern is factored once.
    time_limit : float, optional
        Stop searching after this many seconds, keeping the best found so far.
        The closing comparison against the defaults still runs.
    iterations : int, default 1
        How many full passes over all settings to make. Each pass after the
        first starts again from the settings the previous pass found, because
        once one setting has moved, another may have a new best value. The
        search stops early after a pass that changes nothing, so a large
        value costs nothing once the settings have settled. Each pass takes
        about as long as the first.
    min_gain : float, default 0.03
        The smallest improvement worth keeping, as a fraction: 0.03 keeps a
        change only when it saves more than 3% of the time. When timings are
        noisier than that, the bar rises to the noise, so small wins that may
        be chance are left alone. Raise it to change only what clearly
        matters.
    verbose : bool, default True
        Print every step to stderr, with progress and the estimated time
        left, and the summary at the end.
    progress : callable, optional
        Called with a :class:`Progress` after every step, for a custom
        display. Raising from it stops the optimization, as Ctrl-C does.
    mtype, path, triangle
        As for :func:`sparta.analyze`, applied to every problem.
    note : str, optional
        Stored with the profile and shown by :func:`sparta.profiles`.

    Returns
    -------
    Optimization
        The tuned Config and what it gained.
    """
    if name is not None:
        _profile_check(name)
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be an integer of at least 1")
    start_config = Config.defaults() if start is None else Config.resolve(start)
    prepared = [_prepare(matrix, rhs, mtype, path, triangle) for matrix, rhs in _pairs(problems)]

    def _relay(line: str, done: int, total: int, elapsed: float,
               remaining: float | None) -> None:
        if verbose:
            print(line, file=sys.stderr, flush=True)
        if progress is not None:
            progress(Progress(line, done, total, elapsed, remaining))

    outcome = _optimize(prepared, start_config.to_dict(), gpu, cpu, analysis,
                        None if time_limit is None else float(time_limit), iterations,
                        float(min_gain), _relay)
    if name is not None:
        outcome.save(name, note)
    result = Optimization(outcome, name)
    if verbose:
        print(result, file=sys.stderr, flush=True)
    return result


def _pairs(problems: object) -> list[tuple]:
    """Every accepted form of `problems` as a list of (matrix, rhs or None)."""
    if problems is None:
        return []
    if sparse.issparse(problems) or isinstance(problems, tuple):
        problems = [problems]
    pairs = [p if isinstance(p, tuple) else (p, None) for p in problems]
    if not pairs:
        raise ValueError("give at least one problem, or None for the standard ones")
    for matrix, _ in pairs:
        if not sparse.issparse(matrix):
            raise TypeError("each problem must be a SciPy sparse matrix or a (matrix, rhs) pair")
    return pairs


def _prepare(matrix: sparse.spmatrix, rhs: object, mtype: MatrixType, path: Path,
             triangle: str) -> tuple:
    """One problem as the Rust optimizer takes it: the analyzed CSC storage."""
    a, selected = _prepare_matrix(matrix, MatrixType(mtype), Path(path), triangle)
    n = a.shape[0]
    if rhs is not None:
        rhs = np.asarray(rhs, dtype=np.complex128)
        rhs = np.ascontiguousarray(rhs.reshape(n, -1) if rhs.ndim == 1 else rhs)
        if rhs.shape[0] != n:
            raise ValueError(f"a right-hand side has {rhs.shape[0]} rows for a matrix of order {n}")
    return (n, np.ascontiguousarray(a.indptr, dtype=np.uintp),
            np.ascontiguousarray(a.indices, dtype=np.uintp),
            np.ascontiguousarray(a.data, dtype=np.complex128), rhs, selected == Path.LU)
