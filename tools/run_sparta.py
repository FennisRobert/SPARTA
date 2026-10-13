"""Run one assembled system through SPARTA and report timing and accuracy.

Driven by `run_sparta.sh`. Loads a `.sparta` benchmark container, solves it with
the Python API, and prints a single formatted report: symbolic sizes, memory,
per-stage seconds, and relative residuals measured in complex128 against the
original matrix.
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import sparta
from sparta_io import System

# EMerge's own SuperLU settings; a low pivot tolerance keeps the diagonal.
SUPERLU = dict(permc_spec="MMD_AT_PLUS_A", relax=0, diag_pivot_thresh=0.001,
               options=dict(SymmetricMode=True, Equil=False, IterRefine="SINGLE"))


def arguments() -> argparse.Namespace:
    """Parse the command line into run options; file is the only positional."""
    parser = argparse.ArgumentParser(prog="run_sparta.sh", description=__doc__)
    parser.add_argument("file", type=Path, help="a .sparta benchmark container")
    method = parser.add_mutually_exclusive_group()
    method.add_argument("--lu", action="store_const", dest="path", const=sparta.Path.LU,
                       help="force the LU factorization")
    method.add_argument("--ldlt", action="store_const", dest="path", const=sparta.Path.LDLT,
                       help="force the LDLT factorization (needs a symmetric matrix)")
    parser.add_argument("--verbose", action="store_true",
                        help="let SPARTA print its own per-solve summary as well")
    parser.add_argument("--equilibrate", action="store_true",
                        help="balance row and column magnitudes before factorizing")
    parser.add_argument("--refine", type=int, default=-1, metavar="N",
                        help="correction solves per RHS: -1 for as many as --rtol needs "
                             "(the default), 0 to switch refinement off, N for a cap")
    parser.add_argument("--rtol", type=float, default=1e-6, metavar="T",
                        help="relative residual to stop refinement at (default 1e-6)")
    parser.add_argument("--no-minimize", action="store_false", dest="minimize",
                        help="plain corrections instead of residual-minimizing ones")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="refactorize and solve N times, reporting medians")
    parser.add_argument("--reference", action="store_true",
                        help="also solve on the CPU in complex128 with SuperLU and compare")
    parser.add_argument("--port", type=int, default=None, metavar="J",
                        help="solve only right-hand side J instead of all of them")
    return parser.parse_args()


def relative(x: np.ndarray, y: np.ndarray) -> float:
    """Return the relative 2-norm difference between complex vectors x and y."""
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))


def residuals(a, b: np.ndarray, x: np.ndarray) -> list[float]:
    """Return the relative residual of each column of x against matrix a and RHS b."""
    r = a @ x.astype(np.complex128) - b
    return [float(np.linalg.norm(r[:, j]) / np.linalg.norm(b[:, j])) for j in range(b.shape[1])]


def cpu_reference(a, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve a x = b on the CPU in complex128 with SuperLU; return x and its seconds."""
    from scipy.sparse.linalg import splu
    started = time.perf_counter()
    x = splu(a, **SUPERLU).solve(b)
    return x, time.perf_counter() - started


def rule(title: str = "") -> None:
    """Print a section rule, optionally labelled with a title."""
    print(f"\n{title}" if title else "")
    print("─" * 74)


def report_system(system: System, a, b: np.ndarray, options: argparse.Namespace) -> None:
    """Print the input's size, storage and the options this run used."""
    print(f"SPARTA · {options.file.name}")
    print("─" * 74)
    print(f"  system      n={system.n:,}  nnz(A)={a.nnz:,}  RHS={b.shape[1]}  k0={system.k0:.4f}")
    print(f"  storage     {'full matrix' if system.full else 'lower triangle, complex-symmetric'}")
    if not options.refine:
        refine = "off"
    elif options.refine < 0:
        refine = f"to {options.rtol:.1e}"
    else:
        refine = f"{options.refine} steps, target {options.rtol:.1e}"
    print(f"  requested   path={(options.path or sparta.Path.AUTO).value}  "
          f"ordering={(options.ordering or sparta.Ordering.SCALPEL).value}")
    print(f"  options     refinement={refine}  equilibrate={options.equilibrate}  "
          f"minimize={options.minimize}")


def report_plan(solver: sparta.Solver, analyze_seconds: float) -> None:
    """Print the symbolic plan's sizes and the GPU memory it needs."""
    info = solver.info
    rule("plan")
    print(f"  method      {info['path'].value.upper()}   analyzed in {analyze_seconds:.3f}s")
    print(f"  symbolic    supernodes={info['supernodes']:,}  levels={info['levels']}  "
          f"largest front={info['max_front']:,} rows")
    workspace = info['front_bytes'] / 2**30
    factors = info['factor_bytes'] / 2**30
    print(f"  memory      workspace={workspace:.3f} GiB + factors={factors:.3f} GiB"
          f" = {workspace + factors:.3f} GiB")


def report_timing(analyze: float, factor: list[float], device: list[float],
                  solve: list[float], repeats: int) -> None:
    """Print per-stage seconds; factor/device/solve are medians over the repeats."""
    rule("timing" + (f" (median of {repeats} runs)" if repeats > 1 else ""))
    f, d, s = statistics.median(factor), statistics.median(device), statistics.median(solve)
    print(f"  analyze       {analyze:8.3f}s   once, on the CPU")
    print(f"  factorize     {f:8.3f}s   {d:.3f}s of that on the GPU")
    print(f"  solve         {s:8.3f}s   all right-hand sides")
    print(f"  {'─' * 30}")
    print(f"  first solve   {analyze + f + s:8.3f}s   analyze + factorize + solve")
    print(f"  per repeat    {f + s:8.3f}s   factorize + solve, plan reused")


def report_accuracy(system: System, residual: list[float], solution: np.ndarray,
                    reference: np.ndarray | None, reference_seconds: float,
                    reports: tuple) -> None:
    """Print each RHS's residual, refinement outcome and distance to known solutions."""
    rule("accuracy  (relative, measured in complex128 against the original matrix)")
    print(f"  {'RHS':>4}  {'residual':>11}  {'vs stored x':>12}  {'vs SuperLU x':>13}  refinement")
    for j, value in enumerate(residual):
        stored = (f"{relative(solution[:, j], system.solution[:, j]):.3e}"
                  if system.solution is not None else "-")
        fresh = f"{relative(solution[:, j], reference[:, j]):.3e}" if reference is not None else "-"
        note = "-"
        if j < len(reports):
            note = f"{reports[j].status.value}, {reports[j].corrections} corrections"
        print(f"  {j:>4}  {value:>11.3e}  {stored:>12}  {fresh:>13}  {note}")
    if reports:
        first = statistics.median([r.initial_relative_residual for r in reports])
        print(f"\n  refinement started from a median residual of {first:.3e}")
    if reference is not None:
        print(f"  CPU complex128 SuperLU reference took {reference_seconds:.3f}s")


def main() -> None:
    """Run the requested system and print one report."""
    options = arguments()
    if not sparta.gpu_available():
        sys.exit("no Metal device is visible to this process")
    system = System(options.file)
    a = system.matrix().astype(np.complex128).tocsc()
    b = system.rhs if options.port is None else system.rhs[:, [options.port]]
    report_system(system, a, b, options)

    # The container stores a full matrix or just the lower triangle of a
    # complex-symmetric one; passing the triangle as a full matrix would solve a
    # triangular system instead.
    stored = system.stored.astype(np.complex128)
    settings = dict(verbose=options.verbose, equilibrate=options.equilibrate,
                    refinement_steps=options.refine, refinement_rtol=options.rtol,
                    refinement_minimize=options.minimize,
                    triangle="full" if system.full else "lower")
    if options.path is not None:
        settings["path"] = options.path
    if options.ordering is not None:
        settings["ordering"] = options.ordering

    # Warm the Metal pipelines so compilation does not land in the first timing.
    warm = sparta.analyze(stored, **settings)
    warm.factorize()
    warm.solve(b[:, 0])
    del warm

    started = time.perf_counter()
    solver = sparta.analyze(stored, **settings)
    analyze_seconds = time.perf_counter() - started
    report_plan(solver, analyze_seconds)

    factor, device, solve = [], [], []
    for _ in range(max(1, options.repeat)):
        started = time.perf_counter()
        solver.factorize()
        factor.append(time.perf_counter() - started)
        device.append(solver.factor_device_seconds)
        started = time.perf_counter()
        x = solver.solve_many(b)
        solve.append(time.perf_counter() - started)
    report_timing(analyze_seconds, factor, device, solve, max(1, options.repeat))

    reference, reference_seconds = None, 0.0
    if options.reference:
        reference, reference_seconds = cpu_reference(a, b)
    report_accuracy(system, residuals(a, b, x), x, reference, reference_seconds,
                    solver.refinement_reports)
    if solver.perturbed_pivots:
        print(f"\n  note: {solver.perturbed_pivots} pivot(s) were rescued; these factors "
              f"belong to a slightly perturbed matrix")
    print()


if __name__ == "__main__":
    main()
