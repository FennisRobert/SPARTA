"""Website benchmark: every test system, GPU and CPU backend, through the Python API.

    python tools/website_bench.py [repeats] [system,...] > results.jsonl

Run from the repository root with the package installed.

Per system and backend, each repeat runs a fresh analyze, factorize, a solve
of the first right-hand side without iterative refinement, and a refactorization with the *2 file's
values, or with the same values scaled when there is no *2 file. Times are wall clock, medians over the
repeats. Residuals are ||b - A x|| / ||b|| in complex128 against the original
matrix. Prints one JSON line per system and backend.
"""
import json
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "tools")
import sparta  # noqa: E402
import sparta_io  # noqa: E402

SYSTEMS = [
    ("em_m2_sym1", "em_m2_sym2"),
    ("em_m3_asym1", "em_m3_asym2"),
    ("em_m1_sym1", "em_m1_sym2"),
    ("em_bandcond_sym1", None),
]
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None


def residual(a, x, b):
    return float(np.linalg.norm(b - a @ x) / np.linalg.norm(b))


for name, partner in SYSTEMS:
    if ONLY and name not in ONLY:
        continue
    s = sparta_io.System(f"test_matrices/{name}.sparta")
    a = s.matrix().tocsc()
    b = np.ascontiguousarray(s.rhs[:, 0])
    # Without a partner file, refactor with the same values scaled: the same
    # pattern and the same work, which is all a timing needs.
    a2 = (sparta_io.System(f"test_matrices/{partner}.sparta").matrix().tocsc() if partner
          else (a * 1.001).tocsc())
    for backend in ("gpu", "cpu"):
        rows = []
        for _ in range(REPEATS):
            t = time.perf_counter()
            f = sparta.analyze(a, force_cpu=(backend == "cpu"), refinement_steps=0)
            analyze = time.perf_counter() - t
            t = time.perf_counter()
            f.factorize()
            factor = time.perf_counter() - t
            t = time.perf_counter()
            x = f.solve(b)
            solve = time.perf_counter() - t
            row = dict(analyze=analyze, factor=factor, solve=solve,
                       residual=residual(a, x, b),
                       steps=sum(r.corrections for r in f.refinement_reports))
            t = time.perf_counter()
            f.factorize(matrix=a2)
            row["refactor"] = time.perf_counter() - t
            info = f.info
            f.destroy()
            rows.append(row)
        med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
        print(json.dumps(dict(system=name, backend=backend, n=s.n, nnz=a.nnz,
                              path=info["path"].value, supernodes=info["supernodes"],
                              factor_bytes=info["factor_bytes"], **med)), flush=True)
