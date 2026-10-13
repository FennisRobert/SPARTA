#!/usr/bin/env python3
"""Converts EMerge matrix exports into SPARTA's native benchmark format, so Rust
benchmarks can run on real systems without Python.

    python tools/npz_to_sparta.py SparseData_*.npz -o test_matrices/em_m1_sym
    python tools/npz_to_sparta.py SparseMatrix_*.npz -o test_matrices/em_m3_asym

Writes one `<out><index>.sparta` per input. Every snapshot of one sweep shares a
mesh and therefore a sparsity pattern, which is what makes them useful for
measuring refactorization across a frequency sweep.

Two input layouts are recognised and auto-detected:

*MLPreconData* (`SparseData*.npz`) carries the curl-curl and mass operators
separately along with the mesh and preconditioner data. The solved system is
`A = E - k0^2 B` restricted to the unconstrained DOFs, and it is complex
**symmetric**, so only its lower triangle is stored.

*Bare scipy* (`SparseMatrix_i.npz`, written by `scipy.sparse.save_npz`) carries
an already-assembled matrix with no metadata. Its companions live beside it as
`SolveIds_i.npy`, `SolVector_i.npy` and `Solution_i.npy`. These are the periodic
array models, which are **unsymmetric**, so the full matrix is stored.

Only A, b and x survive the conversion: the mesh, the DOF coordinates, the
separate E/B operators and the G/PI preconditioner blocks are all dropped. Note
what that costs -- with E and B collapsed into one A, the system can no longer be
regenerated at a different k0, only at the frequencies that were exported.

Layout (little-endian, the only byte order SPARTA targets):

    magic    8 bytes  "SPARTA02"
    n        u64      order of the reduced system
    nnz      u64      stored entries
    nrhs     u64      right-hand side columns (excitation ports)
    flags    u64      bit 0: a reference solution block follows
                      bit 1: `values` is the full matrix; when clear it is the
                             lower triangle of a complex-symmetric matrix
    k0       f64      free-space wavenumber, 0 when the source carried none
    col_ptr  u64 x (n+1)
    row_idx  u64 x nnz          sorted and unique within each column
    values   c128 x nnz
    rhs      c128 x (n*nrhs)    port-major: all of port 0, then port 1, ...
    solution c128 x (n*nrhs)    present only when bit 0 of flags is set

`SPARTA01` was this without `nrhs` and the storage flag, implicitly one port and
always a symmetric lower triangle. The Rust loader still reads it.

`sparta_io.write` produces the layout, for systems that do not come from EMerge
as well.
"""
import argparse
import pathlib
import re
import sys

import numpy as np
import scipy.sparse as sp

from sparta_io import canonical, write


def from_mlprecon(source):
    """`SparseData*.npz`: the curl-curl operator, its mesh and its DOF map."""
    with np.load(source, allow_pickle=False) as f:
        k0 = float(f["k0"])
        shape = tuple(f["E_shape"])
        e = sp.csc_matrix((f["E_data"], f["E_indices"], f["E_indptr"]), shape=shape)
        b = sp.csc_matrix((f["B_data"], f["B_indices"], f["B_indptr"]), shape=shape)
        ids = f["solve_ids"]
        rhs = np.atleast_2d(f["b"])[ids, :]
        solution = np.atleast_2d(f["solution"])[ids, :] if "solution" in f else None
    # EMerge eliminates constrained DOFs, so the solved system is the
    # solve_ids submatrix of the curl-curl operator minus the k0^2 mass term.
    a = canonical((e - k0**2 * b)[ids][:, ids])
    return a, rhs, solution, k0, True


def from_bare(source):
    """`SparseMatrix_i.npz` plus its three sidecar vectors."""
    index = re.search(r"(\d+)(?=\.npz$)", source.name)
    if not index:
        raise SystemExit(f"{source.name}: expected a trailing index, e.g. SparseMatrix_1.npz")
    sidecar = lambda stem: source.with_name(f"{stem}_{index.group(1)}.npy")
    missing = [p.name for p in map(sidecar, ("SolveIds", "SolVector")) if not p.exists()]
    if missing:
        raise SystemExit(f"{source.name}: missing companion file(s) {', '.join(missing)}")
    a = canonical(sp.load_npz(source))
    ids = np.load(sidecar("SolveIds"), allow_pickle=False)
    rhs = np.atleast_2d(np.load(sidecar("SolVector"), allow_pickle=False))[ids, :]
    reference = sidecar("Solution")
    solution = (np.atleast_2d(np.load(reference, allow_pickle=False))[ids, :]
                if reference.exists() else None)
    # Storing a lower triangle would silently discard a real asymmetry, so the
    # writer checks rather than assumes (symmetric=None).
    return canonical(a[ids][:, ids]), rhs, solution, 0.0, None


def is_bare_scipy(path):
    """True when the .npz is a bare `scipy.sparse.save_npz` container rather
    than an MLPreconData export."""
    with np.load(path, allow_pickle=False) as z:
        return "format" in z.files and "E_data" not in z.files


def convert(source, target):
    reader = from_bare if is_bare_scipy(source) else from_mlprecon
    a, rhs, solution, k0, symmetric = reader(source)
    stored = write(target, a, rhs, solution=solution, k0=k0, symmetric=symmetric)
    nnz = a.nnz if stored == "full" else sp.tril(a).nnz
    size = target.stat().st_size / 2**20
    kind = "full/unsymmetric" if stored == "full" else "lower/symmetric"
    print(f"{source.name} -> {target}  n={a.shape[0]} nnz={nnz} nrhs={rhs.shape[1]} "
          f"k0={k0:.6g} {kind} {size:.1f} MiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=pathlib.Path)
    parser.add_argument("-o", "--out", default="matrix",
                        help="output prefix; each input gets a numbered .sparta file")
    args = parser.parse_args()
    prefix = pathlib.Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for i, source in enumerate(sorted(args.sources), start=1):
        convert(source, prefix.with_name(f"{prefix.name}{i}.sparta"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
