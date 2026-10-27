"""Finds a BLAS and LAPACK for the CPU solver at import, so no build flags are needed.

The extension links Accelerate on Apple, or whatever SPARTA_BLAS_LIB named when it
was built. Otherwise it loads one at run time from the first of these that works:

1. ``SPARTA_BLAS``, a path to an OpenBLAS or MKL shared library;
2. the ``scipy-openblas32`` wheel, which this package depends on;
3. the OpenBLAS SciPy's own wheel bundles, then a conda environment's OpenBLAS or MKL.

Every candidate is a fixed path, so this costs one import and a library load, a few
milliseconds. Without any, the CPU solver uses its portable loops, which are correct
but several times slower; ``sparta.blas_library()`` reports which one is in use.
"""
from __future__ import annotations
import glob
import os
import sys
import warnings
from ._sparta import _load_blas, blas_library

_PORTABLE = 'portable loops'


def _scipy_openblas32() -> list[str]:
    try:
        import scipy_openblas32
    except ImportError:
        return []
    lib = scipy_openblas32.get_lib_dir()
    names = [n for n in os.listdir(lib) if n.startswith('libscipy_openblas')
             and n.endswith(('.dll', '.so', '.dylib'))]
    return [os.path.normpath(os.path.join(lib, n)) for n in sorted(names)]


def _bundled() -> list[str]:
    """SciPy's vendored OpenBLAS (32-bit integers, ``scipy_`` prefix), then conda's."""
    found = []
    try:
        import scipy
        site = os.path.dirname(os.path.dirname(scipy.__file__))
        found += sorted(glob.glob(os.path.join(site, 'scipy.libs', 'libscipy_openblas-*')))
    except ImportError:
        pass
    if sys.platform == 'win32':
        bin_ = os.path.join(sys.prefix, 'Library', 'bin')
        patterns = ['openblas.dll', 'libopenblas.dll', 'mkl_rt.*.dll', 'mkl_rt.dll']
    else:
        bin_ = os.path.join(sys.prefix, 'lib')
        patterns = ['libopenblas.so*', 'libmkl_rt.so*']
    for pattern in patterns:
        found += sorted(glob.glob(os.path.join(bin_, pattern)))
    return found


def _load() -> None:
    if blas_library() != _PORTABLE:
        return  # linked at build time, or loaded before
    explicit = os.environ.get('SPARTA_BLAS')
    if explicit:
        try:
            _load_blas(explicit)
            return
        except RuntimeError as e:
            warnings.warn(f'SPARTA_BLAS: {e}; looking for another BLAS', RuntimeWarning, stacklevel=3)
    for find in (_scipy_openblas32, _bundled):
        for path in find():
            try:
                _load_blas(path)
                return
            except RuntimeError:
                continue
    warnings.warn('SPARTA found no BLAS for its CPU solver and uses its portable loops, '
                  'which are several times slower. Install scipy-openblas32 or set '
                  'SPARTA_BLAS to an OpenBLAS or MKL library.', RuntimeWarning, stacklevel=3)


if sys.platform != 'darwin':
    _load()
