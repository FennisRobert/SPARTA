<p align="center">
  <img src="documentation/logo_header.svg" alt="SPARTA logo: a Spartan helmet" height="170">
</p>

<h1 align="center">SPARTA</h1>

<p align="center">
  <i><b>SP</b>arse <b>A</b>ccelerated <b>R</b>apid <b>T</b>riangular <b>A</b>nalyser</i>
</p>

<p align="center">
  <b>A sparse direct solver that factors on the GPU</b><br>
  Multifrontal LDLᵀ and LU for large sparse systems, from Rust, Python and C/C++
</p>

<p align="center">
  <img alt="Platform: macOS on Apple Silicon" src="https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20Silicon-1f2328?style=flat-square&logo=apple&logoColor=white">
  <img alt="GPU backend: Metal" src="https://img.shields.io/badge/GPU-Metal-ab141a?style=flat-square">
  <img alt="Languages: Rust, Python, C and C++" src="https://img.shields.io/badge/API-Rust%20%7C%20Python%20%7C%20C%2FC%2B%2B-e0ab32?style=flat-square">
</p>

<p align="center">
  <a href="#installing">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#precision-and-accuracy">Precision</a> ·
  <a href="documentation/user_guide.pdf">User guide (PDF)</a>
</p>

---

SPARTA is a sparse direct solver: it solves large linear systems **Ax = b**,
where almost every entry of A is zero, by factoring A on the GPU. It runs on
Apple Silicon GPUs through Metal today. A Vulkan backend for AMD and NVIDIA
GPUs is planned but not written yet.

The GPU works in single precision, which is fast but carries only about seven
significant digits. When that is not accurate enough, SPARTA can close the gap
in two ways. **Iterative refinement**, on by default in Python, corrects the GPU
answer in double precision on the CPU. A **CPU backend** does the whole
factorization in double precision when you ask for it.

Two factorization methods are implemented:

- **LDLᵀ** for symmetric matrices, real or complex, where A equals its own
  transpose.
- **LU** for every other square matrix: unsymmetric, structurally symmetric or
  Hermitian.

Before factoring, SPARTA reorders the unknowns with **nested dissection** to
keep the factors small, using **SCALPEL**, SPARTA's own parallel
implementation. No other ordering strategies are implemented.

SPARTA can be used from **Rust**, **Python**, and **C or C++**.

## Contents

**Getting started**
1. [How SPARTA works](#how-sparta-works)
2. [Requirements](#requirements)
3. [Installing](#installing)
4. [Quick start](#quick-start)

**Using SPARTA in detail**

5. [Describing your matrix](#describing-your-matrix)
6. [Many matrices and right-hand sides](#many-matrices-and-right-hand-sides)
7. [Configuration](#configuration)
8. [Precision and accuracy](#precision-and-accuracy)
9. [Memory and diagnostics](#memory-and-diagnostics)

**Building and development**

10. [Building from source](#building-from-source)
11. [Development](#development)
12. [Credits and licence](#credits-and-licence)

# Getting started

## How SPARTA works

Every solve goes through the same three steps, in every language:

| Step | Runs on | What it does |
|---|---|---|
| 1. **Analyze** | CPU | Orders the unknowns and plans the computation and memory. Looks only at where the nonzeros are. |
| 2. **Factor** | GPU (or CPU) | Computes the factors of A from its values. |
| 3. **Solve** | GPU (or CPU) | Uses the factors to find x for a right-hand side b. |

Each step can be repeated without redoing the ones before it:

- **Same matrix, new b**: solve again. No refactoring needed.
- **Same pattern of nonzeros, new values** (a frequency sweep, a time loop):
  factor again. The analysis and the GPU memory it reserved are reused.
- **Different pattern**: analyze again.

## Requirements

### Per operating system

**macOS.** An Apple Silicon Mac (M1 or later) with macOS 13 or later runs
everything, including the GPU. An Intel Mac runs the analysis and the
double-precision CPU backend, but has no GPU backend.

**Windows.** Any 64-bit machine runs the analysis and the double-precision CPU
backend. AMD and NVIDIA GPUs run through the [Vulkan](#vulkan) backend, which
is in alpha: tested on an NVIDIA RTX 3080 and an AMD Radeon integrated GPU,
correct but not yet tuned for speed. For a fast CPU backend, install a BLAS
library (see [Building from source](#building-from-source)).

**Linux.** The same as Windows: the analysis, the CPU backend and the alpha
Vulkan backend. Not yet tested on Linux.

### Packages

| Package | What it is for |
|---|---|
| Rust toolchain (`rustc`, `cargo`) | Compiling SPARTA. Needed by every install route except prebuilt binaries. |
| Xcode command line tools | macOS only: compiling the Metal bridge. |
| git | Downloading the source from GitHub. |
| Python 3.9 or later, with pip | The Python package. |
| NumPy and SciPy | Used by the Python package. pip installs them automatically. |
| OpenBLAS or Intel MKL | Optional, Linux and Windows only: makes the CPU backend fast. |

### Installing the packages

**macOS**

```sh
xcode-select --install                                            # Metal bridge tools and git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh    # Rust
```

For Python, use the one you already run your simulations with (python.org,
Homebrew or conda).

**Linux (Debian, Ubuntu)**

```sh
sudo apt install build-essential git python3 python3-pip python3-venv   # linker for Rust, git, Python
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh          # Rust
```

On Fedora or RHEL, the first line is `sudo dnf install gcc git python3 python3-pip`.

**Windows**

1. Rust: download and run `rustup-init.exe` from <https://rustup.rs>. When it
   asks, let it install the Visual Studio C++ Build Tools, which Rust needs to
   link.
2. git: install from <https://git-scm.com/download/win>.
3. Python: install from <https://www.python.org/downloads/>.

**Check** that everything is found by opening a new terminal and running:

```sh
cargo --version
git --version
python3 --version     # "python --version" on Windows
```

## Installing

SPARTA is not published on crates.io or PyPI, so `cargo add sparta-solver` and
`pip install sparta-solver` will not find it. Each language below has the same
three routes:

1. **You have access to the GitHub repository.**
2. **You have a copy of the source, but no GitHub access.**
3. **You have prebuilt binaries**, so nothing needs compiling.

Routes 1 and 2 compile SPARTA and need the Rust toolchain from
[Requirements](#requirements), and on macOS the Xcode command line tools. The first build takes a few minutes.

### Rust

The package is `sparta-solver`, and it is imported as `sparta`. On Apple
Silicon the build picks the Metal backend automatically.

**1. With GitHub access.** Add to your `Cargo.toml`:

```toml
[dependencies]
sparta-solver = { git = "ssh://git@github.com/FennisRobert/SPARTA.git" }
```

This needs an SSH key registered with your GitHub account (`gh auth login`
sets one up). Cargo fetches private repositories most reliably through your
own git, so also put this in `~/.cargo/config.toml`:

```toml
[net]
git-fetch-with-cli = true
```

**2. From a copy of the source.** Point Cargo at the folder containing
SPARTA's `Cargo.toml`:

```toml
[dependencies]
sparta-solver = { path = "../SPARTA" }
```

**3. Prebuilt binaries.** Not applicable: Cargo always compiles Rust libraries
from source. If you were sent a source archive, unpack it and use route 2.

### Python

Install into the Python environment you run your simulations in. The package is
imported as `sparta`. For routes 1 and 2, install the Rust toolchain first (see
[Requirements](#requirements)): pip compiles SPARTA's Rust code during the install.

**1. With GitHub access.**

```sh
git clone git@github.com:FennisRobert/SPARTA.git   # or: gh repo clone FennisRobert/SPARTA
cd SPARTA
pip install .
```

Or in one line, without keeping a copy of the source:

```sh
pip install "git+ssh://git@github.com/FennisRobert/SPARTA.git"
```

Both need an SSH key registered with your GitHub account (`gh auth login`
sets one up).

**2. From a copy of the source.** Unpack it, then install from its top folder,
the one containing `pyproject.toml`:

```sh
cd path/to/SPARTA
pip install .
```

pip downloads its build helper (maturin), NumPy and SciPy from PyPI.

**3. From a prebuilt wheel.** A wheel is a ready-compiled package: no Rust,
no compiler, no source code needed.

```sh
pip install sparta_solver-0.1.0-cp39-abi3-macosx_11_0_arm64.whl
```

One wheel works for every Python from 3.9 up, on Apple Silicon only. To make a
wheel to hand out, run this in the source folder:

```sh
pip wheel . --no-deps -w dist      # writes dist/sparta_solver-<version>-cp39-abi3-macosx_11_0_arm64.whl
```

**Check the install:**

```python
import sparta
print(sparta.__version__, sparta.gpu_available())   # gpu_available() is False if Metal is not visible
```

For development work on the Python layer, `pip install -e .` gives an editable install.

### C and C++

**1. With GitHub access.** Clone the repository and build the library:

```sh
git clone git@github.com:FennisRobert/SPARTA.git
cd SPARTA
cargo build --release -p sparta-capi    # writes target/release/libsparta.dylib
```

Then compile your program against the headers in `crates/sparta-capi/include`:

```sh
cc  -std=c11   -Icrates/sparta-capi/include app.c   -Ltarget/release -lsparta -Wl,-rpath,target/release
c++ -std=c++17 -Icrates/sparta-capi/include app.cpp -Ltarget/release -lsparta -Wl,-rpath,target/release
```

**2. From a copy of the source.** The same `cargo build` and compile commands,
run from the unpacked folder.

**3. From a prebuilt SDK.** The SDK is a folder,
`sparta-<version>-macos-arm64`, holding `include/`, `lib/libsparta.dylib` and
examples. It needs no Rust or Python. Compile against it with:

```sh
cc  -std=c11   -Iinclude app.c   -Llib -lsparta -Wl,-rpath,@loader_path/../lib
c++ -std=c++17 -Iinclude app.cpp -Llib -lsparta -Wl,-rpath,@loader_path/../lib
```

The library finds itself through the rpath, so point `-rpath` at wherever your
application keeps `libsparta.dylib`. To make the SDK to hand out, run
`make package` in `crates/sparta-capi`, which writes it under `dist/`.

More in [crates/sparta-capi/README.md](crates/sparta-capi/README.md).

## Quick start

All three examples solve the same 2×2 system, `[[4, 1], [1, 3]] x = [1, 2]`,
whose answer is `x = [1/11, 7/11]`.

### Rust

```rust
use sparta::{Analysis, Complex32, Config, Solver};

fn main() -> sparta::Result<()> {
    // Symmetric, so pass the lower triangle in CSC form.
    let n = 2;
    let col_ptr = [0, 2, 3];
    let row_idx = [0, 1, 1];
    let values = [4.0, 1.0, 3.0].map(|v| Complex32::new(v, 0.0));

    let analysis = Analysis::ldlt(n, &col_ptr, &row_idx, &Config::default())?; // 1. analyze (CPU)
    let mut solver = Solver::new(analysis)?;
    solver.factor(&values)?;                                                  // 2. factor (GPU)
    let b = [1.0, 2.0].map(|v| Complex32::new(v, 0.0));
    let x = solver.solve(&b)?;                                                // 3. solve (GPU)
    println!("{x:?}");
    solver.destroy();                                                         // 4. free
    Ok(())
}
```

`Analysis` takes the pattern in CSC form, and `factor` takes the values in the
same order. Values are `Complex32`; give real matrices an imaginary part of zero.

### Python

```python
import numpy as np
from scipy.sparse import csc_matrix
import sparta

A = csc_matrix([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])

F = sparta.analyze(A)   # 1. analyze the pattern (CPU)
F.factorize()           # 2. build the factors   (GPU)
x = F.solve(b)          # 3. find x              (GPU)
F.destroy()             # 4. hand back the GPU memory
```

1. **`sparta.analyze(A)`** returns a `Solver`. It reads the pattern and plans
   the work. No GPU memory is claimed yet.
2. **`F.factorize()`** builds the factors on the GPU. It returns the solver, so
   `F = sparta.analyze(A).factorize()` also works.
3. **`F.solve(b)`** finds x. Call it again with another b without factorizing again.
4. **`F.destroy()`** releases the GPU memory straight away rather than whenever
   Python gets round to it. It is optional but recommended for large systems;
   see [Memory and diagnostics](#memory-and-diagnostics).

`analyze` takes any square SciPy sparse matrix holding float32, float64,
complex64 or complex128 values. SPARTA copies it and never modifies yours.
Results are complex arrays, also for real input.

### C and C++

```c
#include "sparta.h"

int64_t col_ptr[]   = {0, 2, 3};                       /* lower triangle, CSC */
int64_t row_index[] = {0, 1, 1};
sparta_complex values[] = {{4, 0}, {1, 0}, {3, 0}};
sparta_complex b[] = {{1, 0}, {2, 0}}, x[2];

sparta_analysis *plan = NULL;
sparta_solver *solver = NULL;
sparta_analyze(2, col_ptr, row_index, SPARTA_SYMMETRIC, NULL, &plan);   /* 1. analyze */
sparta_solver_create(&plan, &solver);
sparta_factor(solver, values);                                          /* 2. factor  */
sparta_solve(solver, b, x);                                             /* 3. solve   */
sparta_solver_free(solver);                                             /* 4. free    */
```

Every call returns a `sparta_status`, and `sparta_last_error()` explains a
failure in a sentence. The C++ header `sparta.hpp` wraps all of this in a
`sparta::Solver` class. Worked examples are in `crates/sparta-capi/examples`.

# Using SPARTA in detail

## Describing your matrix

The kind of matrix decides which method SPARTA uses and how you store it:

| Your matrix | Store | Method |
|---|---|---|
| Symmetric, real or complex: A equals its transpose | Whole matrix, or only the lower triangle | LDLᵀ |
| Anything else square: unsymmetric, only the *pattern* symmetric, or Hermitian | Whole matrix | LU |

Two points catch people out:

- **Symmetric means equal to the transpose**, not the conjugate transpose. A
  complex matrix that equals its conjugate transpose (Hermitian) is solved with
  LU, because LDLᴴ is not implemented.
- **A symmetric pattern is not enough for LDLᵀ.** The values must match too.

LDLᵀ is the faster path. LU needs roughly twice the memory and time, so
describe a symmetric matrix as symmetric when you can.

### In Rust

You pick the method by the constructor you call, and store the matrix to match.
Rust does not check symmetry, because it only ever sees one triangle:

```rust
let analysis = Analysis::ldlt(n, &col_ptr, &row_idx, &config)?; // symmetric: lower triangle only
let analysis = Analysis::lu(n, &col_ptr, &row_idx, &config)?;   // anything else: the whole matrix
```

Row indices must be sorted and unique within each column. For `ldlt` every
entry must be on or below the diagonal.

### In Python

By default SPARTA looks at the matrix and picks LDLᵀ if it equals its
transpose, LU otherwise. You can also say what the matrix is, and optionally
force the method:

```python
F = sparta.analyze(A, mtype=sparta.MatrixType.COMPLEX_SYMMETRIC)
F = sparta.analyze(A, path=sparta.Path.LU)      # force LU, whatever A is
```

**SPARTA checks what you declare.** If you declare A symmetric and it is not,
`analyze` raises `ValueError`. It never silently factors a different matrix
from the one you gave it. `sparta.MatrixType` lists every type you can declare;
[python/API.md](python/API.md) explains each one.

**Only the lower triangle.** Pass the whole matrix by default. If you only
store the lower triangle, say so, and SPARTA reads it as a symmetric matrix
solved with LDLᵀ:

```python
F = sparta.analyze(L, triangle="lower")
```

**Nearly symmetric matrices.** The symmetry check allows a small absolute
difference between mirrored entries, `sparta.EPS`, which starts at `1e-10`.
Change it before calling `analyze`:

```python
sparta.EPS = 0          # demand exact symmetry
```

A larger tolerance accepts matrices that are only nearly symmetric. LDLᵀ then
uses the lower triangle as it is, not an average of the two triangles. Choose
LU if the difference between the triangles matters.

### In C and C++

Pass `SPARTA_SYMMETRIC` with the lower triangle, or `SPARTA_GENERAL` with the
whole matrix, to `sparta_analyze`. The same CSC rules as Rust apply.

## Many matrices and right-hand sides

### A new matrix with the same pattern

For a frequency sweep, or any series of matrices that share one pattern of
nonzeros, analyze once and factor each matrix. The analysis and the GPU memory
are reused.

**Rust**

```rust
let mut solver = Solver::new(Analysis::ldlt(n, &col_ptr, &row_idx, &Config::default())?)?;
solver.factor(&values1)?;
let x1 = solver.solve(&b1)?;
solver.factor(&values2)?;       // same pattern, new values
let x2 = solver.solve(&b2)?;
```

**Python**

```python
F = sparta.analyze(A1).factorize()
x1 = F.solve(b1)
F.factorize(matrix=A2)          # same pattern, new values
x2 = F.solve(b2)
```

**C**

```c
sparta_factor(solver, values1);
sparta_solve(solver, b1, x1);
sparta_factor(solver, values2);   /* same pattern, new values */
sparta_solve(solver, b2, x2);
```

Only the values may change, and they must be in the same order as the first
time. Stored zeros count as part of the pattern, so a matrix that stores an
extra explicit zero has a different pattern. In Python, `factorize()` with no
matrix reuses the values from the last successful factorization, and
`factorize(matrix=...)` with a different pattern runs the analysis again
before it factors, keeping your settings. `F.reanalyses` counts how often that
happened; each one costs a full analysis, so a count that keeps rising means
the pattern is not stable between calls.

### Several right-hand sides at once

Solve many right-hand sides in one call rather than one at a time. The columns
share one pass over the factors, which on the shipped test systems makes an
eight-column block 4× to 6× faster than eight separate solves.

**Rust**

```rust
// The columns are stored end to end: column j is b[j*n .. (j+1)*n].
let x = solver.solve_many(&b, nrhs)?;
```

**Python**

```python
X = F.solve_many(B)       # B.shape == (n, nrhs), one right-hand side per column; X has the same shape
```

**C**

```c
sparta_solve_many(solver, b, x, nrhs);   /* columns end to end, as in Rust */
```

## Configuration

Most users need no settings at all: the defaults are measured, and the ones
that matter most are covered in [Precision and accuracy](#precision-and-accuracy).
This section lists every setting for when you want to tune.

### Setting options in each language

The settings have the same names in every language.

**Rust:** fill in a `Config` struct. Start from the defaults or a preset and
change what you need:

```rust
let config = Config { nemin: 32, ..Config::default() };
let config = sparta::Preset::Sweep.config();
```

The four SCALPEL settings live in `config.scalpel` (for example
`config.scalpel.wide_cycles`).

**Python:** pass them as keyword arguments to `analyze`. `sparta.Config` holds
a whole set at once:

```python
F = sparta.analyze(A, ordering=sparta.Ordering.SCALPEL, nemin=32)

config = sparta.Config.preset("sweep").with_(nemin=32)
F = sparta.analyze(A, config=config)    # keyword arguments still override the config
```

The solve-behaviour settings and `force_cpu` can also be passed to
`factorize`, to change them for an existing solver. Everything else is fixed
when the matrix is analyzed.

**C:** fill a `sparta_config` with `sparta_config_default` or
`sparta_config_preset`, change fields, and pass it to `sparta_analyze`.

### All settings

Types are given language-neutrally: `uint32` and `uint64` are unsigned
integers (`usize` in Rust and `size_t` in C for `uint64`), `float64` is a
double, and `enum` is one of the named choices.

| Name | Type | Default | What it is for |
|---|---|---|---|
| **Solve behaviour** (Python only) | | | |
| `refinement_steps` | int | `-1` | Maximum refinement corrections per right-hand side. |
| `refinement_rtol` | float64 | `1e-6` | Residual at which refinement stops. |
| `refinement_minimize` | bool | `True` | Scale each correction to shrink the residual most. |
| `equilibrate` | bool | `False` | Balance row and column sizes before converting to single precision. |
| `verbose` | bool | `False` | Print a short summary per solve. |
| `safe_mode` | bool | `False` | Re-check every new matrix passed to `factorize`. |
| `mtype` | enum | `AUTO` | What kind of matrix this is. |
| `path` | enum | `AUTO` | Force LDLᵀ or LU. |
| `triangle` | text | `"full"` | Whole matrix, or `"lower"` triangle only. |
| **Ordering** | | | |
| `ordering` | enum | SCALPEL | Nested-dissection implementation. SCALPEL is the only one. |
| `imbalance` | float64 | `0.8` | SCALPEL only. How unequal a split may be. |
| `wide_cycles` | uint32 | `4` | SCALPEL only. Separator refinement cycles per split. |
| `fm_passes` | uint32 | `2` | SCALPEL only. Refinement passes per separator. |
| `separator_trials` | uint32 | `4` | SCALPEL only. Candidate separators tried per split. |
| **Supernodes and fronts** | | | |
| `nemin` | uint64 | `16` | How eagerly groups of columns merge into wider blocks. |
| `max_front` | uint64, optional | unlimited | Largest dense block allowed, in rows. |
| **Numerics** | | | |
| `force_cpu` | bool | `False` | Double precision on the CPU instead of the GPU. Python and Rust only. |
| `lu_diag_threshold` | float64 | `0.0` | LU only. How strongly to prefer the diagonal as pivot. |
| **CPU backend** (Python and Rust) | | | |
| `cpu_panel` | uint64 | `32` | Pivots per panel before an update. |
| `cpu_block_rows` | uint64 | `96` | Rows per block in the trailing update. |
| `cpu_blas_min_rows` | uint64 | `64` | Smallest block handed to BLAS. |
| **GPU scheduling** | | | |
| `small_front_rows` | uint32 | `32` | Block height where the GPU leaves its serial kernel. |
| `schur_batch` | uint32 | `0` | Panels completed between update drains. |
| `panel_threads` | uint32 | `512` | Threads per threadgroup in the panel factorization. |
| `solve_threads` | uint32 | `0` | Threads per block in the solves. |
| **GPU kernel geometry** | | | |
| `panel` | uint32 | `16` | Pivots per panel. |
| `schur_block` | uint32 | `4` | Register block in the Schur update. |
| `strip_block` | uint32 | `4` | Register block in the strip update. |
| `narrow_cols` | uint32 | `32` | Width where a front takes the wide kernels. |
| `pivot_threads` | uint32 | `128` | Threads per pivot search. |
| `schur_chunk` | uint32 | `16` | Panel columns staged per Schur pass. |

The tables further down give the allowed values and more detail for each group.

### Presets

A preset is a named starting point for a kind of workload. It is not a
guarantee: the best settings depend on your matrices, so measure before you
keep a change.

| Preset | Use it for | What it changes |
|---|---|---|
| `balanced` | The default. One factorization of an unstructured mesh | Nothing |
| `sweep` | One pattern factored many times | More time spent on ordering quality |
| `one-shot` | Factor once, then discard | A fast ordering, accepting more fill |
| `structured-grid` | Regular grids | Wider supernodes (`nemin` 32) |
| `careful` | Ill-conditioned systems | Strict LU pivoting and a more balanced split |

### Tuning to your computer

`sparta.optimize` finds the fastest settings for your matrices on your
hardware by measuring them, and saves them under a name:

```python
result = sparta.optimize([A1, (A2, b2)], name="antenna")   # prints progress and the gain
solver = sparta.analyze(A, config="antenna")                # in any later session
```

It tunes the GPU and CPU backends, keeps only changes that save more than 3%
(`min_gain`), beat the timing noise and hold up on a recheck, and reports the
gain over the defaults at the end. `iterations=3` repeats the search from the
settings found, until a pass changes nothing.
Without problems it tunes on two standard finite-element systems. Profiles
are kept in `configs.json` in `~/Library/SPARTA` (macOS) or `~/.sparta`;
`sparta.profiles()` lists them and `sparta.Config.delete(name)` removes one.
In Rust, the same search is `sparta::optimize::optimize`, and
`sparta::profiles` manages the same file. The user guide has the details.

### Solve behaviour (Python only)

These control the Python layer around the solver. Rust and C have no
equivalent yet.

| Setting | Type | Values | What it does |
|---|---|---|---|
| `refinement_steps` | int | `-1`, `0`, or a positive integer | Maximum corrections per right-hand side. `-1` takes as many as `refinement_rtol` needs, `0` switches refinement off. |
| `refinement_rtol` | float64 | between 0 and 1 | Relative residual at which refinement stops. |
| `refinement_minimize` | bool | | Scales each correction to shrink the residual as much as possible. Costs one extra CPU matrix-vector product per step, and roughly halves the steps on badly conditioned matrices. |
| `equilibrate` | bool | | Balances row and column sizes before the conversion to single precision. Can help accuracy, costs extra CPU work and memory, and does not always pay off. |
| `verbose` | bool | | Prints a short summary per solve: method, ordering, fill, memory, timings, residuals. Adds the CPU work of measuring the residuals. |
| `safe_mode` | bool | | Re-checks every new matrix passed to `factorize`. Off, a new matrix stored exactly like the analyzed one is read straight into the analyzed layout, which is much faster, and its values are not checked for infinities or NaNs. Either way a different pattern is detected and analyzed again. |
| `mtype`, `path`, `triangle` | enum, enum, text | see [Describing your matrix](#describing-your-matrix) | How to read and factor the matrix. |

### SCALPEL settings

| Setting | Type | Values | What it does |
|---|---|---|---|
| `imbalance` | float64 | 0.5 to 0.9 | Largest share of the matrix allowed in one half of a split. |
| `wide_cycles` | uint32 | 0 or more | Separator refinement cycles per split. `0` orders faster and gives more fill. |
| `fm_passes` | uint32 | 0 or more | Fiduccia–Mattheyses refinement passes. `0` is much faster and much worse. |
| `separator_trials` | uint32 | 1 or more | Starting separators tried per split. The best one is kept. |

SCALPEL runs in parallel. `RAYON_NUM_THREADS` limits how many CPU threads SCALPEL and the rest of the
analysis use. The ordering decides how much fill-in the factors carry, and so most of
the cost, but a faster ordering does not always mean a faster factorization:
measure both.

### Supernodes and fronts

| Setting | Type | Values | What it does |
|---|---|---|---|
| `nemin` | uint64 | 1 or more | Fewest columns a group of columns (a supernode) must eliminate before it may merge with its neighbour. Larger values give fewer, wider blocks, which suits the GPU but stores more fill. |
| `max_front` | uint64, optional | 1 or more, or unlimited | Largest dense block allowed, in rows. Analysis fails if a block would be bigger. It exists to fail early on a bad ordering. Unlimited is `None` in Python and Rust, `0` in C. |

### Numerics

| Setting | Type | Values | What it does |
|---|---|---|---|
| `force_cpu` | bool | | Factor and solve on the CPU in double precision instead of on the GPU. Python and Rust only. See [Precision and accuracy](#precision-and-accuracy). |
| `lu_diag_threshold` | float64 | 0 to 1 | LU only. `0` is strict partial pivoting. Above 0 the factorization accepts the diagonal entry as pivot if it is within this fraction of the column's best candidate. That is faster, but less stable, and **it changes results**, not just speed. |

### CPU backend tuning (Python and Rust)

These only matter with `force_cpu`.

| Setting | Type | Values | What it does |
|---|---|---|---|
| `cpu_panel` | uint64 | 1 or more | Pivots gathered before a trailing update, on the LU path and on fronts LAPACK cannot factor. |
| `cpu_block_rows` | uint64 | 1 or more | Rows per block in the trailing update, and the unit of work when one front is split across threads. |
| `cpu_blas_min_rows` | uint64 | 1 or more | Smallest trailing block handed to BLAS. Raise it to avoid BLAS overhead on small blocks. |

### GPU tuning (advanced)

The defaults are measured on Apple Silicon. Leave these alone unless you are
tuning for different hardware.

| Setting | Type | Values | What it does |
|---|---|---|---|
| `small_front_rows` | uint32 | 1 or more | Block height at which the GPU switches from its serial kernel to cooperative panels. |
| `schur_batch` | uint32 | 0 or more | Pivot panels completed between two update drains. `0` drains once per level of the tree. |
| `panel_threads` | uint32 | 32 to 1024 | Threads per threadgroup in the panel factorization. |
| `solve_threads` | uint32 | `0`, or 32 to 1024 | Threads per block in the solves. `0` picks per block width. |
| `panel` | uint32 | 16 | Pivots per panel. Other widths need kernel changes. |
| `schur_block` | uint32 | 1 to 8 | Per-thread register block in the Schur update. The tile is `16 × schur_block` square. |
| `strip_block` | uint32 | 1 to 8 | Per-thread register block in the strip update. |
| `narrow_cols` | uint32 | 1 or more | Width above which a front takes the wide-front kernels. |
| `pivot_threads` | uint32 | a multiple of 32, up to 1024 | Threads cooperating on one pivot search. |
| `schur_chunk` | uint32 | 1 or more | Panel columns staged per Schur pass. |

The last six are compiled into the GPU kernels. A combination the process has
not used before costs about 300 ms of shader compilation the first time.
Settings that break a device limit, such as shared memory, are rejected with a
message naming the setting. `SPARTA_SCHUR_BATCH`, `SPARTA_PANEL_THREADS`,
`SPARTA_SOLVE_THREADS` and `SPARTA_LU_DIAG_THRESHOLD` override their defaults
from the environment, for quick experiments.

## Precision and accuracy

### Single versus double precision

The GPU computes in **single precision** (complex64), which holds about seven
significant digits. On the kinds of systems SPARTA targets, a bare GPU solve
lands at a relative residual of about 1e-5. The **CPU backend** computes in
**double precision** (complex128), about sixteen digits, and is much slower.

Between the two sits **iterative refinement**. The CPU measures how far the GPU
answer is off, in double precision, and asks the GPU for corrections. This
gives double-precision accuracy at close to single-precision speed, and works
on all but the most badly conditioned systems.

| Approach | Accuracy | Speed | Available in |
|---|---|---|---|
| GPU, no refinement | residual ~1e-5 | fastest | Rust, Python, C |
| GPU with refinement (Python default) | residual down to 1e-6, or tighter if asked | ~9% slower on the larger test systems | Python |
| CPU backend | full double precision | much slower | Rust, Python |

**The residual is not the error in x.** Refinement stops on the residual,
`‖b − Ax‖ / ‖b‖`. The error in x is larger, by roughly 7× on a
well-conditioned system and by several hundred times on a sensitive one. Ask
for a tighter residual than the accuracy you need.

### Getting more accuracy from the GPU

**Rust and C.** Refinement is not built into these interfaces yet: a solve
returns the bare single-precision answer. `examples/refine.rs` shows how to
refine from Rust yourself.

**Python.** Refinement is on by default and stops at a relative residual of
`1e-6`. Keep A and b in double precision (float64 or complex128), since that
is what the residual is measured against:

```python
F = sparta.analyze(A.astype(np.complex128)).factorize()
x = F.solve(b.astype(np.complex128))

sparta.analyze(A, refinement_rtol=1e-10)   # tighter, at the cost of more corrections
sparta.analyze(A, refinement_steps=2)      # at most two corrections, whatever they reach
sparta.analyze(A, refinement_steps=0)      # off: the bare single-precision solve
```

The factors stay single precision. Only the residuals and the running answer
are kept in double precision, so a refined solve returns complex128 whatever
you passed in. Refinement stops as soon as it reaches the target. If it stops
improving first, it returns the best answer it found and says so in its report
(see [Checking a solve](#checking-a-solve)). If refinement struggles, try
`equilibrate=True`, the `careful` preset, or the CPU backend.

### Solving on the CPU in double precision

`force_cpu` runs the factorization and the solves in double precision on the
CPU. It uses the same analysis and the same kind of factorization as the GPU.
The large dense updates go to Apple Accelerate on a Mac, or to whichever BLAS
library the build linked elsewhere (see
[CPU dense kernels](#cpu-dense-kernels-blas-and-lapack)).

SPARTA never switches to the CPU on its own. If the GPU fails, you get an
error, not a slow answer.

**Rust:** set `force_cpu: true` in the `Config`, and use `factor_c128`,
`solve_c128` or `solve_many_c128` to keep your values in double precision.

**Python:**

```python
F = sparta.analyze(A, force_cpu=True, refinement_steps=0).factorize()

# Or switch an existing solver, reusing its analysis:
F.factorize(force_cpu=True)     # force_cpu=False switches back to the GPU
```

Switching releases the old factors and builds new ones on the chosen backend.
`F.factor_seconds` reports the CPU factorization time.

**C:** not available. The C library is GPU-only.

[CPU architecture and tuning](src/cpu/README.md) covers how the CPU backend
executes and what memory it uses.

### Checking a solve

Check the result before trusting it, especially on a new kind of matrix.

**Rust.**
- `solver.perturbed_pivots()` counts pivots SPARTA had to nudge to avoid a
  breakdown. Nonzero means the factors are of a slightly different matrix.
- `solver.pivot_range()` gives the smallest and largest pivot. A ratio near
  single-precision resolution (about 1e-7) or below is a warning sign.
- Compute `‖b − Ax‖ / ‖b‖` yourself, in double precision.

**Python.** Right after `factorize()`, before any solve, `F.quality` says how
far the factorization is from breaking down. Reading it costs no extra GPU
work:

```python
F.factorize()
q = F.quality
print(q.verdict)        # 'good', 'degraded', 'unusable' or 'unknown'
if not q.usable:
    ...                 # try refinement, the careful preset, or force_cpu=True
```

After a refined solve, `F.refinement_reports` holds one report per right-hand
side:

```python
x = F.solve(b)
for r in F.refinement_reports:
    print(r.status, r.corrections, r.relative_residual, r.estimated_relative_error)
```

`status` is `CONVERGED`, `STAGNATED`, `ITERATION_LIMIT` or `NONFINITE`.
`estimated_relative_error` estimates the error in x itself, which is usually
the number you actually care about.

To check an answer yourself, compute the residual in double precision:

```python
residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
```

**C.** `sparta_perturbed_pivots` reports the same count as Rust. Compute the
residual yourself.

A small residual does not guarantee a small error in x for a very sensitive
matrix. Zero perturbed pivots does not guarantee accuracy either.

## Memory and diagnostics

### Freeing a solver

A factorization can hold a lot of GPU memory, 1.4 GiB on the larger test
system. It is released when the solver is. Free it explicitly when you are done:

- **Rust:** `solver.destroy()`, or let the solver go out of scope.
- **Python:** `F.destroy()`. Otherwise the memory stays claimed for as long as
  anything refers to the solver, and a list, a traceback or a notebook's output
  history all count. `destroy()` is safe to call twice. A solver is also a
  context manager, which frees it even if the block raises an error:

  ```python
  with sparta.analyze(A) as F:
      F.factorize()
      x = F.solve(b)
  ```

- **C:** `sparta_solver_free(solver)`, and `sparta_analysis_free` for an
  analysis that never became a solver.

After `destroy()`, a Python solver still answers `F.n` and `F.path`. `factorize`,
`solve` and `info` raise `ValueError`. Use a solver only on the thread that
created it.

When things go wrong: a bad matrix or a bad option leaves an existing
factorization intact. A failure inside the factorization invalidates it, and
solving raises an error until a later factorization succeeds.

### Looking inside

**Rust.** `Analysis` reports `fill_ratio()`, `factor_nonzeros()`,
`supernodes()`, `max_front_size()` and `device_bytes()`. `Solver` reports
`factor_seconds()`, `factor_device_seconds()`, `perturbed_pivots()`,
`pivot_range()` and `is_cpu()`.

**Python.** `verbose=True` prints a short summary per solve. These properties
report the same things in code:

| Property | Tells you |
|---|---|
| `sparta.gpu_available()` | Whether Metal is visible to this process |
| `F.n`, `F.path` | Matrix size and the chosen method |
| `F.info` | Pattern statistics and estimated memory |
| `F.quality` | How trustworthy the factors are (see [Checking a solve](#checking-a-solve)) |
| `F.factor_device_seconds`, `F.factor_seconds` | GPU time, and wall time, of the last factorization |
| `F.perturbed_pivots` | Pivots that had to be nudged. Zero is not a guarantee of accuracy |
| `F.reanalyses` | How often `factorize` met a new pattern and analyzed it again |
| `F.diagnostics()` | All of the above, plus the current options, the backend and the refinement reports |

The full argument-by-argument reference, including every error, is in
[python/API.md](python/API.md).

**C.** `sparta_analysis_fill_ratio`, `sparta_analysis_factor_bytes`,
`sparta_factor_seconds` and `sparta_perturbed_pivots`.
`sparta_device()` names the GPU backend a library was built for.

`SPARTA_PROFILE=1` in the environment prints per-stage timings of the analysis
and factorization.

# Building and development

## Building from source

This matters when you build on something other than an Apple Silicon Mac, or
want the CPU backend to run fast off Apple.

### GPU backends

One build carries one GPU backend. With no feature named, the target machine
decides:

| Cargo feature | Backend | State |
|---|---|---|
| `metal` | Apple Metal, Apple Silicon | Implemented. The default on Apple Silicon |
| `vulkan` | Vulkan compute, AMD and NVIDIA | Implemented. See [Vulkan](#vulkan) below |
| neither, off Apple Silicon | none | Analysis and the CPU backend only |

```sh
cargo build --release                     # the target decides
cargo build --release --features vulkan   # name one explicitly
```

`F.diagnostics()["backend"]` in Python and `sparta_device()` in C report which
backend a binary carries.

### Vulkan

The Vulkan backend runs the same factorizations and solves as Metal, in
complex64, on AMD and NVIDIA GPUs under Windows and Linux.

- **Building** needs `glslc` from the [Vulkan SDK](https://vulkan.lunarg.com/),
  which compiles the kernels to SPIR-V. `build.rs` finds it through `GLSLC`,
  then `VULKAN_SDK`, then the path.
- **Running** needs a driver with Vulkan 1.3. Nothing is linked at build time:
  the Vulkan loader is opened when the first solver is created, and without
  one `gpu_available()` is false.
- **Choosing a device.** With several GPUs the first discrete one is used.
  `SPARTA_VULKAN_DEVICE` picks another, by index or by part of its name:

```sh
cargo build --release --features vulkan
SPARTA_VULKAN_DEVICE=amd cargo test --release --features vulkan --test shader
maturin develop --release --manifest-path python/Cargo.toml --features vulkan
```

`SPARTA_PROFILE=1` prints the device time per kernel, as on Metal, and
`SPARTA_PROFILE=2` lists every dispatch. The Khronos validation layer
(`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`) runs without messages.
Tested on an NVIDIA RTX 3080 and an AMD Radeon integrated GPU (Windows 11).

### CPU dense kernels (BLAS and LAPACK)

The CPU backend (`force_cpu`) hands its large dense matrix operations to a
BLAS/LAPACK library. Which library is fixed when SPARTA is compiled. Without
one, the CPU backend still gives correct answers, just much more slowly,
because it falls back on plain Rust loops. The GPU path does not use BLAS at
all.

Two build-time environment variables choose the library:

| Variable | Meaning |
|---|---|
| `SPARTA_BLAS_LIB` | Library to link, without the `lib` prefix: `openblas`, or `mkl_rt` for Intel MKL. Several may be given, separated by commas. |
| `SPARTA_BLAS_DIR` | Folder to look for it in, when it is not on the default search path. |

They apply to `cargo build` and to `pip install .` alike, because both compile
the same Rust code. A wrong name fails at link time, not at run time.

#### macOS (Apple Silicon and Intel)

Nothing to do. Every Mac build uses Apple's Accelerate framework, which ships
with macOS. The variables above are ignored.

#### Linux with OpenBLAS

```sh
sudo apt install libopenblas-dev          # Debian, Ubuntu
sudo dnf install openblas-devel           # Fedora, RHEL

SPARTA_BLAS_LIB=openblas cargo build --release
SPARTA_BLAS_LIB=openblas pip install .    # for the Python package
```

If OpenBLAS is installed somewhere non-standard, add
`SPARTA_BLAS_DIR=/path/to/openblas/lib`.

#### Linux with Intel MKL (preferred on Intel and AMD x86)

MKL is the better choice on x86. SPARTA spreads its own work across threads
and needs to set BLAS threading *per thread*, which MKL supports and OpenBLAS
does not.

```sh
# Install Intel oneAPI MKL from https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
source /opt/intel/oneapi/setvars.sh       # sets MKLROOT and the library path

SPARTA_BLAS_DIR=$MKLROOT/lib SPARTA_BLAS_LIB=mkl_rt cargo build --release
SPARTA_BLAS_DIR=$MKLROOT/lib SPARTA_BLAS_LIB=mkl_rt pip install .
```

Run `source /opt/intel/oneapi/setvars.sh` in every shell that runs SPARTA as
well, so the library is found at run time.

#### Windows with Intel MKL

The `mkl-devel` package from PyPI carries the import library and `mkl` the
DLLs. With both installed in a virtual environment at `C:\venv`:

```sh
SPARTA_BLAS_DIR='C:\venv\Library\lib' SPARTA_BLAS_LIB=mkl_rt cargo build --release
```

`C:\venv\Library\bin` must be on `PATH` when the result runs.

#### Other platforms

Only macOS on Apple Silicon is tested. The same two variables apply anywhere
else. With neither set, the build still works, and the CPU backend uses its
portable Rust loops.

## Development

### Build and test

```sh
cargo test                     # symbolic, analysis and allocator checks, no GPU needed
cargo build --release          # the library and the benchmark examples
sh validate.sh                 # all of the above plus the Metal numerical suite
```

`validate.sh` also builds the Python extension and runs the GPU tests against
NumPy double-precision references. The first time, it creates a virtualenv
under the ignored `.local/` folder.

### Test matrices and benchmarks

The benchmarks run on real systems exported from EMerge, kept in
`test_matrices/`. They are too large for git and are not part of the
repository; tests that need a matrix you don't have are skipped. To make your
own test systems, `tools/sparta_io.py` reads and writes the `.sparta` format,
and `tools/npz_to_sparta.py` converts SciPy `.npz` exports.

```sh
cargo run --release --example bench -- --curl-curl 24                   # synthetic 24³ system
cargo run --release --example bench -- test_matrices/em_m1_sym1.sparta
cargo run --release --example refine -- test_matrices/em_m1_sym1.sparta
cargo run --release --example dispatch_audit -- test_matrices/em_m1_sym1.sparta
./run_sparta.sh test_matrices/em_m3_asym1.sparta                        # same, through the Python API
```

| Tool | What it does |
|---|---|
| `bench` | Analyzes the first file once and reuses it for the rest, reporting per-stage timings, the double-precision residual and the distance to a shipped reference solution. |
| `refine` | Mixed-precision refinement in Rust. |
| `dispatch_audit` | Dispatched versus useful GPU threadgroups. |
| `plan_digest` | A fingerprint of the analysis, used to show that a change to the analysis leaves the numerical work untouched. |
| `run_sparta.sh` | One system through the Python API, with a formatted report. Takes `--refine N`, `--minimize`, `--ldlt`, `--verbose`. |
| `tools/npz_to_sparta.py` | Converts EMerge exports (`MLPreconData` snapshots and bare SciPy matrices) into the `.sparta` format these tools read. |
| `tools/sparta_io.py` | Reads and writes `.sparta` files from Python. |

The SCALPEL ordering is a standalone crate with its own options and
measurements: see [crates/scalpel/README.md](crates/scalpel/README.md) and
[documentation/SCALPEL.md](documentation/SCALPEL.md).

### Repository layout

```
src/                 the solver: analysis, GPU bridge, CPU backend
src/kernels/         Metal kernels: assembly, pivoting, updates, solves
src/symbolic/        ordering, elimination tree, column counts, supernodes
crates/scalpel/      the SCALPEL nested-dissection ordering
crates/sparta-capi/  the C and C++ interface
python/              the sparta Python package and its Rust bindings
examples/            benchmark and analysis programs
tools/               converters, accuracy benchmarks, matrix sync
tests/               Rust tests and the Python GPU suite
```

## Credits and licence

SPARTA is Copyright (C) 2026 Robert Fennis and Milan Rother, and is free
software: you can redistribute it and/or modify it under the terms of the GNU
Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. The
licence is in [LICENSE](LICENSE). SPARTA is distributed in the hope that it
will be useful, but WITHOUT ANY WARRANTY.

The AGPL lets you use SPARTA in open-source software under the same licence.
Software that includes SPARTA and is distributed, or offered to users over a
network, must make its own source available under the AGPL too. To include
SPARTA in closed-source software, you need a commercial licence from us: email
robertfennis@emerge-numerics.com.

SPARTA contains no third-party source code. It links a small set of Rust
crates under permissive licences; [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY)
says what is third-party and where it came from, and
[THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) carries the notices that must ship
with any binary.

Everything else is original SPARTA work: the SCALPEL ordering, the symbolic
analysis, the Metal kernels, the GPU storage layout and schedule, the CPU
backend, the Objective-C bridge, and the Python and C bindings. The blocked
pivoting follows the algorithm of LAPACK's `zsytrf`/`zlasyf`.
