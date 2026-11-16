<p align="center">
  <img src="https://raw.githubusercontent.com/FennisRobert/SPARTA/main/documentation/logo_header.svg" alt="SPARTA logo: a Spartan helmet" height="170">
</p>

<h1 align="center">SPARTA</h1>

<p align="center">
  <i><b>SP</b>arse <b>A</b>ccelerated <b>R</b>apid <b>T</b>riangular <b>A</b>nalyser</i>
</p>

<p align="center">
  <b>A sparse direct solver built for finite-element problems</b><br>
  Specialised for FEM systems, tuned to keep every core busy, factorization on the GPU or CPU
</p>

<p align="center">
  <img alt="Platform: macOS on Apple Silicon" src="https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20Silicon-1f2328?style=flat-square&logo=apple&logoColor=white">
  <img alt="GPU backend: Metal" src="https://img.shields.io/badge/GPU-Metal-ab141a?style=flat-square">
  <img alt="Languages: Rust, Python, C, C++, MATLAB and Julia" src="https://img.shields.io/badge/API-Rust%20%7C%20Python%20%7C%20C%2FC%2B%2B%20%7C%20MATLAB%20%7C%20Julia-e0ab32?style=flat-square">
</p>

<p align="center">
  <a href="#installing">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#precision-and-accuracy">Precision</a> ·
  <a href="documentation/user_guide.pdf">User guide (PDF)</a>
</p>

---

SPARTA solves **Ax = b** for the large sparse matrices that finite-element
(FEM) simulations produce. It is a direct solver: it factors A, then solves.
It is built for FEM systems only, and is tuned to run close to the machine's
limits on memory bandwidth and arithmetic.

- **Built for FEM.** SPARTA plans the whole factorization once from the
  sparsity pattern. Every later matrix with the same pattern, such as the
  next frequency of a sweep, reuses that plan.
- **Every core kept busy.** Many small, measured choices keep every core doing
  useful arithmetic.
- **Its own dense kernels.** On one core, the dense kernels reach 82 to 91% of
  the processor's peak. You do not need to install BLAS, LAPACK or any other
  maths library. On Macs before M4, SPARTA uses Apple's Accelerate, which
  comes with macOS.

[Why SPARTA is fast](#why-sparta-is-fast) lists the choices and the
measurements.

**Hardware.** On Apple Silicon, SPARTA factors on the GPU through Metal. On
AMD and NVIDIA GPUs on Windows and Linux, it factors through Vulkan. On any 64-bit
machine, it can factor on the CPU in double precision.

**Precision.** SPARTA takes real or complex matrices and uses real or complex
arithmetic to match. The GPU works in single precision, which carries about
seven significant digits. If you need more, you have two options:

- **Iterative refinement** corrects the GPU answer in double precision on the
  CPU. It is on by default in Python.
- **The CPU backend** does the whole factorization in double precision.

**Factorizations.** For symmetric matrices, where A equals its own transpose,
SPARTA uses **LDLᵀ**. For every other square matrix (unsymmetric, only
structurally symmetric, or Hermitian), it uses **LU**.

**Ordering.** Before factoring, SPARTA reorders the unknowns to keep the
factors small. It uses **SCALPEL**, its own parallel nested dissection. No
other ordering is available.

SPARTA can be used from **Rust**, **Python**, and **C or C++**.

## Contents

**Getting started**
1. [How SPARTA works](#how-sparta-works)
2. [Why SPARTA is fast](#why-sparta-is-fast)
3. [Requirements](#requirements)
4. [Installing](#installing)
5. [Quick start](#quick-start)

**Using SPARTA in detail**

6. [Describing your matrix](#describing-your-matrix)
7. [Many matrices and right-hand sides](#many-matrices-and-right-hand-sides)
8. [Configuration](#configuration)
9. [Precision and accuracy](#precision-and-accuracy)
10. [Memory and diagnostics](#memory-and-diagnostics)

**Building and development**

11. [Building from source](#building-from-source)
12. [Development](#development)
13. [Contributing](#contributing)
14. [Credits and licence](#credits-and-licence)

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

## Why SPARTA is fast

This section shows where SPARTA's speed comes from, so you can judge how it
applies to your own problems.

A direct solver's time is the number of operations divided by the rate at
which they run. For FEM systems, the number of operations is already as low as
it can be. SPARTA is therefore built to raise the rate: to keep every core busy
and close to its peak.

### The number of operations

Factoring a sparse matrix turns some zero entries into nonzeros. The order in
which the unknowns are eliminated decides how many. SPARTA uses nested
dissection: it splits the mesh in two along a small separator, orders each half
the same way, and eliminates the separator last. On FEM meshes, no other order
does better, up to a constant factor:

| Mesh, n unknowns | Operations to factor | Memory for the factors |
|---|---|---|
| 2D | proportional to n<sup>1.5</sup> | proportional to n log n |
| 3D | proportional to n<sup>2</sup> | proportional to n<sup>4/3</sup> |

(George, 1973; Hoffman, Martin and Rose, 1973; Lipton, Rose and Tarjan, 1979.)

### The rate: choices made for FEM systems

Each choice below was measured on FEM test cases and kept because it made them
faster.

- **Plan once, factor many times.** The analysis fixes the ordering, the
  schedule and every block of memory. For a new matrix with the same pattern,
  SPARTA goes straight to the numerical factorization.
- **Wide fronts.** SPARTA merges neighbouring groups of columns into wider
  dense blocks, even when that stores a few extra zeros. Wider blocks run
  closer to peak.
- **One large update per node.** In CPU mode, each node's update goes straight
  into the columns of the nodes above it. No intermediate matrix has to be
  cleared and assembled.
- **Parallelism that never competes for cores.** Near the bottom of the
  elimination tree, there are many small independent blocks, and each runs on
  one core. Near the top, there are a few large blocks, and each spreads over
  every core. The dense kernels never add threads that compete with these.
- **Pivots stay in their front.** When a pivot is tiny, SPARTA raises it
  instead of passing it on to a later block (see `static_pivot` in
  [Numerics](#numerics)). As a result, the plan never changes during
  factoring.
- **Real arithmetic where the values are real.** In many FEM models, only the
  port and boundary unknowns have complex values. SPARTA factors the parts that
  stay real in real arithmetic, which is about four times cheaper (see
  `real_split`).
- **Complex symmetric LDLᵀ.** Lossy materials and absorbing boundaries give
  complex symmetric matrices. SPARTA factors these as LDLᵀ, which costs half as
  much as LU.
- **Solves in blocks.** SPARTA pushes many right-hand sides through the factors
  in one pass. Port projections touch only the parts of the factors they need
  (see [Port matrices](#port-matrices-and-sparse-right-hand-sides)).
- **No round trips in GPU mode.** The whole factorization runs on the GPU,
  without returning to the CPU between steps.

### The speed of each operation: dense kernels near peak

Most of the arithmetic happens in large dense blocks near the top of the
elimination tree, as matrix products. SPARTA runs these on its own dense
kernels, SpartaBLAS, with a separate kernel set for each processor family (see
[CPU dense kernels](#cpu-dense-kernels-spartablas)).

On an Apple M4 Pro, against the processor's measured peak:

| Product | Unit | Peak | SpartaBLAS | Fraction of peak |
|---|---|---|---|---|
| `f64`, one thread | matrix unit (SME) | 499 GFLOP/s | 450 GFLOP/s | 90% |
| `complex128`, one thread | matrix unit (SME) | 499 GFLOP/s | 453 GFLOP/s | 91% |
| `f32`, one thread | matrix unit (SME) | 1996 GFLOP/s | 1634 GFLOP/s | 82% |
| `complex64`, one thread | matrix unit (SME) | 1996 GFLOP/s | 1763 GFLOP/s | 88% |
| `f64`, one core | vector unit (NEON) | 58 GFLOP/s | 52 to 55 GFLOP/s | 90 to 95% |
| `complex128`, all 14 cores | matrix units | 1255 GFLOP/s | 902 GFLOP/s | 72% |

The peak is the most fused multiply-adds the processor can issue, measured with
nothing else running. The products are 512 × 512 × 512 on one thread and
3000 × 3000 × 400 on all cores, in the `C −= A·Bᵀ` form the factorization uses.
To reproduce the numbers, run `examples/peak.rs` and
`examples/vs_accelerate.rs` in `crates/sparta-blas`. The x86 kernels (AVX2,
AVX-512) are written but **not yet measured** on x86 hardware.

In CPU mode, the dense kernels take most of the factorization time. On our
largest FEM test case, 97% of the time goes to the large blocks.

### What SPARTA is not designed for

Two kinds of sparse systems fit SPARTA poorly:

- **Badly scaled, strongly indefinite systems**, such as some saddle-point
  problems. These need pivots that move between blocks to stay accurate.
  SPARTA keeps pivots in their block, so it can lose accuracy here. When that
  happens, SPARTA reports it (see [Checking a solve](#checking-a-solve)).
- **Matrices that do not come from a mesh**, such as circuit or network
  matrices. These have few large dense blocks, so SPARTA's dense kernels hardly
  help.

## Requirements

### Per operating system

**macOS.** An Apple Silicon Mac (M1 or later) with macOS 13 or later runs
everything, including the GPU. An Intel Mac runs the analysis and the
double-precision CPU backend, but has no GPU backend.

**Windows.** Any 64-bit machine runs the analysis and the double-precision CPU
backend. AMD and NVIDIA GPUs run through the [Vulkan](#vulkan) backend, which
is tested on an NVIDIA RTX 3080 and an AMD Radeon integrated GPU.

**Linux.** The same as Windows: the analysis, the CPU backend and the Vulkan
backend, tested on two NVIDIA RTX A5000s. The NVIDIA driver on a server or
headless install often comes without its Vulkan part, so the GPU works for
CUDA but `gpu_available()` is false. `sparta.gpu_status()` then says which part
is missing and, when an NVIDIA driver is loaded, which package to install. To
add it on Ubuntu or Debian, install the Vulkan loader and the Vulkan driver
that match the installed driver version (580 here, or `libnvidia-gl-580-server`
for the server driver):

```sh
sudo apt install libvulkan1 vulkan-tools libnvidia-gl-580
vulkaninfo --summary      # lists the GPUs Vulkan can use
```

On Fedora and RHEL, the loader is `vulkan-loader`, and the NVIDIA driver
package includes its Vulkan part. In a Docker container,
`NVIDIA_DRIVER_CAPABILITIES` must include `graphics`.

### Packages

| Package | What it is for |
|---|---|
| Rust toolchain (`rustc`, `cargo`) | Compiling SPARTA. Not needed for the PyPI wheels or prebuilt binaries. |
| Xcode command line tools | macOS only: compiling the Metal bridge. |
| git | Downloading the source from GitHub. |
| Python 3.9 or later, with pip | The Python package. |
| NumPy and SciPy | Used by the Python package. pip installs them automatically. |

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

For Python, install SPARTA from PyPI (see [Python](#python)). That needs no
Rust and no compiler on the platforms listed there. Every other route compiles
SPARTA from source.

SPARTA is not published on crates.io, so `cargo add sparta-solver` will not
find it. The other languages have three routes:

1. **You have access to the GitHub repository.**
2. **You have a copy of the source, but no GitHub access.**
3. **You have prebuilt binaries**, so nothing needs compiling.

Routes 1 and 2 compile SPARTA and need the Rust toolchain from
[Requirements](#requirements), and on macOS the Xcode command line tools. The
first build takes a few minutes.

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

Install SPARTA into the Python environment you run your simulations in. The
package is imported as `sparta`.

**1. From PyPI.**

```sh
pip install sparta-solver
```

PyPI has prebuilt wheels for Python 3.9 and later on these platforms, so they
need no Rust and no compiler:

| Platform | Backends |
|---|---|
| macOS on Apple Silicon | Metal GPU and CPU |
| macOS on Intel | CPU |
| Linux x86_64 and aarch64 | Vulkan GPU and CPU |
| Windows x86_64 | Vulkan GPU and CPU |

On any other platform, pip downloads the source and compiles it. That needs
the Rust toolchain from [Requirements](#requirements).

**2. From GitHub.** This installs the latest, unreleased code and needs the
Rust toolchain:

```sh
pip install "git+https://github.com/FennisRobert/SPARTA.git"
```

**3. From a copy of the source.** This also needs the Rust toolchain. Unpack
the source, then install from its top folder, the one containing
`pyproject.toml`:

```sh
cd path/to/SPARTA
pip install .
```

pip downloads its build helper (maturin), NumPy and SciPy from PyPI.

**Check the install:**

```python
import sparta
print(sparta.__version__, sparta.gpu_available())   # gpu_available() is False if no GPU is usable
print(sparta.gpu_status())   # the GPU in use, or why there is none and what to install
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

More in [crates/sparta-capi/README.md](https://github.com/FennisRobert/SPARTA/blob/main/crates/sparta-capi/README.md).

### MATLAB and Julia

Both use the C library above, so build it first with
`cargo build --release -p sparta-capi`.

**MATLAB** (untested: written without access to MATLAB). From
`bindings/matlab`, run `build_sparta_mex` once (it needs a C compiler that
`mex -setup C` accepts), then `test_sparta`. Add that folder to the MATLAB path.

**Julia** (1.9 or later). No compiler is needed:

```julia
using Pkg
Pkg.develop(path = "bindings/julia")
```

Details in [bindings/matlab/README.md](https://github.com/FennisRobert/SPARTA/blob/main/bindings/matlab/README.md) and
[bindings/julia/README.md](https://github.com/FennisRobert/SPARTA/blob/main/bindings/julia/README.md).

## Quick start

All the examples solve the same 2×2 system, `[[4, 1], [1, 3]] x = [1, 2]`,
whose answer is `x = [1/11, 7/11]`.

### Rust

```rust
use sparta::{Analysis, Config, Solver};

fn main() -> sparta::Result<()> {
    // Symmetric, so pass the lower triangle in CSC form.
    let n = 2;
    let col_ptr = [0, 2, 3];
    let row_idx = [0, 1, 1];
    let values: [f32; 3] = [4.0, 1.0, 3.0];

    let analysis = Analysis::ldlt(n, &col_ptr, &row_idx, &Config::default())?; // 1. analyze (CPU)
    let mut solver = Solver::new(analysis)?;
    solver.factor_values(&values)?;                                           // 2. factor (GPU)
    let b: [f32; 2] = [1.0, 2.0];
    let x = solver.solve_values(&b, 1)?;                                      // 3. solve (GPU)
    println!("{x:?}");
    solver.destroy();                                                         // 4. free
    Ok(())
}
```

`Analysis` takes the pattern in CSC form, and `factor_values` takes the values
in the same order. The type of the values picks the arithmetic:

- **Real matrix:** pass `f32` values, as above. `x` comes back as `f32`.
- **Complex matrix:** pass `Complex32` values instead, for example
  `[4.0, 1.0, 3.0].map(|v| Complex32::new(v, 0.5))`, and a `Complex32` b.

A complex b may be solved against a real matrix; its x is complex. A real b
against a complex matrix is refused, because its x is complex.

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

`analyze` takes any square SciPy sparse matrix with float32, float64,
complex64 or complex128 values. SPARTA never modifies your matrix.

SPARTA keeps its own copy of the matrix, so that later changes to your array
cannot put refinement out of step with the factors. For a symmetric matrix, it
keeps only the lower triangle. With refinement on, this copy is about the size
of your matrix when it is symmetric, and about twice that when it is not.

**Real matrices.** A real matrix is factored in real arithmetic. In CPU mode,
this is about twice as fast as the same matrix stored as complex. In GPU mode,
real matrices still run through the complex kernels, so they are not faster
yet. When the matrix and b are both real, x is real. Otherwise, x is complex.

### C and C++

```c
#include "sparta.h"

int64_t col_ptr[]   = {0, 2, 3};                       /* lower triangle, CSC */
int64_t row_index[] = {0, 1, 1};
float values[] = {4, 1, 3};                            /* a real matrix */
float b[] = {1, 2}, x[2];

sparta_analysis *plan = NULL;
sparta_solver *solver = NULL;
sparta_analyze(2, col_ptr, row_index, SPARTA_SYMMETRIC, NULL, &plan);   /* 1. analyze */
sparta_solver_create(&plan, &solver);
sparta_factor_real(solver, values);                                     /* 2. factor  */
sparta_solve_real(solver, b, x);                                        /* 3. solve   */
sparta_solver_free(solver);                                             /* 4. free    */
```

For a complex matrix, use `sparta_complex` arrays with `sparta_factor` and
`sparta_solve` instead:

```c
sparta_complex values[] = {{4, 0.5f}, {1, 0}, {3, 0.5f}};
sparta_complex b[] = {{1, 0}, {2, 0}}, x[2];
sparta_factor(solver, values);
sparta_solve(solver, b, x);
```

`sparta_solve` also accepts a complex b against real factors.

Every call returns a `sparta_status`, and `sparta_last_error()` explains a
failure in a sentence. The C++ header `sparta.hpp` wraps all of this in a
`sparta::Solver` class, whose `factor`, `solve` and `solve_many` take either
`std::vector<float>` (real) or `std::vector<std::complex<float>>` (complex).
Worked examples are in `crates/sparta-capi/examples`.

### MATLAB and Julia

```matlab
A = sparse([4 1; 1 3]);
S = SpartaSolver(A);     % analyze and factor
x = S \ [1; 2];          % solve
```

```julia
using SPARTA, SparseArrays
F = SPARTA.Solver(sparse([4.0 1; 1 3]))   # analyze and factor
x = F \ [1.0, 2]                          # solve
close(F)                                  # free the GPU memory
```

For a sweep, `factorize(S, A2)` in MATLAB and `factorize!(F, A2)` in Julia
factor new values of the same pattern without analyzing again.

Julia picks the arithmetic from the element type: a `Float64` matrix is
factored as real and returns `Float64` answers, a `ComplexF64` one as complex.
MATLAB factors every matrix as complex for now.

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
[python/API.md](https://github.com/FennisRobert/SPARTA/blob/main/python/API.md) explains each one.

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
solver.factor_values(&values1)?;
let x1 = solver.solve_values(&b1, 1)?;
solver.factor_values(&values2)?;       // same pattern, new values
let x2 = solver.solve_values(&b2, 1)?;
```

The values may change type between factorizations, for example from a real
matrix to a complex one; the arithmetic follows.

**Python**

```python
F = sparta.analyze(A1).factorize()
x1 = F.solve(b1)
F.factorize(matrix=A2)          # same pattern, new values
x2 = F.solve(b2)
```

**C**

```c
sparta_factor(solver, values1);   /* complex; sparta_factor_real for a real matrix */
sparta_solve(solver, b1, x1);
sparta_factor(solver, values2);   /* same pattern, new values */
sparta_solve(solver, b2, x2);
```

Only the values may change, and they must be in the same order as the first
time. Stored zeros count as part of the pattern: a matrix with one extra
explicit zero has a different pattern.

In Python:

- `factorize()` with no matrix reuses the values of the last successful
  factorization.
- `factorize(matrix=...)` with a different pattern runs the analysis again,
  with your settings, before it factors.
- `F.reanalyses` counts these extra analyses. Each one costs a full analysis.
  If the count keeps rising, your pattern changes between calls.

### Several right-hand sides at once

Solve many right-hand sides in one call, not one at a time. The columns share
one pass over the factors. On our FEM test cases, a block of eight columns is
4× to 6× faster than eight separate solves.

**Rust**

```rust
// The columns are stored end to end: column j is b[j*n .. (j+1)*n].
let x = solver.solve_values(&b, nrhs)?;
```

**Python**

```python
X = F.solve_many(B)       # B.shape == (n, nrhs), one right-hand side per column; X has the same shape
```

**C**

```c
sparta_solve_many(solver, b, x, nrhs);        /* complex b, columns end to end as in Rust */
sparta_solve_many_real(solver, b, x, nrhs);   /* real b against real factors */
```

### Port matrices and sparse right-hand sides

In a frequency sweep, you often need only a few numbers from the solution:
the port matrix of the model, or the field at a few chosen unknowns. SPARTA
can compute those directly, without forming the full solution.

- **B: what you excite.** An `n × p` sparse matrix with one column per
  excitation, such as one port. Each column is nonzero only on the unknowns of
  that port.
- **W: what you measure.** An `n × q` sparse matrix with one column per
  quantity, each a weighted sum of unknowns.

The result is `Wᵀ A⁻¹ B`, a small `q × p` matrix. Entry `(i, j)` is
measurement `i` when column `j` of B is excited. The transpose is plain,
never conjugated. With `W = B` it is the port matrix `Bᵀ A⁻¹ B`. With one
column per chosen unknown (a 1 in its row) it gives the values of the
solution there.

**Python**

```python
X = F.solve_many(B)              # B may be SciPy sparse: the full n x p solution, faster
Z = F.solve_many(B, project=B)   # the p x p port matrix B^T A^-1 B
V = F.solve_many(B, project=W)   # W^T A^-1 B, q x p
v = F.solve(b, project=W)        # one right-hand side: W^T x, length q
```

**Rust**

```rust
use sparta::SparseColumns;
// CSC arrays, numbered as the matrix: column j has rows[ptr[j]..ptr[j+1]].
let b = SparseColumns::new(n, &ptr, &rows, &values)?;
let x = solver.solve_sparse(&b)?;              // n values per column, end to end
let z = solver.solve_projected(&b, None)?;     // B^T A^-1 B, p x p, column-major
let v = solver.solve_projected(&b, Some(&w))?; // W^T A^-1 B, q x p, column-major
```

**How it runs.** For LDLᵀ factors in CPU mode, SPARTA takes a fast route. A
column of B only reaches the blocks on the path from its unknowns to the top of
the elimination tree, so a projection only touches those blocks. It forms no
full solution and never builds an `n × p` block.

For GPU factors, the hybrid factorization and LU, SPARTA solves as usual and
then projects. The answer is the same, but not faster.

`F.diagnostics()['solve_route']` says which route the latest solve took:
`'sparse'` or `'dense'`.

**Accuracy.** Projections are as accurate as an ordinary solve:

- **CPU factors:** In Python, SPARTA takes the fast route only when no pivot
  was perturbed. The double-precision factors then need no refinement.
- **Sparse solve without `project`:** SPARTA measures the residual. If it is
  above `refinement_rtol`, SPARTA refines on the usual route. You get the same
  reports and warnings as from a dense solve.
- **Fast projection:** No full solution is formed, so `refinement_reports` is
  empty.
- **Symmetry:** For a symmetric matrix, `project=B` returns an exactly
  symmetric result on every route.

**What it saves.** Measured on the CPU (M4 Pro, complex128), with ports of six
unknowns each:

| System | Ports | `B.T @ solve_many(B)` | `solve_many(B, project=B)` | Speedup | Including the factorization |
|---|---|---|---|---|---|
| 109k unknowns | 4 | 0.030 s | 0.010 s | 3× | 1.06× |
| 109k unknowns | 100 | 0.363 s | 0.028 s | 13× | 2.0× |
| 887k unknowns | 20 | 0.782 s | 0.126 s | 6× | 1.13× |
| 887k unknowns | 100 | 7.64 s | 0.20 s | 38× | 2.5× |

The more ports you have, the more it saves. With only a few ports, the
factorization takes most of the time either way.

**If you know separate port-matrix calls.** Some solvers offer
`port_matrix(B)`, `solve_sparse(B)` and a "reach" query as separate calls. In
SPARTA, they map as follows:

| Elsewhere | SPARTA |
|---|---|
| `port_matrix(B)` | `solve_many(B, project=B)`; Rust `solve_projected(&b, None)` |
| `solve_sparse(B)` | `solve_many(B)` with a sparse B; Rust `solve_sparse(&b)` |
| A reach query, to decide whether the sparse route pays | Not needed: SPARTA chooses the route, and `diagnostics()['solve_route']` reports it |

SPARTA's versions work on every factorization, including GPU and LU, and
also take a separate `W`. C, Julia and MATLAB do not offer them yet.

## Configuration

Most users need no settings: the defaults are measured. The settings that
matter most are covered in [Precision and accuracy](#precision-and-accuracy).
This section lists every setting, for when you want to tune.

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

To change an existing solver, you can also pass the solve-behaviour settings,
`force_cpu` and `hybrid` to `factorize`. All other settings are fixed at
analysis, and `factorize` ignores them. So you can pass one settings dict to
both calls.

An unknown keyword never stops a run. `analyze`, `factorize` and
`Config.with_` ignore it and issue a `sparta.UnusedSettingWarning`.

**C:** fill a `sparta_config` with `sparta_config_default` or
`sparta_config_preset`, change fields, and pass it to `sparta_analyze`.

### All settings

Types are given language-neutrally: `uint32` and `uint64` are unsigned
integers (`usize` in Rust and `size_t` in C for `uint64`), `float64` is a
double, and `enum` is one of the named choices.

| Name | Type | Default | What it is for |
|---|---|---|---|
| **Solve behaviour** (Python only) | | | |
| `refinement_steps` | int | `-1` | Limits the refinement corrections per right-hand side. `-1` takes as many as needed. |
| `refinement_rtol` | float64 | `1e-6` | Sets the relative residual at which refinement stops. |
| `refinement_minimize` | bool | `True` | Makes each correction shrink the residual as much as possible, so fewer steps are needed. |
| `equilibrate` | bool | `False` | Balances row and column sizes before the conversion to single precision. This can improve accuracy. |
| `verbose` | bool | `False` | Prints a short summary after each solve. |
| `safe_mode` | bool | `False` | Checks every new matrix passed to `factorize`. This is slower, but catches infinities and NaNs. |
| `mtype` | enum | `AUTO` | Tells SPARTA what kind of matrix you pass. |
| `path` | enum | `AUTO` | Forces LDLᵀ or LU. |
| `triangle` | text | `"full"` | Tells SPARTA whether you pass the whole matrix or only the lower triangle. |
| **Ordering** | | | |
| `ordering` | enum | SCALPEL | Selects the nested-dissection implementation. SCALPEL is the only one. |
| `imbalance` | float64 | `0.8` | Only used by SCALPEL. Sets how unequal the two halves of a split may be. |
| `wide_cycles` | uint32 | `4` | Only used by SCALPEL. More cycles give less fill, but the ordering takes longer. |
| `fm_passes` | uint32 | `4` | Only used by SCALPEL. More passes give less fill, but the ordering takes longer. |
| `separator_trials` | uint32 | `4` | Only used by SCALPEL. More trials can find a better split, but the ordering takes longer. |
| **Supernodes and fronts** | | | |
| `nemin` | uint64 | `16` | Controls how eagerly columns merge into wider blocks. Larger values suit the GPU, but store more fill. |
| `max_front` | uint64, optional | unlimited | Makes the analysis fail when a dense block would have more rows than this. |
| **Numerics** | | | |
| `force_cpu` | bool | `False` | Factors and solves on the CPU in double precision instead of on the GPU. Only available in Python and Rust. |
| `hybrid_cpu` | float64 | `0.0` | Lets the CPU do this share of the factorization while the GPU does the rest. Only used on Metal. In Python, `hybrid=True` turns it on. |
| `lu_diag_threshold` | float64 | `0.0` | Only affects LU factorization. Above 0, SPARTA prefers the diagonal as pivot, which is faster but less stable. |
| `supernodal` | bool | `True` | Speeds up LDLᵀ factorization and lowers its memory. Expected to be strictly better than the multifrontal factorization, which is used when it is turned off, with the same accuracy. Turn it off to check whether your matrix is an exception (see [Supernodes and fronts](#supernodes-and-fronts)). |
| `real_split` | bool | `True` | Speeds up complex LDLᵀ factorization when parts of the matrix are real. Turn it off only if the GPU's first factorizations must all take the same time. |
| `static_pivot` | float64 | `1e-12` | Only affects LDLᵀ factorization. Raises tiny pivots to this size, relative to the largest entry, so SPARTA never divides by a near-zero. `0` replaces exact zeros only. |
| **CPU backend** (Python and Rust) | | | |
| `cpu_panel` | uint64 | `32` | Only used in CPU mode. Sets how many pivots are gathered before an update. |
| `cpu_block_rows` | uint64 | `96` | Only used in CPU mode. Sets the rows per block of the trailing update. |
| `cpu_blas_min_rows` | uint64 | `64` | Only used in CPU mode. Blocks smaller than this skip the dense kernels. |
| **GPU scheduling** | | | |
| `small_front_rows` | uint32 | `32` | Sets the block height where the GPU switches from its serial kernel to cooperative panels. |
| `schur_batch` | uint32 | `0` | Sets how many panels the GPU completes between two update drains. |
| `panel_threads` | uint32 | `512` | Sets the threads per threadgroup in the GPU panel factorization. |
| `solve_threads` | uint32 | `0` | Sets the threads per block in the GPU solves. |
| **GPU kernel geometry** | | | |
| `panel` | uint32 | `16` | Sets the pivots per GPU panel. |
| `schur_block` | uint32 | `4` | Sets the register block of the GPU Schur update. |
| `strip_block` | uint32 | `4` | Sets the register block of the GPU strip update. |
| `narrow_cols` | uint32 | `32` | Sets the width above which a block uses the GPU's wide kernels. |
| `pivot_threads` | uint32 | `128` | Sets the GPU threads that search for one pivot together. |
| `schur_chunk` | uint32 | `16` | Sets the panel columns staged per GPU Schur pass. |

The tables further down give the allowed values and more detail for each group.

### Presets

A preset is a named starting point for a kind of workload. The best settings
still depend on your matrices, so measure before you keep a change.

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

It tunes both the GPU and the CPU backend. It keeps a change only if the
change saves more than 3% (`min_gain`), beats the timing noise and holds up on
a recheck. At the end, it reports the gain over the defaults.

- `iterations=3` repeats the search from the settings found, until a pass
  changes nothing.
- Without matrices, it tunes on two standard finite-element systems.
- Profiles are kept in `configs.json`, in `~/Library/SPARTA` on macOS and in
  `~/.sparta` elsewhere. `sparta.profiles()` lists them, and
  `sparta.Config.delete(name)` removes one.
- In Rust, the same search is `sparta::optimize::optimize`, and
  `sparta::profiles` manages the same file.

The user guide has the details.

### Solve behaviour (Python only)

These settings control the Python layer around the solver. Rust and C do
not have them yet.

| Setting | Type | Values | What it does |
|---|---|---|---|
| `refinement_steps` | int | `-1`, `0`, or a positive integer | Limits the corrections per right-hand side. `-1` takes as many as `refinement_rtol` needs. `0` turns refinement off. |
| `refinement_rtol` | float64 | between 0 and 1 | Refinement stops when the relative residual falls below this value. |
| `refinement_minimize` | bool | | Scales each correction to shrink the residual as much as possible. Costs one extra CPU matrix-vector product per step, and roughly halves the steps on badly conditioned matrices. | | Scales each correction to shrink the residual as much as possible. Each step costs one extra matrix-vector product on the CPU. On badly conditioned matrices, it roughly halves the number of steps. |
| `equilibrate` | bool | | Balances row and column sizes before the conversion to single precision. Can help accuracy, costs extra CPU work and memory, and does not always pay off. | | Balances row and column sizes before the conversion to single precision. This can improve accuracy, but costs extra CPU time and memory, and does not always help. |
| `verbose` | bool | | Prints a short summary per solve: method, ordering, fill, memory, timings, residuals. Adds the CPU work of measuring the residuals. | | Prints a short summary after each solve: method, ordering, fill, memory, timings and residuals. Measuring the residuals adds some CPU time. |
| `safe_mode` | bool | | Re-checks every new matrix passed to `factorize`. Off, a new matrix stored exactly like the analyzed one is read straight into the analyzed layout, which is much faster, and its values are not checked for infinities or NaNs. Either way a different pattern is detected and analyzed again. | | Checks every new matrix passed to `factorize`. When it is off, a matrix stored exactly like the analyzed one is read straight into the analyzed layout. That is much faster, but its values are not checked for infinities or NaNs. In both cases, SPARTA detects a different pattern and analyzes it again. |
| `mtype`, `path`, `triangle` | enum, enum, text | see [Describing your matrix](#describing-your-matrix) | These tell SPARTA how to read and factor the matrix. |

### SCALPEL settings

| Setting | Type | Values | What it does |
|---|---|---|---|
| `imbalance` | float64 | 0.5 to 0.9 | Sets the largest share of the matrix allowed in one half of a split. |
| `wide_cycles` | uint32 | 0 or more | Sets the separator refinement cycles per split. With `0`, the ordering is faster, but the factors get more fill. |
| `fm_passes` | uint32 | 0 or more | Sets the Fiduccia–Mattheyses refinement passes. With `0`, the ordering is much faster, but the factors get much more fill. |
| `separator_trials` | uint32 | 1 or more | Sets how many starting separators are tried per split. SPARTA keeps the best one. |

SCALPEL runs in parallel, on the threads that [`threads`](#cpu-threads)
sets. The ordering decides how much fill the factors get, and so most of the
cost. A faster ordering does not always give a faster factorization, so
measure both.

### Supernodes and fronts

| Setting | Type | Values | What it does |
|---|---|---|---|
| `nemin` | uint64 | 1 or more | Sets how many columns a group of columns (a supernode) must eliminate before it may merge with its neighbour. Larger values give fewer, wider blocks. These suit the GPU, but store more fill. |
| `max_front` | uint64, optional | 1 or more, or unlimited | Makes the analysis fail when a dense block would have more rows than this. Use it to catch a bad ordering early. In Python and Rust, `None` means unlimited. In C, `0` means unlimited. |
| `supernodal` | bool | | Only affects LDLᵀ factorization. In CPU mode, it factors each block of columns in place instead of in a separate frontal matrix, which makes the factorization 25 to 43% faster and lowers peak memory. In GPU mode, it halves the work of setting up each front, which makes the factorization about 15% faster. It is expected to be strictly better than the multifrontal factorization, which is used when it is turned off: the pivots and the accuracy are the same, but the two can differ slightly in how they round. One exception is possible in CPU mode. Threads that update the same rows of the factor take turns, so on a matrix where many independent parts feed into the same rows, the threads can end up waiting on each other. None of our FEM test cases shows this. If the CPU factorization of your matrix is slower than expected, turn it off to check whether this is the cause. LU, the hybrid's CPU share and Vulkan ignore it. |
| `real_split` | bool | | Only affects complex LDLᵀ factorization. It factors the parts of the matrix without imaginary entries in real arithmetic. In CPU mode, this makes the factorization 1.1 to 1.8 times faster and lowers peak memory by 12 to 28%, but only when `supernodal` is on. In GPU mode, the solver runs its second factorization without it and its third with it, then keeps whichever was faster, so the first three factorizations can differ in time. The results are equally accurate either way, so turn it off only if those first factorizations must all take the same time. |

### Numerics

| Setting | Type | Values | What it does |
|---|---|---|---|
| `force_cpu` | bool | | Factor and solve on the CPU in double precision instead of on the GPU. Python and Rust only. See [Precision and accuracy](#precision-and-accuracy). | | Factors and solves on the CPU in double precision instead of on the GPU. Only available in Python and Rust. See [Precision and accuracy](#precision-and-accuracy). |
| `hybrid_cpu` | float64 | 0 to below 1 | Lets the CPU do this share of the factorization while the GPU does the rest. `0` uses the GPU alone. The value is only a starting point: SPARTA adapts the share after every factorization. In Python, `hybrid=True` sets 30%. Only used on Metal. See [CPU and GPU together](#cpu-and-gpu-together-hybrid). |
| `lu_diag_threshold` | float64 | 0 to 1 | Only affects LU factorization. With `0`, SPARTA uses strict partial pivoting. Above 0, SPARTA takes the diagonal entry as pivot if it is within this fraction of the best candidate in its column. This is faster but less stable, and **it changes the results**, not just the speed. |
| `static_pivot` | float64 | 0 to below 1 | Only affects LDLᵀ factorization. SPARTA keeps each pivot in its block, so a pivot can be tiny without being zero. This happens, for example, with zero diagonal entries in saddle-point or constrained systems. Dividing by such a pivot can ruin the factorization. SPARTA therefore raises any pivot smaller than `static_pivot` times the largest matrix entry to that size, and counts it in `perturbed_pivots`. In CPU mode, the raise is also capped at `static_pivot` times the largest entry below the pivot, so a small pivot on a badly scaled diagonal is left alone. In GPU mode, single precision cannot tell such a pivot from rounding noise, so the threshold applies as is. The default, `1e-12`, is far below the pivots of ordinary matrices. With `0`, SPARTA replaces exact zeros only, as before version 0.3.5. LU and Vulkan ignore it. Only available in Python and Rust. In C, Julia and MATLAB, `SPARTA_STATIC_PIVOT` overrides the default. |

### CPU threads

| Setting | Type | Values | What it does |
|---|---|---|---|
| `threads` | uint64 (uint32 in C) | 0 to 4096 | Sets the CPU threads for SPARTA's own work: the analysis, the CPU factorization and solves, and the hybrid's CPU share. With `0`, the default, SPARTA uses the process-wide Rayon pool, which `RAYON_NUM_THREADS` sizes. With a positive count, each analysis and each solver gets a pool of its own. |

A positive count is a hard limit. The dense kernels start no threads of
their own and run on the same pool, so SPARTA never uses more threads than
this.

`RAYON_NUM_THREADS` is read once, when the process first uses Rayon, and
applies to everything in the process. `threads` can differ per solver and does
not affect other code that uses Rayon.

If `SPARTA_NUM_THREADS` is set to a positive count, it overrides `threads`
everywhere, in every language. Use it in wrappers and job scripts that cannot
change the call.

```python
solver = sparta.analyze(A, force_cpu=True, threads=6)
```

### CPU backend tuning (Python and Rust)

These settings only apply in CPU mode (`force_cpu`).

| Setting | Type | Values | What it does |
|---|---|---|---|
| `cpu_panel` | uint64 | 1 or more | Sets how many pivots are gathered before a trailing update. It applies to LU, and to the LDLᵀ blocks that the blocked route cannot take (an exactly zero or tiny pivot). |
| `cpu_block_rows` | uint64 | 1 or more | Sets the rows per block of the trailing update. When one block is split across threads, this is also the unit of work. |
| `cpu_blas_min_rows` | uint64 | 1 or more | Blocks with fewer trailing rows than this skip the dense kernels. Raise it to avoid the call overhead on small blocks. |

### GPU tuning (advanced)

The defaults are measured on Apple Silicon. Change these settings only when
you tune for different hardware.

| Setting | Type | Values | What it does |
|---|---|---|---|
| `small_front_rows` | uint32 | 1 or more | Sets the block height at which the GPU switches from its serial kernel to cooperative panels. |
| `schur_batch` | uint32 | 0 or more | Sets how many pivot panels the GPU completes between two update drains. With `0`, it drains once per level of the tree. |
| `panel_threads` | uint32 | 32 to 1024 | Sets the threads per threadgroup in the panel factorization. |
| `solve_threads` | uint32 | `0`, or 32 to 1024 | Sets the threads per block in the solves. With `0`, SPARTA picks it per block width. |
| `panel` | uint32 | 16 | Sets the pivots per panel. Other widths need kernel changes. |
| `schur_block` | uint32 | 1 to 8 | Sets the per-thread register block in the Schur update. The tile is `16 × schur_block` square. |
| `strip_block` | uint32 | 1 to 8 | Sets the per-thread register block in the strip update. |
| `narrow_cols` | uint32 | 1 or more | Sets the width above which a block uses the wide-front kernels. |
| `pivot_threads` | uint32 | a multiple of 32, up to 1024 | Sets the threads that search for one pivot together. |
| `schur_chunk` | uint32 | 1 or more | Sets the panel columns staged per Schur pass. |

The last six settings are compiled into the GPU kernels. The first time a
process uses a new combination, shader compilation takes about 300 ms.

If a setting breaks a device limit, such as shared memory, SPARTA rejects it
with a message that names the setting.

For quick experiments, these environment variables override their defaults:
`SPARTA_SCHUR_BATCH`, `SPARTA_PANEL_THREADS`, `SPARTA_SOLVE_THREADS`,
`SPARTA_LU_DIAG_THRESHOLD`, `SPARTA_STATIC_PIVOT` and `SPARTA_HYBRID_CPU`.

## Precision and accuracy

### Single versus double precision

The GPU computes in **single precision**: float32 for a real matrix,
complex64 for a complex one. That holds about seven significant digits. On FEM
systems, a GPU solve without refinement reaches a relative residual of about
1e-5.

The **CPU backend** computes in **double precision**: float64 or complex128,
about sixteen digits. It is much slower.

**Iterative refinement** sits between the two. The CPU measures the error of
the GPU answer in double precision, and the GPU computes corrections. This
gives double-precision accuracy at close to single-precision speed. It works on
all but the most badly conditioned systems.

| Approach | Accuracy | Speed | Available in |
|---|---|---|---|
| GPU, no refinement | residual ~1e-5 | fastest | Rust, Python, C |
| GPU with refinement (Python default) | residual down to 1e-6, or tighter if asked | ~9% slower on the larger test systems | Python |
| CPU backend | full double precision | much slower | Rust, Python |

**The residual is not the error in x.** Refinement stops on the residual,
`‖b − Ax‖ / ‖b‖`. The error in x is larger: about 7× on a well-conditioned
system, and several hundred times on a sensitive one. So ask for a tighter
residual than the accuracy you need.

### Getting more accuracy from the GPU

**Rust and C.** These interfaces have no built-in refinement yet. A solve
returns the single-precision answer. `examples/refine.rs` shows how to refine
from Rust yourself.

**Python.** Refinement is on by default and stops at a relative residual of
`1e-6`. Keep A and b in double precision (float64 or complex128), because the
residual is measured against them.

```python
F = sparta.analyze(A.astype(np.float64)).factorize()      # a real matrix
x = F.solve(b.astype(np.float64))                         # x is float64

F = sparta.analyze(A.astype(np.complex128)).factorize()   # a complex matrix
x = F.solve(b.astype(np.complex128))                      # x is complex128

sparta.analyze(A, refinement_rtol=1e-10)   # tighter, at the cost of more corrections
sparta.analyze(A, refinement_steps=2)      # at most two corrections, whatever they reach
sparta.analyze(A, refinement_steps=0)      # off: the bare single-precision solve
```

The factors stay in single precision. Only the residuals and the answer are
kept in double precision. So a refined solve always returns double precision:
float64 when A and b are both real, complex128 otherwise.

Refinement stops as soon as it reaches the target. If it stops improving
first, it returns the best answer it found and says so in its report (see
[Checking a solve](#checking-a-solve)). If refinement struggles, try
`equilibrate=True`, the `careful` preset, or the CPU backend.

### Solving on the CPU in double precision

`force_cpu` runs the factorization and the solves on the CPU, in double
precision. It uses the same analysis and the same kind of factorization as the
GPU. The large dense updates run on SPARTA's own dense kernels (see
[CPU dense kernels](#cpu-dense-kernels-spartablas)).

SPARTA never switches to the CPU on its own. If the GPU fails, you get an
error, not a slow answer.

**Rust:** Set `force_cpu: true` in the `Config`. Pass double-precision values
to `factor_values` and `solve_values`: `f64` for a real matrix, `Complex64` for
a complex one. The CPU computes in the type you pass, so with `f32` or
`Complex32` it computes in single precision.

**Python:**

```python
F = sparta.analyze(A, force_cpu=True, refinement_steps=0).factorize()
F.compute_dtype     # float64 for a real A, complex128 for a complex one

# Or switch an existing solver, reusing its analysis:
F.factorize(force_cpu=True)     # force_cpu=False switches back to the GPU
```

When you switch, SPARTA releases the old factors and builds new ones on the
chosen backend. `F.factor_seconds` reports the CPU factorization time.

**C:** Not available. The C library only uses the GPU.

[CPU architecture and tuning](https://github.com/FennisRobert/SPARTA/blob/main/src/cpu/README.md) covers how the CPU backend
executes and what memory it uses.

### CPU and GPU together (hybrid)

On a Mac, the CPU and the GPU share one memory, so they can factor one matrix
together. With `hybrid` on, the CPU factors whole branches of the elimination
tree while the GPU factors the rest. The GPU then finishes from where the two
meet.

The CPU computes in the GPU's precision, so the answers match a GPU-only
factorization up to single-precision rounding. It works for LDLᵀ and LU, real
and complex.

**Python:**

```python
F = sparta.analyze(A, hybrid=True).factorize()   # the CPU starts at 30% of the work
F.factorize(A2, hybrid=0.2)                      # or start it at 20%
F.factorize(A3, hybrid=False)                    # the GPU alone again
F.hybrid_timing                                  # how the last factorization split
```

**Rust:** set `hybrid_cpu: 0.3` in the `Config`. **C:** set
`config.hybrid_cpu` before `sparta_analyze`. **Julia:** `Solver(A; hybrid = true)`.
**MATLAB:** the `'hybrid_cpu'` setting.

The first factorization already runs split. After that, SPARTA adapts the
split to the measured speed of each device. In a sweep, the split settles from
the third factorization on. If you change `hybrid` in `factorize`, SPARTA
rebuilds the GPU factors.

With `force_cpu`, on Vulkan, or without a GPU, SPARTA ignores `hybrid` instead
of refusing it.

Measured on an Apple M4 Pro (10 performance cores, 20-core GPU), settled sweep
steps, with `RAYON_NUM_THREADS=10` (the performance cores, which beat the
default of all cores):

| System | GPU alone | Hybrid | Speed-up |
|---|---:|---:|---:|
| 887k unknowns, LDLᵀ | 1.49 s | 1.10 s | 1.37× |
| 457k unknowns, LDLᵀ | 271 ms | 217 ms | 1.25× |
| 109k unknowns, LDLᵀ | 94 ms | 86 ms | 1.10× |
| 887k unknowns, LU | 2.07 s | 1.94 s | 1.07× |
| 86k unknowns, LU | 329 ms | 309 ms | 1.06× |

It helps most on large symmetric systems. For LU, the CPU is slower relative
to the GPU, so LU gains less.

### Checking a solve

Check the result before trusting it, especially on a new kind of matrix.

**Rust.**
- `solver.perturbed_pivots()` counts pivots SPARTA had to nudge to avoid a
  breakdown: exact zeros, and pivots below `static_pivot`. Nonzero means the
  factors are of a slightly different matrix.
- `solver.pivot_range()` gives the smallest and largest pivot. A ratio near
  single-precision resolution (about 1e-7) or below is a warning sign.
- Compute `‖b − Ax‖ / ‖b‖` yourself, in double precision.

**Python.** Right after `factorize()`, before any solve, `F.quality` says how
close the factorization came to breaking down. Reading it costs no extra GPU
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
`estimated_relative_error` estimates the error in x itself. That is usually
the number you care about.

**Zero diagonal entries.** SPARTA is built for FEM systems such as curl-curl
and Helmholtz problems, whose diagonal is nonzero. Matrices with zero diagonal
entries (saddle-point, Lagrange-multiplier or mixed formulations) are not
supported yet. Such a matrix can come out inaccurate or fail, even when it is
well conditioned. For such a matrix, `factorize()` issues a
`sparta.UnsupportedMatrixWarning`.

**Refinement falls short.** When refinement does not reach `refinement_rtol`,
the solve issues a `sparta.AccuracyWarning`. The warning says how far off the
answer is and the most likely cause:

- a singular or ill-conditioned matrix, or
- zero diagonal entries whose pivot partner sits in another block, which
  SPARTA cannot pivot across.

To turn the warning into an error, use
`warnings.simplefilter('error', sparta.AccuracyWarning)`.

**Singular matrices.** For a singular matrix, `factorize()` fails with an
error that names the unknown whose pivot broke down, in your numbering. For
example: `factorization failed at unknown 40 ...: zero column (rank
deficient). The matrix is singular...`. Rust (`Error::Pivot`'s `unknown`), C
and Julia get the same text.

To check an answer yourself, compute the residual in double precision:

```python
residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
```

**C.** `sparta_perturbed_pivots` reports the same count as Rust. Compute the
residual yourself.

For a very sensitive matrix, a small residual does not guarantee a small
error in x. Zero perturbed pivots does not guarantee accuracy either.

## Memory and diagnostics

### Freeing a solver

A factorization can hold a lot of GPU memory: 1.4 GiB on our larger test
case. SPARTA releases it when the solver is released. Free it explicitly when
you are done:

- **Rust:** `solver.destroy()`, or let the solver go out of scope.
- **Python:** `F.destroy()`. Without it, the memory stays claimed as long as
  anything refers to the solver, including a list, a traceback or a
  notebook's output history. `destroy()` is safe to call twice. A solver is
  also a context manager, which frees it even if the block raises an error:

  ```python
  with sparta.analyze(A) as F:
      F.factorize()
      x = F.solve(b)
  ```

- **C:** `sparta_solver_free(solver)`, and `sparta_analysis_free` for an
  analysis that never became a solver.

After `destroy()`, a Python solver still answers `F.n` and `F.path`, but
`factorize`, `solve` and `info` raise `ValueError`. Use a solver only on the
thread that created it.

**When things go wrong.** A bad matrix or a bad option leaves an existing
factorization intact. A failure inside the factorization invalidates it. Until
a later factorization succeeds, solving raises an error.

### Looking inside

**Rust.** `Analysis` reports `fill_ratio()`, `factor_nonzeros()`,
`supernodes()`, `max_front_size()` and `device_bytes()`. `Solver` reports
`factor_seconds()`, `factor_device_seconds()`, `perturbed_pivots()`,
`pivot_range()` and `is_cpu()`.

**Python.** `verbose=True` prints a short summary per solve. These properties
report the same things in code:

| Property | Tells you |
|---|---|
| `sparta.gpu_available()` | Whether this build's GPU backend (Metal or Vulkan) can run in this process |
| `sparta.gpu_status()` | The GPU in use, for example `"Vulkan on NVIDIA RTX A5000"`, or `"unavailable: "` and the reason, such as a missing Vulkan loader or driver with the package to install |
| `sparta.blas_library()` | The CPU dense kernels in use, for example `"SpartaBLAS (sme512)"` |
| `F.n`, `F.path` | Matrix size and the chosen method |
| `F.info` | Pattern statistics and estimated memory |
| `F.quality` | How trustworthy the factors are (see [Checking a solve](#checking-a-solve)) |
| `F.factor_device_seconds`, `F.factor_seconds` | GPU time, and wall time, of the last factorization |
| `F.perturbed_pivots` | Pivots that had to be nudged. Zero is not a guarantee of accuracy |
| `F.reanalyses` | How often `factorize` met a new pattern and analyzed it again |
| `F.diagnostics()` | All of the above, plus the current options, the backend and the refinement reports |

The full argument-by-argument reference, including every error, is in
[python/API.md](https://github.com/FennisRobert/SPARTA/blob/main/python/API.md).

**C.** `sparta_analysis_fill_ratio`, `sparta_analysis_factor_bytes`,
`sparta_factor_seconds` and `sparta_perturbed_pivots`.
`sparta_device()` names the GPU backend a library was built for.

`SPARTA_PROFILE=1` in the environment prints per-stage timings of the analysis
and factorization.

# Building and development

## Building from source

Read this when you build on something other than an Apple Silicon Mac.

### GPU backends

One build carries one GPU backend. With no feature named, the target machine
decides:

| Cargo feature | Backend | State |
|---|---|---|
| `metal` | Apple Metal, Apple Silicon | Implemented. The default on Apple Silicon |
| `vulkan` | Vulkan compute, AMD and NVIDIA | Implemented. The default on Windows and Linux. See [Vulkan](#vulkan) below |
| neither, on any other target (Intel Mac) | none | Analysis and the CPU backend only |
| `--no-default-features`, neither named | none | Analysis and the CPU backend only, on any target |

```sh
cargo build --release                         # the target decides
cargo build --release --features vulkan       # name one explicitly
cargo build --release --no-default-features   # CPU backend only
```

The default `gpu` feature lets the target choose. If you turn default
features off without naming a backend, the build leaves out the GPU code, its
kernels and the Metal framework links.

- **Rust:** A crate that only uses the CPU path depends on `sparta-solver`
  with `default-features = false`.
- **Python and C:** The same switch works. For example,
  `maturin build --release --manifest-path python/Cargo.toml --no-default-features`
  builds a CPU-only wheel.

A CPU-only build has no device, so every solver needs `force_cpu`
(`Config::force_cpu` in Rust, `force_cpu=True` in Python, `config.force_cpu`
in C).

`F.diagnostics()["backend"]` in Python and `sparta_device()` in C report which
backend a binary carries.

### Vulkan

The Vulkan backend runs the same factorizations and solves as Metal, in
float32 for real matrices and complex64 for complex ones, on AMD and NVIDIA
GPUs under Windows and Linux.

- **Building** needs nothing extra. On Windows or Linux, a plain
  `pip install` or `cargo build` includes Vulkan. The kernels ship compiled to
  SPIR-V in `src/kernels/vulkan/spirv/`.
- **Editing kernels.** After you edit a `.comp` file, you need `glslc` from the
  [Vulkan SDK](https://vulkan.lunarg.com/). `build.rs` looks for it in
  `GLSLC`, then `VULKAN_SDK`, then the path. Run
  `SPARTA_WRITE_SPIRV=1 cargo build --features vulkan` to refresh the shipped
  copies. If `glslc` is missing and the kernels no longer match the shipped
  SPIR-V, the build stops with an error instead of using stale kernels.
- **Running** needs a driver with Vulkan 1.3. Nothing is linked at build time.
  SPARTA opens the Vulkan loader when the first solver is created. Without a
  loader, `gpu_available()` is false, and `gpu_status()` says why. On Linux,
  the loader and the NVIDIA Vulkan driver are separate packages; see
  [Per operating system](#per-operating-system).
- **Memory.** All buffers together must fit in the GPU's memory. The four
  that grow with the problem (the front workspace, the factors, the panel
  workspace and LU's `U`) are reached by device address, so each can be up to
  16 GiB, or the device's largest buffer if that is smaller. The others
  are descriptor bindings, which NVIDIA limits to 4 GiB each, but they hold
  only the matrix and the analysis tables.
- **Choosing a device.** With several GPUs, SPARTA uses the first discrete
  one. To pick another, set `SPARTA_VULKAN_DEVICE` to its index or part of its
  name:

```sh
cargo build --release                     # Vulkan by default on Windows and Linux
SPARTA_VULKAN_DEVICE=amd cargo test --release --features vulkan --test shader
maturin develop --release --manifest-path python/Cargo.toml --features vulkan
```

`SPARTA_PROFILE=1` prints the device time per kernel, as on Metal, and
`SPARTA_PROFILE=2` lists every dispatch. The Khronos validation layer
(`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`) runs without messages.
Tested on an NVIDIA RTX 3080 and an AMD Radeon integrated GPU on Windows 11, and two NVIDIA RTX A5000s on Linux.

**Testing the Vulkan path on a Mac.** MoltenVK translates Vulkan to Metal,
so you can check the Vulkan kernels on Apple Silicon. Use it for testing only:
on an M4 Pro, it is 1.0 to 1.7 times slower than the Metal backend, with the
same residuals.

```sh
brew install molten-vk vulkan-loader
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib   # where the loader lives
cargo test --release --no-default-features --features vulkan
maturin develop --release --manifest-path python/Cargo.toml --features vulkan
```

You need the variable because macOS does not search `/opt/homebrew/lib` for
libraries by default. The Metal-only hybrid tests skip themselves.

### CPU dense kernels (SpartaBLAS)

In CPU mode (`force_cpu`) and in the hybrid's CPU share, SPARTA runs its
dense work on SpartaBLAS, its own kernels in `crates/sparta-blas`. The dense
work is the matrix products, the triangular solves, and the LU and LDLᵀ
factorization of each block's diagonal. You do not need to install or link
anything, on any platform. GPU mode does not use these kernels.

On Apple M1 to M3 and on Intel Macs, SPARTA uses Apple's Accelerate instead,
which comes with macOS. These Macs have a matrix unit that only Apple's
libraries can program, so Accelerate is faster there than SpartaBLAS's `neon`
set. From M4 on, the same unit is programmable through SME, and the `sme512`
set is as fast as Accelerate on our FEM test cases.

SpartaBLAS has one kernel set per processor family. On first use, it picks
the best set the processor supports:

| Kernel set | Processors | Status |
|---|---|---|
| `sme512` | Arm SME2 matrix unit at a 512-bit vector length: Apple M4 and later | Hand-written; measured |
| `neon` | Every 64-bit Arm processor | Measured on Apple M4 Pro |
| `avx512` | x86-64 with AVX-512: Intel Xeon (Skylake-SP and later), AMD Zen 4 and later | Written; not yet run on x86 hardware |
| `avx2` | x86-64 with AVX2 and FMA: Intel since 2013 (Haswell), AMD since Zen 1 | Written; not yet run on x86 hardware |
| `scalar` | Anything else | Portable fallback |

`sparta.blas_library()` (Python) and `sparta::blas_library()` (Rust) name the
set in use, for example `"SpartaBLAS (sme512)"`, or `"Accelerate"`. To force a
set for testing, set `SPARTA_KERNEL` before the first CPU factorization. On a
Mac, `SPARTA_KERNEL=accelerate` forces Accelerate:

```sh
SPARTA_KERNEL=neon python my_simulation.py     # sme512, avx512, avx2, neon, scalar or accelerate
```

If the processor cannot run the forced set, SPARTA prints a message and uses
the best available set instead.

SpartaBLAS starts no threads of its own. Large calls split across the same
Rayon pool as the rest of SPARTA (see [CPU threads](#cpu-threads)), so the
dense work never competes with the factorization for cores. Accelerate uses
its own threads, but only for a block that has the whole machine to itself.
On macOS 15 and later, setting `threads` keeps Accelerate on one thread per call.

The `sme512` kernels follow Arm's architecture, not one vendor's, but they need
a 512-bit vector length. A processor with SME at another length uses `neon`
until a kernel set for its length exists. On Linux and Windows, SPARTA does not
detect the matrix unit yet, so those systems also use `neon`.

## Development

### Build and test

```sh
cargo test                     # symbolic, analysis and allocator checks, no GPU needed
cargo build --release          # the library and the benchmark examples
sh validate.sh                 # all of the above plus the Metal numerical suite
```

`validate.sh` also builds the Python extension and runs the GPU tests against
NumPy double-precision references. On its first run, it creates a virtualenv
in the ignored `.local/` folder.

### Test matrices and benchmarks

The benchmarks run on real FEM systems in `test_matrices/`. These are too
large for git and are not part of the repository. Tests that need a matrix you
do not have are skipped. To make your own test systems, use
`tools/sparta_io.py`, which reads and writes the `.sparta` format, and
`tools/npz_to_sparta.py`, which converts SciPy `.npz` exports.

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
measurements. See [crates/scalpel/README.md](https://github.com/FennisRobert/SPARTA/blob/main/crates/scalpel/README.md) and
[documentation/SCALPEL.md](https://github.com/FennisRobert/SPARTA/blob/main/documentation/SCALPEL.md).

### Repository layout

```
src/                 the solver: analysis, GPU bridge, CPU backend
src/kernels/         Metal kernels: assembly, pivoting, updates, solves
src/symbolic/        ordering, elimination tree, column counts, supernodes
crates/scalpel/      the SCALPEL nested-dissection ordering
crates/sparta-blas/  SpartaBLAS: the CPU dense kernels, one set per processor family
crates/sparta-capi/  the C and C++ interface
python/              the sparta Python package and its Rust bindings
examples/            benchmark and analysis programs
tools/               converters, accuracy benchmarks, matrix sync
tests/               Rust tests and the Python GPU suite
```

## Contributing

This repository does not accept pull requests. It is published from a private
development repository, so changes made here cannot be merged back. To request
a feature or report a problem, open an
[issue](https://github.com/FennisRobert/SPARTA/issues) or start a
[discussion](https://github.com/FennisRobert/SPARTA/discussions).

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
