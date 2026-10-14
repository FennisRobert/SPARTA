# SPARTA for MATLAB

> **Untested in MATLAB.** This binding was written without access to MATLAB.
> The C gateway (`sparta_mex.c`) has been run against the real SPARTA library
> through a stand-in for MATLAB's `mex.h`, in both complex layouts. The
> MATLAB class has not been run at all. If `test_sparta` fails for you,
> please [open an issue](https://github.com/FennisRobert/SPARTA/issues) with
> its output.

`SpartaSolver` is a handle class over the SPARTA C library, in the style of
MATLAB's `decomposition`:

```matlab
S = SpartaSolver(A);          % plan and factor on the GPU
x = S \ b;                    % solve; b may have several columns

for f = frequencies           % a sweep: same pattern, new values
    factorize(S, K - f^2 * M);
    x = S \ b;
end
```

## Requirements

- MATLAB R2018a or later, with a C compiler set up for `mex -setup C`.
  - On Apple Silicon the MATLAB must be the native build (R2023b or later).
    An Intel MATLAB running under Rosetta cannot load the library.
  - On Windows, use Visual Studio's compiler. Rust builds the library with
    MSVC, and MinGW cannot link that import library.
- A GPU SPARTA supports:
  - Apple Silicon (Metal).
  - On Windows and Linux, an AMD or NVIDIA GPU with a Vulkan 1.3 driver.
- Rust, to build the library.

## Building

From the repository root, build the shared library:

```sh
cargo build --release -p sparta-capi
```

On Windows and Linux this needs nothing beyond Rust. The Vulkan kernels ship
precompiled.

Then in MATLAB, from this folder:

```matlab
build_sparta_mex     % compiles sparta_mex against target/release
test_sparta          % checks that everything works
example_sweep        % a worked example
```

Add this folder to the MATLAB path to use `SpartaSolver` elsewhere. On Windows,
`build_sparta_mex` copies `sparta.dll` into this folder, because Windows finds
a DLL beside the MEX file that loads it.

## Settings

Settings go in as name-value pairs:

```matlab
S = SpartaSolver(A, 'kind', 'symmetric', 'preset', 'sweep', 'nemin', 32);
```

| Name | Default | Meaning |
|---|---|---|
| `kind` | `'auto'` | `'auto'` tests `issymmetric(A)` and picks one of the two below. `'symmetric'` factors `A = L D L.'` and reads only the lower triangle. `'general'` factors `A = L U` with pivoting. |
| `preset` | `'balanced'` | Starting point for the other settings: `'balanced'`, `'sweep'`, `'one_shot'`, `'structured_grid'` or `'careful'`. |
| `refinement_steps` | `-1` | The most double-precision corrections per solve. `-1` corrects until `refinement_rtol` is met, at most 50 times. `0` returns the single-precision answer as it is. |
| `refinement_rtol` | `1e-6` | The relative residual `norm(b - A*x)/norm(b)` at which refinement stops. |
| any `sparta_config` field | | For example `nemin`, `imbalance`, `max_front` or `wide_cycles`. See `crates/sparta-capi/include/sparta.h` and the user guide. |

`RefinementSteps` and `RefinementRtol` can also be changed on a live solver.
The read-only properties report the following:

- `Rows`: the number of unknowns.
- `Kind`: the path in use.
- `FillRatio`: the fill-in of the factors.
- `FactorBytes`: the GPU memory the factors use.
- `FactorSeconds`: the GPU time of the last factorization.
- `PerturbedPivots`: pivots the factorization had to nudge off zero. Normally 0.

## Things to know

- **Pass the whole matrix**, both triangles, even for `'symmetric'`.
  Refinement multiplies by it.
- **Precision.**
  - The GPU computes in single precision (complex64).
  - The answer is refined in double precision. On a well-conditioned system
    this reaches the default `1e-6` after about one extra solve.
  - On an ill-conditioned system a small residual does not mean a small
    error. Lower `refinement_rtol` when the error matters.
- **Changing pattern.**
  - MATLAB drops entries that compute to exactly zero, so a later matrix in a
    sweep can hold fewer entries than the first. `factorize` fills the
    missing ones with zeros and keeps the plan.
  - A matrix with an entry outside the plan is planned again, which costs one
    analysis.
- **Errors.** Every error has an identifier:
  - `sparta:pivot`: the matrix is singular or too badly scaled.
  - `sparta:unsupported`: no GPU is available.
  - `sparta:device`: the GPU failed.
  - `sparta:invalid`: an input or setting was rejected.
- **Threads.** A solver may be used only from the MATLAB session that created
  it. `parfor` workers are separate processes, so each must make its own.
- **Memory.** The GPU memory is released when the last reference to a solver
  goes (`clear S`). `clear mex` does not unload the gateway while solvers are
  alive, so it cannot strand their memory.

## Files

| File | What it is |
|---|---|
| `SpartaSolver.m` | The class. `help SpartaSolver` explains it. |
| `sparta_mex.c` | The C gateway between MATLAB and `sparta.h`. Not called directly. |
| `build_sparta_mex.m` | Compiles the gateway. |
| `test_sparta.m` | Checks the binding end to end. |
| `example_sweep.m` | A solve, a frequency sweep, and several right-hand sides. |
