# SPARTA for Julia

`SPARTA.jl` calls the SPARTA C library directly through `ccall`, so no
compiler is involved:

```julia
using SPARTA, SparseArrays

F = SPARTA.Solver(A)          # plan and factor on the GPU
x = F \ b                     # solve; b may have several columns

for f in frequencies          # a sweep: same pattern, new values
    factorize!(F, K - f^2 * M)
    x = F \ b
end
close(F)                      # release the GPU memory now
```

## Requirements

- Julia 1.9 or later.
- A GPU SPARTA supports:
  - Apple Silicon (Metal).
  - On Windows and Linux, an AMD or NVIDIA GPU with a Vulkan 1.3 driver.
- Rust, to build the library.

## Installing

From the repository root, build the shared library:

```sh
cargo build --release -p sparta-capi
```

Then add this folder to a Julia environment as a development package:

```julia
using Pkg
Pkg.develop(path = "bindings/julia")
Pkg.test("SPARTA")
```

The package loads `target/release/libsparta.dylib` (or `libsparta.so`, or
`sparta.dll`) from the repository it sits in. To use a library elsewhere, set
`SPARTA_LIB` to its full path before `using SPARTA`.

## Settings

```julia
F = SPARTA.Solver(A; kind = :symmetric, preset = :sweep, nemin = 32)
```

| Keyword | Default | Meaning |
|---|---|---|
| `kind` | `:auto` | `:auto` tests `issymmetric(A)` and picks one of the two below. `:symmetric` factors `A = L D Lᵀ` and reads only the lower triangle. `:general` factors `A = L U` with pivoting. |
| `preset` | `:balanced` | Starting point for the other settings: `:balanced`, `:sweep`, `:one_shot`, `:structured_grid` or `:careful`. |
| `refinement_steps` | `-1` | The most double-precision corrections per solve. `-1` corrects until `refinement_rtol` is met, at most 50 times. `0` returns the single-precision answer as it is. |
| `refinement_rtol` | `1e-6` | The relative residual `norm(b - A*x)/norm(b)` at which refinement stops. |
| any `sparta_config` field | | For example `nemin`, `imbalance`, `max_front` or `wide_cycles`. See `crates/sparta-capi/include/sparta.h` and the user guide. |

`F.refinement_steps` and `F.refinement_rtol` can also be changed on a live
solver. A solver also reports the following:

- `F.rows`: the number of unknowns.
- `F.kind`: the path in use.
- `F.fill_ratio`: the fill-in of the factors.
- `F.factor_bytes`: the GPU memory the factors use.
- `F.factor_seconds`: the GPU time of the last factorization.
- `F.perturbed_pivots`: pivots the factorization had to nudge off zero.
  Normally 0.

## Things to know

- **Pass the whole matrix**, both triangles, even for `:symmetric`.
  Refinement multiplies by it.
- **Precision.**
  - The GPU computes in single precision (`ComplexF32`).
  - Answers are refined and returned in `ComplexF64`.
  - On an ill-conditioned system a small residual does not mean a small
    error. Lower `refinement_rtol` when the error matters.
- **Changing pattern.**
  - A matrix with entries missing from the planned pattern reuses the plan,
    with zeros in their place.
  - A matrix with an entry outside the pattern is planned again.
  - Explicitly stored zeros count as entries.
- **Errors.**
  - A failed library call throws `SpartaError`. Its `kind` is one of:
    - `:pivot`: the matrix is singular or too badly scaled.
    - `:unsupported`: no GPU is available.
    - `:device`: the GPU failed.
    - `:invalid`: an input or setting was rejected.
  - Bad keyword arguments throw `ArgumentError`.
- **Tasks and threads.** A solver serializes the calls made on it, so tasks
  may share one. Different solvers run independently on different threads.
- **Memory.** `close(F)` releases the GPU memory at once. Otherwise the
  garbage collector does it when `F` is no longer reachable.

## Files

| File | What it is |
|---|---|
| `src/SPARTA.jl` | The module. |
| `src/library.jl` | Loading the library, the settings struct, errors. |
| `src/solver.jl` | `Solver`, `factorize!`, solving and refinement. |
| `test/runtests.jl` | The checks `Pkg.test` runs. |
| `examples/sweep.jl` | A solve, a frequency sweep, and several right-hand sides. |
