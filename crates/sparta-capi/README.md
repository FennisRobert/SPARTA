# SPARTA for C and C++

A sparse direct solver for Apple Silicon GPUs, as a shared library with a C
header. `libsparta.dylib` is self-contained: linking it needs no Rust
toolchain, no Python, and no other library.

```
include/sparta.h      the C interface, and the reference for what it all means
include/sparta.hpp    a header-only C++ layer over it
lib/libsparta.dylib   the library
examples/             a worked example in each language
```

## Requirements

Apple Silicon running macOS 13 or later. Arithmetic is single-precision complex
(complex64). There is no CPU fallback: without a Metal GPU the calls fail
rather than running slowly.

## Using it

```c
#include "sparta.h"

sparta_analysis *plan = NULL;
sparta_solver *solver = NULL;

sparta_analyze(n, col_ptr, row_index, SPARTA_SYMMETRIC, NULL, &plan);
sparta_solver_create(&plan, &solver);
sparta_factor(solver, values);
sparta_solve(solver, b, x);
sparta_solver_free(solver);
```

Each step is separate because each is reusable. The analysis depends only on
where the nonzeros are, so a frequency sweep pays for it once and then calls
`sparta_factor` and `sparta_solve` per frequency. Every call returns a
`sparta_status`, and `sparta_last_error()` explains a failure in a sentence.

```cpp
#include "sparta.hpp"

sparta::Solver solver(n, col_ptr, row_index, sparta::Matrix::Symmetric);
solver.factor(values);
std::vector<sparta::complex> x = solver.solve(b);
```

## Building against it

```sh
cc  -std=c11   -Iinclude app.c   -Llib -lsparta -Wl,-rpath,@loader_path/../lib
c++ -std=c++17 -Iinclude app.cpp -Llib -lsparta -Wl,-rpath,@loader_path/../lib
```

The library's install name is `@rpath/libsparta.dylib`, so point an rpath at
wherever the application keeps it.

## Working on this crate

```sh
make test       # the API checks, 53 of them
make run-c      # the C example
make run-cpp    # the C++ example
make package    # assemble dist/sparta-<version>-macos-arm64
```

The shared library exports only the documented `sparta_*` functions and
nothing else. The static library (`libsparta.a`, built alongside) exposes every
internal symbol instead, so ship the shared one unless a customer insists.

## Accuracy

The GPU computes in single precision, so a solve lands near a relative residual
of 1e-6 on a well-conditioned system. The Python package adds double-precision
iterative refinement on top of that and reaches 1e-14; **that layer is not yet
in this interface**. A C caller wanting more accuracy than single precision
gives must refine the answer itself, or wait for the refinement to move down
into the Rust core.

## Licence

The library is licensed under the GNU AGPL, version 3 or later; see `LICENSE`
at the repository root. Closed-source use needs a commercial licence; see
"Credits and licence" in the main README.
`THIRD-PARTY-NOTICES` covers the third-party code compiled into the library and
must be redistributed with it.
