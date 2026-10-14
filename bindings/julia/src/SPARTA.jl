"""
    SPARTA

Sparse direct solves on the GPU, through the SPARTA C library.

```julia
using SPARTA, SparseArrays
F = SPARTA.Solver(A)        # plan and factor on the GPU
x = F \\ b                   # solve; b may have several columns
factorize!(F, A2)           # new values, same pattern: reuses the plan
close(F)                    # release the GPU memory now rather than at GC
```

The library is found through the `SPARTA_LIB` environment variable, a path to
`libsparta.dylib`, `libsparta.so` or `sparta.dll`, or else in `target/release`
of the repository this package sits in.
"""
module SPARTA

using Libdl
using LinearAlgebra
using SparseArrays

export factorize!, SpartaError

include("library.jl")
include("solver.jl")

end
