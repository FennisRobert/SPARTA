# Solver: one plan and its GPU memory, with the factors of the last matrix.

"CPU share of the work that `hybrid = true` starts from; see `Solver`."
const HYBRID_SHARE = 0.3

"""
    Solver(A; kind = :auto, preset = :balanced, refinement_steps = -1,
           refinement_rtol = 1e-6, settings...)

Plan and factor the square sparse matrix `A` on the GPU. Solve with `F \\ b`
or `ldiv!(x, F, b)`; factor new values of the same pattern with
[`factorize!`](@ref).

- `kind`: `:auto` tests `issymmetric(A)` and picks one of the two below.
  `:symmetric` factors `A = L D Lᵀ` and reads only the lower triangle.
  `:general` factors `A = L U` with pivoting.
- `preset`: starting point for the settings: `:balanced`, `:sweep`,
  `:one_shot`, `:structured_grid` or `:careful`.
- `refinement_steps`: the most double-precision corrections per solve. `-1`
  corrects until `refinement_rtol` is met, at most 50 times; `0` returns the
  single-precision answer as it is.
- `refinement_rtol`: the relative residual `norm(b - A*x)/norm(b)` at which
  refinement stops.
- `hybrid`: factor with the CPU and the GPU together: `true` starts the CPU
  at 30% of the work, a number from 0 to below 1 starts it there, `false`
  (the default) is the GPU alone. The share adapts to the measured speed of
  each device after every factorization. Ignored where the library has no
  hybrid path (anything but Metal).
- `settings`: any field of `sparta_config` in sparta.h, such as `nemin`,
  `imbalance` or `max_front`. The SPARTA user guide explains each.

Pass the whole matrix, both triangles, even for `:symmetric`: refinement
multiplies by it. The GPU computes in single precision: a real matrix in
`Float32` with real arithmetic, a complex one in `ComplexF32`. Answers are
refined and returned in double precision, `Float64` when the matrix and the
right-hand side are both real and `ComplexF64` otherwise.

A solver is used by one task at a time; calls on it are serialized. Release
its GPU memory with `close(F)`, or leave that to the garbage collector.
"""
mutable struct Solver
    handle::Ptr{Cvoid}
    rows::Int
    kind::Symbol
    auto_kind::Bool
    config::RawConfig
    # the analyzed pattern, zero-based, and scratch for the values placed in
    # it: Float32 for a real matrix, ComplexF32 for a complex one
    col_ptr::Vector{Int64}
    row_index::Vector{Int64}
    values::Union{Vector{Float32},Vector{ComplexF32}}
    matrix::Union{SparseMatrixCSC{Float64,Int},SparseMatrixCSC{ComplexF64,Int}}
    factored::Bool
    fill_ratio::Float64
    factor_bytes::Int
    factor_seconds::Float64
    perturbed_pivots::Int
    refinement_steps::Int
    refinement_rtol::Float64
    lock::ReentrantLock
end

function Solver(A::SparseMatrixCSC; kind::Symbol = :auto, preset::Symbol = :balanced,
                refinement_steps::Integer = -1, refinement_rtol::Real = 1e-6,
                hybrid::Union{Bool,Real} = false, settings...)
    check_matrix(A, 0)
    if hybrid !== false
        share = hybrid === true ? HYBRID_SHARE : Float64(hybrid)
        0 <= share < 1 || throw(ArgumentError("hybrid must be true, false or a share from 0 to below 1"))
        settings = (; settings..., hybrid_cpu = share)
    end
    kind in (:auto, :symmetric, :general) ||
        throw(ArgumentError("kind must be :auto, :symmetric or :general"))
    auto = kind == :auto
    F = Solver(C_NULL, size(A, 1), auto ? detect_kind(A) : kind, auto, config(preset, settings),
               Int64[], Int64[], ComplexF32[], spzeros(ComplexF64, 0, 0), false, 0.0, 0, 0.0, 0,
               refinement_steps, refinement_rtol, ReentrantLock())
    finalizer(release!, F)
    plan!(F, A)
    factorize!(F, A)
    return F
end

"""
    factorize!(F, A) -> F

Factor new values. A matrix with the analyzed pattern, or with entries
missing from it, reuses the plan and the GPU memory; missing entries count as
zero. A matrix with an entry outside the pattern is planned again. A real
matrix is factored with real arithmetic, a complex one with complex
arithmetic, whichever the previous one was.

- `A`: square sparse matrix of the size `F` was made for
"""
function factorize!(F::Solver, A::SparseMatrixCSC)
    check_matrix(A, F.rows)
    lock(F.lock) do
        if F.auto_kind && detect_kind(A) != F.kind
            F.kind = detect_kind(A)
            plan!(F, A)
        end
        stored = F.kind == :symmetric ? tril(A) : A
        real = eltype(A) <: Real
        if real != (F.values isa Vector{Float32})
            F.values = Vector{real ? Float32 : ComplexF32}(undef, length(F.row_index))
        end
        place_values!(F, stored) || (plan!(F, A); place_values!(F, stored))
        F.factored = false
        if real
            check("factor", ccall(sym(:sparta_factor_real), Cint, (Ptr{Cvoid}, Ptr{Float32}), F.handle, F.values))
        else
            check("factor", ccall(sym(:sparta_factor), Cint, (Ptr{Cvoid}, Ptr{ComplexF32}), F.handle, F.values))
        end
        F.factored = true
        F.factor_seconds = ccall(sym(:sparta_factor_seconds), Cdouble, (Ptr{Cvoid},), F.handle)
        F.perturbed_pivots = Int(ccall(sym(:sparta_perturbed_pivots), UInt64, (Ptr{Cvoid},), F.handle))
        F.matrix = SparseMatrixCSC{real ? Float64 : ComplexF64,Int}(A)
    end
    return F
end

"""
    solve(F, b) -> Vector or Matrix of Float64 or ComplexF64

Solve `A x = b` with the current factors. The same as `F \\ b`. The answer
is `Float64` when the matrix and `b` are both real, `ComplexF64` otherwise.

- `b`: vector, or matrix with one right-hand side per column
"""
function solve(F::Solver, b::AbstractVecOrMat)
    size(b, 1) == F.rows || throw(DimensionMismatch("b has $(size(b, 1)) rows; the matrix has $(F.rows)"))
    real = F.matrix isa SparseMatrixCSC{Float64} && eltype(b) <: Real
    rhs = Matrix{real ? Float64 : ComplexF64}(reshape(b, size(b, 1), :))
    x = lock(F.lock) do
        refine(F, rhs, gpu_solve(F, rhs))
    end
    return b isa AbstractVector ? vec(x) : x
end

Base.:\(F::Solver, b::AbstractVecOrMat) = solve(F, b)
LinearAlgebra.ldiv!(x::AbstractVecOrMat, F::Solver, b::AbstractVecOrMat) = copyto!(x, solve(F, b))
Base.size(F::Solver) = (F.rows, F.rows)
Base.size(F::Solver, d::Integer) = d <= 2 ? F.rows : 1

"""
    close(F)

Release the GPU memory now. `F` cannot be used afterwards.
"""
Base.close(F::Solver) = lock(() -> release!(F), F.lock)

function Base.show(io::IO, F::Solver)
    print(io, "SPARTA.Solver(", F.rows, " unknowns, ", F.kind,
          F.handle == C_NULL ? ", closed)" : ", fill ratio $(round(F.fill_ratio, digits = 2)))")
end

# ---- internals -------------------------------------------------------------

"""
    release!(F)

Free the C solver, once. Also the finalizer: the GPU objects need exclusive
ownership, not a particular thread, and nothing else holds `F` when it runs.
"""
function release!(F::Solver)
    if F.handle != C_NULL
        ccall(sym(:sparta_solver_free), Cvoid, (Ptr{Cvoid},), F.handle)
        F.handle = C_NULL
        F.factored = false
    end
    return nothing
end

"""
    plan!(F, A)

Plan `A`'s pattern for `F.kind` and reserve GPU memory, replacing any
earlier plan.
"""
function plan!(F::Solver, A::SparseMatrixCSC)
    release!(F)
    stored = F.kind == :symmetric ? tril(A) : A
    F.col_ptr = Vector{Int64}(stored.colptr) .- 1
    F.row_index = Vector{Int64}(rowvals(stored)) .- 1
    F.values = Vector{eltype(A) <: Real ? Float32 : ComplexF32}(undef, length(F.row_index))
    kind = F.kind == :symmetric ? Cint(0) : Cint(1)
    plan = Ref{Ptr{Cvoid}}(C_NULL)
    check("analyze", ccall(sym(:sparta_analyze), Cint,
                           (Int64, Ptr{Int64}, Ptr{Int64}, Cint, Ref{RawConfig}, Ptr{Ptr{Cvoid}}),
                           F.rows, F.col_ptr, F.row_index, kind, F.config, plan))
    F.fill_ratio = ccall(sym(:sparta_analysis_fill_ratio), Cdouble, (Ptr{Cvoid},), plan[])
    F.factor_bytes = Int(ccall(sym(:sparta_analysis_factor_bytes), Csize_t, (Ptr{Cvoid},), plan[]))
    solver = Ref{Ptr{Cvoid}}(C_NULL)
    check("reserve GPU memory", ccall(sym(:sparta_solver_create), Cint,
                                      (Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}), plan, solver))
    F.handle = solver[]
    return nothing
end

"""
    place_values!(F, A) -> Bool

Copy `A`'s values into `F.values` at their places in the analyzed pattern,
with zero where `A` has no entry. False when `A` has an entry outside it.
"""
function place_values!(F::Solver, A::SparseMatrixCSC)
    rows, vals = rowvals(A), nonzeros(A)
    fill!(F.values, zero(eltype(F.values)))
    @inbounds for j in 1:F.rows
        p, stop = F.col_ptr[j] + 1, F.col_ptr[j + 1]
        for q in nzrange(A, j)
            row = rows[q] - 1
            while p <= stop && F.row_index[p] < row
                p += 1
            end
            (p > stop || F.row_index[p] != row) && return false
            F.values[p] = vals[q]
            p += 1
        end
    end
    return true
end

"""
    gpu_solve(F, b) -> Matrix of b's element type

One single-precision solve per column of `b`, all in one call. A real `b`
needs real factors.
"""
function gpu_solve(F::Solver, b::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
    F.handle == C_NULL && throw(SpartaError(:invalid, "this solver has been closed"))
    F.factored || throw(SpartaError(:invalid, "the last factorization failed; factor again before solving"))
    if T == Float64
        x = Matrix{Float32}(b)
        size(x, 2) == 0 && return Matrix{T}(x)
        check("solve", ccall(sym(:sparta_solve_many_real), Cint,
                             (Ptr{Cvoid}, Ptr{Float32}, Ptr{Float32}, Int64),
                             F.handle, x, x, size(x, 2)))
    else
        x = Matrix{ComplexF32}(b)
        size(x, 2) == 0 && return Matrix{T}(x)
        check("solve", ccall(sym(:sparta_solve_many), Cint,
                             (Ptr{Cvoid}, Ptr{ComplexF32}, Ptr{ComplexF32}, Int64),
                             F.handle, x, x, size(x, 2)))
    end
    return Matrix{T}(x)
end

"""
    refine(F, b, x) -> x

Correct `x` in double precision, column by column, until each relative
residual meets `F.refinement_rtol` or stops improving.
"""
function refine(F::Solver, b::Matrix{T}, x::Matrix{T}) where {T<:Union{Float64,ComplexF64}}
    steps = F.refinement_steps < 0 ? 50 : F.refinement_steps
    steps == 0 && return x
    norms(m) = vec(sqrt.(sum(abs2, m; dims = 1)))
    scale = max.(norms(b), floatmin(Float64))
    r = b - F.matrix * x
    error_now = norms(r) ./ scale
    open = error_now .> F.refinement_rtol
    for _ in 1:steps
        any(open) || break
        cols = findall(open)
        trial = x[:, cols] + gpu_solve(F, r[:, cols])
        residual = b[:, cols] - F.matrix * trial
        error_trial = norms(residual) ./ scale[cols]
        better = error_trial .< error_now[cols]
        x[:, cols[better]] = trial[:, better]
        r[:, cols[better]] = residual[:, better]
        error_now[cols[better]] = error_trial[better]
        open[cols] = better .& (error_trial .> F.refinement_rtol)
    end
    return x
end

"""
    check_matrix(A, rows)

Throw unless `A` is square and nonempty and, when `rows > 0`, `rows` by `rows`.
"""
function check_matrix(A::SparseMatrixCSC, rows::Integer)
    size(A, 1) == size(A, 2) && size(A, 1) > 0 ||
        throw(ArgumentError("the matrix must be square and nonempty"))
    rows > 0 && size(A, 1) != rows &&
        throw(DimensionMismatch("the matrix has $(size(A, 1)) rows; this solver was made for $rows"))
    return nothing
end

detect_kind(A::SparseMatrixCSC) = issymmetric(A) ? :symmetric : :general
