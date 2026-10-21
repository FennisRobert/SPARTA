# The C library: where it is, its settings struct, and its errors.

const HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

"""
    RawConfig

Mirror of `sparta_config` in sparta.h, field for field. Julia lays out an
immutable struct of plain fields as C does, so a `Ref` of it passes as a
pointer. Use the keyword arguments of [`Solver`](@ref) rather than this.
"""
struct RawConfig
    struct_size::Csize_t
    ordering::Cint
    nemin::Csize_t
    max_front::Csize_t
    imbalance::Cdouble
    wide_cycles::UInt32
    fm_passes::UInt32
    separator_trials::UInt32
    lu_diag_threshold::Cdouble
    small_front_rows::UInt32
    schur_batch::UInt32
    panel_threads::UInt32
    solve_threads::UInt32
    panel::UInt32
    schur_block::UInt32
    strip_block::UInt32
    narrow_cols::UInt32
    pivot_threads::UInt32
    schur_chunk::UInt32
    hybrid_cpu::Cdouble
end

"""
    library_path() -> String

The shared library to load: `ENV["SPARTA_LIB"]` when set, else the release
build in `target/release` of the repository holding this package.
"""
function library_path()
    haskey(ENV, "SPARTA_LIB") && return ENV["SPARTA_LIB"]
    name = Sys.isapple() ? "libsparta.dylib" : Sys.iswindows() ? "sparta.dll" : "libsparta.so"
    return normpath(joinpath(@__DIR__, "..", "..", "..", "target", "release", name))
end

function __init__()
    path = library_path()
    isfile(path) || error("SPARTA library not found at $path. Build it with " *
                          "`cargo build --release -p sparta-capi`, or point SPARTA_LIB at it.")
    HANDLE[] = Libdl.dlopen(path)
    default = Ref{RawConfig}()
    ccall(sym(:sparta_config_default), Cvoid, (Ptr{RawConfig},), default)
    # A mismatch means this file and sparta.h disagree on the struct.
    default[].struct_size == sizeof(RawConfig) ||
        error("sparta_config is $(default[].struct_size) bytes in the library and " *
              "$(sizeof(RawConfig)) here; this package does not match the library")
end

"""
    sym(name) -> Ptr{Cvoid}

The address of a function in the loaded library.
"""
sym(name::Symbol) = Libdl.dlsym(HANDLE[], name)

const PRESETS = (:balanced, :sweep, :one_shot, :structured_grid, :careful)

"""
    config(preset, settings) -> RawConfig

The settings a preset stands for, with each of `settings` (pairs of a
`sparta_config` field name and a value) placed on top.

- `preset`: one of `:balanced`, `:sweep`, `:one_shot`, `:structured_grid`, `:careful`
- `settings`: for example `(nemin = 32, imbalance = 0.7)`
"""
function config(preset::Symbol, settings)
    index = findfirst(==(preset), PRESETS)
    index === nothing && throw(ArgumentError("unknown preset :$preset; use one of $(PRESETS)"))
    base = Ref{RawConfig}()
    ccall(sym(:sparta_config_default), Cvoid, (Ptr{RawConfig},), base)
    check("preset", ccall(sym(:sparta_config_preset), Cint, (Ptr{RawConfig}, Cint), base, index - 1))
    values = Dict{Symbol,Any}(n => getfield(base[], n) for n in fieldnames(RawConfig))
    for (name, value) in pairs(settings)
        (name in (:struct_size, :ordering) || !haskey(values, name)) &&
            throw(ArgumentError("unknown setting $name"))
        values[name] = convert(fieldtype(RawConfig, name), value)
    end
    return RawConfig((values[n] for n in fieldnames(RawConfig))...)
end

"""
    SpartaError(kind, message)

A failed SPARTA call. `kind` is one of `:invalid`, `:unsupported`, `:device`,
`:pivot` (singular or badly scaled matrix) or `:internal`.
"""
struct SpartaError <: Exception
    kind::Symbol
    message::String
end

Base.showerror(io::IO, e::SpartaError) = print(io, "SpartaError(:", e.kind, "): ", e.message)

const STATUS = (:invalid, :unsupported, :device, :pivot, :internal)

"""
    check(step, status)

Throws a [`SpartaError`](@ref) carrying the library's explanation unless
`status` is `SPARTA_OK`.
"""
function check(step::AbstractString, status::Cint)
    status == 0 && return nothing
    reason = unsafe_string(ccall(sym(:sparta_last_error), Cstring, ()))
    kind = 1 <= status <= length(STATUS) ? STATUS[status] : :internal
    throw(SpartaError(kind, "$step failed: $reason"))
end

"""
    version() -> String

The SPARTA library version, such as `"0.2.0"`.
"""
version() = unsafe_string(ccall(sym(:sparta_version), Cstring, ()))

"""
    device() -> String

The GPU backend the library was built for: `"metal"` or `"vulkan"`.
"""
device() = unsafe_string(ccall(sym(:sparta_device), Cstring, ()))

"""
    gpu_available() -> Bool

Whether this machine has a GPU the library can use.
"""
gpu_available() = ccall(sym(:sparta_gpu_available), Cint, ()) != 0
