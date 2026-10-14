# Solve a complex-symmetric system with SPARTA, then sweep it.
#
# The matrix is a 2D Laplacian with an imaginary shift on its diagonal, a
# stand-in for a damped wave problem. Its pattern stays fixed while the shift
# changes, so the sweep reuses the plan and the GPU memory and only factors
# again.
#
#     julia --project=bindings/julia bindings/julia/examples/sweep.jl

using SPARTA, SparseArrays, LinearAlgebra, Printf

side = 300
d = spdiagm(-1 => -ones(side - 1), 0 => 2ones(side), 1 => -ones(side - 1))
L = kron(sparse(I, side, side), d) + kron(d, sparse(I, side, side))
n = size(L, 1)
Id = sparse(I, n, n)
want = ones(n)

println("SPARTA $(SPARTA.version()) on $(SPARTA.device()), $n unknowns")
SPARTA.gpu_available() || error("no GPU that SPARTA can use is visible")

# One system: plan, factor and solve.
A = L + 0.05im * Id
b = A * want
t = @elapsed F = SPARTA.Solver(A; preset = :sweep)
@printf("plan + factor  %.3f s (GPU factor %.4f s), fill ratio %.1f, %s, %.1f MB of factors\n",
        t, F.factor_seconds, F.fill_ratio, F.kind, F.factor_bytes / 1e6)
x = F \ b
@printf("solve          relative residual %.1e, largest error %.1e\n",
        norm(b - A * x) / norm(b), maximum(abs.(x .- want)))

# A sweep: new values, same pattern, no new plan.
t = @elapsed for s in range(0.05, 1, length = 8)
    global A = L + s * im * Id
    factorize!(F, A)
    y = F \ (A * want)
    @printf("shift %.3f    factor %.4f s on the GPU, largest error %.1e\n",
            s, F.factor_seconds, maximum(abs.(y .- want)))
end
@printf("8 factor+solve pairs in %.3f s\n", t)

# Several right-hand sides at once are cheaper than one by one.
W = [want 2want 3want]
X = F \ (A * W)
@printf("3 right-hand sides, largest error %.1e\n", maximum(abs.(X .- W)))

# The single-precision answer without refinement, for comparison.
F.refinement_steps = 0
x = F \ b
@printf("unrefined      relative residual %.1e\n", norm(b - A * x) / norm(b))
close(F)
