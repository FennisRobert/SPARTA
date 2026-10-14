using SPARTA, SparseArrays, LinearAlgebra, Test

"""2D Laplacian on a `side` by `side` grid, 5-point stencil."""
function laplacian(side)
    d = spdiagm(-1 => -ones(side - 1), 0 => 2ones(side), 1 => -ones(side - 1))
    return kron(sparse(I, side, side), d) + kron(d, sparse(I, side, side))
end

residual(A, x, b) = norm(b - A * x) / norm(b)

@testset "SPARTA $(SPARTA.version()) on $(SPARTA.device())" begin
    @test SPARTA.gpu_available()
    L = laplacian(60)
    n = size(L, 1)
    Id = sparse(I, n, n)
    want = (1:n) ./ n .+ 0.5im

    A = L + 0.05im * Id
    F = SPARTA.Solver(A)
    @testset "symmetric" begin
        @test F.kind == :symmetric
        @test F.fill_ratio > 1 && F.factor_bytes > 0 && F.perturbed_pivots == 0
        @test residual(A, F \ (A * want), A * want) < 1e-6
        F.refinement_steps = 0
        r = residual(A, F \ (A * want), A * want)
        @test 1e-9 < r < 1e-4
        F.refinement_steps = -1
        x = similar(want, ComplexF64)
        ldiv!(x, F, A * want)
        @test residual(A, x, A * want) < 1e-6
    end

    @testset "several right-hand sides" begin
        B = [A * want real(A * want) zeros(n)]
        X = F \ B
        @test size(X) == (n, 3)
        @test residual(A, X[:, 1], B[:, 1]) < 1e-6
        @test residual(A, X[:, 2], B[:, 2]) < 1e-6
        @test all(iszero, X[:, 3])
        @test size(F \ zeros(n, 0)) == (n, 0)
    end

    @testset "sweep" begin
        for s in (0.2, 0.7)
            A = L + s * im * Id
            factorize!(F, A)
            @test residual(A, F \ (A * want), A * want) < 1e-6
        end
        fill = F.fill_ratio
        Az = dropzeros!(copy(A))
        Az[2, 1] = Az[1, 2] = 0
        dropzeros!(Az)
        factorize!(F, Az)
        @test F.fill_ratio == fill
        @test residual(Az, F \ (Az * want), Az * want) < 1e-6
        Ag = copy(A)
        Ag[n, 1] = Ag[1, n] = -0.01
        factorize!(F, Ag)
        @test residual(Ag, F \ (Ag * want), Ag * want) < 1e-6
    end

    @testset "general" begin
        U = L + 0.05im * Id + sparse(2:n, 1:n-1, fill(0.3, n - 1), n, n)
        G = SPARTA.Solver(U; preset = :careful, nemin = 24)
        @test G.kind == :general
        @test residual(U, G \ (U * want), U * want) < 1e-6
        factorize!(F, U)
        @test F.kind == :general
        @test residual(U, F \ (U * want), U * want) < 1e-6
        close(G)
    end

    @testset "real input" begin
        R = SPARTA.Solver(L; kind = :symmetric, preset = :structured_grid)
        @test residual(L, R \ ones(n), ones(n)) < 1e-6
        @test eltype(R \ ones(n)) == ComplexF64
    end

    @testset "errors" begin
        @test_throws ArgumentError SPARTA.Solver(A; nemn = 3)
        @test_throws ArgumentError SPARTA.Solver(A; preset = :fast)
        @test_throws ArgumentError SPARTA.Solver(A; kind = :hermitian)
        @test_throws ArgumentError SPARTA.Solver(spzeros(3, 4))
        @test_throws DimensionMismatch F \ ones(n + 1)
        @test_throws DimensionMismatch factorize!(F, laplacian(5))
        e = try SPARTA.Solver(A; panel = 8); nothing catch err; err end
        @test e isa SpartaError && e.kind == :invalid
        e = try SPARTA.Solver(spzeros(ComplexF64, n, n)); nothing catch err; err end
        @test e isa SpartaError && e.kind == :pivot
        @test_throws InexactError SPARTA.Solver(A; nemin = -1)
    end

    @testset "release" begin
        close(F)
        close(F)
        @test_throws SpartaError F \ ones(n)
        @test occursin("closed", sprint(show, F))
        for _ in 1:20
            SPARTA.Solver(L + 0.1im * Id)
        end
        GC.gc()
        @test true
    end
end
