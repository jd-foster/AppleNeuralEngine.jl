using AppleNeuralEngine, Reactant
using LinearAlgebra, Random, Test

relerr(a, b) = norm(Float32.(a) .- Float32.(b)) / max(norm(Float32.(b)), eps(Float32))
const ANE = get(ENV, "APPLENEURALENGINE_SKIP_ANE", "0") != "1" && functional()

@testset "Reactant front end" begin
    @test Base.get_extension(AppleNeuralEngine, :AppleNeuralEngineReactantExt) !== nothing

    Random.seed!(1)
    C = randn(Float16, 32, 16)
    # Keep values small: slices with a nonzero start on the innermost axis saturate
    # above 4094, and the constant pads behind `concatenate` are lossy below 2^-7.
    cases = [
        ("dense tanh layer", (W, x) -> tanh.(W * x .+ Float16(1)), (randn(Float16, 32, 64), randn(Float16, 64, 16)), 2e-3),
        ("adjoint matmul", (W, x) -> W' * x, (randn(Float16, 64, 32), randn(Float16, 64, 16)), 2e-3),
        ("matrix-vector", (W, v) -> W * v, (randn(Float16, 32, 64), randn(Float16, 64)), 2e-3),
        ("column softmax", x -> (e = exp.(x .- maximum(x; dims=1)); e ./ sum(e; dims=1)), (randn(Float16, 32, 16),), 3e-3),
        ("captured constant", x -> C .* x .+ C, (randn(Float16, 32, 16),), 2e-3),
        ("ifelse", x -> ifelse.(x .> 0, x, Float16(0.1) .* x), (randn(Float16, 32, 16),), 1e-3),
        ("row sums", x -> sum(x; dims=2), (randn(Float16, 32, 16),), 3e-3),
        ("Float32 in and out", x -> 2 .* x .+ 1, (randn(Float32, 16, 8),), 2e-3),
        ("rank 3", x -> x .* 2 .+ 1, (randn(Float16, 8, 4, 3),), 2e-3),
        ("unrolled circshift", (a, b) -> (for _ in 1:4; a = max.(a .* Float16(0.5), b); b = circshift(b, 1); end; a),
            (randn(Float16, 128), randn(Float16, 128)), 1e-3),
        ("3-point stencil ×16", x -> (for _ in 1:16
                x = Float16(0.5) .* x .+ Float16(0.25) .* (vcat(x[2:end], x[end:end]) .+ vcat(x[1:1], x[1:end-1]))
            end; x), (Float16.(1 .+ rand(256)),), 1e-2),     # fp16 rounding accumulates over 16 steps
    ]
    for (label, f, args, tol) in cases
        @testset "$label" begin
            g = trace(f, args...; frontend=:reactant)
            @test g isa AppleNeuralEngine.MIL.Graph
            if ANE
                p = compile(f, args...; frontend=:reactant)
                y = p(args...)
                ref = f(map(a -> Float32.(a), args)...)
                @test size(y) == size(ref)
                @test eltype(y) == eltype(first(args))
                @test relerr(y, ref) < tol
                close!(p)
            end
        end
    end

    if ANE
        @testset "shifts are exact" begin
            # Pure data movement must copy fp16 values bit for bit, tiny and huge ones included.
            x = Float16.(randn(256) .* 1e-3)
            x[1:6] .= (nextfloat(Float16(0)), -floatmin(Float16), Float16(1e-6), Float16(-3e-5), Float16(5000), Float16(-60000))
            A = Float16.(randn(32, 8) .* 1e-3); A[2, 1] = nextfloat(Float16(0)); A[3, 2] = Float16(40000)
            for (label, f, a) in (("right shift", v -> vcat(v[1:1], v[1:end-1]), x),
                                  ("left shift", v -> vcat(v[2:end], v[end:end]), x),
                                  ("offset slices", v -> vcat(v[101:end], v[15:100], v[2:14], v[1:1]), x),
                                  ("strided slice", v -> v[2:3:end], x),
                                  ("ring", v -> circshift(v, 3), x),
                                  ("zero padding", v -> vcat(zero(v[1:2]), v[1:end-2]), x),
                                  ("matrix, shift along dim 1", B -> vcat(B[2:end, :], B[end:end, :]), A),
                                  ("matrix, shift along dim 2", B -> hcat(B[:, 2:end], B[:, end:end]), A),
                                  ("matrix block", B -> B[3:17, 2:6], A))
                @testset "$label" begin
                    p = compile(f, a; frontend=:reactant)
                    @test p(a) == f(a)
                    close!(p)
                end
            end
        end
    end

    @testset "unsupported input" begin
        @test_throws ArgumentError trace(x -> x .+ 1, (2, 2, 2, 2); frontend=:reactant)
        @test_throws ArgumentError trace(x -> x .+ 1, rand(Int, 4); frontend=:reactant)
    end
end
