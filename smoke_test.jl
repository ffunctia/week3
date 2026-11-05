include(joinpath(@__DIR__, "chol_gpu.jl"))
using Printf

function test_precision(::Type{T}; n::Int = 203, block_size::Int = 64) where {T<:Union{Float32,Float64}}
    A = random_spd(T, n)
    CUDA.synchronize()

    F = copy(A)
    elapsed = @elapsed begin
        tile_cholesky!(F; block_size)
        CUDA.synchronize()
    end

    err = factorization_error(A, F)
    tolerance = 10n * eps(T)
    isfinite(err) || error("$T residual is not finite: $err")
    err <= tolerance || error("$T accuracy check failed: error=$err, tolerance=$tolerance")

    @printf("PASS precision=%s N=%d block=%d time=%.4f s residual=%.3e tolerance=%.3e\n",
            string(T), n, block_size, elapsed, err, tolerance)
end

function main()
    CUDA.functional() || error("CUDA.jl cannot access a functional CUDA device")
    println("GPU smoke test")
    println("Device: ", CUDA.device())
    println("Julia: ", VERSION)

    test_precision(Float32)
    test_precision(Float64)
    CUDA.synchronize()
    println("SMOKE TEST PASSED")
end

main()
