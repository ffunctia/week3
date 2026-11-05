include(joinpath(@__DIR__, "chol_gpu.jl"))
using Printf

function main()
    CUDA.functional() || error("CUDA is not functional on this node")

    n = parse(Int, get(ENV, "N", "0"))
    n > 0 || error("Set N to one matrix size, e.g. N=1000")
    precision = get(ENV, "PRECISION", "")
    T = precision == "Float32" ? Float32 :
        precision == "Float64" ? Float64 :
        error("Set PRECISION to Float32 or Float64")
    block_size = parse(Int, get(ENV, "BLOCK_SIZE", "256"))
    reps = parse(Int, get(ENV, "REPS", "3"))
    block_size > 0 || error("BLOCK_SIZE must be positive")
    reps > 0 || error("REPS must be positive")

    println("GPU: ", CUDA.device())
    println("Case: precision=$precision N=$n block=$block_size reps=$reps")

    check_n = 203
    check_block = min(block_size, 64)
    Acheck = random_spd(T, check_n)
    Fcheck = copy(Acheck)
    tile_cholesky!(Fcheck; block_size = check_block)
    CUDA.synchronize()
    err = factorization_error(Acheck, Fcheck)
    tolerance = 10check_n * eps(T)
    isfinite(err) && err <= tolerance ||
        error("Accuracy check failed: error=$err, tolerance=$tolerance")
    @printf("Accuracy check passed: residual=%.3e tolerance=%.3e\n", err, tolerance)

    A = random_spd(T, n)
    warm_n = min(max(2 * block_size, 64), n)
    warm = random_spd(T, warm_n)
    tile_cholesky!(warm; block_size)
    CUDA.synchronize()

    B = similar(A)
    best = Inf
    for _ in 1:reps
        copyto!(B, A)
        CUDA.synchronize()
        elapsed = @elapsed begin
            tile_cholesky!(B; block_size)
            CUDA.synchronize()
        end
        best = min(best, elapsed)
    end
    rate = (n^3 / 3 + n^2 / 2 + n / 6) / best / 1e9

    mkpath("results")
    job_id = get(ENV, "SLURM_JOB_ID", "local")
    outfile = "results/bench_$(precision)_N$(n)_$(job_id).csv"
    open(outfile, "w") do io
        println(io, "precision,N,block_size,time_s,gflops,check_residual")
        println(io, "$precision,$n,$block_size,$best,$rate,$err")
    end
    @printf("RESULT precision=%s N=%d time=%.6f s throughput=%.2f Gflop/s\n",
            precision, n, best, rate)
    println("Wrote ", abspath(outfile))
end

main()
