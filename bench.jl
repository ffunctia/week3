include(joinpath(@__DIR__, "chol_gpu.jl"))
using Printf

const DEFAULT_SIZES = collect(1000:500:7000)
const DEFAULT_BLOCK_SIZE = 256
const DEFAULT_REPS = 3

function parse_options(args)
    options = Dict(
        "--sizes" => join(DEFAULT_SIZES, ","),
        "--reps" => string(DEFAULT_REPS),
        "--block-size" => string(DEFAULT_BLOCK_SIZE),
        "--out" => "results/bench.csv",
    )
    i = 1
    while i <= length(args)
        key = args[i]
        haskey(options, key) || error("Unknown option: $key")
        i < length(args) || error("Missing value for $key")
        options[key] = args[i + 1]
        i += 2
    end
    sizes = parse.(Int, split(options["--sizes"], ','))
    reps = parse(Int, options["--reps"])
    block_size = parse(Int, options["--block-size"])
    reps > 0 || error("--reps must be positive")
    block_size > 0 || error("--block-size must be positive")
    return sizes, reps, block_size, options["--out"]
end

chol_flops(n) = n^3 / 3 + n^2 / 2 + n / 6

function timed_factorization(A, reps, block_size)
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
    return best
end

function check_accuracy(::Type{T}, block_size) where {T<:Union{Float32,Float64}}
    n = 203
    check_block_size = min(block_size, 64)
    A = random_spd(T, n)
    CUDA.synchronize()
    F = copy(A)
    tile_cholesky!(F; block_size = check_block_size)
    CUDA.synchronize()
    err = factorization_error(A, F)
    tolerance = 10n * eps(T)
    err <= tolerance || error("$T accuracy check failed: error=$err, tolerance=$tolerance")
    @printf("accuracy %-7s block=%d error=%.4e tolerance=%.4e\n",
            string(T), check_block_size, err, tolerance)
end

function main(args)
    CUDA.functional() || error("CUDA is not functional on this node")
    sizes, reps, block_size, outfile = parse_options(args)
    mkpath(dirname(abspath(outfile)))

    println("GPU: ", CUDA.name(CUDA.device()))
    println("Julia threads: ", Threads.nthreads())
    println("Block size: ", block_size, "; repetitions: ", reps)

    for T in (Float32, Float64)
        check_accuracy(T, block_size)
    end

    open(outfile, "w") do io
        println(io, "precision,N,block_size,time_s,gflops")
        for T in (Float32, Float64), n in sizes
            A = random_spd(T, n)
            CUDA.synchronize()

            warm_n = min(max(2 * block_size, 64), n)
            warm = random_spd(T, warm_n)
            tile_cholesky!(warm; block_size)
            CUDA.synchronize()

            elapsed = timed_factorization(A, reps, block_size)
            rate = chol_flops(n) / elapsed / 1e9
            @printf("%-7s N=%d time=%.6f s throughput=%.2f Gflop/s\n",
                    string(T), n, elapsed, rate)
            println(io, "$(T),$(n),$(block_size),$(elapsed),$(rate)")
            flush(io)
        end
    end
    println("Wrote ", abspath(outfile))
end

main(ARGS)
