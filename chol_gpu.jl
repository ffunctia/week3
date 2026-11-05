using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using LinearAlgebra

"""
    tile_cholesky!(A; block_size=256)

In-place, right-looking blocked Cholesky factorization of a dense GPU matrix.
The lower triangle stores L on return; the upper triangle is unspecified.
"""
function tile_cholesky!(A::CUDA.CuMatrix{T}; block_size::Int = 256) where {T<:Union{Float32,Float64}}
    n, m = size(A)
    n == m || throw(DimensionMismatch("Cholesky requires a square matrix"))
    block_size > 0 || throw(ArgumentError("block_size must be positive"))

    for k0 in 1:block_size:n
        k1 = min(k0 + block_size - 1, n)
        Akk = view(A, k0:k1, k0:k1)

        CUSOLVER.potrf!('L', Akk)

        if k1 == n
            continue
        end

        for i0 in k1+1:block_size:n
            i1 = min(i0 + block_size - 1, n)
            Aik = view(A, i0:i1, k0:k1)
            CUBLAS.trsm!('R', 'L', 'T', 'N', one(T), Akk, Aik)
        end

        for j0 in k1+1:block_size:n
            j1 = min(j0 + block_size - 1, n)
            Ajk = view(A, j0:j1, k0:k1)
            Ajj = view(A, j0:j1, j0:j1)
            CUBLAS.syrk!('L', 'N', -one(T), Ajk, one(T), Ajj)

            for i0 in j0+block_size:block_size:n
                i1 = min(i0 + block_size - 1, n)
                Aik = view(A, i0:i1, k0:k1)
                Aij = view(A, i0:i1, j0:j1)
                CUBLAS.gemm!('N', 'T', -one(T), Aik, Ajk, one(T), Aij)
            end
        end
    end
    return A
end

function random_spd(::Type{T}, n::Int) where {T<:Union{Float32,Float64}}
    G = CUDA.rand(T, n, n)
    A = G * transpose(G)
    A .+= T(n) .* CUDA.CuArray(Matrix{T}(I, n, n))
    return A
end

function factorization_error(A::CUDA.CuMatrix{T}, F::CUDA.CuMatrix{T}) where {T}
    Ahost = Float64.(Array(A))
    Lhost = Float64.(tril(Array(F)))
    return opnorm(Ahost - Lhost * transpose(Lhost), 1) / opnorm(Ahost, 1)
end
