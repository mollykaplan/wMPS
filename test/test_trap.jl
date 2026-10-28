#the Lieb-Liniger model in a harmonic trap, LL_D6_trap and LL_D6_trap_minimal

@testset "scaling function moments" begin
    h = wMPS.h6
    M = wMPS.scaling_moments(h, 2)
    w = length(h) - 2
    M1(k) = M[2, k + w + 1]
    M2(k) = M[3, k + w + 1]
    @test M[1, :] == [k == 0 for k in -w:w]                                 #orthonormality
    @test sum(M[2, :]) ≈ sum((0:(length(h) - 1)) .* h) / sqrt(2)           #Σₖ M₁(k) = ∫ x s(x) dx
    for k in 1:w
        @test M1(-k) ≈ M1(k) atol = 1e-14
        @test M2(-k) ≈ M2(k) - 2k * M1(k) atol = 1e-14
    end
    #against the tabulated scaling function, up to the accuracy of the quadrature
    xs = (0:(length(wMPS.sxdata) - 1)) .* wMPS.dx
    shifted(k) = [0 <= x - k <= 5 ? wMPS.sxdata[round(Int, (x - k) / wMPS.dx) + 1] : 0.0 for x in xs]
    for q in 0:2, k in 0:w
        @test wMPS.dx * sum(xs .^ q .* wMPS.sxdata .* shifted(k)) ≈ M[q + 1, k + w + 1] atol = 5e-3
    end
end

@testset "trap matrix" begin
    V = wMPS.trap_matrix(30; ω=0.7, r=1, x0=6.0)
    @test V ≈ V' atol = 1e-14
    @test all(iszero(V[i, j]) for i in 1:30, j in 1:30 if abs(i - j) > 4)
    #one particle in the trap, ∫ ∂ₓψ⁺∂ₓψ + ½ω²x² ψ⁺ψ: a harmonic oscillator with frequency √2 ω
    r, ω = 3, 1.0
    L = 20 * 2^r
    K = zeros(L, L)
    for n in 1:L, s in 0:4
        n + s <= L && (K[n, n + s] = K[n + s, n] = wMPS.K6[1 + s])
    end
    E = eigvals(Symmetric(2^(2r) .* K .+ wMPS.trap_matrix(L; ω, r)))
    @test E[1:3] ≈ sqrt(2) * ω .* (0.5:1:2.5) rtol = 5e-4    #discretization error at r = 3
end

@testset "LL_D6_trap" begin
    L, d, r, ω, x0 = 6, 3, 1, 0.7, 0.8
    kw = (; μ=1.0, c=3.0, r, cutoff=d - 1, ω, x0)
    @test exact_model(LL_D6_trap_minimal) === LL_D6_trap
    @test_throws ArgumentError LL_D6_trap(InfiniteChain(1); kw...)
    @test_throws ArgumentError LL_D6_trap_minimal(InfiniteChain(1); kw...)
    @test_throws ArgumentError LL_D6_trap_minimal(FiniteChain(4); kw...)    #shorter than the range

    He = LL_D6_trap(FiniteChain(L); kw...)
    Hm = LL_D6_trap_minimal(FiniteChain(L); kw...)
    @test maximum(dim(left_virtualspace(Hm, i)) for i in 1:L) <= 45 + 8

    #the trap adds exactly Σₙₘ 2ʳ Vₙₘ a⁺ₙ aₘ, built here from dense matrices; the dense array of
    #an MPO has site 1 as its fastest index, i.e. as the last factor of kron
    dense(H) = reshape(convert(TensorMap, H).data, d^L, d^L)
    a = diagm(1 => sqrt.(1:(d - 1)))
    op(o, n) = foldl(kron, [L + 1 - j == n ? o : Matrix(1.0I, d, d) for j in 1:L])
    V = wMPS.trap_matrix(L; ω, r, x0)
    trap = sum(2^r * V[n, m] * op(a', n) * op(a, m) for n in 1:L, m in 1:L)
    H0 = dense(LL_D6(FiniteChain(L); μ=1.0, c=3.0, r, cutoff=d - 1))
    @test dense(He) - H0 ≈ trap atol = 1e-10
    @test dense(Hm) ≈ dense(He) atol = 1e-10

    ψ = FiniteMPS(rand, ComplexF64, 10, ℂ^d, ℂ^4)
    E = expectation_value(ψ, LL_D6_trap(FiniteChain(10); kw...))
    @test isreal_energy(E)
    @test expectation_value(ψ, LL_D6_trap_minimal(FiniteChain(10); kw...)) ≈ E rtol = 1e-12
end
