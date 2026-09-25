#the exact, compact MPOs built by LL_wavelet_minimal

@testset "bond dimensions" begin
    @test dim(left_virtualspace(LL_D6_minimal(; cutoff=2), 1)) == 45
    @test dim(left_virtualspace(LL_D8_minimal(; cutoff=2), 1)) == 84
    #the alphabet only goes up to (a⁺)² a², so the bond dimension does not grow with the cutoff
    @test dim(left_virtualspace(LL_D6_minimal(; cutoff=3), 1)) == 45
end

@testset "exact_mpo_data residual: $name" for (name, K, Γ4) in (("D6", wMPS.K6, wMPS.Γ4_D6), ("D8", wMPS.K8, wMPS.Γ4_D8))
    _, ops, index = wMPS.boson_alphabet(Float64, 2, 2, 2)
    coeffs, R = wMPS.LL_terms(K, Γ4, index; μ=1.5, c=2.0, r=1)
    @test R == length(K) - 1
    data = @test_logs wMPS.exact_mpo_data(coeffs, length(ops), R)    #no warning about discarded directions
    @test data.residual < 1e-13
    @test maximum(data.dropped) < 1e-14
end

@testset "constructor call forms: $model" for model in (LL_D6_minimal, LL_D8_minimal)
    @test model() isa InfiniteMPOHamiltonian
    @test model(InfiniteChain(1)) isa InfiniteMPOHamiltonian
    @test model(ComplexF64, InfiniteChain(1)) isa InfiniteMPOHamiltonian
    @test model(Trivial) isa InfiniteMPOHamiltonian
    @test model(FiniteChain(10)) isa FiniteMPOHamiltonian
    @test_throws ArgumentError model(U1Irrep)
    @test_throws ArgumentError model(InfiniteChain(2))
    @test_throws ArgumentError model(FiniteChain(3))    #shorter than the interaction range
end

#energy density of the product state ⊗ₙ |φ⟩, straight from the definition of H (see LL_terms)
function product_energy(K, Γ4, m; μ, c, r)
    w = (size(Γ4, 1) - 1) ÷ 2
    e = 2^(3r) * (K[1] * m[2, 2] + 2 * sum(K[2:end]) * m[2, 1] * m[1, 2]) - 2^r * μ * m[2, 2]
    for I in CartesianIndices(Γ4)
        a, b, c′ = Tuple(I) .- (w + 1)
        e += 2^(2r) * c * Γ4[I] * product_expectation(m, (0, a), (b, c′))
    end
    return e
end

#checks the MPOs against the Hamiltonian itself, without going through LL_D6 or LL_D8
@testset "product state energies: $name" for (name, model, K, Γ4) in (("D6", LL_D6_minimal, wMPS.K6, wMPS.Γ4_D6),
                                                                      ("D8", LL_D8_minimal, wMPS.K8, wMPS.Γ4_D8))
    for (μ, c, r) in ((2.0, 1.0, 0), (1.5, 3.0, 1)), cutoff in (2, 3)
        φ = normalize(randn(ComplexF64, cutoff + 1))
        E = expectation_value(product_mps(φ), model(; μ, c, r, cutoff))
        @test E ≈ product_energy(K, Γ4, boson_moments(φ); μ, c, r) rtol = 1e-12
    end
end

#LL_D6 builds its MPO term by term with @mpoham, so this is an independent check of the construction
@testset "LL_D6_minimal agrees with LL_D6" begin
    for (μ, c, r) in ((2.0, 1.0, 0), (1.5, 3.0, 1))
        ψ = InfiniteMPS(ℂ^3, ℂ^6)
        E = expectation_value(ψ, LL_D6(; μ, c, r, cutoff=2))
        @test expectation_value(ψ, LL_D6_minimal(; μ, c, r, cutoff=2)) ≈ E rtol = 1e-12
    end

    ψ = InfiniteMPS(ℂ^4, ℂ^5)
    @test expectation_value(ψ, LL_D6_minimal(; cutoff=3)) ≈ expectation_value(ψ, LL_D6(; cutoff=3)) rtol = 1e-12

    L = 12
    ψ = FiniteMPS(L, ℂ^3, ℂ^4)
    E = expectation_value(ψ, LL_D6(FiniteChain(L); cutoff=2))
    @test expectation_value(ψ, LL_D6_minimal(FiniteChain(L); cutoff=2)) ≈ E rtol = 1e-12
end

@testset "energies are real: LL_D8_minimal" begin
    @test isreal_energy(expectation_value(InfiniteMPS(ℂ^3, ℂ^4), LL_D8_minimal(; cutoff=2)))
    L = 10
    @test isreal_energy(expectation_value(FiniteMPS(L, ℂ^3, ℂ^2), LL_D8_minimal(FiniteChain(L); cutoff=2)))
end

#LL_D8 has bond dimension 2466, which makes this very slow
if SLOW
    @testset "LL_D8_minimal agrees with LL_D8" begin
        ψ = InfiniteMPS(ℂ^3, ℂ^4)
        @test expectation_value(ψ, LL_D8_minimal(; μ=1.5, c=3.0, r=1)) ≈ expectation_value(ψ, LL_D8(; μ=1.5, c=3.0, r=1)) rtol = 1e-12
        L = 10
        ψ = FiniteMPS(L, ℂ^3, ℂ^2)
        @test expectation_value(ψ, LL_D8_minimal(FiniteChain(L))) ≈ expectation_value(ψ, LL_D8(FiniteChain(L))) rtol = 1e-12
    end
end
