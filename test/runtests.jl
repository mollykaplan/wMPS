using wMPS
using Test, Random, LinearAlgebra
using MPSKit, MPSKitModels, TensorKit

#every test that builds LL_D8 takes a long time (its MPO has bond dimension 2466), so those only
#run with `WMPS_SLOW_TESTS=true julia --project -e 'using Pkg; Pkg.test()'`
const SLOW = get(ENV, "WMPS_SLOW_TESTS", "false") == "true"

#MPSrefine, single_site and refinement_alg print progress; keep the test output readable
quietly(f) = redirect_stdout(f, devnull)

isreal_energy(E) = abs(imag(E)) < 1e-10 * max(1, abs(real(E)))

#translation invariant product state ⊗ₙ |φ⟩, with the single-site state φ in the Fock basis
product_mps(φ) = InfiniteMPS([TensorMap(reshape(φ, 1, length(φ), 1), ℂ^1 ⊗ ℂ^length(φ) ← ℂ^1)])

#m[p+1, q+1] = ⟨φ|(a⁺)ᵖ aᑫ|φ⟩, built independently of MPSKitModels and wMPS
function boson_moments(φ; maxpow=4)
    a = diagm(1 => sqrt.(1:(length(φ) - 1)))
    return [dot(φ, a'^p * a^q * φ) for p in 0:maxpow, q in 0:maxpow]
end

#⟨ψ⁺(c₁)⋯ψ⁺(cₙ) ψ(a₁)⋯ψ(aₘ)⟩ in a product state: every site holds a normal ordered (a⁺)ᵖ aᑫ
function product_expectation(m, creators, annihilators)
    sites = union(creators, annihilators)
    return prod(m[count(==(s), creators) + 1, count(==(s), annihilators) + 1] for s in sites)
end

Random.seed!(1234)

@testset "wMPS" begin
    @testset "data" begin include("test_data.jl") end
    @testset "models" begin include("test_models.jl") end
    @testset "minimal MPOs" begin include("test_minimal.jl") end
    @testset "refinement" begin include("test_refinement.jl") end
    @testset "correlators" begin include("test_correlators.jl") end
    @testset "timing" begin include("test_timing.jl") end
    @testset "Aqua" begin include("test_aqua.jl") end
end
