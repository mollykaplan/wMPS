#reference values from ψ(y) = Σᵢ 2^(r/2) s(2ʳy - i) aᵢ in a product state, summing over every
#site where s is nonzero

#scaling function on the tabulated grid, zero outside its support [0, 5]
function s_grid(y)
    k = round(Int, y / wMPS.dx) + 1
    return 1 <= k <= length(wMPS.sxdata) ? wMPS.sxdata[k] : 0.0
end

#sites i with s(2ʳy - i) ≠ 0, and the corresponding weights
function field_sites(y, r)
    Δ = 2^r
    sites = (floor(Int, Δ * y) - 5):(ceil(Int, Δ * y) + 1)
    return [(i, sqrt(Δ) * s_grid(Δ * y - i)) for i in sites if !iszero(s_grid(Δ * y - i))]
end

#⟨ψ⁺(x) ψ(0)⟩
function product_correlator(m, x, r)
    return sum(wi * wj * product_expectation(m, (i,), (j,)) for (i, wi) in field_sites(x, r), (j, wj) in field_sites(0, r))
end

#⟨ψ⁺(x) ψ⁺(0) ψ(x) ψ(0)⟩
function product_density_correlator(m, x, r)
    fx, f0 = field_sites(x, r), field_sites(0, r)
    return sum(wi * wk * wj * wl * product_expectation(m, (i, k), (j, l))
               for (i, wi) in fx, (k, wk) in f0, (j, wj) in fx, (l, wl) in f0)
end

φ = normalize(randn(ComplexF64, 3))    #cutoff = 2
ψp = product_mps(φ)
m = boson_moments(φ)

@testset "correlator_x (r = $r, x = $x)" for r in (0, 1), x in (0.0, 1.35, 6.25)
    C = correlator_x(x, ψp, wMPS.dx, r)
    @test C ≈ real(product_correlator(m, x, r)) rtol = 1e-10
end

#the cutoff is read off the physical space of the state
@testset "correlators at cutoff = 3" begin
    φ3 = normalize(randn(ComplexF64, 4))
    ψ3 = product_mps(φ3)
    m3 = boson_moments(φ3)
    @test correlator_x(1.35, ψ3, wMPS.dx, 1) ≈ real(product_correlator(m3, 1.35, 1)) rtol = 1e-10
    @test density_correlator(0.0, ψ3, wMPS.dx, 0) ≈ real(product_density_correlator(m3, 0.0, 0)) rtol = 1e-10
end

@testset "dx must match the scaling function table" begin
    @test_throws ArgumentError correlator_x(1.0, ψp, 0.1, 0)
    @test_throws ArgumentError density_correlator(1.0, ψp, 0.1, 0)
end

@testset "correlator_x at long distance" begin
    #the field operators no longer overlap, so the correlator factorizes into |⟨ψ⟩|² = 2ʳ |⟨a⟩|²
    for r in (0, 1)
        @test correlator_x(10.0, ψp, wMPS.dx, r) ≈ 2^r * abs2(m[1, 2]) rtol = 1e-8
    end
end

#each call makes ~10³ four-point expectation values, so only a couple of points are checked
@testset "density_correlator (x = $x)" for x in (0.0, 6.25)
    G = density_correlator(x, ψp, wMPS.dx, 0)
    @test G ≈ real(product_density_correlator(m, x, 0)) rtol = 1e-10
end

@testset "correlator_x on an entangled state" begin
    ψ = first(find_groundstate(InfiniteMPS(ℂ^3, ℂ^4), LL_D6_minimal(; cutoff=2), VUMPS(; verbosity=0, maxiter=50)))
    #at x = 0 the correlator is the density ⟨ψ⁺(0)ψ(0)⟩ > 0
    @test correlator_x(0.0, ψ, wMPS.dx, 0) > 0
    @test isfinite(correlator_x(2.5, ψ, wMPS.dx, 0))
end
