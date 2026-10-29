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

#continuum_density of an InfiniteMPS returns the whole curve on the grid x = t / 2^(r+J); the reference uses the tabulated s,
#so it is compared at points where 2^r x is a multiple of 1/4, where the table is exact
g_at(x, r; ψ=ψp, J=4) = ((xs, g) = continuum_density(ψ, r, x; J); g[end])
nn_at(x, r; ψ=ψp, J=4) = ((xs, g) = nn_correlator(ψ, r, x; J); g[end])

@testset "continuum_density, InfiniteMPS (r = $r, x = $x)" for r in (0, 1), x in (0.0, 1.25, 6.25)
    @test g_at(x, r) ≈ real(product_correlator(m, x, r)) rtol = 1e-9
end

@testset "continuum_density, InfiniteMPS: the grid" begin
    xs, g = continuum_density(ψp, 1, 2.0; J=3)
    @test xs == (0:32) ./ 16
    @test length(g) == length(xs)
    @test g[[1, 21]] ≈ [g_at(0.0, 1; J=3), g_at(1.25, 1; J=3)]
end

#the cutoff is read off the physical space of the state
@testset "correlators at cutoff = 3" begin
    φ3 = normalize(randn(ComplexF64, 4))
    ψ3 = product_mps(φ3)
    m3 = boson_moments(φ3)
    @test g_at(1.25, 1; ψ=ψ3) ≈ real(product_correlator(m3, 1.25, 1)) rtol = 1e-9
    @test nn_at(1.25, 1; ψ=ψ3) ≈ real(product_density_correlator(m3, 1.25, 1)) rtol = 1e-9
end

@testset "continuum_density, InfiniteMPS: long distance" begin
    #the field operators no longer overlap, so the correlator factorizes into |⟨ψ⟩|² = 2ʳ |⟨a⟩|²
    for r in (0, 1)
        @test g_at(10.0, r) ≈ 2^r * abs2(m[1, 2]) rtol = 1e-8
    end
end

@testset "nn_correlator (r = $r, x = $x)" for r in (0, 1), x in (0.0, 1.25, 6.25)
    @test nn_at(x, r) ≈ real(product_density_correlator(m, x, r)) rtol = 1e-9
end

@testset "nn_correlator: the grid, and Trivial symmetry only" begin
    xs, g = nn_correlator(ψp, 1, 2.0; J=3)
    @test xs == (0:32) ./ 16
    @test g[[1, 21]] ≈ [nn_at(0.0, 1; J=3), nn_at(1.25, 1; J=3)]
    P = U1Space(n => 1 for n in 0:2)
    vacuum = InfiniteMPS([TensorMap(reshape(ComplexF64[1, 0, 0], 1, 3, 1), U1Space(0 => 1) ⊗ P ← U1Space(0 => 1))])
    @test_throws ArgumentError nn_correlator(vacuum, 0, 1.0)
end

@testset "continuum_density, InfiniteMPS: entangled state" begin
    ψ = first(find_groundstate(InfiniteMPS(ℂ^3, ℂ^4), LL_D6_minimal(; cutoff=2), VUMPS(; verbosity=0, maxiter=50)))
    xs, g = continuum_density(ψ, 0, 2.5)
    #at x = 0 the correlator is the density ⟨ψ⁺(0)ψ(0)⟩ > 0
    @test g[1] > 0
    @test all(isfinite, g)
    #the same with a two-site unit cell
    xs2, g2 = continuum_density(InfiniteMPS([ψ.AL[1], ψ.AL[1]]), 0, 2.5)
    @test g2 ≈ g rtol = 1e-8
    #ψ⁺(x)ψ⁺(0)ψ(x)ψ(0) = B⁺B with B = ψ(x)ψ(0), so the density-density correlator is non-negative
    xs, nn = nn_correlator(ψ, 0, 2.5)
    @test all(>=(-1e-12), nn)
    xs2, nn2 = nn_correlator(InfiniteMPS([ψ.AL[1], ψ.AL[1]]), 0, 2.5)
    @test nn2 ≈ nn rtol = 1e-8
end

@testset "scaling_function" begin
    J = 4
    s = wMPS.scaling_function(wMPS.h6, J)
    @test length(s) == 5 * 2^J + 1
    @test s[1] == 0 && s[end] == 0
    #the table at the multiples of 1/4, where both grids meet
    @test s[1:4:end] ≈ wMPS.sxdata[1:5:end] atol = 1e-10
    #partition of unity Σₖ s(x + k) = 1
    @test all(t -> sum(s[t + k * 2^J + 1] for k in 0:4) ≈ 1, 0:(2^J - 1))
end

#a FiniteMPS product state ⊗ₙ |φ⟩ on L sites
finite_product_mps(φ, L) = FiniteMPS([TensorMap(reshape(φ, 1, length(φ), 1), ℂ^1 ⊗ ℂ^length(φ) ← ℂ^1) for _ in 1:L])

@testset "continuum_density, FiniteMPS" begin
    #in a product state ⟨aₙ⁺aₘ⟩ = |⟨a⟩|² for n ≠ m, so ρ = 2ʳ [Σₙ (⟨n⟩ - |⟨a⟩|²) s²ₙ + |⟨a⟩|² (Σₙ sₙ)²]
    L, r, J = 7, 1, 4
    x, ρ = continuum_density(finite_product_mps(φ, L), r; J)
    @test x == (0:((L + 4) * 2^J)) ./ 2^(r + J)
    s = wMPS.scaling_function(wMPS.h6, J)
    sn(n) = [0 <= t - n * 2^J <= 5 * 2^J ? s[t - n * 2^J + 1] : 0.0 for t in 0:((L + 4) * 2^J)]
    ρref = 2^r .* (real(m[2, 2] - abs2(m[1, 2])) .* sum(sn(n) .^ 2 for n in 0:(L - 1)) .+ abs2(m[1, 2]) .* sum(sn, 0:(L - 1)) .^ 2)
    @test ρ ≈ ρref rtol = 1e-10

    #with U(1) symmetry, ∫ρ dx = N (up to the Riemann sum of the overlaps sₙsₘ)
    N, d = 2, 3
    ψ = FiniteMPS(8, U1Space(n => 1 for n in 0:(d - 1)), U1Space(n => 2 for n in 0:N); right=U1Space(N => 1))
    x, ρ = continuum_density(ψ, 2; J=8)
    @test sum(ρ) * step(x) ≈ N rtol = 1e-5
end
