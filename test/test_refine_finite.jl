#refinement of a finite MPS, MPSrefine_finite

#dense state vector of a finite MPS, with site 1 as the fastest index; the last virtual space can carry a
#charge (one-dimensional), so the MPS is contracted by hand rather than with convert(TensorMap, ψ)
function dense_state(ψ)
    v = ones(ComplexF64, 1, 1)
    for A in [ψ.AL[1:(end - 1)]..., ψ.AC[end]]
        a = convert(Array, A)
        Dl, d, Dr = size(a)
        v = reshape(v * reshape(a, Dl, d * Dr), :, Dr)
    end
    return vec(v)
end

#the occupations of the modes 0, …, L-1 in the basis state k (1-based) of the dense vector
occupations(k, d, L) = digits(k - 1; base=d, pad=L)

#one-body density matrix ρ[m, m′] = ⟨a⁺ₘ aₘ′⟩ of a dense state on L modes with at most d - 1 bosons each
function onebody_density(v, d, L)
    ρ = zeros(ComplexF64, L, L)
    for k in eachindex(v), m′ in 1:L, m in 1:L
        iszero(v[k]) && continue
        occ = occupations(k, d, L)
        occ[m′] == 0 && continue
        amp = sqrt(occ[m′])
        occ[m′] -= 1
        occ[m] == d - 1 && continue
        amp *= sqrt(occ[m] + 1)
        occ[m] += 1
        ρ[m, m′] += conj(v[1 + sum(occ[j] * d^(j - 1) for j in 1:L)]) * amp * v[k]
    end
    return ρ
end

#the coarse mode n is Σₖ hₖ times the fine mode 2n + k: T[m+1, n+1] = h[m - 2n + 1]
refinement_matrix(Lfine, L) = [0 <= m - 2n <= 5 ? wMPS.h6[m - 2n + 1] : 0.0 for m in 0:(Lfine - 1), n in 0:(L - 1)]

#random U(1) state with N particles on L sites, whose last `empty` sites are in the vacuum
function random_state(L, N, d; empty=0)
    P = U1Space(n => 1 for n in 0:(d - 1))
    ψ = FiniteMPS(L - empty, P, U1Space(n => 2 for n in 0:N); right=U1Space(N => 1))
    empty == 0 && return ψ
    V = right_virtualspace(ψ, L - empty)
    vac = TensorMap(reshape(Matrix{ComplexF64}(I, dim(V), dim(V)), dim(V), 1, dim(V)) .* reshape([1.0; zeros(d - 1)], 1, d, 1),
                    V ⊗ P ← V)
    return FiniteMPS([ψ.AL[1:(end - 1)]..., ψ.AC[end], fill(vac, empty)...])
end

@testset "single particle: amplitudes refined with h" begin
    L, d = 4, 3
    ψ = random_state(L, 1, d)
    vc = dense_state(ψ)
    φc = [vc[1 + d^n] for n in 0:(L - 1)]
    ψf, p = quietly(() -> MPSrefine_finite(ψ, 64))
    @test length(ψf) == 2L
    vf = dense_state(ψf)
    φf = [vf[1 + d^m] for m in 0:(2L - 1)]
    φexp = refinement_matrix(2L, L) * φc
    @test p ≈ norm(φexp)^2 / norm(φc)^2 rtol = 1e-10
    @test abs(dot(φf, φexp)) ≈ norm(φexp) rtol = 1e-10    #same up to a phase
end

@testset "Trivial and U1Irrep refine to the same state" begin
    L, d = 5, 3
    ψ = random_state(L, 2, d)
    #the same state without symmetry
    trivial(A) = TensorMap(convert(Array, A), ℂ^dim(left_virtualspace(A)) ⊗ ℂ^d ← ℂ^dim(right_virtualspace(A)))
    ψt = FiniteMPS([trivial.(ψ.AL[1:(end - 1)])..., trivial(ψ.AC[end])])
    @test abs(dot(dense_state(ψt), dense_state(ψ))) ≈ 1 rtol = 1e-10
    ψf, p = quietly(() -> MPSrefine_finite(ψ, 64))
    ψtf, pt = quietly(() -> MPSrefine_finite(ψt, 64))
    @test pt ≈ p rtol = 1e-10
    @test abs(dot(dense_state(ψtf), dense_state(ψf))) ≈ 1 rtol = 1e-10
end

@testset "two particles away from the wall: ρ ↦ T ρ Tᵀ" begin
    L, d, N = 5, 3, 2
    ψ = random_state(L, N, d; empty=2)    #modes n ≤ L-3 refine into m ≤ 2L-1
    ρc = onebody_density(dense_state(ψ), d, L)
    ψf, p = quietly(() -> MPSrefine_finite(ψ, 64))
    @test p ≈ 1 rtol = 1e-10
    ρf = onebody_density(dense_state(ψf), d, 2L)
    T = refinement_matrix(2L, L)
    @test ρf ≈ T * ρc * T' atol = 1e-10
    @test real(tr(ρf)) ≈ N rtol = 1e-10
end

@testset "refined ground state keeps its energy" begin
    #the space at resolution r is contained in that at r+1, and H at r is H at r+1 restricted to it, so the
    #continuum energy is unchanged, up to the modes projected at the right wall. The coarse modes n = L-2, L-1
    #reach past the fine chain, so the narrow trap is put close to the left wall (at x0 = 1, the modes n = 6, 7
    #are ≈ 2.5 away), where the refinement is exact; at x0 = 1.5 they hold ~1e-6 particles, which already
    #shifts the energy by 1e-4 relative, since they sit high in the trap
    #
    L, d, N, r = 8, 3, 2, 1
    kw = (; μ=0.0, c=3.0, cutoff=d - 1, ω=4.0, x0=1.0)
    P = U1Space(n => 1 for n in 0:(d - 1))
    ψ = FiniteMPS(L, P, U1Space(n => 4 for n in 0:N); right=U1Space(N => 1))
    ψ, = find_groundstate(ψ, LL_D6_trap_minimal(U1Irrep, FiniteChain(L); r, kw...), DMRG(; tol=1e-10, verbosity=0))
    E = real(expectation_value(ψ, LL_D6_trap_minimal(U1Irrep, FiniteChain(L); r, kw...))) / 2^r
    ψf, p = quietly(() -> MPSrefine_finite(ψ, 64))
    @test 1 - p < 1e-6
    Ef = real(expectation_value(ψf, LL_D6_trap_minimal(U1Irrep, FiniteChain(2L); r=r + 1, kw...))) / 2^(r + 1)
    @test Ef ≈ E rtol = 1e-5
    n = [real(expectation_value(ψf, i => a_number(U1Irrep; cutoff=d - 1))) for i in 1:(2L)]
    @test sum(n) ≈ N rtol = 1e-10
end
