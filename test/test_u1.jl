#U(1) particle number symmetry: every model that conserves the particle number can be built with
#symmetry U1Irrep, and is then the same operator as with Trivial symmetry

#dense matrix of a finite chain MPO; the physical space U1Space(n => 1 for n in 0:cutoff) has the
#same basis order as the Fock basis, so dense matrices of both symmetries can be compared directly
dense_u1(H, d, L) = reshape(convert(Array, convert(TensorMap, H)), d^L, d^L)

@testset "exact_mpo_data with charges: $name" for (name, K, Γ4) in (("D6", wMPS.K6, wMPS.Γ4_D6), ("D8", wMPS.K8, wMPS.Γ4_D8))
    _, ops, index = wMPS.boson_alphabet(Float64, 2, 2, 2)
    charges = wMPS.operator_charge.(ops)
    @test charges == [p - q for (p, q) in wMPS.boson_alphabet(Float64, 2, 2, 2)[1]]
    coeffs, R = wMPS.LL_terms(K, Γ4, index; μ=0.7, c=2.3, r=1)
    data = @test_logs wMPS.exact_mpo_data(coeffs, length(ops), R; charges)
    @test data.residual < 1e-13
    @test maximum(data.dropped) < 1e-14
    #same number of states as without charges
    @test data.nstates == wMPS.exact_mpo_data(coeffs, length(ops), R).nstates
    #every nonzero coefficient changes the charge of the state by that of its operator
    q = data.statecharges
    n = length(q)
    @test all(iszero(data.A[o, α, β]) || q[β] == q[α] + charges[o] for o in eachindex(ops), α in 1:n, β in 1:n)
    @test all(iszero(data.B[o, α]) || q[α] + charges[o] == 0 for o in eachindex(ops), α in 1:n)
    @test all(iszero(data.C[o, α]) || q[α] == charges[o] for o in eachindex(ops), α in 1:n)
end

@testset "U1Irrep and Trivial give the same operator" begin
    L, d, r = 6, 3, 1
    kw = (; μ=1.0, c=3.0, r, cutoff=d - 1)
    tkw = (; kw..., ω=0.7, x0=0.8)
    for (model, k) in ((LL_D6, kw), (LL_D6_minimal, kw), (LL_D6_trap, tkw), (LL_D6_trap_minimal, tkw))
        Ht = model(FiniteChain(L); k...)
        Hu = model(U1Irrep, FiniteChain(L); k...)
        @test sectortype(physicalspace(Hu, 1)) == U1Irrep
        @test dense_u1(Hu, d, L) ≈ dense_u1(Ht, d, L) atol = 1e-10
        @test maximum(dim(left_virtualspace(Hu, i)) for i in 1:L) == maximum(dim(left_virtualspace(Ht, i)) for i in 1:L)
    end
    @test_throws ArgumentError LL_D6(SU2Irrep, FiniteChain(L); kw...)
    @test_throws ArgumentError LL_D6_trap_minimal(SU2Irrep, FiniteChain(L); tkw...)
    @test_throws ArgumentError quad_model(U1Irrep)    #a⁺a⁺ + aa does not conserve the particle number
end

@testset "DMRG in a fixed particle number sector" begin
    L, d, N = 6, 3, 2
    kw = (; μ=1.0, c=3.0, r=1, cutoff=d - 1, ω=0.7, x0=1.5)
    #exact diagonalization of the dense Trivial Hamiltonian, restricted to N particles; site i is
    #the i-th fastest index of the dense matrix
    H = dense_u1(LL_D6_trap(FiniteChain(L); kw...), d, L)
    sector = [k for k in 0:(d^L - 1) if sum(digits(k; base=d, pad=L)) == N] .+ 1
    E0 = eigmin(Hermitian(H[sector, sector]))

    P = U1Space(n => 1 for n in 0:(d - 1))
    ψ = FiniteMPS(L, P, U1Space(0 => 4, 1 => 8, 2 => 4); right=U1Space(N => 1))
    ψ, = find_groundstate(ψ, LL_D6_trap_minimal(U1Irrep, FiniteChain(L); kw...), DMRG(; tol=1e-10, verbosity=0))
    n = [real(expectation_value(ψ, i => a_number(U1Irrep; cutoff=d - 1))) for i in 1:L]
    @test sum(n) ≈ N atol = 1e-10
    @test real(expectation_value(ψ, LL_D6_trap(U1Irrep, FiniteChain(L); kw...))) ≈ E0 rtol = 1e-8
end
