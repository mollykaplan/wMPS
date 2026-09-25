#building blocks of the refinement circuit

#restriction of a (ℂ^d ⊗ ℂ^d) ← (ℂ^d ⊗ ℂ^d) tensor to input states with at most d-1 bosons in
#total, which is the subspace the two-mode rotation maps into the truncated Fock space
function number_conserving_block(T, d)
    M = reshape(convert(Array, T), d^2, d^2)
    cols = [n1 + 1 + d * n2 for n1 in 0:(d - 1) for n2 in 0:(d - 1) if n1 + n2 <= d - 1]
    return M[:, cols]
end

@testset "Afock is an isometry on the number conserving subspace (d = $d)" for d in (3, 4)
    for A in (wMPS.A1, wMPS.A2, wMPS.A3)
        B = number_conserving_block(wMPS.Afock(A, d), d)
        @test B' * B ≈ I atol = 1e-14
    end
end

@testset "Vfock" begin
    V = wMPS.Vfock(3, 2)
    @test V' * V ≈ id(ℂ^3)
    #the wavelet mode (second leg) is left in the vacuum
    @test all(iszero, convert(Array, V)[:, 2:end, :])
end

@testset "trunc_bonds" begin
    ψ = InfiniteMPS(ℂ^3, ℂ^12)
    ψt = trunc_bonds(ψ, 4)
    @test dim(left_virtualspace(ψt, 1)) == 4
    @test norm(ψt) ≈ 1
    @test abs(dot(ψt, ψ)) > 0.99
    #nothing to truncate
    for D in (12, 20)
        ψt = trunc_bonds(ψ, D)
        @test dim(left_virtualspace(ψt, 1)) == 12
        @test abs(dot(ψt, ψ)) ≈ 1
    end

    ψ2 = InfiniteMPS([ℂ^3, ℂ^3], [ℂ^10, ℂ^12])
    @test [dim(left_virtualspace(trunc_bonds(ψ2, 5), i)) for i in 1:2] == [5, 5]
end

#refining a ground state from r = 0 to r = 1
μ, c, dmax, d, D = 2.0, 1.0, 2, 3, 8
H0 = LL_D6_minimal(; μ, c, r=0, cutoff=dmax)
H1 = LL_D6_minimal(; μ, c, r=1, cutoff=dmax)
H1_2 = InfiniteMPOHamiltonian([H1[1], H1[1]])
groundstate(H) = first(find_groundstate(InfiniteMPS(ℂ^(dmax + 1), ℂ^D), H, VUMPS(; verbosity=0)))
ψ0 = groundstate(H0)
E0 = real(expectation_value(ψ0, H0))
E1 = real(expectation_value(groundstate(H1), H1))    #variational bound for every state at r = 1

ψf, Ef = quietly(() -> MPSrefine(ψ0, H1_2, dmax, d, D))
ψs, Es = quietly(() -> single_site(ψf, H1))

@testset "MPSrefine" begin
    @test length(ψf) == 2
    @test all(i -> dim(left_virtualspace(ψf, i)) <= D, 1:2)
    @test Ef ≈ real(expectation_value(ψf, H1_2)) / 2
    #the refined state describes the same continuum state, up to Fock space and bond truncation
    @test Ef ≈ E0 rtol = 1e-2
    @test Ef > E1
end

@testset "single_site" begin
    @test length(ψs) == 1
    @test dim(left_virtualspace(ψs, 1)) == D
    @test Es ≈ real(expectation_value(ψs, H1))
    @test Es > E1
    #close to the two-site state it came from
    @test abs(dot(ψf, InfiniteMPS([ψs.AL[1], ψs.AL[1]]))) > 0.99
end

@testset "refinement_alg" begin
    @test_throws ArgumentError refinement_alg(ψ0, 0, 1, μ, c, dmax, d + 1, D)
    ψr = quietly(() -> refinement_alg(ψ0, 0, 1, μ, c, dmax, d, D; model=LL_D6_minimal))
    @test length(ψr) == 1
    #one step of refinement_alg is MPSrefine followed by single_site
    @test real(expectation_value(ψr, H1)) ≈ Es rtol = 1e-8
end

@testset "refine_optim" begin
    @test_throws ArgumentError refine_optim(ψ0, 0, 1, μ, c, dmax, d + 1, D, 10)
    states, history = quietly(() -> refine_optim(ψ0, 0, 1, μ, c, dmax, d, D, 10; verbosity=0, model=LL_D6_minimal))
    @test size(states) == (1, D, dmax + 1, D)
    #columns: time, energy density, gradient norm; the time includes the refinement itself
    @test size(history, 2) == 3
    @test issorted(history[:, 1])
    @test history[1, 1] > 0
    AL = TensorMap(states[1, :, :, :], ℂ^D ⊗ ℂ^(dmax + 1) ← ℂ^D)
    @test AL' * AL ≈ id(ℂ^D)    #the stored tensor is a left isometry
    #the L-BFGS steps lower the energy of the refined single-site state
    Eopt = real(expectation_value(InfiniteMPS([AL]), H1))
    @test E1 - 1e-8 < Eopt < Es
    @test history[end, 2] ≈ Eopt rtol = 1e-8
end

@testset "verbosity = 0 prints nothing" begin
    out = mktemp() do path, io
        redirect_stdout(io) do
            MPSrefine(ψ0, H1_2, dmax, d, D; verbosity=0)
            single_site(ψf, H1; verbosity=0)
            refinement_alg(ψ0, 0, 1, μ, c, dmax, d, D; verbosity=0)
        end
        flush(io)
        read(path, String)
    end
    @test isempty(out)
end
