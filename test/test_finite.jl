#finite_groundstates and the Tonks-Girardeau limit

@testset "Tonks-Girardeau densities hold N particles" begin
    N, l, ω = 4, 6.0, 1.5
    xs = range(-10, 20, length=30001)
    @test sum(x -> 0 <= x <= l ? tg_box_density(x, N, l) : 0.0, xs) * step(xs) ≈ N rtol = 1e-6
    @test sum(x -> tg_trap_density(x, N, ω, l / 2), xs) * step(xs) ≈ N rtol = 1e-6
    @test box_levels(N, l) ≈ [(j * π / l)^2 for j in 1:N]
    @test sum(trap_levels(N, ω)) ≈ sqrt(2) * ω * N^2 / 2
end

@testset "finite_groundstates" begin
    #a small trap, refined from r = 1 to r = 2
    l, N = 4, 2
    kw = (; μ=0.0, c=3.0, ω=2.0, x0=1.5)
    saved = []
    results = quietly(() -> finite_groundstates(LL_D6_trap_minimal, l, N, 1:2; D=8, cutoff=2, tol=1e-8,
                                                callback=res -> push!(saved, res.r), kw...))
    @test [res.r for res in results] == [1, 2] == saved
    @test [res.L for res in results] == [8, 16]
    for res in results
        @test sum(res.n) ≈ N rtol = 1e-10
        #E is the energy of the continuum Hamiltonian, 2^-r times that of the lattice one
        H = LL_D6_trap(U1Irrep, FiniteChain(res.L); r=res.r, cutoff=2, kw...)
        @test res.E ≈ real(expectation_value(res.ψ, H)) / 2^res.r rtol = 1e-10
    end
    #the space at r = 2 contains that at r = 1, so the energy can only go down
    @test results[2].E < results[1].E
end
