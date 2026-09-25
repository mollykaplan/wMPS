#columns of a history array: time, energy, error
@testset "add_histories" begin
    h1 = [0.0 1.0 0.1; 2.0 0.5 0.01]
    h2 = [0.0 0.5 0.01; 1.0 0.4 0.001; 3.0 0.3 0.0001]
    h2_orig = copy(h2)

    h = add_histories(h1, h2)
    @test h == [0.0 1.0 0.1; 2.0 0.5 0.01; 3.0 0.4 0.001; 5.0 0.3 0.0001]
    @test h2 == h2_orig    #inputs are left untouched

    hkf = add_histories_kf(h1, h2)
    @test hkf == [0.0 1.0 0.1; 2.0 0.5 0.01; 2.0 0.5 0.01; 3.0 0.4 0.001; 5.0 0.3 0.0001]
    @test h2 == h2_orig
end

H = LL_D6_minimal(; μ=2.0, c=1.0, r=0, cutoff=2)

function check_history(hist, ψ, H)
    @test size(hist, 2) == 3
    @test issorted(hist[:, 1])
    @test hist[end, 2] ≈ real(expectation_value(ψ, H)) rtol = 1e-8
end

@testset "find_groundstate_t, VUMPS" begin
    ψ0 = InfiniteMPS(ℂ^3, ℂ^8)
    alg = VUMPS(; maxiter=100, verbosity=0)
    ψ, envs, ϵ, hist = find_groundstate_t(ψ0, H, alg)
    ψref, = find_groundstate(ψ0, H, alg)
    check_history(hist, ψ, H)
    @test real(expectation_value(ψ, H)) ≈ real(expectation_value(ψref, H)) rtol = 1e-8
end

@testset "precondition" begin
    ψ0 = InfiniteMPS(ℂ^3, ℂ^8)
    envs = environments(ψ0, H)
    _, g = wMPS.GrassmannMPS.fg(ψ0, H, envs)
    Pg = wMPS.precondition(ψ0, g; metric_regulator=1)
    Pgref = wMPS.GrassmannMPS.precondition(ψ0, g)
    @test all(Pg[i].Z ≈ Pgref[i].Z for i in eachindex(ψ0))
end

@testset "find_groundstate_t, GradientGrassmann" begin
    ψ0 = InfiniteMPS(ℂ^3, ℂ^8)
    method = wMPS.OptimKit.LBFGS(8; maxiter=50, verbosity=0)
    #metric_regulator = 1 is the MPSKit default preconditioner
    ψ, envs, hist = find_groundstate_t(ψ0, H, GradientGrassmann(; method); metric_regulator=1)
    ψref, = find_groundstate(ψ0, H, GradientGrassmann(; method))
    check_history(hist, ψ, H)
    @test real(expectation_value(ψ, H)) ≈ real(expectation_value(ψref, H)) rtol = 1e-8

    ψ, envs, hist = find_groundstate_t(ψ0, H, GradientGrassmann(; method))
    check_history(hist, ψ, H)

    #the timed optimizer is only implemented for L-BFGS
    @test_throws MethodError find_groundstate_t(ψ0, H, GradientGrassmann(; maxiter=2))
end
