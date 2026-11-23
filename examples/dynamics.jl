#Two quick examples of dynamics with wavelet MPS: the single-particle dispersion of the wavelet kinetic
#term, and the release of a few particles from a Gaussian well on a finite chain, against free fermions.
#Run from the package root with `julia --project=examples examples/dynamics.jl`.

using wMPS, MPSKit, MPSKitModels, TensorKit, LinearAlgebra
using Plots, Printf

################################################################################################
#1. Single-particle dispersion of the kinetic term at resolution r
#
#In the one-particle sector, the kinetic term restricted to the modes at resolution r is
#2^(2r) Σₙₘ Kₙ₋ₘ a⁺ₙ aₘ, with the D6 coefficients K₀, …, K₄ (wMPS.K6). Its Bloch dispersion
#
#       εᵣ(k) = 2^(2r) [K₀ + 2 Σ_{a=1}^{4} Kₐ cos(k a 2^-r)],    |k| ≤ π 2^r,
#
#is ≥ k², since the modes span a subspace of the continuum, and agrees with k² at small k. Its
#relative error fixes the momenta that resolution r describes: those of a quench must lie well inside.
################################################################################################

ε(k, r) = 2.0^(2r) * (wMPS.K6[1] + 2 * sum(wMPS.K6[a + 1] * cos(k * a / 2^r) for a in 1:4))
relerror(k, r) = ε(k, r) / k^2 - 1

println(" r | largest k with |εᵣ/k² - 1| < 1e-3 | error at k = 2^r/8")
plt = plot(xlabel="k", ylabel="εᵣ(k)/k² - 1", xscale=:log10, yscale=:log10, legend=:bottomright,
           title="single-particle dispersion at resolution r")
for r in 1:5
    ks = range(1e-3, π * 2^r, length=100_000)
    kmax = ks[findfirst(k -> abs(relerror(k, r)) >= 1e-3, ks) - 1]
    println(" $r | ", round(kmax; sigdigits=3), " | ", round(relerror(2^r / 8, r); sigdigits=3))
    #below k ≈ 2^r/100 the relative error is dominated by round-off
    k = exp10.(range(log10(2^r / 100), log10(π * 2^r), length=400))
    plot!(plt, k, relerror.(k, r), label="r = $r")
end
savefig(plt, joinpath(@__DIR__, "dispersion.png"))

################################################################################################
#2. Release from a Gaussian well on a finite chain, against free fermions
#
#N particles in a box of length l with hard walls, in the well V(x) = -V₀ exp(-(x - x₀)²/2w²), are
#released at t = 0 and evolve with the Lieb-Liniger Hamiltonian alone. At c = 200, close to the
#Tonks-Girardeau limit, the density is that of N free fermions, whose orbitals are the N lowest
#eigenfunctions of -∂ₓ² + V(x) in the box, evolved with -∂ₓ².
################################################################################################

#parameters
c = 200.0
N = 2 #number of particles
l = 6 #length of the box
r = 2 #L = l 2^r = 24 sites
dmax = 2 #maximum number of bosons per site
V0, w = 4.0, 1.0 #depth and width of the well
x0 = l / 2
D = 16 #bond dimension of the ground state
χ = 32 #maximal bond dimension during the evolution
dt, T = 0.02, 0.5 #time step and final time, in continuum units

#N free fermions in the box [0, l] with hard walls, in the well V, released at t = 0, with finite
#differences on M points: the points x, the energy in the well, and ρ(t) on x
function free_fermions(l, N, V; M=2000)
    x = range(0, l, length=M + 2)[2:(end - 1)]
    h = step(x)
    kinetic = SymTridiagonal(fill(2 / h^2, M), fill(-1 / h^2, M - 1))
    well = eigen(SymTridiagonal(kinetic.dv .+ V.(x), kinetic.ev))
    φ = well.vectors[:, 1:N] ./ sqrt(h)    #∫|φ|² = 1
    free = eigen(kinetic)
    ρ(t) = vec(sum(abs2, free.vectors * (exp.(-im * free.values * t) .* (free.vectors' * φ)); dims=2))
    return x, sum(well.values[1:N]), ρ
end

#ground state with N particles in the well, with U(1) symmetric DMRG
(; L, E, ψ) = only(finite_groundstates(LL_D6_gaussian_minimal, l, N, r:r; D, cutoff=dmax, label="well",
                                       μ=0.0, c, V0, w, x0))
xff, Eff, ρff = free_fermions(l, N, x -> -V0 * exp(-(x - x0)^2 / 2w^2))
println("energy in the well: wMPS $E, free fermions $Eff")

#the well is switched off at t = 0. expectation_value(ψ, H) is 2^r times the continuum energy, so a
#continuum time step dt is dt/2^r for the MPO. Two-site TDVP lets the bond dimension grow up to χ
H = LL_D6_minimal(U1Irrep, FiniteChain(L); r, μ=0.0, c, cutoff=dmax)
number = a_number(U1Irrep; cutoff=dmax)
alg = TDVP2(; trscheme=truncrank(χ) & truncerror(; atol=1e-5))
densities = [(0.0, continuum_density(ψ, r))]
for step in 1:round(Int, T / dt)
    global ψ, = timestep(ψ, H, (step - 1) * dt, dt / 2^r, alg)
    #the energy and particle number are conserved
    @printf("t = %.2f: E = %.8f, N = %.10f, D = %d\n", step * dt, real(expectation_value(ψ, H)) / 2^r,
            sum(real(expectation_value(ψ, i => number)) for i in 1:L), maximum(dim(left_virtualspace(ψ, i)) for i in 1:L))
    step % 5 == 0 && push!(densities, (step * dt, continuum_density(ψ, r)))
end

#wMPS (solid) against free fermions (dashed)
plt = plot(xlabel="x", ylabel="density", title="release from a Gaussian well, c = $c, r = $r, N = $N",
           legend=:topright)
for (i, (t, (x, ρ))) in enumerate(densities[1:2:end])
    plot!(plt, x, ρ, color=i, label="t = $(round(t; digits=2))")
    plot!(plt, xff, ρff(t), color=i, linestyle=:dash, label="")
end
savefig(plt, joinpath(@__DIR__, "release_density.png"))
