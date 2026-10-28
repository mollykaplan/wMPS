#The wavelet Lieb-Liniger model on a finite chain, with DMRG, compared with the Tonks-Girardeau limit.
#Run from the package root with `julia --project=examples examples/finite_chain.jl`.

using wMPS, MPSKit, MPSKitModels, TensorKit
using Plots

#parameters
c = 200.0 #strong repulsion, close to the Tonks-Girardeau limit c → ∞
dmax = 2 #maximum number of bosons per site
D = 16 #maximal MPS bond dimension
l = 16 #length of the box, in continuum units
resolutions = (1, 3, 5) #the modes are 2^-r apart, so the number of sites L = l 2^r grows with r

#mode i of the chain (n = i - 1) has its centre of mass at (i - 1 + s̄) / 2^r, with s̄ = ∫ x s(x)² dx;
#the density 2^r nᵢ is plotted at these points
s̄ = wMPS.scaling_moments(wMPS.h6, 1)[2, 5]
positions(L, r) = ((1:L) .- 1 .+ s̄) ./ 2^r

#ground state with DMRG, and the occupation of each mode
function groundstate(H, L)
    ψ, = find_groundstate(FiniteMPS(L, ℂ^(dmax + 1), ℂ^D), H, DMRG(; tol=1e-5, maxiter=100, verbosity=1))
    n = [real(expectation_value(ψ, i => a_number(; cutoff=dmax))) for i in 1:L]
    return ψ, n
end

################################################################################################
#The Tonks-Girardeau limit c → ∞: N free fermions, filling the N lowest levels of -∂ₓ² + V(x)
#(the kinetic term ∫ ∂ₓψ⁺∂ₓψ corresponds to a mass 1/2)
################################################################################################

#hard wall box [0, l]: levels (jπ/l)², eigenfunctions √(2/l) sin(jπx/l), j = 1, 2, …
box_levels(N) = [(j * π / l)^2 for j in 1:N]
tg_box_density(x, N) = sum(2 / l * sin(j * π * x / l)^2 for j in 1:N)

#harmonic trap ½ω²(x - x₀)²: an oscillator with frequency Ω = √2 ω and length b = √(2/Ω), levels
#Ω(j + ½), j = 0, 1, …; the eigenfunctions are Hermite functions, built with their recursion
trap_levels(N, ω) = [sqrt(2) * ω * (j + 1 / 2) for j in 0:(N - 1)]
function tg_trap_density(x, N, ω, x0)
    b = sqrt(2 / (sqrt(2) * ω))
    ξ = (x - x0) / b
    φprev, φ = 0.0, π^(-1 / 4) * exp(-ξ^2 / 2)
    ρ = 0.0
    for j in 0:(N - 1)
        ρ += φ^2 / b
        φprev, φ = φ, sqrt(2 / (j + 1)) * ξ * φ - sqrt(j / (j + 1)) * φprev
    end
    return ρ
end

xs = range(0, l, length=1000)

################################################################################################
#Finite chain with hard wall boundary conditions, aka an infinite well
################################################################################################

#μ lies halfway between the 10th and 11th levels of the box, so that N = 10 in the TG limit
μ = (10.5π / l)^2
Ntg = count(<(μ), box_levels(20))

#both LL_D6 and LL_D6_minimal work on finite chains: only the terms that fit inside the chain are
#kept, which amounts to hard wall boundary conditions at the level of the wavelet modes. They are
#the same operator, but the minimal MPO has bond dimension 45 instead of 562, so DMRG is much faster.
#We optimize with the minimal MPO, and evaluate the final energies with the exact one, LL_D6
plt = plot(xlabel="x", ylabel="density", title="infinite well, c = $c")
for r in resolutions
    L = round(Int, l * 2^r)
    ψ, n = groundstate(LL_D6_minimal(FiniteChain(L); μ, c, r, cutoff=dmax), L)
    #expectation_value is 2^r times the energy of the continuum Hamiltonian, including -μN
    E = real(expectation_value(ψ, LL_D6(FiniteChain(L); μ, c, r, cutoff=dmax))) / 2^r
    #the modes are orthonormal, so the total particle number is the sum of the occupations; H
    #conserves it, so the ground state has a definite number of particles
    println("infinite well, r = $r, L = $L: N = $(sum(n)), E = $E")
    plot!(plt, positions(L, r), 2^r .* n, label="r = $r", marker=(r == first(resolutions) ? :circle : :none))
end
println("infinite well, Tonks-Girardeau: N = $Ntg, E = $(sum(box_levels(Ntg)) - μ * Ntg)")
plot!(plt, xs, tg_box_density.(xs, Ntg), linestyle=:dash, color=:black, label="Tonks-Girardeau")
savefig(plt, joinpath(@__DIR__, "infinite_well_density.png"))

################################################################################################
#The same gas in a harmonic trap V(x) = ½ω²(x - x₀)², in the middle of the box
################################################################################################

ω = 1.0
x0 = l / 2 #fixed, so that the trap is the same at every resolution

#μ lies halfway between the 10th and 11th levels of the trap, so that N = 10 in the TG limit; the
#gas then extends to |x - x₀| ≈ √(2μ)/ω ≈ 5.3, well inside the box
μtrap = 10 * sqrt(2) * ω
Ntrap = count(<(μtrap), trap_levels(20, ω))

#as above, we optimize with the minimal MPO (bond dimension 53 instead of 582), and evaluate the
#energy with the exact one, LL_D6_trap
plt = plot(xlabel="x", ylabel="density", title="harmonic trap, ω = $ω, c = $c")
for r in resolutions
    L = round(Int, l * 2^r)
    ψ, n = groundstate(LL_D6_trap_minimal(FiniteChain(L); μ=μtrap, c, r, cutoff=dmax, ω, x0), L)
    E = real(expectation_value(ψ, LL_D6_trap(FiniteChain(L); μ=μtrap, c, r, cutoff=dmax, ω, x0))) / 2^r
    println("harmonic trap, r = $r, L = $L: N = $(sum(n)), E = $E")
    plot!(plt, positions(L, r), 2^r .* n, label="r = $r", marker=(r == first(resolutions) ? :circle : :none))
end
println("harmonic trap, Tonks-Girardeau: N = $Ntrap, E = $(sum(trap_levels(Ntrap, ω)) - μtrap * Ntrap)")
plot!(plt, xs, tg_trap_density.(xs, Ntrap, ω, x0), linestyle=:dash, color=:black, label="Tonks-Girardeau")
savefig(plt, joinpath(@__DIR__, "harmonic_trap_density.png"))
