#The wavelet Lieb-Liniger model in a harmonic trap on a finite chain, with DMRG at a fixed particle number,
#compared with the Tonks-Girardeau limit.
#Run from the package root with `julia --project=examples examples/finite_chain.jl`.

using wMPS, MPSKit, TensorKit
using Plots

#parameters
c = 200.0 #strong repulsion, close to the Tonks-Girardeau limit c → ∞
N = 3 #number of particles
l = 6 #length of the box, in continuum units
r = 2 #the modes are 2^-r apart, so the chain has L = l 2^r = 24 sites
dmax = 2 #maximum number of bosons per site
D = 8 #maximal MPS bond dimension, summed over the particle number sectors

#harmonic trap V(x) = ½ω²(x - x₀)², in the middle of the box. In the Tonks-Girardeau limit the particles
#fill the levels up to ≈ N√2 ω, so the gas extends to |x - x₀| ≈ √(2(N - ½)√2/ω) ≈ 1.9, well inside the box
ω = 2.0
x0 = l / 2

#the Hamiltonian conserves the particle number, so finite_groundstates builds it with U(1) symmetry and
#runs DMRG directly in the sector with N particles; the chemical potential would only shift the energy
#by -μN, and is set to zero. DMRG uses the minimal MPO, and the energy is evaluated with the exact one
(; L, n, E, ψ) = only(finite_groundstates(LL_D6_trap_minimal, l, N, r:r; D, cutoff=dmax, label="trap",
                                          μ=0.0, c, ω, x0))
Etg = sum(trap_levels(N, ω))
#at r = 2 the energy is still well above the Tonks-Girardeau one: at large c the wavefunction has a cusp
#where two particles meet, which the modes only resolve at higher r (see refinement with MPSrefine_finite)
println("N = $(sum(n)), E = $E; Tonks-Girardeau limit: E = $Etg")

#the continuum density ρ(x) = ⟨ψ⁺(x)ψ(x)⟩ on a grid of spacing 2^-(r+5), against free fermions
x, ρ = continuum_density(ψ, r)
plt = plot(x, ρ, xlabel="x", ylabel="density", label="wMPS, r = $r, E = $(round(E; digits=3))",
           title="harmonic trap, ω = $ω, c = $c, N = $N")
plot!(plt, x, tg_trap_density.(x, N, ω, x0), linestyle=:dash, color=:black,
      label="Tonks-Girardeau, E = $(round(Etg; digits=3))")
savefig(plt, joinpath(@__DIR__, "harmonic_trap_density.png"))
