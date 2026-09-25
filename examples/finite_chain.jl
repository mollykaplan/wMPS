#The wavelet Lieb-Liniger model on a finite chain, with DMRG.
#Run from the package root with `julia --project=examples examples/finite_chain.jl`.

using wMPS, MPSKit, MPSKitModels, TensorKit
using Plots

#parameters
μ = 1.0
c = 8.0
r = 1
dmax = 2 #maximum number of bosons per site
D = 16 #maximal MPS bond dimension
L = 48 #number of sites, i.e. a box of length L / 2^r

#the minimal MPOs work on finite chains too: only the terms that fit inside the chain are kept,
#which amounts to hard wall boundary conditions at the level of the wavelet modes
H = LL_D6_minimal(FiniteChain(L); μ, c, r, cutoff=dmax)
ψ0 = FiniteMPS(L, ℂ^(dmax + 1), ℂ^D)
ψ, = find_groundstate(ψ0, H, DMRG(; tol=1e-5, maxiter=100, verbosity=1))

#on a finite chain expectation_value gives the total energy, in units where the energy per site
#is the energy density
E = real(expectation_value(ψ, H))
println("Energy per site on $L sites: $(E / L)")

#compare with the infinite chain, which has no boundary effects
H∞ = LL_D6_minimal(; μ, c, r, cutoff=dmax)
ψ∞, = find_groundstate(InfiniteMPS(ℂ^(dmax + 1), ℂ^D), H∞, VUMPS(; tol=1e-8, verbosity=0))
println("Energy density on the infinite chain: $(real(expectation_value(ψ∞, H∞)))")

#occupation of each wavelet mode; the continuum density is 2^r times the occupation in the bulk
n = [real(expectation_value(ψ, i => a_number(; cutoff=dmax))) for i in 1:L]
n∞ = real(expectation_value(ψ∞, 1 => a_number(; cutoff=dmax)))
x = ((1:L) .- 1) ./ 2^r
plot(x, 2^r .* n, marker=:circle, label="finite chain", xlabel="x", ylabel="density")
hline!([2^r * n∞], linestyle=:dash, label="infinite chain")
savefig(joinpath(@__DIR__, "finite_chain_density.png"))
