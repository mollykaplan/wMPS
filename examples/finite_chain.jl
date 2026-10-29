#The wavelet Lieb-Liniger model on a finite chain, with DMRG, compared with the Tonks-Girardeau limit.
#Run from the package root with `julia --project=examples examples/finite_chain.jl`. The ground
#states of each model are computed by their own Julia process, in parallel, and saved in
#examples/finite_chain_data; those of one model can be (re)computed alone with e.g.
#`julia --project=examples examples/finite_chain.jl trap`.

using wMPS, MPSKit, MPSKitModels, TensorKit
using Plots, Serialization

#parameters
c = 200.0 #strong repulsion, close to the Tonks-Girardeau limit c → ∞
N = 10 #number of particles
dmax = 2 #maximum number of bosons per site
D = 16 #maximal MPS bond dimension, summed over the particle number sectors
l = 16 #length of the box, in continuum units
rmin, rmax = 1, 5 #the modes are 2^-r apart, so the number of sites L = l 2^r grows with r
#the ground state at the lowest resolution is found from a random state, and each of the following
#ones from the previous one, refined to the next resolution with MPSrefine_finite

#the Hamiltonian conserves the particle number, so finite_groundstates builds it with U(1) symmetry
#and looks for the ground state directly in the sector with N particles. The chemical potential then
#only shifts the energy by -μN, and is set to zero
μ = 0.0

#harmonic trap V(x) = ½ω²(x - x₀)², in the middle of the box; x₀ is fixed, so that the trap is the
#same at every resolution. In the TG limit the N = 10 particles fill the levels up to ≈ 10√2 ω, so
#the gas extends to |x - x₀| ≈ √(2 × 10√2)/ω ≈ 5.3, well inside the box
ω = 1.0
x0 = l / 2

#both LL_D6 and LL_D6_minimal work on finite chains: only the terms that fit inside the chain are
#kept, which amounts to hard wall boundary conditions at the level of the wavelet modes. They are
#the same operator, but the minimal MPO has bond dimension 45 instead of 562 (53 instead of 582 with
#the trap), so DMRG is much faster. We optimize with the minimal MPO, and evaluate the final
#energies with the exact one, exact_model(model)
models = Dict("well" => (LL_D6_minimal, (; μ, c)),
              "trap" => (LL_D6_trap_minimal, (; μ, c, ω, x0)))

datadir = joinpath(@__DIR__, "finite_chain_data")
datafile(name, r) = joinpath(datadir, "$(name)_r$r.jls")

#the ground states of a single model with DMRG, from rmin to rmax, when called by the loop below. At
#rmin DMRG starts from a random state, and at each following resolution from the previous ground
#state, refined with MPSrefine_finite (see finite_groundstates); each one is saved as soon as it is
#computed
if length(ARGS) == 1
    name = only(ARGS)
    model, kw = models[name]
    finite_groundstates(model, l, N, rmin:rmax; D, cutoff=dmax, label=name,
                        callback=result -> serialize(datafile(name, result.r), result), kw...)
    exit()
end

#all ground states, one process per model
mkpath(datadir)
julia = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__)`
jobs = [run(pipeline(`$julia $name`; stdout, stderr); wait=false) for name in keys(models)]
foreach(wait, jobs)

################################################################################################
#Plots of the density
################################################################################################

xs = range(0, l, length=1000)
#the Tonks-Girardeau limit c → ∞: N free fermions in the N lowest levels
tg = Dict("well" => (sum(box_levels(N, l)), x -> tg_box_density(x, N, l)),
          "trap" => (sum(trap_levels(N, ω)), x -> tg_trap_density(x, N, ω, x0)))
titles = Dict("well" => "infinite well, c = $c", "trap" => "harmonic trap, ω = $ω, c = $c")
files = Dict("well" => "infinite_well_density.png", "trap" => "harmonic_trap_density.png")

for name in keys(models)
    plt = plot(xlabel="x", ylabel="density", title=titles[name], legend=:bottom)
    for r in rmin:rmax
        (; L, n, E, ψ) = deserialize(datafile(name, r))
        println("$name, r = $r, L = $L: N = $(sum(n)), E = $E")
        #the continuum density ρ(x) = ⟨ψ⁺(x)ψ(x)⟩, with the same spacing 2^-9 in x at every resolution
        x, ρ = continuum_density(ψ, r; J=9 - r)
        plot!(plt, x, ρ, label="r = $r, E = $(round(E; digits=2))")
    end
    E, ρ = tg[name]
    println("$name, Tonks-Girardeau: N = $N, E = $E")
    plot!(plt, xs, ρ.(xs), linestyle=:dash, color=:black, label="Tonks-Girardeau, E = $(round(E; digits=2))")
    savefig(plt, joinpath(@__DIR__, files[name]))
end
