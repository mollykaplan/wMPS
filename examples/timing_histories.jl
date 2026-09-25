#Convergence of VUMPS followed by L-BFGS, versus wall time, using the timed solvers.
#Run from the package root with `julia --project=examples examples/timing_histories.jl`.

using wMPS, MPSKit, TensorKit, OptimKit
using Plots, LaTeXStrings

#parameters
μ = 1.0
c = 8.0
r = 2
dmax = 2 #maximum number of bosons per site
D = 12 #MPS bond dimension

exact = -0.230661378983648 #exact energy density of the Lieb-Liniger model at μ = 1, c = 8
relerr(E) = abs((E - exact) / exact)

H = LL_D6_minimal(; μ, c, r, cutoff=dmax)
ψ0 = InfiniteMPS(ℂ^(dmax + 1), ℂ^D)

#each history is an array with columns (wall time in s, energy density, error)
ψ, _, _, hist_vumps = find_groundstate_t(ψ0, H, VUMPS(; tol=1e-3, verbosity=1))
alg = GradientGrassmann(; method=OptimKit.LBFGS(100; gradtol=1e-5, maxiter=500, verbosity=1))
ψ, _, hist_lbfgs = find_groundstate_t(ψ, H, alg)

#one history for the whole run: the L-BFGS times are shifted to start where VUMPS ended, and the
#first L-BFGS row (the state VUMPS returned) is dropped since it is already in hist_vumps
hist = add_histories(hist_vumps, hist_lbfgs)
nv = size(hist_vumps, 1)

plot(hist[1:nv, 1], relerr.(hist[1:nv, 2]), label="VUMPS", yscale=:log10, marker=:circle, markersize=2)
plot!(hist[nv:end, 1], relerr.(hist[nv:end, 2]), label="L-BFGS")
xlabel!("time (s)")
ylabel!(L"|E - E_{exact}| / |E_{exact}|")
savefig(joinpath(@__DIR__, "timing_histories.png"))
println("final relative error: $(relerr(hist[end, 2])), total time $(hist[end, 1]) s")
