#Ground state of the Lieb-Liniger model at a fixed resolution, with D6 and D8 wavelets.
#Run from the package root with `julia --project=examples examples/groundstate.jl`.

using wMPS, MPSKit, TensorKit, OptimKit

#parameters
μ = 1.0
c = 8.0
r = 2
dmax = 2 #maximum number of bosons per site
D = 12 #MPS bond dimension

#exact energy density of the Lieb-Liniger model at μ = 1, c = 8 (Bethe ansatz)
exact = -0.230661378983648
relerr(E) = abs((E - exact) / exact)

for (name, model) in (("D6", LL_D6_minimal), ("D8", LL_D8_minimal))
    H = model(; μ, c, r, cutoff=dmax)
    ψ0 = InfiniteMPS(ℂ^(dmax + 1), ℂ^D) #random initial state

    #a few VUMPS iterations to get close to the ground state, then L-BFGS to converge
    ψ, = find_groundstate(ψ0, H, VUMPS(; tol=1e-3, verbosity=1))
    E_vumps = real(expectation_value(ψ, H))
    ψ, = find_groundstate(ψ, H, GradientGrassmann(; method=OptimKit.LBFGS(100; gradtol=1e-5, maxiter=500, verbosity=1)))
    E = real(expectation_value(ψ, H)) #expectation_value(ψ, H) is the energy density

    println("$name, r = $r, D = $D:")
    println("  after VUMPS:  E = $E_vumps, relative error $(relerr(E_vumps))")
    println("  after L-BFGS: E = $E, relative error $(relerr(E))")
end

