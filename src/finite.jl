################################################################################################
#Ground states on a finite chain, refined from one resolution to the next
################################################################################################

"""
    finite_groundstates(model, l, N, rs; D, cutoff=2, tol=1e-5, maxiter=100, label=string(model),
                        callback=Returns(nothing), kwargs...)

Ground states with N particles of `model` (e.g. `LL_D6_minimal` or `LL_D6_trap_minimal`) in a box of
length l, at each resolution r in the range `rs`, i.e. on a `FiniteChain(L)` with L = l 2^r sites, at
maximal MPS bond dimension D (summed over the particle number sectors). `cutoff` and `kwargs` (e.g.
`μ`, `c`, `ω`, `x0`) are passed to `model`.

The Hamiltonian is built with `U1Irrep` symmetry, and the ground state is looked for directly in the
sector with N particles. At the lowest resolution DMRG starts from a random state, and two-site DMRG
lets its bond spaces grow up to D, distributed over the particle number sectors as the ground state
needs. Each following resolution starts from the previous ground state, refined with
[`MPSrefine_finite`](@ref), and two sweeps of two-site DMRG let the bond spaces adapt to it. Single-site
DMRG then converges with tolerance `tol`, in at most `maxiter` sweeps.

Returns a vector with, for each r, the named tuple `(; r, L, n, E, time, ψ)`: the occupation n of each
mode, the continuum energy E evaluated with `exact_model(model)`, the DMRG time in s, and the state.
`callback` is called with each of them as soon as it is computed, e.g. to save it. Progress is printed,
with `label` to tell runs apart.
"""
function finite_groundstates(model, l, N, rs::AbstractUnitRange; D, cutoff=2, tol=1e-5, maxiter=100,
                             label=string(model), callback=Returns(nothing), kwargs...)
    symmetry = U1Irrep
    results = []
    local ψ
    for r in rs
        L = round(Int, l * 2^r)
        H = model(symmetry, FiniteChain(L); r, cutoff, kwargs...)
        time = @elapsed begin
            if r == first(rs)
                #random state with N particles: the charge of the last virtual space is N
                P = U1Space(n => 1 for n in 0:cutoff)
                ψ = FiniteMPS(L, P, U1Space(n => 1 for n in 0:N); right=U1Space(N => 1))
                ψ, = find_groundstate(ψ, H, DMRG2(; trscheme=truncrank(D), tol, maxiter=5, verbosity=0))
            else
                #the ground state at r - 1, on L/2 sites, refined to r on L sites; it keeps the
                #particle number and, up to the modes projected at the right wall, the energy
                ψ, p = MPSrefine_finite(ψ, D; verbosity=0)
                println("$label, r = $r: probability lost in the refinement 1 - p = $(1 - p)")
                ψ, = find_groundstate(ψ, H, DMRG2(; trscheme=truncrank(D), tol, maxiter=2, verbosity=0))
            end
            #verbosity=2 prints the number of sweeps
            ψ, = find_groundstate(ψ, H, DMRG(; tol, maxiter, verbosity=2))
        end
        n = [real(expectation_value(ψ, i => a_number(symmetry; cutoff))) for i in 1:L]
        #expectation_value is 2^r times the energy of the continuum Hamiltonian
        E = real(expectation_value(ψ, exact_model(model)(symmetry, FiniteChain(L); r, cutoff, kwargs...))) / 2^r
        #the modes are orthonormal, so the total particle number is the sum of the occupations,
        #which is N up to round-off
        println("$label, r = $r, L = $L: N = $(sum(n)), E = $E, ",
                "D = $(maximum(dim(left_virtualspace(ψ, i)) for i in 1:L)), DMRG time = $(round(time; digits=1)) s")
        flush(stdout)
        result = (; r, L, n, E, time, ψ)
        callback(result)
        push!(results, result)
    end
    return results
end

################################################################################################
#The Tonks-Girardeau limit c → ∞: N free fermions, filling the N lowest levels of -∂ₓ² + V(x)
#(the kinetic term ∫ ∂ₓψ⁺∂ₓψ corresponds to a mass 1/2)
################################################################################################

"""
    box_levels(N, l)

The N lowest levels (jπ/l)², j = 1, …, N, of -∂ₓ² in a hard wall box [0, l]. In the Tonks-Girardeau
limit c → ∞, the ground state energy of N particles is their sum.
"""
box_levels(N, l) = [(j * π / l)^2 for j in 1:N]

"""
    tg_box_density(x, N, l)

The density of N particles in a hard wall box [0, l] in the Tonks-Girardeau limit c → ∞, that of N free
fermions with eigenfunctions √(2/l) sin(jπx/l).
"""
tg_box_density(x, N, l) = sum(2 / l * sin(j * π * x / l)^2 for j in 1:N)

"""
    trap_levels(N, ω)

The N lowest levels √2 ω (j + ½), j = 0, …, N - 1, of -∂ₓ² + ½ω²(x - x₀)², an oscillator with frequency
Ω = √2 ω. In the Tonks-Girardeau limit c → ∞, the ground state energy of N particles is their sum.
"""
trap_levels(N, ω) = [sqrt(2) * ω * (j + 1 / 2) for j in 0:(N - 1)]

"""
    tg_trap_density(x, N, ω, x0)

The density of N particles in the harmonic trap ½ω²(x - x₀)² in the Tonks-Girardeau limit c → ∞, that of
N free fermions. The eigenfunctions are Hermite functions of length b = √(2/Ω), Ω = √2 ω, built with
their recursion.
"""
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
