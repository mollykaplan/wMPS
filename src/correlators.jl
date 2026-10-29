################################################################################################
#Daubechies wavelet values
################################################################################################

#values of the N=6 Daubechies scaling function s(x) on [0, 5], in steps of dx
const sxdata=[0.,0.0477135636885,0.101396505763,0.158604211489,0.215399070084,0.284716624233,0.337127934667,0.394080128947,0.457836727203,0.530679681933,0.605178468388,0.660144859778,0.716586371058,0.775155036186,0.837684705982,0.898113049518,0.973137367421,1.05016315871,1.12798099254,1.20528197587,1.28633506943,1.22948248334,1.15723970541,1.07711397914,1.00074725511,0.889916048133,0.82991718245,0.758831434055,0.670055664191,0.556984446439,0.441122481462,0.386695807121,0.330364448733,0.2700926328,0.200333452582,0.139418881901,0.037237256857,-0.06845927815,-0.1740327981,-0.275556401991,-0.385836961046,-0.366179422596,-0.333668189291,-0.296157303242,-0.264946951462,-0.202979934521,-0.189570037759,-0.167645406296,-0.13097774463,-0.0734424884942,-0.0149705913866,-0.0211077543959,-0.0278821949469,-0.033098796308,-0.0311199431713,-0.040567571439,-0.0112874414092,0.0190279678607,0.0467214663125,0.0678657092793,0.0952675460038,0.0856559726186,0.0726300715015,0.0591546552953,0.0486546894467,0.0299440598983,0.0234756216905,0.0150748268266,0.00304042006306,-0.0141882578704,-0.0315413029749,-0.0258525194967,-0.0190756083477,-0.0121016183393,-0.00690046314216,0.00302513129194,0.000912474753948,-0.000731903906359,-0.000669676298295,0.0024087161316,0.0042343456164,0.00332740295281,0.00240190662056,0.00128445731713,0.000145936821122,-0.00159679774383,-0.000950701048111,-0.000340983533211,0.0000449331724483,-0.0000333820070649,0.000210944511631,0.000119606992823,6.98350365964*10^(-6),-0.0000472543387936,2.24774952535*10^(-6),0.0000105087281527,3.42377605237*10^(-7),5.54896037911*10^(-8),1.55451934543*10^(-8),7.12088762269*10^(-10),0.]
const sint=[sxdata[21],sxdata[41],sxdata[61],sxdata[81]] #[s(1), s(2), s(3), s(4)]
const dx=0.05 #step size between values

#maximum number of bosons per site of the MPS ψ
state_cutoff(ψ) = dim(physicalspace(ψ, 1)) - 1

"""
    scaling_function(h, J)

Values of the scaling function s(x) of the wavelet with filter coefficients `h` (e.g. `wMPS.h6`),
supported on [0, w] with w = length(h) - 1, at the points x = t / 2^J: `s[t + 1] = s(t / 2^J)` for
t = 0, …, w 2^J. They are exact up to round-off, and computed with the cascade algorithm: the values
at the integers are the eigenvector of the refinement equation s(x) = √2 Σₖ hₖ s(2x - k) with
eigenvalue 1, normalized to Σₖ s(k) = 1, and the refinement equation then gives the values at the
half-integers, the quarter-integers, …
"""
function scaling_function(h::AbstractVector, J::Integer)
    w = length(h) - 1
    M = [1 <= 2k - j + 1 <= w + 1 ? sqrt(2) * h[2k - j + 1] : 0.0 for k in 1:(w - 1), j in 1:(w - 1)]
    v = vec(nullspace(M - I))
    s = [0.0; v ./ sum(v); 0.0]
    for j in 0:(J - 1)
        s = [iseven(t) ? s[t ÷ 2 + 1] :
             sqrt(2) * sum(h[k + 1] * s[t - k * 2^j + 1] for k in 0:w if 0 <= t - k * 2^j <= w * 2^j; init=0.0)
             for t in 0:(w * 2^(j + 1))]
    end
    return s
end

#a⁺ ⊗ a and n on the physical space of ψ, with its symmetry
function onebody_operators(ψ)
    P = physicalspace(ψ, 1)
    cutoff = state_cutoff(ψ)
    hopping = symmetric_operator(a_plus(; cutoff) ⊗ a_min(; cutoff), sectortype(P); cutoff)
    number = symmetric_operator(a_number(; cutoff), sectortype(P); cutoff)
    return hopping, number
end

################################################################################################
#Correlation functions
################################################################################################

"""
    continuum_density(ψ::InfiniteMPS, r, xmax; J=5)

The one-body density matrix ⟨ψ⁺(x)ψ(0)⟩ of an `InfiniteMPS` ψ at resolution r, in the basis of the D6
scaling functions, at the points x = t / 2^(r+J) with 0 ≤ x ≤ xmax; at x = 0 it is the density at 0.
Returns `(x, g)`, with g its real part. For a `FiniteMPS`, `continuum_density(ψ, r)` gives the density
⟨ψ⁺(x)ψ(x)⟩ instead.

With ψ(y) = Σₙ 2^(r/2) s(2^r y - n) aₙ, ψ(0) involves the modes n = -4, …, -1 only. All the ⟨aₘ⁺aₙ⟩
needed are computed once, with one call to MPSKit's `correlator` for each of these n, and s(x) is
computed exactly on the grid with [`scaling_function`](@ref), so a whole curve costs about as much as
a single point. Works with `Trivial` states, and with states of other symmetries for which
`symmetric_operator` is implemented.
"""
function continuum_density(ψ::InfiniteMPS, r::Integer, xmax::Real; J::Integer=5)
    hopping, number = onebody_operators(ψ)
    w = length(h6) - 1
    T = floor(Int, xmax * 2^(r + J))
    #the modes of ψ(0), and those of ψ⁺(x), which overlap some 0 ≤ x ≤ xmax
    ns = (1 - w):(-1)
    ms = (1 - w):max(cld(T, 2^J) - 1, 0)
    row(m) = m - first(ms) + 1
    #G[row(m), k] = ⟨aₘ⁺aₙ⟩ with n = ns[k], from ⟨aₙ⁺aₘ⟩ with m > n
    G = zeros(ComplexF64, length(ms), length(ns))
    for (k, n) in enumerate(ns)
        G[row(n), k] = expectation_value(ψ, n => number)
        for (m, c) in zip((n + 1):last(ms), correlator(ψ, hopping, n, (n + 1):last(ms)))
            G[row(m), k] = conj(c)
            m in ns && (G[row(n), m - first(ns) + 1] = c)
        end
    end
    s = scaling_function(h6, J)
    g = zeros(T + 1)
    for (k, n) in enumerate(ns), m in ms, t in max(m * 2^J, 0):min((m + w) * 2^J, T)
        g[t + 1] += 2^r * real(G[row(m), k]) * s[t - m * 2^J + 1] * s[-n * 2^J + 1]
    end
    return (0:T) ./ 2^(r + J), g
end

"""
    continuum_density(ψ::FiniteMPS, r; J=5)

The density ρ(x) = ⟨ψ⁺(x)ψ(x)⟩ of a `FiniteMPS` ψ at resolution r, e.g. a ground state of a model on a
`FiniteChain(L)`, whose site n + 1 holds the D6 mode sₙ(x) = 2^(r/2) s(2^r x - n). Returns `(x, ρ)` at
the points x = t / 2^(r+J), from 0 to the end of the last mode at (L + 4) / 2^r.

ρ(x) = Σₙₘ ⟨aₙ⁺aₘ⟩ sₙ(x) sₘ(x), where two modes only overlap for |n - m| ≤ 4, so only these
correlations are computed, and s(x) is computed exactly on the grid with [`scaling_function`](@ref).
Works with `Trivial` and `U1Irrep` states. For an `InfiniteMPS`, `continuum_density(ψ, r, xmax)` gives
the one-body density matrix ⟨ψ⁺(x)ψ(0)⟩ instead.
"""
function continuum_density(ψ::FiniteMPS, r::Integer; J::Integer=5)
    hopping, number = onebody_operators(ψ)
    L = length(ψ)
    w = length(h6) - 1
    G = zeros(L, L)
    for i in 1:L
        G[i, i] = real(expectation_value(ψ, i => number))
        js = (i + 1):min(i + w - 1, L)
        isempty(js) || (G[i, js] = G[js, i] = real.(correlator(ψ, hopping, i, js)))
    end
    s = scaling_function(h6, J)
    ρ = zeros((L - 1 + w) * 2^J + 1)
    for n in 0:(L - 1), m in max(0, n - w + 1):min(L - 1, n + w - 1), t in (max(n, m) * 2^J):((min(n, m) + w) * 2^J)
        ρ[t + 1] += 2^r * G[n + 1, m + 1] * s[t - n * 2^J + 1] * s[t - m * 2^J + 1]
    end
    return (0:(length(ρ) - 1)) ./ 2^(r + J), ρ
end

#⟨ψ| ∏ₚ Oₚ |ψ⟩ for an InfiniteMPS with Trivial symmetry, as dense arrays: the left environment E[α, β]
#(ket, bra) after site p, from the one after site p - 1, with the operator O (a matrix, or nothing for the
#identity) on site p; the environment left of any site is the identity, since ψ.AL is left canonical,
#and the one after site p is closed with the bond matrix C[p]
function transfer(E, A, O)
    O === nothing && return @tensor E2[a2, b2] := E[a, b] * A[a, s, a2] * conj(A[b, s, b2])
    return @tensor E2[a2, b2] := E[a, b] * A[a, s, a2] * O[t, s] * conj(A[b, t, b2])
end
close_environment(E, C) = @tensor E[a, b] * C[a, c] * conj(C[b, c])

"""
    nn_correlator(ψ::InfiniteMPS, r, xmax; J=5)

The density-density correlation function ⟨ψ⁺(x)ψ⁺(0)ψ(x)ψ(0)⟩ of an `InfiniteMPS` ψ with `Trivial`
symmetry at resolution r, in the basis of the D6 scaling functions, at the points x = t / 2^(r+J) with
0 ≤ x ≤ xmax. Returns `(x, g)`, with g its real part.

With ψ(y) = Σₙ 2^(r/2) s(2^r y - n) aₙ, it is a sum of ⟨aᵢ⁺aₖ⁺aⱼaₗ⟩ over the modes k, l = -4, …, -1 of
ψ(0) and the pairs of overlapping modes i, j of ψ(x). For each (k, l), the left environment with aₖ⁺aₗ
inserted is carried along the chain once, and each (i, j) only adds the few sites between i and j, so
all the expectation values cost about as much as a few correlation functions. s(x) is computed exactly
on the grid with [`scaling_function`](@ref).
"""
function nn_correlator(ψ::InfiniteMPS, r::Integer, xmax::Real; J::Integer=5)
    sectortype(physicalspace(ψ, 1)) === Trivial ||
        throw(ArgumentError("nn_correlator is only implemented for states with Trivial symmetry"))
    cutoff = state_cutoff(ψ)
    a = diagm(1 => sqrt.(1:cutoff))
    w = length(h6) - 1
    T = floor(Int, xmax * 2^(r + J))
    ns = (1 - w):(-1)
    ms = (1 - w):max(cld(T, 2^J) - 1, 0)
    AL(p) = convert(Array, ψ.AL[p])
    C(p) = convert(Array, ψ.C[p])
    #the environment left of site p
    E0(p) = Matrix{ComplexF64}(I, dim(left_virtualspace(ψ, p)), dim(left_virtualspace(ψ, p)))
    #(a⁺)ᵖ aᑫ on each site, with p and q the number of creators and annihilators on that site
    function site_operators(creators, annihilators)
        return Dict(p => a'^count(==(p), creators) * a^count(==(p), annihilators) for p in union(creators, annihilators))
    end
    #⟨∏ₚ Oₚ⟩ over the sites first:last, starting from the environment E after site first - 1
    function sweep(E, sites, ops)
        for p in sites
            E = transfer(E, AL(p), get(ops, p, nothing))
        end
        return E
    end
    s = scaling_function(h6, J)
    #H[i, j] = Σₖₗ s(-k) s(-l) ⟨aᵢ⁺aₖ⁺aⱼaₗ⟩, for the pairs i, j that overlap
    H = Dict{Tuple{Int, Int}, Float64}()
    pairs = [(i, j) for i in ms, j in ms if abs(i - j) < w]
    for k in ns, l in ns
        weight = s[-k * 2^J + 1] * s[-l * 2^J + 1]
        add!(ij, value) = H[ij] = get(H, ij, 0.0) + weight * real(value)
        #pairs that overlap the modes of ψ(0): all four operators in one sweep
        for (i, j) in pairs
            min(i, j) <= -1 || continue
            ops = site_operators((i, k), (j, l))
            lo, hi = extrema(keys(ops))
            add!((i, j), close_environment(sweep(E0(lo), lo:hi, ops), C(hi)))
        end
        #the other pairs: the environment with aₖ⁺aₗ, carried along the chain, Eₚ after site p
        Ek = sweep(E0(min(k, l)), min(k, l):(-1), site_operators((k,), (l,)))
        envs = Dict(-1 => Ek)
        for p in 0:last(ms)
            envs[p] = transfer(envs[p - 1], AL(p), nothing)
        end
        for (i, j) in pairs
            min(i, j) >= 0 || continue
            lo, hi = minmax(i, j)
            add!((i, j), close_environment(sweep(envs[lo - 1], lo:hi, site_operators((i,), (j,))), C(hi)))
        end
    end
    g = zeros(T + 1)
    for ((i, j), Hij) in H, t in (max(i, j) * 2^J):((min(i, j) + w) * 2^J)
        0 <= t <= T || continue
        g[t + 1] += 2^(2r) * Hij * s[t - i * 2^J + 1] * s[t - j * 2^J + 1]
    end
    return (0:T) ./ 2^(r + J), g
end
