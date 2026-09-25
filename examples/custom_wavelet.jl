#Building exact, compact MPOs from your own coefficients.
#Run from the package root with `julia --project=examples examples/custom_wavelet.jl`.

using wMPS, MPSKit, MPSKitModels, TensorKit

################################################################################################
#1. The Lieb-Liniger model for any wavelet order
################################################################################################

#LL_wavelet_minimal takes the kinetic coefficients K and the four-point overlaps Γ4 of the
#scaling functions, in the same layout as the arrays stored in data/:
#  K[1 + s]       = Kₙ,ₙ₊ₛ for s = 0, …, S
#  Γ4[a+w+1, b+w+1, c+w+1] = overlap of the scaling functions at positions (0, a, b, c), for offsets -w:w
#Here we use the D8 coefficients shipped with the package, but they could come from anywhere.
K = wMPS.K8
Γ4 = wMPS.Γ4_D8

#verbose=true reports the bond dimension, the number of internal states per grade, and the
#largest error on any term of H (the residual), which should be at round-off level
H = LL_wavelet_minimal(K, Γ4; μ=1.0, c=8.0, r=2, cutoff=2, verbose=true)
println("bond dimension: ", dim(left_virtualspace(H, 1)))

#it is the same operator as LL_D8_minimal
ψ = InfiniteMPS(ℂ^3, ℂ^6)
println("LL_wavelet_minimal: ", expectation_value(ψ, H))
println("LL_D8_minimal:      ", expectation_value(ψ, LL_D8_minimal(; μ=1.0, c=8.0, r=2, cutoff=2)))

################################################################################################
#2. Any translation invariant, finite range Hamiltonian
################################################################################################

#Underneath, the MPO is built from a list of terms over an alphabet of on-site operators
#(a⁺)ᵖ aᑫ. As an example, here is the quadratic test model quad_model, written as such a list:
#  H = 2³ʳ Σₙₘ aₙ⁺ Kₙₘ aₘ + 2ʳλ Σₙ (aₙ⁺aₙ⁺ + aₙaₙ) + 2ʳμ Σₙ aₙ⁺aₙ
μ, λ, r, cutoff = 1.0, 0.5, 1, 2
labels, ops, index = wMPS.boson_alphabet(ComplexF64, cutoff, 2, 2) #ops[index[(p, q)]] = (a⁺)ᵖ aᑫ
K6 = wMPS.K6
R = length(K6) - 1 #range of the Hamiltonian: every term fits in a window of R + 1 sites

#each term is a window of R + 1 alphabet indices, anchored on its leftmost site (index 1 is the identity)
function window(ops_at)
    s = ones(Int, R + 1)
    for (site, pq) in ops_at
        s[site] = index[pq]
    end
    return s
end

coeffs = Dict{Vector{Int}, Float64}()
coeffs[window([1 => (1, 1)])] = 2^(3r) * K6[1] + 2^r * μ #on-site kinetic term and chemical potential
coeffs[window([1 => (2, 0)])] = 2^r * λ #a⁺a⁺
coeffs[window([1 => (0, 2)])] = 2^r * λ #aa
for s in 1:R #hopping over s sites, in both directions
    coeffs[window([1 => (1, 0), 1 + s => (0, 1)])] = 2^(3r) * K6[1 + s]
    coeffs[window([1 => (0, 1), 1 + s => (1, 0)])] = 2^(3r) * K6[1 + s]
end

Hquad = wMPS.exact_mpo_hamiltonian(coeffs, ops, R, InfiniteChain(1); verbose=true)

ψ = InfiniteMPS(ℂ^(cutoff + 1), ℂ^6)
println("term list:  ", expectation_value(ψ, Hquad), ", bond dimension ", dim(left_virtualspace(Hquad, 1)))
Href = quad_model(; μ, λ, r, cutoff)
println("quad_model: ", expectation_value(ψ, Href), ", bond dimension ", dim(left_virtualspace(Href, 1)))
