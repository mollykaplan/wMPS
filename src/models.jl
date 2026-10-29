################################################################################################
#Loading coefficients
################################################################################################

const DATA_DIR = joinpath(@__DIR__, "..", "data")

#reads a dataset from a file in data/, and rebuilds the precompile cache if the file changes
function load_coefficients(filename, dataset)
    path = joinpath(DATA_DIR, filename)
    include_dependency(path)
    return h5open(f -> real(read(f, dataset)), path, "r")
end

#kinetic term coefficients
const K6 = [295/56, -356/105, 92/105, -4/35, -3/560]

#Γ4 for N=6 wavelets
const Γ4_D6 = load_coefficients("gamma4_D6_save_arr.h5", "Γ4arr")

#kinetic coefficients for N=8 wavelets
const K8 = load_coefficients("K_D8_save_arr.h5", "K8arr")

#Γ4 for N=8 wavelets
const Γ4_D8 = load_coefficients("gamma4_D8_save_arr.h5", "Γ4_D8arr")

################################################################################################
#Definitions: mapping points on the lattice & different combinations of a⁺ a⁺ a⁻ a⁻
################################################################################################

#Translating an infinite chain by k sites
k_neighbours(k::Int, chain::InfiniteChain) = map(v -> v => v + k, vertices(chain))
#Translating an infinite chain by k1 and k2 sites
threept_neighbours(k1::Int, k2::Int, chain::InfiniteChain) = map(v -> v => (v + k1, v + k2), vertices(chain))
#Translating an infinite chain by k1, k2, and k3 sites
fourpt_neighbours(k1::Int, k2::Int, k3::Int, chain::InfiniteChain) = map(v -> v => (v + k1, v + k2, v + k3), vertices(chain))

#Translating an infinite chain by k sites
k_neighbours(k::Int, chain::FiniteChain) = map(v -> v => v + k, vertices(chain)[1:(end-k)])
#Translating an infinite chain by k1 and k2 sites
threept_neighbours(k1::Int, k2::Int, chain::FiniteChain) = map(v -> v => (v + k1, v + k2), vertices(chain)[1:(end-max(k1,k2))])
#Translating an infinite chain by k1, k2, and k3 sites
fourpt_neighbours(k1::Int, k2::Int, k3::Int, chain::FiniteChain) = map(v -> v => (v + k1, v + k2, v + k3), vertices(chain)[1:(end-max(k1,k2,k3))])


function a_pp(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁺a⁺
    return MPSKitModels.contract_onesite(a_plus(elt; cutoff=cutoff), a_plus(elt; cutoff=cutoff))
end

function a_mm(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁻a⁻
    return MPSKitModels.contract_onesite(a_min(elt; cutoff=cutoff), a_min(elt; cutoff=cutoff))
end

#Interaction term, all operators on different sites
a_ppmm(symmetry::Type{<:Sector}; kwargs...) = a_ppmm(ComplexF64, symmetry; kwargs...)
function a_ppmm(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁺ ⊗ a⁺ ⊗ a⁻ ⊗ a⁻
    return a_plus(elt; cutoff=cutoff) ⊗ a_plus(elt; cutoff=cutoff) ⊗ a_min(elt; cutoff=cutoff) ⊗ a_min(elt; cutoff=cutoff)
end
a_mmpp(symmetry::Type{<:Sector}; kwargs...) = a_mmpp(ComplexF64, symmetry; kwargs...)
function a_mmpp(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁻ ⊗ a⁻ ⊗ a⁺ ⊗ a⁺
    return a_min(elt; cutoff=cutoff) ⊗ a_min(elt; cutoff=cutoff) ⊗ a_plus(elt; cutoff=cutoff) ⊗ a_plus(elt; cutoff=cutoff)
end

#Interaction term, two of the operators on the same site
function a_ppmm_2mm(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁺a⁺ ⊗ a⁻ ⊗ a⁻
    return MPSKitModels.contract_onesite(a_plus(elt; cutoff=cutoff), a_plus(elt; cutoff=cutoff)) ⊗ a_min(elt; cutoff=cutoff) ⊗ a_min(elt; cutoff=cutoff)
end
function a_ppmm_2pm(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁺ ⊗ a⁺a⁻ ⊗ a⁻
    return a_plus(elt; cutoff=cutoff) ⊗ a_number(elt; cutoff=cutoff) ⊗ a_min(elt; cutoff=cutoff)
end
function a_ppmm_2mp(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁻ ⊗ a⁺a⁻ ⊗ a⁺
    return a_min(elt; cutoff=cutoff) ⊗ a_number(elt; cutoff=cutoff) ⊗ a_plus(elt; cutoff=cutoff) 
end
function a_ppmm_2pp(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁺ ⊗ a⁺ ⊗ a⁻a⁻
    return MPSKitModels.contract_onesite(a_min(elt; cutoff=cutoff), a_min(elt; cutoff=cutoff))  ⊗ a_plus(elt; cutoff=cutoff) ⊗ a_plus(elt; cutoff=cutoff)
end

#Interaction term, three of the operators on the same site
function a_ppmm_3m(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁺a⁺a⁻ ⊗ a⁻
    return MPSKitModels.contract_onesite(a_plus(elt; cutoff=cutoff), a_number(elt; cutoff=cutoff)) ⊗ a_min(elt; cutoff=cutoff)
end
function a_ppmm_3p(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁺a⁻a⁻ ⊗ a⁺ 
    return MPSKitModels.contract_onesite(a_number(elt; cutoff=cutoff), a_min(elt; cutoff=cutoff)) ⊗ a_plus(elt; cutoff=cutoff)
end
function a_ppmm_3pm(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁺a⁻ ⊗ a⁺a⁻
    return MPSKitModels.contract_onesite(a_plus(elt; cutoff=cutoff), a_min(elt; cutoff=cutoff)) ⊗ MPSKitModels.contract_onesite(a_plus(elt; cutoff=cutoff), a_min(elt; cutoff=cutoff))
end
function a_ppmm_3ppmm(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁺a⁺ ⊗ a⁻a⁻
    return MPSKitModels.contract_onesite(a_plus(elt; cutoff=cutoff), a_plus(elt; cutoff=cutoff)) ⊗ MPSKitModels.contract_onesite(a_min(elt; cutoff=cutoff), a_min(elt; cutoff=cutoff))
end
function a_ppmm_3mmpp(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁻a⁻ ⊗ a⁺a⁺
    return MPSKitModels.contract_onesite(a_min(elt; cutoff=cutoff), a_min(elt; cutoff=cutoff)) ⊗ MPSKitModels.contract_onesite(a_plus(elt; cutoff=cutoff), a_plus(elt; cutoff=cutoff))
end

#Interaction term, all operators on the same site
function a_ppmm_4(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; cutoff::Integer=2)
    #a⁺a⁺a⁻a⁻
    return MPSKitModels.contract_onesite(MPSKitModels.contract_onesite(a_plus(elt; cutoff=cutoff), a_plus(elt; cutoff=cutoff)), MPSKitModels.contract_onesite(a_min(elt; cutoff=cutoff), a_min(elt; cutoff=cutoff)))
end

#The operators above are built without symmetry. Every term of the Lieb-Liniger Hamiltonian
#conserves the particle number, so they can be turned into U(1) symmetric operators, acting on the
#physical space U1Space(n => 1 for n in 0:cutoff), whose basis is the Fock basis in the same order
symmetric_operator(op::AbstractTensorMap, ::Type{Trivial}; cutoff::Integer=2) = op
function symmetric_operator(op::AbstractTensorMap, ::Type{U1Irrep}; cutoff::Integer=2)
    P = U1Space(n => 1 for n in 0:cutoff)
    k = numout(op)
    #TensorMap checks that the operator really is U(1) symmetric
    return TensorMap(convert(Array, op), P^k ← P^k)
end
symmetric_operator(op::AbstractTensorMap, symmetry::Type{<:Sector}; kwargs...) =
    throw(ArgumentError("only Trivial and U1Irrep symmetries are implemented, got $symmetry"))

################################################################################################
#Defining the Lieb-Liniger Hamiltonian for D6 and D8, output is an MPO
################################################################################################

const LL_DOC = """
MPO for the hamiltonian of the Lieb-Liniger model in the wavelet basis, for D6 (`LL_D6`) and D8 (`LL_D8`) wavelets.
The wavelet Lieb-Liniger model is defined as

H = 2³ʳ Σₙₘ (ψₙʳ)⁺ Kₙₘ ψₘʳ + 2²ʳc Σₙₘₗₖ Γ₄[m-n][l-n][k-n] (ψₙʳ)⁺ (ψₘʳ)⁺ ψₗʳ ψₖʳ - 2ʳμ Σₙ (ψₙʳ)⁺ ψₙʳ

i.e. 2ʳ times the continuum Hamiltonian restricted to the scaling functions at resolution r, so that the energy
per site is the continuum energy density.

ψₙʳ (ψₙʳ⁺) is a bosonic annihilation (creation) operator, on a truncated Hilbert space (maximum of "cutoff" bosons per site),
and where r is the resolution.

By default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with
`ComplexF64` entries of the tensors. The Hamiltonian conserves the particle number, and can also be built with `symmetry = U1Irrep`;
the physical space is then `U1Space(n => 1 for n in 0:cutoff)`, and a finite chain ground state
can be searched for directly in the sector with N particles.
"""

################################# N=6 Daubechies Wavelet ######################################
"""
    LL_D6(elt::Type{<:Number}=ComplexF64,
                        symmetry::Type{<:Sector}=Trivial,
                        lattice::AbstractLattice=InfiniteChain(1);
                        μ=2.0, c=1.0, r=0.0, cutoff=2)

$LL_DOC
"""
function LL_D6 end
function LL_D6(lattice::AbstractLattice; kwargs...)
    return LL_D6(ComplexF64, Trivial, lattice; kwargs...)
end
function LL_D6(symmetry::Type{<:Sector}, lattice::AbstractLattice=InfiniteChain(1);
                        kwargs...)
    return LL_D6(ComplexF64, symmetry, lattice; kwargs...)
end
function LL_D6(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return LL_D6(elt, Trivial, lattice; kwargs...)
end
function LL_D6(elt::Type{<:Number}=ComplexF64,
                        symmetry::Type{<:Sector}=Trivial,
                        lattice::AbstractLattice=InfiniteChain(1);
                        μ=2.0, c=1.0, r=0.0, cutoff=2)
    return LL_D6_hamiltonian(nothing, elt, symmetry, lattice; μ, c, r, cutoff)
end

#LL_D6, with the local operators `extra` (a SumOfLocalOperators, or nothing) added to the sum of
#terms before it is turned into an MPO
function LL_D6_hamiltonian(extra, elt::Type{<:Number}, symmetry::Type{<:Sector}, lattice::AbstractLattice;
                           μ, c, r, cutoff)
    #all terms in the Hamiltonian
    chem_pot = rmul!(a_number(elt; cutoff=cutoff), -(2^r)*μ)
    kinetic =  rmul!(a_plusmin(elt; cutoff=cutoff)+ a_minplus(elt; cutoff=cutoff), 2^(3*r))
    kinetic_ss = rmul!(a_number(elt; cutoff=cutoff), 2^(3*r))
    interaction = rmul!(a_ppmm(elt; cutoff=cutoff) +
                    a_mmpp(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺ ⊗ a⁺ ⊗ a⁻ ⊗ a⁻ + a⁻ ⊗ a⁻ ⊗ a⁺ ⊗ a⁺
    interaction_ss_2 = rmul!(a_ppmm_2mm(elt; cutoff=cutoff) +
                        a_ppmm_2pp(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺a⁺ ⊗ a⁻ ⊗ a⁻ + a⁻a⁻ ⊗ a⁺ ⊗ a⁺
    interaction_ss_2pm = rmul!(a_ppmm_2pm(elt; cutoff=cutoff) +
                            a_ppmm_2mp(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺ ⊗ a⁺a⁻ ⊗ a⁻ + a⁻ ⊗ a⁺a⁻ ⊗ a⁺
    interaction_ss_3 = rmul!(a_ppmm_3p(elt; cutoff=cutoff) +
                         a_ppmm_3m(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺a⁻a⁻ ⊗ a⁺ + a⁺a⁺a⁻ ⊗ a⁻
    interaction_ss_3pm = rmul!(a_ppmm_3pm(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺a⁻ ⊗ a⁺a⁻
    interaction_ss_3ppmm = rmul!(a_ppmm_3ppmm(elt; cutoff=cutoff) +
                            a_ppmm_3mmpp(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺a⁺ ⊗ a⁻a⁻ + a⁻a⁻ ⊗ a⁺a⁺
    interaction_ss_4 = rmul!(a_ppmm_4(elt; cutoff=cutoff), 2^(2*r)*c)
    #the operators are built without symmetry, and then made symmetric under `symmetry`
    ops = (chem_pot, kinetic, kinetic_ss, interaction, interaction_ss_2, interaction_ss_2pm, interaction_ss_3,
           interaction_ss_3pm, interaction_ss_3ppmm, interaction_ss_4)
    chem_pot, kinetic, kinetic_ss, interaction, interaction_ss_2, interaction_ss_2pm, interaction_ss_3,
        interaction_ss_3pm, interaction_ss_3ppmm, interaction_ss_4 = map(o -> symmetric_operator(o, symmetry; cutoff=cutoff), ops)

    #building the Hamiltonian
    return @mpoham begin
        #kinetic terms
        mpo_sum= sum(vertices(lattice)) do i
            return K6[1]*kinetic_ss{i} #kinetic term, a⁺a⁻ on same site
        end
        for seps=1:4
            term = sum(k_neighbours(seps, lattice)) do (i, j)
                return K6[1+seps]*kinetic{i,j} #kinetic term, a⁺ ⊗ a⁻ on different sites
            end
            mpo_sum += term
        end
        
        #interaction terms
        mpo_sum += sum(vertices(lattice)) do i #all operators on the same site
            return Γ4_D6[5,5,5]*interaction_ss_4{i}
        end
        for m=1:4
            term3 = sum(k_neighbours(m, lattice)) do (i, j) #two-site terms
                return 2*Γ4_D6[5,5,m+5]*interaction_ss_3{i,j} + 2*Γ4_D6[5,5,-m+5]*interaction_ss_3{j,i} +
                        4*Γ4_D6[m+5,5,m+5]*interaction_ss_3pm{i,j} +
                        Γ4_D6[5,m+5,m+5]*interaction_ss_3ppmm{i,j}
            end
            mpo_sum+=term3
            for n=1:4
                if n!=m
                    term2 = sum(threept_neighbours(m,n,lattice)) do (i,(j,k))  #three-site terms
                        return Γ4_D6[5,m+5,n+5]*interaction_ss_2{i,j,k} + 2*Γ4_D6[5,-m+5,n-m+5]*interaction_ss_2{j,i,k} +
                                4*Γ4_D6[m+5,m+5,n+5]*interaction_ss_2pm{i,j,k} +
                                2*Γ4_D6[m+5,5,n+5]*interaction_ss_2pm{j,i,k}
                    end
                    mpo_sum+=term2
                    for o=1:4
                        if o!=n && o!=m
                            term1 = sum(fourpt_neighbours(m,n,o, lattice)) do (i,(j,k,l))  #four-site terms
                                return 2*Γ4_D6[m+5,n+5,o+5]*interaction{i,j,k,l}
                            end
                            mpo_sum+=term1
                        end
                    end
                end
            end
        end

        #adding the chemical potential term and the extra terms, and outputting
        mpo_sum += sum(vertices(lattice)) do i
            return chem_pot{i}
        end
        isnothing(extra) ? mpo_sum : mpo_sum + extra
    end
end

################################# N=8 Daubechies Wavelet ######################################
"""
    LL_D8(elt::Type{<:Number}=ComplexF64,
                        symmetry::Type{<:Sector}=Trivial,
                        lattice::AbstractLattice=InfiniteChain(1);
                        μ=2.0, c=1.0, r=0.0, cutoff=2)

$LL_DOC
"""
function LL_D8 end
function LL_D8(lattice::AbstractLattice; kwargs...)
    return LL_D8(ComplexF64, Trivial, lattice; kwargs...)
end
function LL_D8(symmetry::Type{<:Sector}, lattice::AbstractLattice=InfiniteChain(1);
                        kwargs...)
    return LL_D8(ComplexF64, symmetry, lattice; kwargs...)
end
function LL_D8(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return LL_D8(elt, Trivial, lattice; kwargs...)
end
function LL_D8(elt::Type{<:Number}=ComplexF64,
                        symmetry::Type{<:Sector}=Trivial,
                        lattice::AbstractLattice=InfiniteChain(1);
                        μ=2.0, c=1.0, r=0.0, cutoff=2)
    #all terms in the Hamiltonian
    chem_pot = rmul!(a_number(elt; cutoff=cutoff), -(2^r)*μ)
    kinetic =  rmul!(a_plusmin(elt; cutoff=cutoff)+ a_minplus(elt; cutoff=cutoff), 2^(3*r))
    kinetic_ss = rmul!(a_number(elt; cutoff=cutoff), 2^(3*r))
    interaction = rmul!(a_ppmm(elt; cutoff=cutoff) +
                    a_mmpp(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺ ⊗ a⁺ ⊗ a⁻ ⊗ a⁻ + a⁻ ⊗ a⁻ ⊗ a⁺ ⊗ a⁺
    interaction_ss_2 = rmul!(a_ppmm_2mm(elt; cutoff=cutoff) + 
                        a_ppmm_2pp(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺a⁺ ⊗ a⁻ ⊗ a⁻ + a⁻a⁻ ⊗ a⁺ ⊗ a⁺
    interaction_ss_2pm = rmul!(a_ppmm_2pm(elt; cutoff=cutoff) +
                            a_ppmm_2mp(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺ ⊗ a⁺a⁻ ⊗ a⁻ + a⁻ ⊗ a⁺a⁻ ⊗ a⁺
    interaction_ss_3 = rmul!(a_ppmm_3p(elt; cutoff=cutoff) +
                         a_ppmm_3m(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺a⁻a⁻ ⊗ a⁺ + a⁺a⁺a⁻ ⊗ a⁻
    interaction_ss_3pm = rmul!(a_ppmm_3pm(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺a⁻ ⊗ a⁺a⁻
    interaction_ss_3ppmm = rmul!(a_ppmm_3ppmm(elt; cutoff=cutoff) +
                            a_ppmm_3mmpp(elt; cutoff=cutoff), 2^(2*r)*c) #a⁺a⁺ ⊗ a⁻a⁻ + a⁻a⁻ ⊗ a⁺a⁺
    interaction_ss_4 = rmul!(a_ppmm_4(elt; cutoff=cutoff), 2^(2*r)*c)
    #the operators are built without symmetry, and then made symmetric under `symmetry`
    ops = (chem_pot, kinetic, kinetic_ss, interaction, interaction_ss_2, interaction_ss_2pm, interaction_ss_3,
           interaction_ss_3pm, interaction_ss_3ppmm, interaction_ss_4)
    chem_pot, kinetic, kinetic_ss, interaction, interaction_ss_2, interaction_ss_2pm, interaction_ss_3,
        interaction_ss_3pm, interaction_ss_3ppmm, interaction_ss_4 = map(o -> symmetric_operator(o, symmetry; cutoff=cutoff), ops)

    #building the Hamiltonian
    return @mpoham begin 
        #kinetic terms
        mpo_sum= sum(vertices(lattice)) do i
            return K8[1]*kinetic_ss{i} #kinetic term, a⁺a⁻ on same site
        end
        for seps=1:6
            term = sum(k_neighbours(seps, lattice)) do (i, j)
                return K8[1+seps]*kinetic{i,j} #kinetic term, a⁺ ⊗ a⁻ on different sites
            end
            mpo_sum += term
        end
        
        #interaction terms
        mpo_sum += sum(vertices(lattice)) do i #all operators on the same site
            return Γ4_D8[7,7,7]*interaction_ss_4{i}
        end
        for m=1:6
            term3 = sum(k_neighbours(m, lattice)) do (i, j) #two-site terms
                return 2*Γ4_D8[7,7,m+7]*interaction_ss_3{i,j} + 2*Γ4_D8[7,7,-m+7]*interaction_ss_3{j,i} +
                        4*Γ4_D8[m+7,7,m+7]*interaction_ss_3pm{i,j} +
                        Γ4_D8[7,m+7,m+7]*interaction_ss_3ppmm{i,j}
            end
            mpo_sum+=term3
            for n=1:6
                if n!=m
                    term2 = sum(threept_neighbours(m,n,lattice)) do (i,(j,k))  #three-site terms
                        return Γ4_D8[7,m+7,n+7]*interaction_ss_2{i,j,k} + 2*Γ4_D8[7,-m+7,n-m+7]*interaction_ss_2{j,i,k} +
                                4*Γ4_D8[m+7,m+7,n+7]*interaction_ss_2pm{i,j,k} +
                                2*Γ4_D8[m+7,7,n+7]*interaction_ss_2pm{j,i,k}
                    end
                    mpo_sum+=term2
                    for o=1:6
                        if o!=n && o!=m
                            term1 = sum(fourpt_neighbours(m,n,o, lattice)) do (i,(j,k,l))  #four-site terms
                                return 2*Γ4_D8[m+7,n+7,o+7]*interaction{i,j,k,l} 
                            end
                            mpo_sum+=term1
                        end
                    end
                end
            end
        end
        
        #adding the chemical potential term, and outputting
        mpo_sum + sum(vertices(lattice)) do i
            return chem_pot{i}
        end
    end
end

################################################################################################
#Simple gaussian Hamiltonian for testing, using D6 and D8 wavelets
################################################################################################

const QUAD_DOC = """
MPO for the hamiltonian of a simple Gaussian model in the wavelet basis, for D6 (`quad_model`) and D8 (`quad_model_D8`)
wavelets. The wavelet model is defined as

H = 2³ʳ Σₙₘ (ψₙʳ)⁺ Kₙₘ ψₘʳ + 2ʳλ Σₙ [(ψₙʳ)⁺(ψₙʳ)⁺ + ψₙʳψₙʳ] + 2ʳμ Σₙ (ψₙʳ)⁺ ψₙʳ

ψₙʳ (ψₙʳ⁺) is a bosonic annihilation (creation) operator, on a truncated Hilbert space (maximum of "cutoff" bosons per site), 
and where r is the resolution.

By default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with 
`ComplexF64` entries of the tensors.
"""

"""
    quad_model(elt::Type{<:Number}=ComplexF64,
                        symmetry::Type{<:Sector}=Trivial,
                        lattice::AbstractLattice=InfiniteChain(1);
                        μ=1.0, λ=1.0, r=0.0, cutoff=2)

$QUAD_DOC
"""
function quad_model end
function quad_model(lattice::AbstractLattice; kwargs...)
    return quad_model(ComplexF64, Trivial, lattice; kwargs...)
end
function quad_model(symmetry::Type{<:Sector}, lattice::AbstractLattice=InfiniteChain(1);
                        kwargs...)
    return quad_model(ComplexF64, symmetry, lattice; kwargs...)
end
function quad_model(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return quad_model(elt, Trivial, lattice; kwargs...)
end
function quad_model(elt::Type{<:Number}=ComplexF64,
                            symmetry::Type{<:Sector}=Trivial,
                            lattice::AbstractLattice=InfiniteChain(1);
                            μ=1.0, λ=1.0, r=0.0, cutoff=2)
    #a⁺a⁺ + aa does not conserve the particle number
    symmetry === Trivial || throw(ArgumentError("only Trivial symmetry is implemented, got $symmetry"))
    #all terms in the Hamiltonian
    attraction = rmul!(a_number(elt, symmetry; cutoff=cutoff), (2^r)*μ)
    interaction = rmul!(a_pp(elt, symmetry; cutoff=cutoff)+a_mm(elt, symmetry; cutoff=cutoff), (2^r)*λ)
    kinetic =  rmul!(a_plusmin(elt, symmetry; cutoff=cutoff)+ a_minplus(elt, symmetry; cutoff=cutoff), 2^(3*r))
    kinetic_ss = rmul!(a_number(elt, symmetry; cutoff=cutoff), 2^(3*r))
    #building the Hamiltonian
    return @mpoham begin 
        #kinetic terms
        mpo_sum = sum(vertices(lattice)) do i
            return K6[1]*kinetic_ss{i} #kinetic term, a⁺a⁻ on same site
        end
        for seps=1:4
            term = sum(k_neighbours(seps, lattice)) do (i, j)
                return K6[1+seps]*kinetic{i,j} #kinetic term, a⁺ ⊗ a⁻ on different sites
            end
            mpo_sum += term
        end

        #adding the attraction and interaction terms, and outputting
        mpo_sum + sum(vertices(lattice)) do i
            return attraction{i}+interaction{i}
        end
    end
end

################################################################################################
#Simple gaussian Hamiltonian for testing, using D8 wavelets
################################################################################################
"""
    quad_model_D8(elt::Type{<:Number}=ComplexF64,
                        symmetry::Type{<:Sector}=Trivial,
                        lattice::AbstractLattice=InfiniteChain(1);
                        μ=1.0, λ=1.0, r=0.0, cutoff=2)

$QUAD_DOC
"""
function quad_model_D8 end
function quad_model_D8(lattice::AbstractLattice; kwargs...)
    return quad_model_D8(ComplexF64, Trivial, lattice; kwargs...)
end
function quad_model_D8(symmetry::Type{<:Sector}, lattice::AbstractLattice=InfiniteChain(1);
                        kwargs...)
    return quad_model_D8(ComplexF64, symmetry, lattice; kwargs...)
end
function quad_model_D8(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return quad_model_D8(elt, Trivial, lattice; kwargs...)
end
function quad_model_D8(elt::Type{<:Number}=ComplexF64,
                            symmetry::Type{<:Sector}=Trivial,
                            lattice::AbstractLattice=InfiniteChain(1);
                            μ=1.0, λ=1.0, r=0.0, cutoff=2)
    #a⁺a⁺ + aa does not conserve the particle number
    symmetry === Trivial || throw(ArgumentError("only Trivial symmetry is implemented, got $symmetry"))
    #all terms in the Hamiltonian
    attraction = rmul!(a_number(elt, symmetry; cutoff=cutoff), (2^r)*μ)
    interaction = rmul!(a_pp(elt, symmetry; cutoff=cutoff)+a_mm(elt, symmetry; cutoff=cutoff), (2^r)*λ)
    kinetic =  rmul!(a_plusmin(elt, symmetry; cutoff=cutoff)+ a_minplus(elt, symmetry; cutoff=cutoff), 2^(3*r))
    kinetic_ss = rmul!(a_number(elt, symmetry; cutoff=cutoff), 2^(3*r))
    #building the Hamiltonian
    return @mpoham begin 
        #kinetic terms
        mpo_sum = sum(vertices(lattice)) do i
            return K8[1]*kinetic_ss{i} #kinetic term, a⁺a⁻ on same site
        end
        for seps=1:6
            term = sum(k_neighbours(seps, lattice)) do (i, j)
                return K8[1+seps]*kinetic{i,j} #kinetic term, a⁺ ⊗ a⁻ on different sites
            end
            mpo_sum += term
        end

        #adding the attraction and interaction terms, and outputting
        mpo_sum + sum(vertices(lattice)) do i
            return attraction{i}+interaction{i}
        end
    end
end

################################################################################################
# Exact, small bond dimension MPO for the wavelet Lieb-Liniger model.
#
# `LL_D6` / `LL_D8` above build the Hamiltonian with `@mpoham`, which gives every single term of the sum
# its own private set of virtual levels.  For the N=6 wavelet model that is 134 distinct terms
# spread over windows of 5 sites, and the MPO comes out at bond dimension 562 even though the
# same operator fits in 45.
#
# Here the Hamiltonian is turned into a finite state automaton *before* the MPO is
# instantiated.  Writing a translation invariant, finite range Hamiltonian as
#
#       H = Σ_n Σ_s  γ[s] O_{s₀}(n) O_{s₁}(n+1) ⋯ O_{s_R}(n+R)
#
# over an alphabet of on-site operators `O`, the MPO in Jordan form
#
#       W = [ 1  C  D ;  0  A  B ;  0  0  1 ]
#
# is exactly such an automaton: `C` starts a term, `A` carries it along, `B` closes it, and `D`
# holds the purely on-site part.  The internal states are the *partially completed terms*.  A
# state reached after emitting the prefix u = (s₀, …, s_{ℓ-1}) has to remember nothing about u
# except the linear functional
#
#       row(u) : v ↦ γ[u·v]         (v = the suffix still to be emitted),
#
# because that is all the rest of the chain ever sees.  So the states of grade ℓ (= terms that
# started ℓ sites ago) can be taken to be a basis of
#
#       S_ℓ = span{ row(u) : u a prefix of length ℓ },
#
# which is what is done below: one orthonormal basis per grade, obtained from one SVD of the
# rows of that grade.  The prefixes of a given grade are massively linearly dependent
# (61 → 7 at grade 4 for N=6), and those dependencies are *exact*: the discarded singular
# values sit at 1e-17 of the largest, several orders below the smallest one kept, so nothing
# is approximated away.  `exact_mpo_data` checks the gap and warns rather than silently
# truncating anything real.
#
# Why this works, and why it is cheap: appending an operator o to a prefix acts on the rows as
# the shift (S_o f)[v] = f[o·v], and row(u·o) = S_o row(u).  So S_o maps S_ℓ into S_{ℓ+1} and
# `A` is read off as its matrix between the two bases — no linear system is ever solved, which
# is what keeps the whole thing accurate to round-off.  It also makes `A` block *bidiagonal*
# (grade ℓ only talks to grade ℓ+1), hence exactly nilpotent and strictly upper triangular,
# which is what MPSKit's environment recursion needs.
#
# Result for N=6 Daubechies wavelets, cutoff = 2:  bond dimension 562 → 45, with every term of
# H reproduced to 1e-13 absolute (5e-16 relative), i.e. to floating point round-off, and
# VUMPS about 9x faster.  The absolute lower bound, the rank of the full straddling matrix
# 𝓗 (all grades at once, rather than grade by grade), is 36 — but reaching it mixes states of
# different grades, which costs both the nilpotency of A and, because the singular values of
# 𝓗 span twelve orders of magnitude, roughly eight digits of accuracy.  Not worth 45 → 36.
#
# U(1) symmetry: every term conserves the particle number, so a prefix u that changes it by q
# can only be completed by suffixes that change it by -q.  The rows of a grade therefore split
# into charge sectors with disjoint supports, and taking one SVD per (grade, charge) instead of
# one per grade spans the same space, with the same number of states, but now every state has
# a definite charge.  That is exactly what a U(1) symmetric MPO needs: the state is a virtual
# level carrying that charge.
#
# The construction is generic: `exact_mpo_hamiltonian` takes any finite range term list.
# `LL_D6_minimal` / `LL_D8_minimal` feed it the wavelet Lieb-Liniger terms and are drop-in
# replacements for `LL_D6` / `LL_D8`.
################################################################################################

################################################################################################
# Local operator alphabet
################################################################################################

"""
    boson_alphabet(elt, cutoff, maxp, maxq)

Basis of normal ordered on-site operators `(a⁺)ᵖ aᑫ`, `0 ≤ p ≤ maxp`, `0 ≤ q ≤ maxq`, on a
Hilbert space truncated to `cutoff` bosons per site.  The identity is always the first entry,
which the MPO construction below relies on.  Returns `(labels, matrices, index)` where
`index[(p, q)]` is the position of `(a⁺)ᵖ aᑫ` in the list.

Every term of the wavelet Lieb-Liniger Hamiltonian is a tensor product of such operators: in
`ψₙ⁺ψₘ⁺ψₗψₖ` the creation operators all stand to the left of the annihilation operators, so
restricting a term to one site always leaves a normal ordered `(a⁺)ᵖ aᑫ`, and operators on
different sites commute.
"""
function boson_alphabet(elt::Type{<:Number}, cutoff::Integer, maxp::Integer, maxq::Integer)
    d = cutoff + 1
    a = zeros(elt, d, d)
    a⁺ = zeros(elt, d, d)
    for n in 1:cutoff                 # same convention as MPSKitModels.a_min / a_plus
        a[n, n + 1] = sqrt(n)
        a⁺[n + 1, n] = sqrt(n)
    end

    labels = NTuple{2, Int}[]
    matrices = Matrix{elt}[]
    for p in 0:maxp, q in 0:maxq
        push!(labels, (p, q))
        push!(matrices, a⁺^p * a^q)
    end
    # make sure the identity comes first
    i0 = findfirst(==((0, 0)), labels)
    labels[1], labels[i0] = labels[i0], labels[1]
    matrices[1], matrices[i0] = matrices[i0], matrices[1]

    index = Dict(l => i for (i, l) in enumerate(labels))
    return labels, matrices, index
end

################################################################################################
# Exact MPO for a translation invariant, finite range Hamiltonian
################################################################################################

"""
    exact_mpo_data(coeffs, nops, R; charges=zeros(Int, nops), rtol=1e-13, verbose=false)

Turn the window term list `coeffs` into the blocks `(A, B, C, D)` of a Jordan form MPO.

`coeffs` maps a window string `s = [s₀, s₁, …, s_R]` of alphabet indices (index 1 = identity,
`s₀ ≠ 1`, i.e. every term is anchored on its leftmost site) to its coefficient, and encodes

    H = Σ_n Σ_s coeffs[s] O_{s₀}(n) O_{s₁}(n+1) ⋯ O_{s_R}(n+R).

The internal states are grouped by grade ℓ = 1, …, R (how many sites a partially completed
term already spans) and, within a grade, are an orthonormal basis of the span of the rows

    row(u)[v] = coeffs[u·v],   |u| = ℓ,

obtained from one SVD per charge sector.  `charges[o]` is the change in particle number of the
alphabet operator `o`; every term must conserve it, and the states then have a definite charge
(see the comment above).  With the default, all charges zero, there is one SVD per grade.
Prefixes of the same grade are strongly linearly dependent and
only the dependent directions are dropped: `rtol` is the relative singular value below which a
direction counts as exactly zero.  It has to separate round-off (~1e-16) from the smallest
genuine direction; the gap is several orders wide here, and a warning is raised if it is not.

Returns a named tuple with `A[o, α, β]`, `B[o, α]`, `C[o, α]`, `D[o]` the coefficient of the
alphabet operator `o` in the corresponding block, `grades[α]` and `statecharges[α]` the grade
and the charge of internal state `α`, and `residual`, the largest error on any term of `H` (see [`mpo_data_residual`](@ref)).
"""
function exact_mpo_data(coeffs::Dict{Vector{Int}, T}, nops::Integer, R::Integer;
                        charges::AbstractVector{<:Integer} = zeros(Int, nops),
                        rtol::Real = 1.0e-13, verbose::Bool = false) where {T <: Real}
    charge(u) = sum(charges[o] for o in u; init = 0)
    # ---- split off the purely on-site terms -------------------------------------------------
    D = zeros(T, nops)
    multi = Vector{Int}[]
    for (s, γ) in coeffs
        iszero(γ) && continue
        @assert length(s) == R + 1 "window strings must have length R+1 = $(R + 1)"
        @assert s[1] != 1 "terms must be anchored on their leftmost site"
        @assert charge(s) == 0 "terms must conserve the particle number"
        if all(==(1), @view s[2:end])
            D[s[1]] += γ
        else
            push!(multi, s)
        end
    end

    # ---- the matrix 𝓗[u, v] = γ[u·v] of everything that crosses one bond -------------------
    # rows u = prefixes (what the term does on the ℓ sites left of the bond; trailing
    # identities are kept, they say how far left the term started)
    # columns v = suffixes with trailing identities stripped (those are the same operator)
    rowindex = Dict{Vector{Int}, Int}()
    rowkeys = Vector{Int}[]
    colindex = Dict{Vector{Int}, Int}()
    colkeys = Vector{Int}[]
    entries = Tuple{Int, Int, T}[]
    for s in multi
        γ = coeffs[s]
        stop = findlast(!=(1), s)::Int
        for ℓ in 1:(stop - 1)                       # bond after the ℓ-th site of the window
            u = s[1:ℓ]
            v = s[(ℓ + 1):stop]
            iu = get!(rowindex, u) do
                push!(rowkeys, u)
                length(rowkeys)
            end
            iv = get!(colindex, v) do
                push!(colkeys, v)
                length(colkeys)
            end
            push!(entries, (iu, iv, γ))
        end
    end

    nrow, ncol = length(rowkeys), length(colkeys)
    𝓗 = zeros(T, nrow, ncol)
    for (iu, iv, γ) in entries
        𝓗[iu, iv] += γ
    end

    # ---- one orthonormal basis per grade, made of one basis per charge sector ----------------
    # rows of different charges have disjoint supports, so the stacked bases are orthonormal.
    # The SVD leaves round-off (~1e-16) outside the support of a sector, which is set to zero, so
    # that A, B and C exactly respect the charges
    colcharges = charge.(colkeys)
    rowsof = [findall(u -> length(u) == ℓ, rowkeys) for ℓ in 1:R]
    E = Vector{Matrix{T}}(undef, R)                 # E[ℓ] = n_ℓ × ncol, orthonormal rows
    Q = Vector{Vector{Int}}(undef, R)               # Q[ℓ][α] = charge of state α of grade ℓ
    kept = fill(zero(T), R)                         # σ_n / σ_1, per grade
    dropped = fill(zero(T), R)                      # σ_{n+1} / σ_1, per grade
    for ℓ in 1:R
        q = charge.(rowkeys[rowsof[ℓ]])
        sectors = sort(unique(q))
        Fs = [svd(𝓗[rowsof[ℓ][q .== qs], :]) for qs in sectors]
        σ1 = maximum(F.S[1] for F in Fs)            # the largest singular value of the grade
        ns = [count(>(rtol * σ1), F.S) for F in Fs]
        E[ℓ] = reduce(vcat, [F.Vt[1:n, :] .* (colcharges .== -qs)' for (F, n, qs) in zip(Fs, ns, sectors)])
        Q[ℓ] = reduce(vcat, [fill(qs, n) for (qs, n) in zip(sectors, ns)])
        kept[ℓ] = minimum(F.S[n] for (F, n) in zip(Fs, ns) if n > 0) / σ1
        dropped[ℓ] = maximum(n < length(F.S) ? F.S[n + 1] : zero(T) for (F, n) in zip(Fs, ns)) / σ1
    end
    nstates = size.(E, 1)
    off = cumsum([0; nstates])                      # state α of grade ℓ ↦ off[ℓ] + α
    r = last(off)
    grades = reduce(vcat, [fill(ℓ, nstates[ℓ]) for ℓ in 1:R]; init = Int[])
    statecharges = reduce(vcat, Q; init = Int[])

    # the directions that were dropped have to be zero to machine precision, otherwise `rtol`
    # is throwing away a real piece of the Hamiltonian rather than a linear dependency
    maximum(dropped) > 1.0e-14 &&
        @warn "discarded directions are above round-off, so the MPO is *not* exact; lower \
               `rtol` (currently $rtol)" largest_discarded = maximum(dropped) smallest_kept = minimum(kept)

    # ---- blocks of the MPO ------------------------------------------------------------------
    # state α of grade ℓ ↔ the functional E[ℓ][α, :] on the suffixes still to be emitted
    A = zeros(T, nops, r, r)
    B = zeros(T, nops, r)
    C = zeros(T, nops, r)
    for o in 1:nops
        # A: the shift v ↦ o·v, read between the basis of grade ℓ and that of grade ℓ+1.
        # It maps S_ℓ into S_{ℓ+1} because S_o row(u) = row(u·o), so nothing is solved for.
        for ℓ in 1:(R - 1)
            shifted = zeros(T, nstates[ℓ], ncol)
            for (iv, v) in enumerate(colkeys)
                iov = get(colindex, vcat(o, v), 0)
                iov == 0 && continue                # o·v never completes a term
                @views shifted[:, iv] .= E[ℓ][:, iov]
            end
            A[o, (off[ℓ] + 1):off[ℓ + 1], (off[ℓ + 1] + 1):off[ℓ + 2]] = shifted * E[ℓ + 1]'
        end
        # B: the term stops here, i.e. evaluate the functional on the one-site suffix [o]
        io = get(colindex, [o], 0)
        if io != 0
            for ℓ in 1:R
                @views B[o, (off[ℓ] + 1):off[ℓ + 1]] .= E[ℓ][:, io]
            end
        end
        # C: the term starts here, i.e. the coordinates of row([o]) in the grade 1 basis
        iu = get(rowindex, [o], 0)
        iu != 0 && (@views C[o, 1:off[2]] .= E[1] * 𝓗[iu, :])
    end

    residual = mpo_data_residual(A, B, C, D, coeffs, R, 𝓗, E, off, rowsof, rowkeys, rowindex)

    if verbose
        @info "exact MPO" range = R terms = length(coeffs) bond_dimension = r + 2 prefixes_per_grade = length.(rowsof) states_per_grade = nstates smallest_kept = minimum(kept) largest_discarded = maximum(dropped) residual
    end

    return (; A, B, C, D, grades, statecharges, residual, nstates, kept, dropped)
end

"""
    mpo_data_residual(A, B, C, D, coeffs, R, 𝓗, E, off, rowsof, rowkeys, rowindex)

Largest error made by the automaton `(A, B, C, D)` on the Hamiltonian it represents.  Three
things are checked:

  * every row of `𝓗` lies in the span of the basis of its grade, so that no part of `H` was
    lost when the dependent prefixes were dropped;
  * the realization identity `P[u·o, :] = P[u, :] Aᵒ` holds for every prefix `u` and every
    alphabet operator `o`, where `P[u, :]` are the coordinates of `row(u)`.  In particular
    `P[u, :] Aᵒ = 0` once `u` spans the full range, so the automaton cannot run past it;
  * every term of `coeffs` is reproduced by `C Aᵒ¹ ⋯ Aᵒᵏ⁻¹ B` (resp. by `D` on a single site).

The first two also rule out *spurious* terms: if a string is not a prefix of anything its row
vanishes, so by induction the state is zero and the automaton emits nothing.  A residual at
the 1e-13 level therefore means the MPO is the Hamiltonian, up to floating point round-off.
"""
function mpo_data_residual(A, B, C, D, coeffs::Dict{Vector{Int}, T}, R::Integer,
                           𝓗, E, off, rowsof, rowkeys, rowindex) where {T <: Real}
    nops = length(D)
    r = size(B, 2)

    # coordinates of every row in the basis of its grade
    P = zeros(T, size(𝓗, 1), r)
    res = zero(T)
    for ℓ in 1:R
        Hℓ = @view 𝓗[rowsof[ℓ], :]
        Pℓ = Hℓ * E[ℓ]'
        res = max(res, norm(Hℓ - Pℓ * E[ℓ], Inf))
        P[rowsof[ℓ], (off[ℓ] + 1):off[ℓ + 1]] = Pℓ
    end

    for o in 1:nops
        shifted = zeros(T, size(P)...)
        for (iu, u) in enumerate(rowkeys)
            length(u) == R && continue              # no room left in the window
            iuo = get(rowindex, vcat(u, o), 0)
            iuo == 0 && continue
            @views shifted[iu, :] .= P[iuo, :]
        end
        res = max(res, norm(shifted - P * (@view A[o, :, :]), Inf))
    end

    for (s, γ) in coeffs
        stop = findlast(!=(1), s)::Int
        if stop == 1
            res = max(res, abs(D[s[1]] - γ))
        else
            x = C[s[1], :]
            for k in 2:(stop - 1)
                x = transpose(@view A[s[k], :, :]) * x
            end
            res = max(res, abs(dot(x, @view B[s[stop], :]) - γ))
        end
    end

    return res
end

"""
    exact_mpo_hamiltonian(coeffs, ops, R, lattice; elt=ComplexF64, symmetry=Trivial,
                          rtol=1e-13, verbose=false)

Assemble the `MPOHamiltonian` of the finite range, translation invariant Hamiltonian described
by the window term list `coeffs` over the operator alphabet `ops` (a vector of matrices in the
Fock basis, whose first entry is the identity).  Works on `InfiniteChain(1)` and on a
`FiniteChain`, where the automaton is simply entered at the first site and left at the last one,
so that exactly the terms fitting inside the chain survive.  With `symmetry = U1Irrep`, every
operator of the alphabet must change the particle number by a definite amount, and every term
must conserve it.
"""
function exact_mpo_hamiltonian(coeffs::Dict{Vector{Int}, T}, ops::Vector{<:Matrix},
                               R::Integer, lattice::AbstractLattice;
                               elt::Type{<:Number} = ComplexF64,
                               symmetry::Type{<:Sector} = Trivial, rtol::Real = 1.0e-13,
                               verbose::Bool = false) where {T <: Real}
    W = exact_mpo_matrix(coeffs, ops, R; elt = elt, symmetry = symmetry, rtol = rtol,
                         verbose = verbose)

    if lattice isa InfiniteChain
        length(lattice) == 1 ||
            throw(ArgumentError("only a one-site unit cell is supported, got $lattice"))
        return InfiniteMPOHamiltonian([W])
    elseif lattice isa FiniteChain
        L = length(lattice)
        L > R || throw(ArgumentError("chain of length $L is shorter than the range $R"))
        return finite_mpo_hamiltonian(fill(W, L))
    else
        throw(ArgumentError("unsupported lattice $lattice"))
    end
end

"""
    finite_mpo_hamiltonian(Ws)

`FiniteMPOHamiltonian` from the Jordan form MPO matrices `Ws[i]` of every site, keeping only the
first row at the first site and the last column at the last one.  The rows of levels that cannot
be reached from the left end of the chain, i.e. terms that would have started before the first
site, are dropped: they never contribute, and MPSKit, which deduces the space of a level from
the entries leading to it, would give them a trivial space, inconsistent with the charge of the
entries leaving them.
"""
function finite_mpo_hamiltonian(Ws::AbstractVector{<:AbstractMatrix})
    L = length(Ws)
    Ws = [i == 1 ? W[1:1, :] : i == L ? W[:, end:end] : copy(W) for (i, W) in enumerate(Ws)]
    reached = trues(1)
    for W in Ws
        W[.!reached, :] .= missing
        reached = [any(!ismissing, @view W[:, j]) for j in axes(W, 2)]
    end
    return FiniteMPOHamiltonian(Ws)
end

#the change in particle number of an on-site operator, given as a matrix in the Fock basis
function operator_charge(mat::AbstractMatrix)
    Δ = unique(i - j for (i, j) in Tuple.(findall(!iszero, mat)))
    length(Δ) <= 1 ||
        throw(ArgumentError("the operator does not change the particle number by a definite amount"))
    return isempty(Δ) ? 0 : only(Δ)
end

#an on-site operator, given as a matrix in the Fock basis, as an entry of an MPO matrix, between
#virtual levels of charges qin and qout (only used with U(1) symmetry)
function mpo_entry(mat::AbstractMatrix, elt::Type{<:Number}, ::Type{Trivial} = Trivial,
                   qin::Integer = 0, qout::Integer = 0)
    pspace = ComplexSpace(size(mat, 1))
    return MPSKit.add_util_leg(TensorMap(convert(Matrix{elt}, mat), pspace ← pspace))
end
function mpo_entry(mat::AbstractMatrix, elt::Type{<:Number}, ::Type{U1Irrep}, qin::Integer,
                   qout::Integer)
    d = size(mat, 1)
    pspace = U1Space(n => 1 for n in 0:(d - 1))
    #TensorMap checks that the operator changes the charge by qout - qin
    return TensorMap(reshape(convert(Matrix{elt}, mat), 1, d, d, 1),
                     U1Space(qin => 1) ⊗ pspace ← pspace ⊗ U1Space(qout => 1))
end
mpo_entry(mat::AbstractMatrix, elt::Type{<:Number}, symmetry::Type{<:Sector}, args...) =
    throw(ArgumentError("only Trivial and U1Irrep symmetries are implemented, got $symmetry"))

"""
    exact_mpo_matrix(coeffs, ops, R; elt=ComplexF64, symmetry=Trivial, rtol=1e-13, verbose=false)

The Jordan form MPO matrix `W` built by [`exact_mpo_hamiltonian`](@ref), whose entries are
`missing`, numbers (multiples of the identity) or MPO tensors.  The first and last virtual
levels are the "not started" and "done" states, both of charge 0.
"""
function exact_mpo_matrix(coeffs::Dict{Vector{Int}, T}, ops::Vector{<:Matrix}, R::Integer;
                          elt::Type{<:Number} = ComplexF64, symmetry::Type{<:Sector} = Trivial,
                          rtol::Real = 1.0e-13, verbose::Bool = false) where {T <: Real}
    nops = length(ops)
    charges = symmetry === Trivial ? zeros(Int, nops) : operator_charge.(ops)
    (; A, B, C, D, grades, statecharges) = exact_mpo_data(coeffs, nops, R; charges = charges,
                                                          rtol = rtol, verbose = verbose)
    r = length(grades)
    q = [0; statecharges; 0]                        # charges of all levels

    d = size(first(ops), 1)
    TO = typeof(mpo_entry(zeros(elt, d, d), elt, symmetry, 0, 0))

    # γ ↦ MPO matrix entry: `missing` when it vanishes, a number when it is a multiple of the
    # identity (MPSKit then stores a BraidingTensor rather than a dense tensor), a tensor else
    function entry(γ::AbstractVector, qin, qout)
        iszero(norm(γ)) && return missing
        if iszero(norm(@view γ[2:end]))
            return convert(elt, γ[1])
        end
        mat = zeros(elt, d, d)
        for o in 1:nops
            iszero(γ[o]) || axpy!(γ[o], ops[o], mat)
        end
        return mpo_entry(mat, elt, symmetry, qin, qout)
    end

    W = Matrix{Union{Missing, elt, TO}}(missing, r + 2, r + 2)
    W[1, 1] = one(elt)
    W[end, end] = one(elt)
    W[1, end] = entry(D, 0, 0)
    for α in 1:r
        W[1, 1 + α] = entry((@view C[:, α]), 0, q[1 + α])
        W[1 + α, end] = entry((@view B[:, α]), q[1 + α], 0)
        for β in 1:r
            W[1 + α, 1 + β] = entry((@view A[:, α, β]), q[1 + α], q[1 + β])
        end
    end
    @assert all(ismissing, W[i, j] for i in 1:(r + 2), j in 1:(r + 2) if j < i) "the MPO must \
        be upper triangular for MPSKit's environments to be correct"

    return W
end

################################################################################################
# Wavelet Lieb-Liniger term list
################################################################################################

"""
    LL_terms(K, Γ4, index; μ=2.0, c=1.0, r=0.0)

Window term list of the wavelet Lieb-Liniger Hamiltonian

  H = 2³ʳ Σₙₘ (ψₙʳ)⁺ Kₙₘ ψₘʳ + 2²ʳc Σₙₘₗₖ Γ₄[m-n][l-n][k-n] (ψₙʳ)⁺ (ψₘʳ)⁺ ψₗʳ ψₖʳ
      - 2ʳμ Σₙ (ψₙʳ)⁺ ψₙʳ

i.e. the Hamiltonian of `LL_D6` / `LL_D8`, with `K[1+s] = Kₙ,ₙ₊ₛ` and `Γ4` indexed by offsets `-w:w`
(so `Γ4[a+w+1, b+w+1, c+w+1]`).  The quartic sum is taken over all offsets directly rather
than case by case; `Γ₄` is fully symmetric under permutations of the four wavelet positions,
which is what makes the two enumerations agree.

Each term is anchored on its leftmost site and stored as a string of alphabet indices.  Also
returns the interaction range `R`.
"""
function LL_terms(K::AbstractVector, Γ4::AbstractArray{<:Real, 3},
                  index::Dict{NTuple{2, Int}, Int}; μ = 2.0, c = 1.0, r = 0.0)
    w = (size(Γ4, 1) - 1) ÷ 2

    # the four wavelets must pairwise overlap, so every nonzero Γ₄ fits in a window of w+1
    # sites; the hopping range is fixed by K
    R = length(K) - 1
    for I in CartesianIndices(Γ4)
        iszero(Γ4[I]) && continue
        a, b, c′ = Tuple(I) .- (w + 1)
        R = max(R, maximum((0, a, b, c′)) - minimum((0, a, b, c′)))
    end

    coeffs = Dict{Vector{Int}, Float64}()
    add!(s, γ) = (coeffs[s] = get(coeffs, s, 0.0) + γ)
    string_of(p, q) = [index[(p[i], q[i])] for i in 1:(R + 1)]

    # --- chemical potential and on-site kinetic term -----------------------------------------
    p = zeros(Int, R + 1)
    q = zeros(Int, R + 1)
    p[1] = 1
    q[1] = 1
    add!(string_of(p, q), 2^(3r) * K[1] - 2^r * μ)

    # --- hopping:  2³ʳ K[1+s] (a⁺ₙ aₙ₊ₛ + aₙ a⁺ₙ₊ₛ) -----------------------------------------
    for s in 1:(length(K) - 1)
        iszero(K[1 + s]) && continue
        for (p0, q0) in ((1, 0), (0, 1))
            p = zeros(Int, R + 1)
            q = zeros(Int, R + 1)
            p[1] = p0
            q[1] = q0
            p[1 + s] = q0
            q[1 + s] = p0
            add!(string_of(p, q), 2^(3r) * K[1 + s])
        end
    end

    # --- interaction:  2²ʳc Γ₄[a,b,c] ψₙ⁺ ψₙ₊ₐ⁺ ψₙ₊ᵦ ψₙ₊_c -----------------------------------
    for I in CartesianIndices(Γ4)
        γ = Γ4[I]
        iszero(γ) && continue
        a, b, c′ = Tuple(I) .- (w + 1)
        sites = (0, a, b, c′)
        shift = minimum(sites)
        maximum(sites) - shift <= R || error("Γ₄ support exceeds the range R = $R")
        p = zeros(Int, R + 1)
        q = zeros(Int, R + 1)
        p[1 - shift] += 1                # ψₙ⁺
        p[1 + a - shift] += 1            # ψₙ₊ₐ⁺
        q[1 + b - shift] += 1            # ψₙ₊ᵦ
        q[1 + c′ - shift] += 1           # ψₙ₊_c
        add!(string_of(p, q), 2^(2r) * c * γ)
    end

    filter!(kv -> !iszero(kv.second), coeffs)
    return coeffs, R
end

################################################################################################
# Defining the Lieb-Liniger Hamiltonian for D6 and D8, output is an exact, compact MPO
################################################################################################

const LL_MINIMAL_DOC = """
MPO for the hamiltonian of the Lieb-Liniger model in the wavelet basis, for D6 and D8 wavelets
respectively.  Drop-in replacements for `LL_D6` / `LL_D8`: same operator, same
arguments, but bond dimension 45 instead of 562 (D6) and 84 instead of 2466 (D8), at
`cutoff = 2`.  Nothing is approximated — the MPO reproduces every term of `H` to round-off
(~1e-13 absolute, 5e-16 relative); see [`exact_mpo_data`](@ref) for why, and pass
`verbose=true` to have the construction report its own residual.

The wavelet Lieb-Liniger model is defined as

H = 2³ʳ Σₙₘ (ψₙʳ)⁺ Kₙₘ ψₘʳ + 2²ʳc Σₙₘₗₖ Γ₄[m-n][l-n][k-n] (ψₙʳ)⁺ (ψₘʳ)⁺ ψₗʳ ψₖʳ - 2ʳμ Σₙ (ψₙʳ)⁺ ψₙʳ

i.e. 2ʳ times the continuum Hamiltonian restricted to the scaling functions at resolution r, so that the energy
per site is the continuum energy density.

where ψₙʳ (ψₙʳ⁺) is a bosonic annihilation
(creation) operator on a Hilbert space truncated to `cutoff` bosons per site, and r is the
resolution.

By default, the model is defined on an infinite chain with unit lattice spacing, without any
symmetries and with `ComplexF64` entries of the tensors.  The Hamiltonian conserves the particle number, and can also be built with `symmetry = U1Irrep`;
the physical space is then `U1Space(n => 1 for n in 0:cutoff)`, and a finite chain ground state
can be searched for directly in the sector with N particles.
"""

"""
    LL_D6_minimal(elt::Type{<:Number}=ComplexF64,
                symmetry::Type{<:Sector}=Trivial,
                lattice::AbstractLattice=InfiniteChain(1);
                μ=2.0, c=1.0, r=0.0, cutoff=2, rtol=1e-13, verbose=false)

$LL_MINIMAL_DOC
"""
function LL_D6_minimal end
function LL_D6_minimal(lattice::AbstractLattice; kwargs...)
    return LL_D6_minimal(ComplexF64, Trivial, lattice; kwargs...)
end
function LL_D6_minimal(symmetry::Type{<:Sector}, lattice::AbstractLattice = InfiniteChain(1);
                     kwargs...)
    return LL_D6_minimal(ComplexF64, symmetry, lattice; kwargs...)
end
function LL_D6_minimal(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return LL_D6_minimal(elt, Trivial, lattice; kwargs...)
end
function LL_D6_minimal(elt::Type{<:Number} = ComplexF64,
                     symmetry::Type{<:Sector} = Trivial,
                     lattice::AbstractLattice = InfiniteChain(1);
                     kwargs...)
    return LL_wavelet_minimal(K6, Γ4_D6, elt, symmetry, lattice; kwargs...)
end

"""
    LL_D8_minimal(elt::Type{<:Number}=ComplexF64,
                symmetry::Type{<:Sector}=Trivial,
                lattice::AbstractLattice=InfiniteChain(1);
                μ=2.0, c=1.0, r=0.0, cutoff=2, rtol=1e-13, verbose=false)

$LL_MINIMAL_DOC
"""
function LL_D8_minimal end
function LL_D8_minimal(lattice::AbstractLattice; kwargs...)
    return LL_D8_minimal(ComplexF64, Trivial, lattice; kwargs...)
end
function LL_D8_minimal(symmetry::Type{<:Sector}, lattice::AbstractLattice = InfiniteChain(1);
                     kwargs...)
    return LL_D8_minimal(ComplexF64, symmetry, lattice; kwargs...)
end
function LL_D8_minimal(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return LL_D8_minimal(elt, Trivial, lattice; kwargs...)
end
function LL_D8_minimal(elt::Type{<:Number} = ComplexF64,
                     symmetry::Type{<:Sector} = Trivial,
                     lattice::AbstractLattice = InfiniteChain(1);
                     kwargs...)
    return LL_wavelet_minimal(K8, Γ4_D8, elt, symmetry, lattice; kwargs...)
end

"""
    LL_wavelet_minimal(K, Γ4, elt, symmetry, lattice; μ, c, r, cutoff, rtol, verbose)

Exact MPO of the wavelet Lieb-Liniger model for an arbitrary wavelet order, given the kinetic
coefficients `K` and the four-point overlaps `Γ4`.
"""
function LL_wavelet_minimal(K::AbstractVector, Γ4::AbstractArray{<:Real, 3},
                          elt::Type{<:Number} = ComplexF64,
                          symmetry::Type{<:Sector} = Trivial,
                          lattice::AbstractLattice = InfiniteChain(1);
                          μ = 2.0, c = 1.0, r = 0.0, cutoff = 2, rtol::Real = 1.0e-13,
                          verbose::Bool = false)
    symmetry in (Trivial, U1Irrep) ||
        throw(ArgumentError("only Trivial and U1Irrep symmetries are implemented, got $symmetry"))
    # at most two creation and two annihilation operators of a quartic term meet on one site
    _, ops, index = boson_alphabet(elt, cutoff, 2, 2)
    coeffs, R = LL_terms(K, Γ4, index; μ = μ, c = c, r = r)
    return exact_mpo_hamiltonian(coeffs, ops, R, lattice; elt = elt, symmetry = symmetry,
                                 rtol = rtol, verbose = verbose)
end

################################################################################################
# Lieb-Liniger model in a harmonic trap, for D6 wavelets, on a finite chain
#
# The trap adds ∫ V(x) ψ⁺(x)ψ(x) dx, V(x) = ½ω²(x - x₀)², to the Hamiltonian.  In the basis of
# scaling functions sₙʳ(x) = 2^{r/2} s(2ʳx - n) this is the quadratic term Σₙₘ Vₙₘ ψₙ⁺ψₘ with
#
#       Vₙ,ₙ₊ₖ = ∫ V(x) sₙʳ(x) sₙ₊ₖʳ(x) dx = ½ω² 2⁻²ʳ [(n - y₀)² δₖ₀ + 2(n - y₀) M₁(k) + M₂(k)],
#
# with y₀ = 2ʳx₀ and M_q(k) = ∫ x^q s(x) s(x - k) dx, which follow exactly from the refinement
# equation.  The trap breaks translation invariance, so these models are only defined on a
# FiniteChain, whose site i holds the mode n = i - 1.
################################################################################################

"""
    scaling_moments(h, p)

Moments `M[q+1, k+w+1] = ∫ x^q s(x) s(x-k) dx`, for `q = 0, …, p` and all overlapping shifts
`k = -w:w` (`w = length(h) - 2`), of the scaling function with filter `h`, i.e.
`s(x) = √2 Σⱼ hⱼ s(2x - j)`.

They are exact up to round-off: the refinement equation turns each moment into a linear system,
M_q(k) = 2⁻ᵠ Σᵢⱼ hᵢhⱼ Σ_{q′≤q} binomial(q, q′) i^{q-q′} M_{q′}(2k + j - i), starting from the
orthonormality M₀(k) = δₖ₀.
"""
function scaling_moments(h::AbstractVector, p::Integer)
    w = length(h) - 2
    nk = 2w + 1
    M = zeros(p + 1, nk)
    M[1, w + 1] = 1
    for q in 1:p
        T = zeros(nk, nk)
        rhs = zeros(nk)
        for (ik, k) in enumerate(-w:w), (i, hi) in enumerate(h), (j, hj) in enumerate(h)
            m = 2k + (j - 1) - (i - 1)              # filter indices start at 0
            abs(m) <= w || continue
            T[ik, m + w + 1] += hi * hj / 2^q
            for q′ in 0:(q - 1)
                rhs[ik] += hi * hj / 2^q * binomial(q, q′) * (i - 1)^(q - q′) * M[q′ + 1, m + w + 1]
            end
        end
        M[q + 1, :] = (I - T) \ rhs
    end
    return M
end

"""
    trap_matrix(L; ω=1.0, x0=nothing, r=0.0, h=h6)

The matrix `Vₙₘ = ∫ V(x) sₙʳ(x) sₘʳ(x) dx` of the harmonic potential `V(x) = ½ω²(x - x0)²`
between the `L` scaling functions `n = 0, …, L-1` at resolution `r`.  By default the trap is
centred on the chain, halfway between the centres of mass of the first and the last mode.
"""
function trap_matrix(L::Integer; ω=1.0, x0=nothing, r=0.0, h::AbstractVector=h6)
    M = scaling_moments(h, 2)
    w = length(h) - 2
    y0 = isnothing(x0) ? (L - 1) / 2 + M[2, w + 1] : 2^r * x0
    V = zeros(L, L)
    for n in 0:(L - 1), m in max(0, n - w):min(L - 1, n + w)
        k = m - n
        V[n + 1, m + 1] = ω^2 / 2 * 2.0^(-2r) *((n - y0)^2 * M[1, k + w + 1] + 2 * (n - y0) * M[2, k + w + 1] + M[3, k + w + 1])
    end
    return V
end

check_trap_lattice(lattice) = lattice isa FiniteChain ||
    throw(ArgumentError("the harmonic trap breaks translation invariance, so it needs a FiniteChain, got $lattice"))

"""
    harmonic_trap(V, elt, symmetry, lattice::FiniteChain; cutoff=2)

The one-body term `Σₙₘ Vₙₘ ψₙ⁺ψₘ`, for a symmetric matrix `V` with `Vₙₘ = 0` for `|n - m| > 4`,
as a sum of local operators (one per term, as in `@mpoham`).
"""
function harmonic_trap(V::AbstractMatrix, elt::Type{<:Number}, symmetry::Type{<:Sector},
                       lattice::FiniteChain; cutoff=2)
    w = length(h6) - 2
    number = symmetric_operator(a_number(elt; cutoff=cutoff), symmetry; cutoff=cutoff)
    hopping = symmetric_operator(a_plusmin(elt; cutoff=cutoff) + a_minplus(elt; cutoff=cutoff), symmetry;
                                 cutoff=cutoff) #a⁺ ⊗ a⁻ + a⁻ ⊗ a⁺
    site(i) = i.coordinates[1]
    terms = sum(vertices(lattice)) do i
        return V[site(i), site(i)]*MPSKitModels.LocalOperator(number, i)
    end
    for k=1:w
        terms += sum(k_neighbours(k, lattice)) do (i, j)
            return V[site(i), site(j)]*MPSKitModels.LocalOperator(hopping, i, j)
        end
    end
    return terms
end

"""
    onebody_mpo_hamiltonian(W, V, entry)

Finite chain MPO of `H_W + Σₙₘ Vₙₘ a⁺ₙ aₘ`, where `H_W` is the translation invariant Hamiltonian
with Jordan form MPO matrix `W` (see [`exact_mpo_matrix`](@ref)) and `V` is a symmetric, banded
matrix with `Vₙₘ = 0` for `|n - m| > w`.  `entry((p, q), qin, qout)` is the MPO entry of
`(a⁺)ᵖ aᑫ` between virtual levels of charges `qin` and `qout`.

The one-body term gets 2w virtual levels of its own: "a⁺ was emitted d sites ago" and "a was
emitted d sites ago", for d = 1, …, w.  They are opened with a⁺ or a, carried along with the
identity, and closed at site m with V[m-d, m] a or V[m-d, m] a⁺; the diagonal goes into the on-site
block.  Nothing is compressed, so the bond dimension is that of `W` plus 2w.  The new levels have
charge +1 (after a⁺) and -1 (after a).
"""
function onebody_mpo_hamiltonian(W::AbstractMatrix, V::AbstractMatrix, entry)
    L = size(V, 1)
    w = maximum(abs(i - j) for i in 1:L, j in 1:L if !iszero(V[i, j]); init = 0)
    nW = size(W, 1)                        # 1 = not started, nW = done
    n = nW + 2w
    P(d) = nW - 1 + d                      # a⁺ was emitted d sites ago
    Q(d) = nW - 1 + w + d                  # a was emitted d sites ago
    id = W[1, 1]                           # the identity, stored as a number
    num = entry((1, 1), 0, 0)

    Ws = map(1:L) do m
        Wm = similar(W, n, n)
        fill!(Wm, missing)
        Wm[1:(nW - 1), 1:(nW - 1)] = W[1:(nW - 1), 1:(nW - 1)]
        Wm[1:(nW - 1), n] = W[1:(nW - 1), nW]
        Wm[n, n] = W[nW, nW]
        # the on-site block of W is a tensor or missing, never a multiple of the identity, since
        # exact_mpo_data rejects terms that start with the identity
        D = W[1, nW]
        Wm[1, n] = ismissing(D) ? V[m, m] * num : D + V[m, m] * num
        Wm[1, P(1)] = entry((1, 0), 0, 1)
        Wm[1, Q(1)] = entry((0, 1), 0, -1)
        for d in 1:w
            if d < w
                Wm[P(d), P(d + 1)] = id
                Wm[Q(d), Q(d + 1)] = id
            end
            if m - d >= 1
                Wm[P(d), n] = V[m - d, m] * entry((0, 1), 1, 0)
                Wm[Q(d), n] = V[m - d, m] * entry((1, 0), -1, 0)
            end
        end
        return Wm
    end
    return finite_mpo_hamiltonian(Ws)
end

const LL_TRAP_DOC = """
MPO for the hamiltonian of the Lieb-Liniger model in a harmonic trap, in the basis of D6 wavelets,
on a finite chain.  The Hamiltonian is that of `LL_D6` plus the trap,

H = 2³ʳ Σₙₘ (ψₙʳ)⁺ Kₙₘ ψₘʳ + 2²ʳc Σₙₘₗₖ Γ₄[m-n][l-n][k-n] (ψₙʳ)⁺ (ψₘʳ)⁺ ψₗʳ ψₖʳ - 2ʳμ Σₙ (ψₙʳ)⁺ ψₙʳ
    + 2ʳ Σₙₘ Vₙₘ (ψₙʳ)⁺ ψₘʳ,

where Vₙₘ = ∫ V(x) sₙʳ(x) sₘʳ(x) dx is the matrix of V(x) = ½ω²(x - x₀)², see [`trap_matrix`](@ref).
As for `LL_D6`, this is 2ʳ times the continuum Hamiltonian.  The site i of the chain holds the mode
n = i - 1, and by default the trap is centred on the chain (`x0 = nothing`).

`LL_D6_trap` adds the trap terms to those of `LL_D6`, one by one, before building the MPO, and
`LL_D6_trap_minimal` adds them to the MPO of `LL_D6_minimal` with 8 extra virtual levels (see
[`onebody_mpo_hamiltonian`](@ref)).  They are the same operator, and
`exact_model(LL_D6_trap_minimal) === LL_D6_trap`.  The lattice must be a `FiniteChain`.  The Hamiltonian conserves the particle number, and can also be built with `symmetry = U1Irrep`;
the physical space is then `U1Space(n => 1 for n in 0:cutoff)`, and a finite chain ground state
can be searched for directly in the sector with N particles.
"""

"""
    LL_D6_trap([elt::Type{<:Number}=ComplexF64,] [symmetry::Type{<:Sector}=Trivial,]
               lattice::FiniteChain;
               μ=2.0, c=1.0, r=0.0, cutoff=2, ω=1.0, x0=nothing)

$LL_TRAP_DOC
"""
function LL_D6_trap end
function LL_D6_trap(lattice::AbstractLattice; kwargs...)
    return LL_D6_trap(ComplexF64, Trivial, lattice; kwargs...)
end
function LL_D6_trap(symmetry::Type{<:Sector}, lattice::AbstractLattice; kwargs...)
    return LL_D6_trap(ComplexF64, symmetry, lattice; kwargs...)
end
function LL_D6_trap(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return LL_D6_trap(elt, Trivial, lattice; kwargs...)
end
function LL_D6_trap(elt::Type{<:Number}, symmetry::Type{<:Sector}, lattice::AbstractLattice;
                    μ=2.0, c=1.0, r=0.0, cutoff=2, ω=1.0, x0=nothing)
    check_trap_lattice(lattice)
    V = rmul!(trap_matrix(length(lattice); ω, x0, r), 2^r)
    trap = harmonic_trap(V, elt, symmetry, lattice; cutoff)
    return LL_D6_hamiltonian(trap, elt, symmetry, lattice; μ, c, r, cutoff)
end

"""
    LL_D6_trap_minimal([elt::Type{<:Number}=ComplexF64,] [symmetry::Type{<:Sector}=Trivial,]
                       lattice::FiniteChain;
                       μ=2.0, c=1.0, r=0.0, cutoff=2, ω=1.0, x0=nothing, rtol=1e-13, verbose=false)

$LL_TRAP_DOC
"""
function LL_D6_trap_minimal end
function LL_D6_trap_minimal(lattice::AbstractLattice; kwargs...)
    return LL_D6_trap_minimal(ComplexF64, Trivial, lattice; kwargs...)
end
function LL_D6_trap_minimal(symmetry::Type{<:Sector}, lattice::AbstractLattice; kwargs...)
    return LL_D6_trap_minimal(ComplexF64, symmetry, lattice; kwargs...)
end
function LL_D6_trap_minimal(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return LL_D6_trap_minimal(elt, Trivial, lattice; kwargs...)
end
function LL_D6_trap_minimal(elt::Type{<:Number}, symmetry::Type{<:Sector}, lattice::AbstractLattice;
                            μ=2.0, c=1.0, r=0.0, cutoff=2, ω=1.0, x0=nothing, rtol::Real=1.0e-13,
                            verbose::Bool=false)
    check_trap_lattice(lattice)
    symmetry in (Trivial, U1Irrep) ||
        throw(ArgumentError("only Trivial and U1Irrep symmetries are implemented, got $symmetry"))
    _, ops, index = boson_alphabet(elt, cutoff, 2, 2)
    coeffs, R = LL_terms(K6, Γ4_D6, index; μ = μ, c = c, r = r)
    L = length(lattice)
    L > R || throw(ArgumentError("chain of length $L is shorter than the range $R"))
    W = exact_mpo_matrix(coeffs, ops, R; elt = elt, symmetry = symmetry, rtol = rtol,
                         verbose = verbose)
    V = rmul!(trap_matrix(L; ω, x0, r), 2^r)
    entry(pq, qin, qout) = mpo_entry(ops[index[pq]], elt, symmetry, qin, qout)
    return onebody_mpo_hamiltonian(W, V, entry)
end

"""
    exact_model(model)

The `@mpoham` constructor of the same Hamiltonian as `model`: `LL_D6` for `LL_D6_minimal`,
`LL_D8` for `LL_D8_minimal`, and `model` itself for any other constructor.

Optimize with the minimal MPO, which is much faster, and evaluate the final energies with
`exact_model(model)`, whose MPO is built term by term without the compression of
[`exact_mpo_data`](@ref), so that reported energies are variational independently of it.
"""
exact_model(model) = model
exact_model(::typeof(LL_D6_minimal)) = LL_D6
exact_model(::typeof(LL_D8_minimal)) = LL_D8
exact_model(::typeof(LL_D6_trap_minimal)) = LL_D6_trap
