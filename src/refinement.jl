############## Functions/definitions for refinement ##############

#weights for the D6 wavelets
const h6 = [(1 + sqrt(10) + sqrt(5 + 2 * sqrt(10))) / (16 * sqrt(2)), (5 + sqrt(10) + 3 * sqrt(5 + 2 * sqrt(10))) / (16 * sqrt(2)),
    (10 - 2 * sqrt(10) + 2 * sqrt(5 + 2 * sqrt(10))) / (16 * sqrt(2)), (10 - 2 * sqrt(10) - 2 * sqrt(5 + 2 * sqrt(10))) / (16 * sqrt(2)),
    (5 + sqrt(10) - 3 * sqrt(5 + 2 * sqrt(10))) / (16 * sqrt(2)), (1 + sqrt(10) - sqrt(5 + 2 * sqrt(10))) / (16 * sqrt(2))]

#IWT matrices, which act on operators
const A1 = 1 / sqrt(h6[2]^2 + h6[5]^2) * [h6[2] -h6[5]; h6[5] h6[2]]
const A2 = -[sqrt(h6[3]^2 + h6[4]^2) -sqrt(1 - h6[3]^2 - h6[4]^2); sqrt(1 - h6[3]^2 - h6[4]^2) sqrt(h6[3]^2 + h6[4]^2)]
const A3 = 1 / sqrt(h6[5]^2 + h6[6]^2) * [h6[5] h6[6]; h6[6] -h6[5]]

"""
Converting the IWT operator matrix A into a matrix acting on Fock state inputs n1, n2. 
Output legs act in the Fock space and have dimension ℂ^(cutoff + 1) ← ℂ^(cutoff + 1).
Cutoff=d-1 is the maximum particle number.
"""
function Afock_one(A, n1, n2, cutoff)
    Itm = TensorMap(Matrix{Float64}(I, cutoff + 1, cutoff + 1), ℂ^(cutoff + 1) ← ℂ^(cutoff + 1)) #identity tensor
    An1n2 = 1 / sqrt(factorial(n1) * factorial(n2)) * (A[1, 1] * (a_plus(ComplexF64; cutoff=cutoff) ⊗ Itm) + A[2, 1] * (Itm ⊗ a_plus(ComplexF64; cutoff=cutoff)))^n1 * 
            (A[1, 2] * (a_plus(ComplexF64; cutoff=cutoff) ⊗ Itm) + A[2, 2] * (Itm ⊗ a_plus(ComplexF64; cutoff=cutoff)))^n2
    return An1n2[:, :, 1, 1] #acting on the vacuum of each mode
end

"""
Converting an IWT operator matrix into a four-legged tensor acting on the Fock space.
Inputs are A, the operator matrix A (taking values A1, A2, or A3), and d, the number of states in the truncated Fock space.
The output is a tensor of dimensions (ℂ^d ⊗ ℂ^d) ← (ℂ^d ⊗ ℂ^d)
"""
function Afock(A, d)
    Atm = TensorMap(zeros(d, d, d, d), (ℂ^d ⊗ ℂ^d) ← (ℂ^d ⊗ ℂ^d))
    for n1 = 0:(d-1), n2 = 0:(d-1)
        Atm[:, :, n1+1, n2+1] = Afock_one(A, n1, n2, d - 1)
    end
    return Atm
end

"""
Trivial isometry, mapping leg 1 (scaling states) with the identity and setting leg 2 (wavelet states) to the vacuum.
Inputs are d, the number of states on each input leg, and dmax, the max number of particles on the output leg
"""
function Vfock(d, dmax)
    Vi = zeros(ComplexF64, d, d, dmax + 1)
    for x = 1:(dmax+1)
        Vi[x, 1, x] = 1.0
    end
    return TensorMap(Vi, (ℂ^d ⊗ ℂ^d) ← ℂ^(dmax + 1))
end

"""
Refine an initial MPS state ψi via the introduction of wavelet modes, and then successive application of 
two-site matrices A1, A2, and A3.

Inputs
ψi = initial MPS state
H = Hamiltonian at the new refined scale
dmax = maximum number of bosons at each site in the original MPS (physical space dimension dmax+1)
d = number of physical states in each leg of the refinement
Dtrunc = bond dimension of the truncated MPS output
trunc_err = truncation argument in each tsvd
verbosity = prints progress if ≥ 1

Outputs
ψf = 2-site MPS at one scale above ψi, has same energy density as ψi
E = energy density of ψf with respect to Hamiltonian H
"""
function MPSrefine(ψi, H, dmax, d, Dtrunc; trunc_err=1e-8, verbosity=1)

    #convert A's to Fock space matrices
    A1fock = Afock(A1, d)
    A2fock = Afock(A2, d)
    A3fock = Afock(A3, d)

    ALr = ψi.AL[1]
    Vfock_i = Vfock(d, dmax)

    #first circuit layer
    @plansor step1[-1 -2 -3; -4] := ALr[-1 1; -4] * Vfock_i[2 3; 1] * A1fock[-2 -3; 2 3]
    U1, s1, V1 = svd_trunc!(permute(step1, ((1, 2), (3, 4))); trunc=truncerror(; atol=trunc_err))
    s1 = s1 / sqrt(tr(s1's1))
    s1inv = inv(s1)
    V1 = permute(V1, ((1, 2), (3,)))

    #second circuit layer
    @plansor step2[-1 -2 -3; -4] := s1[-1; 1] * V1[1 2; 3] * U1[3 4; 5] * s1[5; -4] * A2fock[-2 -3; 2 4]
    U2t, s2, V2t = svd_trunc!(permute(step2, ((1, 2), (3, 4))); trunc=truncerror(; atol=trunc_err))
    s2 = s2 / sqrt(tr(s2's2))
    s2inv = inv(s2)
    V2t = permute(V2t, ((1, 2), (3,)))

    #third circuit layer
    @plansor step3[-1 -2 -3; -4] := s2[-1; 1] * V2t[1 2; 3] * s1inv[3; 4] * U2t[4 5; 6] * s2[6; -4] * A3fock[-2 -3; 2 5]
    U3t, s3, V3t = svd_trunc!(permute(step3, ((1, 2), (3, 4))); trunc=truncerror(; atol=trunc_err))
    s3 = s3 / sqrt(tr(s3's3))
    V3t = permute(V3t, ((1, 2), (3,)))

    #apply singular value matrices
    @plansor U3[-1 -2; -3] := U3t[-1 -2; -3]
    @plansor V3[-1 -2; -3] := s3[-1; 1] * V3t[1 -2; 2] * s2inv[2; -3]

    ψf = InfiniteMPS([U3, V3]) #create two-site MPS

    verbosity >= 1 && println("Reducing bond dimension to $Dtrunc")
    ψf = trunc_bonds(ψf, Dtrunc) #truncate bonds
    E = real(expectation_value(ψf, H)) / 2 #groundstate energy
    verbosity >= 1 && println("Groundstate energy density after refinement: $E")
    return (ψf, E)
end

"""
Bond truncation, from MPSKit
"""
function trunc_bonds(ψ, D ; alg_svd = DivideAndConquer(; driver = LAPACK()))
    copied = copy.(ψ.AL)
    ncr = ψ.C[1]

    for i in 1:length(ψ)
        U, ncr, = svd_trunc(ψ.C[i]; trunc=truncrank(D), alg = alg_svd)
        copied[i] = copied[i] * U
        copied[i + 1] = MPSKit._transpose_front(U' * MPSKit._transpose_tail(copied[i + 1]))
    end

    # make sure everything is full rank:
    MPSKit.makefullrank!(copied)

    # if the bond dimension is not changed, we can keep the same center, otherwise recompute
    ψ = if space(ncr, 1) != space(copied[1], 1)
        InfiniteMPS(copied)
    else
        C₀ = ncr isa TensorMap ? ncr : TensorMap(ncr)       
        InfiniteMPS(copied, C₀)
    end
    return normalize!(ψ)
end

############ Functions for making a single-site state #############

"""
Acting with the mixed transfer matrix on a unitary.
Inputs: unitary Udag, Udag dimension chi, MPS tensors A and B
Output: the action as a vector
"""
function MTMaction(Udag, chi, A, B)
    Udag=reshape(Udag,chi,chi)
    Udag=TensorMap(Udag, ℂ^(chi) ← ℂ^(chi))
    @plansor MTMact[-1 -2] := B[-1 1 ; 2] * A[2 3 ; 5] * conj(A[-2 1 ; 4]) * conj(B[4 3 ; 6]) * Udag[5 ; 6]
    return vec(convert(Array, MTMact))
end

"""
Map a two-site MPS to a single-site MPS.
Inputs are a two-site state ψ and the single-site Hamiltonian H; prints the energies if verbosity ≥ 1,
and the leading eigenvalue of the mixed transfer matrix if verbosity ≥ 2.
Output is a tuple containing: the single-site MPS with lowest energy, and its energy. 
"""
function single_site(ψ, H; verbosity=1)
    A=ψ.AR[1]
    B=ψ.AR[2]
    @plansor MTM[-1 -2 ; -3 -4] := B[-1 1 ; 2] * A[2 3 ; -3] * conj(A[-2 1 ; 4]) * conj(B[4 3 ; -4])

    chi=dim(left_virtualspace(MTM))

    I_flat = (Matrix{ComplexF64}(I, chi, chi))[:]
    MTMact(x)=MTMaction(x,chi,A,B)
    MTMmap=LinearMap{ComplexF64}(MTMact,chi^2; ismutating=false, issymmetric=false, ishermitian=false, isposdef=false)

    main_eval, Udag_evec = eigsolve(MTMmap, I_flat, 1, :LM, tol=1e-10)
    #the leading eigenvalue is one when ψ is exactly invariant under translation by one site
    verbosity >= 2 && println("Leading eigenvalue of the mixed transfer matrix: $(first(main_eval))")
    Udag=TensorMap(reshape(Udag_evec[1],chi,chi), ℂ^(chi) ← ℂ^(chi))
    U = Udag'

    #finding the single-site tensor, either BU or U†A
    @plansor CfromB[-1 -2 ; -3] := B[-1 -2 ; 1] * U[1 ; -3]
    @plansor CfromA[-1 -2 ; -3] := Udag[-1 ; 1] * A[1 -2 ; -3]
    MPS_A=InfiniteMPS([CfromA])
    MPS_B=InfiniteMPS([CfromB])
    E_A = real(expectation_value(MPS_A, H))
    E_B = real(expectation_value(MPS_B, H))

    if verbosity >= 1
        println("Energy density of single-site MPS from tensor A: $E_A")
        println("Energy density of single-site MPS from tensor B: $E_B")
    end

    #return whichever MPS has lower energy
    if E_A < E_B
        return (MPS_A, E_A)
    else
        return (MPS_B, E_B)
    end
    
end

################# optimization routines #####################

"""
Iterative refinement.

Inputs
ψ = input single-site state
r_start = resolution at the beginning
n_scale = number of refinement iterations
μ, c = LL parameters
dmax = maximum number of bosons at each site in the original MPS (physical space dimension dmax+1)
d = number of physical states in each leg of the refinement, must equal dmax+1
D_refine = bond dimension of the truncated MPS output
Einc = if true, will stop the refinement loop if the single-site state has higher energy than the previous, lower-resolution state
verbosity = prints progress if ≥ 1
model = Hamiltonian at each resolution, called as model(; μ, c, r, cutoff). Defaults to LL_D6_minimal, the same operator as LL_D6
        but much faster to build and use. The refinement circuit uses D6 wavelets, so this should be a D6 model

Output: final single-site state
"""
function refinement_alg(ψ, r_start, n_scale, μ, c, dmax, d, D_refine ; Einc=false, model=LL_D6_minimal, verbosity=1)
    d == dmax + 1 || throw(ArgumentError("the refinement loop needs d == dmax + 1 (got d = $d, dmax = $dmax), since each step's output is the next step's input"))

    E_end = real(expectation_value(ψ, model(; μ=μ, c=c, r=r_start, cutoff=dmax)))
    verbosity >= 1 && println("Starting energy: $E_end")
    for i=1:n_scale   
        Hri=model(; μ=μ, c=c, r=r_start+i, cutoff=dmax)
        Hri_2 = InfiniteMPOHamiltonian([Hri[1], Hri[1]]) 

        verbosity >= 1 && println("Refining in scale, ending with a 2-site state at r=$(r_start+i) with bond dimension $D_refine")
        state_refined, refine_elapsed, = @timed MPSrefine(ψ, Hri_2, dmax, d, D_refine; verbosity)
        verbosity >= 1 && println("Time to refine, in seconds: $refine_elapsed")
        state_2site = state_refined[1]

        #writing the two-site MPS as a single-site MPS
        state_single, single_elapsed, = @timed single_site(state_2site, Hri; verbosity)
        verbosity >= 1 && println("Time to make a single-site MPS, in seconds: $single_elapsed")

        if Einc && (state_single[2] > E_end)
            verbosity >= 1 && println("Energy increased; exiting refinement loop")
            break
        end

        ψ=state_single[1]
        E_end=state_single[2]
    end
    return ψ
end

"""
Iterative refinement, with L-BFGS optimization at each resolution.

Inputs
ψ = input single-site state
r_start = resolution at the beginning
n_scale = number of refinement iterations
μ, c = LL parameters
dmax = maximum number of bosons at each site in the original MPS (physical space dimension dmax+1)
d = number of physical states in each leg of the refinement, must equal dmax+1
D_refine = bond dimension of the truncated MPS output
gg_fg = number of L-BFGS iterations at each resolution
lbfgs_memory = number of previous steps kept by L-BFGS
verbosity = verbosity of L-BFGS; progress is printed if ≥ 1
metric_regulator = regularization of the L-BFGS preconditioner, see precondition()
model = Hamiltonian at each resolution, called as model(; μ, c, r, cutoff). Defaults to LL_D6_minimal, the same operator as LL_D6
        but much faster to build and use. The refinement circuit uses D6 wavelets, so this should be a D6 model

Output: tuple (state_array, history)
state_array = array of size (n_scale, D_refine, dmax+1, D_refine) holding the AL tensor of the state at each resolution.
    Tensors with bond dimension below D_refine are zero-padded, and resolutions not reached (early exit) are all zeros.
history = combined L-BFGS history of all resolutions, with columns (time, energy density, gradient norm); the time
    includes the refinement and single-site steps
"""
function refine_optim(ψ, r_start, n_scale, μ, c, dmax, d, D_refine, gg_fg ; verbosity=3, lbfgs_memory=8, model=LL_D6_minimal, metric_regulator=1e-8)
    d == dmax + 1 || throw(ArgumentError("the refinement loop needs d == dmax + 1 (got d = $d, dmax = $dmax), since each step's output is the next step's input"))
    E_end = real(expectation_value(ψ, model(; μ=μ, c=c, r=r_start, cutoff=dmax)))
    state_array = zeros(ComplexF64, n_scale, D_refine, dmax + 1, D_refine)
    verbosity >= 1 && println("Starting energy: $E_end")
    delta_old=zeros(Float64, 0, 3) #(time, energy density, gradient norm), empty until the first step completes
    for i=1:n_scale   
        Hri=model(; μ=μ, c=c, r=r_start+i, cutoff=dmax)
        Hri_2 = InfiniteMPOHamiltonian([Hri[1], Hri[1]]) 

        verbosity >= 1 && println("Refining in scale, ending with a 2-site state at r=$(r_start+i) with bond dimension $D_refine")
        state_refined, refine_elapsed, = @timed MPSrefine(ψ, Hri_2, dmax, d, D_refine; verbosity)
        verbosity >= 1 && println("Time to refine, in seconds: $refine_elapsed")
        state_2site = state_refined[1]

        #writing the two-site MPS as a single-site MPS
        state_single, single_elapsed, = @timed single_site(state_2site, Hri; verbosity)
        verbosity >= 1 && println("Time to make a single-site MPS, in seconds: $single_elapsed")
        state_fg=state_single[1]

        alg_gg_fg = OptimKit.LBFGS(lbfgs_memory; maxiter=gg_fg, verbosity=verbosity) #algorithm for gradient descent
        verbosity >= 1 && println("Running GG for $gg_fg iterations on the single-site refinement, with: bond dimension $D_refine and resolution $(r_start+i)")
        ψ, cache, delta = find_groundstate_t(state_fg, Hri, GradientGrassmann(; method=alg_gg_fg) ; metric_regulator);
        AL = convert(Array, ψ.AL[1])
        state_array[i, axes(AL, 1), :, axes(AL, 3)] = AL #bond dimension can end up below D_refine; the rest stays zero

        if state_single[2] > E_end
            verbosity >= 1 && println("Energy increased; exiting refinement loop")
            break
        end

        delta[:,1] .+= refine_elapsed+single_elapsed
        delta_old = isempty(delta_old) ? delta : add_histories_kf(delta_old, delta)

        E_end = real(expectation_value(ψ, Hri)) #groundstate energy
        verbosity >= 1 && println("Groundstate energy: $E_end")
  
    end
    return state_array, delta_old
end
