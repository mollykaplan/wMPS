"""
MPSKit's find_groundstate(), but now returning timing data for plotting
"""
function find_groundstate_t(
        mps::InfiniteMPS, operator, alg::VUMPS, envs = environments(mps, operator)
    )
    return dominant_eigsolve_t(operator, mps, alg, envs; which = :SR)
end

"""
MPSKit's dominant_eigsolve(), but now returning timing data for plotting:
ehistory = array of errors
fhistory = array of energy densities
thistory = array of timestamps
"""
function dominant_eigsolve_t(
        operator, mps, alg::VUMPS, envs = environments(mps, operator);
        which
    )
    log = MPSKit.IterLog("VUMPS")
    iter = 0
    mps = copy(mps)
    ϵ = MPSKit.calc_galerkin(mps, operator, mps, envs)
    alg_environments = MPSKit.updatetol(alg.alg_environments, iter, ϵ)
    MPSKit.recalculate!(envs, mps, operator, mps; alg_environments.tol)

    state = MPSKit.VUMPSState(mps, operator, envs, iter, ϵ, which)
    it = MPSKit.IterativeSolver(alg, state)

    return MPSKit.LoggingExtras.withlevel(; alg.verbosity) do
        obj_init=sum(expectation_value(mps, operator, envs))
        @MPSKit.infov 2 MPSKit.loginit!(log, ϵ, obj_init)
        ehistory=[ϵ]
        fhistory=[obj_init]
        thistory=[Base.time()-log.t_init]

        for (mps, envs, ϵ) in it
            objective=expectation_value(mps, operator, envs)
            push!(ehistory, ϵ)
            push!(fhistory, objective)
            push!(thistory, Base.time()-log.t_init)

            if ϵ ≤ alg.tol
                @MPSKit.infov 2 MPSKit.logfinish!(log, it.iter, ϵ, objective)
                return mps, envs, ϵ, real.([thistory fhistory ehistory])
            end
            if it.iter ≥ alg.maxiter
                @MPSKit.warnv 1 MPSKit.logcancel!(log, it.iter, ϵ, objective)
                return mps, envs, ϵ, real.([thistory fhistory ehistory])
            end
            @MPSKit.infov 3 MPSKit.logiter!(log, it.iter, ϵ, objective)
        end

        # this should never be reached
        return it.state.mps, it.state.envs, it.state.ϵ, real.([thistory fhistory ehistory])
    end
end

"""
MPSKit's optimize(), but now returning timing data for plotting:
ehistory = array of errors
fhistory = array of energy densities
thistory = array of timestamps
"""
function optimize_t(fg, x, alg::OptimKit.LBFGS;
                  precondition=OptimKit._precondition,
                  (finalize!)=OptimKit._finalize!,
                  shouldstop=OptimKit.DefaultShouldStop(alg.maxiter),
                  hasconverged=OptimKit.DefaultHasConverged(alg.gradtol),
                  retract=OptimKit._retract, inner=OptimKit._inner, (transport!)=OptimKit._transport!,
                  (scale!)=OptimKit._scale!, (add!)=OptimKit._add!,
                  isometrictransport=(transport! == OptimKit._transport! && inner == OptimKit._inner))
    t₀ = time()
    verbosity = alg.verbosity
    f, g = fg(x)
    numfg = 1
    numiter = 0
    innergg = inner(x, g, g)
    normgrad = sqrt(innergg)
    fhistory = [f]
    normgradhistory = [normgrad]
    t = time() - t₀
    thistory = [t]
    _hasconverged = hasconverged(x, f, g, normgrad)
    _shouldstop = shouldstop(x, f, g, numfg, numiter, t)

    TangentType = typeof(g)
    ScalarType = typeof(innergg)
    m = alg.m
    H = OptimKit.LBFGSInverseHessian(m, TangentType[], TangentType[], ScalarType[])

    verbosity >= 2 &&
        @info @OptimKit.sprintf("LBFGS: initializing with f = %.12e, ‖∇f‖ = %.4e", f, normgrad)

    while !(_hasconverged || _shouldstop)
        told = t
        # compute new search direction
        if length(H) > 0
            Hg = let x = x
                H(g, ξ -> precondition(x, ξ), (ξ1, ξ2) -> inner(x, ξ1, ξ2), add!, scale!)
            end
            η = scale!(Hg, -1)
        else
            Pg = precondition(x, deepcopy(g))
            normPg = sqrt(inner(x, Pg, Pg))
            η = scale!(Pg, -0.01 / normPg) # initial guess: scale invariant
        end

        # store current quantities as previous quantities
        xprev = x
        gprev = g
        ηprev = η

        # perform line search
        OptimKit._xlast[] = x # store result in global variables to debug linesearch failures
        OptimKit._glast[] = g
        OptimKit._dlast[] = η
        x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
                                            initialguess=one(f),
                                            acceptfirst=alg.acceptfirst,
                                            # for some reason, line search seems to converge to solution alpha = 2 in most cases if acceptfirst = false. If acceptfirst = true, the initial value of alpha can immediately be accepted. This typically leads to a more erratic convergence of normgrad, but to less function evaluations in the end.
                                            retract=retract, inner=inner)
        numfg += nfg
        numiter += 1
        x, f, g = finalize!(x, f, g, numiter)
        innergg = inner(x, g, g)
        normgrad = sqrt(innergg)
        push!(fhistory, f)
        push!(normgradhistory, normgrad)
        t = time() - t₀
        Δt = t - told
        push!(thistory, t)
        _hasconverged = hasconverged(x, f, g, normgrad)
        _shouldstop = shouldstop(x, f, g, numfg, numiter, t)                                       

        # check stopping criteria and print info
        if _hasconverged || _shouldstop
            break
        end

        verbosity >= 3 &&
            @info @OptimKit.sprintf("LBFGS: iter %4d, Δt %s: f = %.12e, ‖∇f‖ = %.4e, α = %.2e, m = %d, nfg = %d",
                           numiter, OptimKit.format_time(Δt), f, normgrad, α, length(H), nfg)

        # transport gprev, ηprev and vectors in Hessian approximation to x
        gprev = transport!(gprev, xprev, ηprev, α, x)
        for k in 1:length(H)
            @inbounds s, y, ρ = H[k]
            s = transport!(s, xprev, ηprev, α, x)
            y = transport!(y, xprev, ηprev, α, x)
            # QUESTION:
            # Do we need to recompute ρ = inv(inner(x, s, y)) if transport is not isometric?
            H[k] = (s, y, ρ)
        end
        ηprev = transport!(deepcopy(ηprev), xprev, ηprev, α, x)

        if isometrictransport
            # TRICK TO ENSURE LOCKING CONDITION IN THE CONTEXT OF LBFGS
            #-----------------------------------------------------------
            # (see A BROYDEN CLASS OF QUASI-NEWTON METHODS FOR RIEMANNIAN OPTIMIZATION)
            # define new isometric transport such that, applying it to transported ηprev,
            # it returns a vector proportional to ξ but with the norm of ηprev
            # still has norm normη because transport is isometric
            normη = sqrt(inner(x, ηprev, ηprev))
            normξ = sqrt(inner(x, ξ, ξ))
            β = normη / normξ
            if !(inner(x, ξ, ηprev) ≈ normξ * normη) # ξ and η are not parallel
                ξ₁ = ηprev
                ξ₂ = scale!(ξ, β)
                ν₁ = add!(ξ₁, ξ₂, +1)
                ν₂ = scale!(deepcopy(ξ₂), -2)
                squarednormν₁ = inner(x, ν₁, ν₁)
                squarednormν₂ = inner(x, ν₂, ν₂)
                # apply Householder transforms to gprev, ηprev and vectors in H
                gprev = add!(gprev, ν₁, -2 * inner(x, ν₁, gprev) / squarednormν₁)
                gprev = add!(gprev, ν₂, -2 * inner(x, ν₂, gprev) / squarednormν₂)
                for k in 1:length(H)
                    @inbounds s, y, ρ = H[k]
                    s = add!(s, ν₁, -2 * inner(x, ν₁, s) / squarednormν₁)
                    s = add!(s, ν₂, -2 * inner(x, ν₂, s) / squarednormν₂)
                    y = add!(y, ν₁, -2 * inner(x, ν₁, y) / squarednormν₁)
                    y = add!(y, ν₂, -2 * inner(x, ν₂, y) / squarednormν₂)
                    H[k] = (s, y, ρ)
                end
                ηprev = ξ₂
            end
        else
            # use cautious update below; see "A Riemannian BFGS Method without
            # Differentiated Retraction for Nonconvex Optimization Problems"
            β = one(normgrad)
        end

        # set up quantities for LBFGS update
        y = add!(scale!(deepcopy(g), 1 / β), gprev, -1)
        s = scale!(ηprev, α)
        innersy = inner(x, s, y)
        innerss = inner(x, s, s)

        if innersy / innerss > normgrad / 10000
            norms = sqrt(innerss)
            ρ = innerss / innersy
            push!(H, (scale!(s, 1 / norms), scale!(y, 1 / norms), ρ))
        end
    end
    if _hasconverged
        verbosity >= 2 &&
            @info @OptimKit.sprintf("LBFGS: converged after %d iterations and time %s: f = %.12e, ‖∇f‖ = %.4e",
                           numiter, OptimKit.format_time(t), f, normgrad)
    else
        verbosity >= 1 &&
            @warn @OptimKit.sprintf("LBFGS: not converged to requested tol after %d iterations and time %s: f = %.12e, ‖∇f‖ = %.4e",
                           numiter, OptimKit.format_time(t), f, normgrad)
    end
    history = [thistory fhistory normgradhistory]
    return x, f, g, numfg, history
end

"""
MPSKit's GrassmannMPS.precondition(), but with the regularization of the inverse metric
scaled by sqrt(metric_regulator). metric_regulator = 1 reproduces the MPSKit default.
"""
function precondition(state, g; metric_regulator = 1e-8)
    g′ = similar(g)
    rtolmin = eps(real(scalartype(state)))^(3 / 4)
    MPSKit.tforeach(eachindex(state); scheduler = MPSKit.Defaults.scheduler[]) do i
        rtol = max(rtolmin, sqrt(metric_regulator) * norm(g[i]))
        ρ = GrassmannMPS.rho_inv_regularized(state.C[i]; rtol)
        g′[i] = GrassmannMPS.rmul(g[i], ρ)
        return nothing
    end
    return g′
end

"""
MPSKit's find_groundstate(), but now returning timing data for plotting.
metric_regulator is passed on to the preconditioner (see precondition()).
"""
function find_groundstate_t(
        ψ::S, H, alg::GradientGrassmann, envs::P = environments(ψ, H);
        metric_regulator = 1e-8
    )::Tuple{S, P, Matrix{Float64}} where {S, P}
    !isa(ψ, FiniteMPS) || dim(ψ.C[end]) == 1 ||
        @warn "This is not fully supported - split the mps up in a sum of mps's and optimize separately"
    normalize!(ψ)

    fg(x) = GrassmannMPS.fg(x, H, envs)
    x, _, _, _, history = optimize_t(
        fg, ψ, alg.method;
        GrassmannMPS.transport!,
        GrassmannMPS.retract,
        GrassmannMPS.inner,
        GrassmannMPS.scale!,
        GrassmannMPS.add!,
        precondition = (x, g) -> precondition(x, g; metric_regulator),
        alg.finalize!,
        isometrictransport = true
    )
    return x, envs, history
end

"""
Adding two history arrays (h1 and h2) without double counting the initialization
"""
function add_histories(h1, h2)
    h2 = h2[2:end,:] #don't double-count initialization
    h2[:,1] .+= h1[end,1]
    return vcat(h1, h2)
end

"""
Adding two history arrays (h1 and h2), keeping the first entry in h2
"""
function add_histories_kf(h1, h2)
    h2 = copy(h2)
    h2[:,1] .+= h1[end,1]
    return vcat(h1, h2)
end
