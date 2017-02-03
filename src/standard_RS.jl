type OrderParams <: AbstractParams
    q0::Float64
    qh0::Float64
    qh1::Float64
end

type ExtParams <:AbstractParams
    α::Float64
    β::Float64
    qs::Float64 # norm2 of weights
end


type ThermFunc <:AbstractParams
    ϕ::Float64
end

# ENERGETIC TERM
argGe_theta(z0, q0, qs) = logH(-z0 * √q0 / √(qs - q0))

function Ge_theta(β, q0, qs)
    ∫D(z0->begin
        argGe_theta(z0, q0, qs)
    end)
end

function ∂q0_Ge_theta(β, q0, qs)
    return -0.5/(qs-q0)*∫D(z0->begin
        x= z0 * √q0 / √(qs - q0)
        GH(-x)^2
    end)
end

function argGe_loglike(z0, β, q0, qs)
    log(∫D(η->begin
        H(-(√(qs-q0)*η+√q0*z0)/√(1-qs))^β
    end))
end

function Ge_loglike(β, q0, qs)
    ∫D(z0->begin
        argGe_loglike(z0, β, q0, qs)
    end)
end

# ENTROPIC TERM
function argGs_continuous1(z0, qh0, qh1)
    b = √qh0 * z0
    if qh1 > qh0
        a = qh1 - qh0
        res = -b^2/(2a) + 0.5*log(π/2) - 0.5*log(a)
        res += log(erfi((a-b)/√(2a)) + erfi((a+b)/√(2a)))
    else
        a = qh0 - qh1
        res = b^2/(2a) + 0.5*log(π/2) - 0.5*log(a)
        res += log(erf((a-b)/√(2a)) + erf((a+b)/√(2a)))
    end
    return res
end

function Gs_continuous1(qh0, qh1)
    ∫D(z0->begin
        argGs_continuous1(z0, qh0, qh1)
    end)
end

function Gs_continuous(qh0, qh1)
    0.5*(log(2π) + qh0/(qh0-qh1) - log(qh0-qh1))
end
function ∂qh0_Gs_continuous(qh0, qh1)
    -0.5*qh0/(qh0-qh1)^2
end

function Gs_binary(qh0, qh1)
    gs = 0.5*(qh1-qh0) #qh1=0 usually
    gs += ∫D(z0->begin
        log(2cosh(z0 * √qh0))
    end)
    return gs
end

function ∂qh0_Gs_binary(qh0, qh1)
    -0.5*∫D(z0->begin
        tanh(z0 * √qh0)^2
    end)
end
####

function free_entropy(ep, op, vartype, enetype)
    @extract ep : α β qs
    @extract op : q0 qh0 qh1
    ϕ = -0.5*(qs*qh1-q0*qh0)
    if vartype == :continuous
        ϕ +=  Gs_continuous(qh0, qh1)
    elseif vartype == :continuous1
        ϕ +=  Gs_continuous1(qh0, qh1)
    elseif vartype == :binary
        ϕ +=  Gs_binary(qh0, qh1)
    end
    if enetype == :theta
        ϕ += α*Ge_theta(β, q0, qs)
    elseif enetype == :loglike
        ϕ += α*Ge_loglike(β, q0, qs)
    end
    return ϕ
end

####
fq0_continuous(qh0, qh1) = -2*∂qh0_Gs_continuous(qh0, qh1)
fqh1_continuous(qs, qh0, qh1) = (-1+2qh0*qs-√(1+4qh0*qs)) / (2qs)

fq0_continuous1(qh0, qh1) = -2*deriv(Gs_continuous1, 1, qh0, qh1)
fqs_continuous1(qh0, qh1) = 2*deriv(Gs_continuous1, 2, qh0, qh1)

fq0_binary(qh0, qh1) = -2*∂qh0_Gs_binary(qh0, qh1)

fqh0_theta(α, β, q0, qs) = -2α*∂q0_Ge_theta(β, q0, qs)
fqh0_loglike(α, β, q0, qs) = -2α*deriv(Ge_loglike, 2, β, q0, qs)

function iqh1_continuous1(qs, qh0, qh1_0)
    ok, qh1, it, normf0 = newton(qh1 -> qs - fqs_continuous1(qh0, qh1), qh1_0)
    # ok, qh1, it, normf0 = findzero_interp(qh1 -> qs - fs1PF_binary(qh0, qh1), qh1_0)

    ok || normf0 < 1e-6 || warn("newton failed: z=$qh1, it=$it, normf0=$normf0")
    ok = true
    if normf0 > 1e-4
        ok = false
    end
    return ok, qh1
end

function all_therm_func(ep::ExtParams, op::OrderParams, vartype, enetype)
    ϕ = free_entropy(ep, op, vartype, enetype)
    return ThermFunc(ϕ)
end
###########


function converge!(ep::ExtParams, op::OrderParams, pars::Params,
        vartype::Symbol, enetype::Symbol)
    @extract pars : maxiters verb ϵ ψ

    Δ = Inf
    ok = false

    it = 0
    verb > 0 && println("$op")
    for it = 1:maxiters
        Δ = 0.0
        verb > 1 && println("it=$it")

        if enetype == :theta
            @update  op.qh0     fqh0_theta       Δ ψ verb  ep.α ep.β op.q0 ep.qs
        elseif enetype == :loglike
            @update  op.qh0     fqh0_loglike     Δ ψ verb  ep.α ep.β op.q0 ep.qs
        end

        if vartype == :continuous
            @update  op.qh1     fqh1_continuous     Δ ψ verb  ep.qs op.qh0 op.qh1
            @update  op.q0      fq0_continuous     Δ ψ verb  op.qh0 op.qh1
        elseif vartype == :continuous1
            @updateI  op.qh1 ok     iqh1_continuous1     Δ ψ verb  ep.qs op.qh0 op.qh1
            @update  op.q0          fq0_continuous1      Δ ψ verb  op.qh0 op.qh1
        elseif vartype == :binary
            @update  op.q0      fq0_binary      Δ ψ verb  op.qh0 op.qh1
        end

        verb > 1 && println(" Δ=$Δ\n")
        ok = Δ < ϵ
        ok && break
    end

    if verb > 0
        println(ok ? "converged" : "failed", " (it=$it Δ=$Δ)")
        println(ep)
        println(op)
    end
    return ok
end

function converge(; α = 0.1, β=Inf, qs = 1.,
                    q0 = 0.1,
                    qh0 = 0.1, qh1 = 0.,
                    vartype = :continuous, # :binary, :continuous, :continuous1
                    enetype = :theta, # :theta, :loglike
                    ϵ = 1e-5, ψ = 0.0, maxiters = 1_000, verb = 3)

    @assert vartype in [:continuous, :continuous1, :binary]
    @assert enetype in [:loglike, :theta]

    ep = ExtParams(α, β, qs)
    pars = Params(ϵ, ψ, maxiters, verb)
    op = OrderParams(q0, qh0, qh1)

    converge!(ep, op, pars, vartype, enetype)

    tf = all_therm_func(ep, op, vartype, enetype)
    println(tf)

    return ep, op, tf, pars
end


function span(; αlist = [0.1],
                β=Inf, qs = 1.,
                q0 = 0.1,
                qh0 = 0.1, qh1 = 0.,
                vartype = :binary, # :binary, :continuous, :continuous1
                enetype = :theta, # :theta, :loglike
                ϵ = 1e-5, ψ = 0.0, maxiters = 1_000, verb = 3,
                resfile = "results_$(vartype)_$(enetype).txt")

    lockfile = resfile *".lock"
    if !isfile(resfile)
        open(resfile, "w") do f
            allheadersshow(f, ExtParamsPF, OrderParamsPF, ThermFuncPF)
        end
    end

    ep = ExtParams(first(αlist), β, qs)
    pars = Params(ϵ, ψ, maxiters, verb)
    op = OrderParams(q0, qh0, qh1)
    results = []

    for α in αlist
        println("\n########  NEW ITER  ########\n")
        ep.α = α
        verb > 0 && println(ep)

        ok = converge!(ep, op, pars, vartype, enetype)
        tf = all_therm_func(ep, op, vartype, enetype)
        push!(results, (ok, deepcopy(ep), deepcopy(op), deepcopy(tf)))
        verb > 0 && println(tf)
        if ok
            exclusive(lockfile) do
                open(resfile, "a") do rf
                    println(rf, plainshow(ep), " ", plainshow(op), " ", plainshow(tf))
                end
            end
        end
        ok || break
    end
    return results
end
