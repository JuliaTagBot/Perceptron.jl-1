type OrderParamsPF <: AbstractParams
    q0::Float64
    s0::Float64
    s1::Float64
    Q::Float64
    qh0::Float64
    qh1::Float64
    sh0::Float64
    sh1::Float64
    Sh::Float64
    Qh::Float64
end

type ExtParamsPF <:AbstractParams
    α::Float64
    β::Float64
    qs::Float64 # norm2 of weights
    S::Float64
end

type ThermFuncPF <:AbstractParams
    ϕ::Float64
end

# ENERGETIC TERM
function argGePF_theta(z0, q0, qs, s0, s1, Q)
    a = √(qs-q0 - (s1-s0)^2/(Q-s0) * (1 - s0*(q0-s0)/(Q*q0-s0^2)))
    b = √((Q*q0-s0^2)/q0)
    ∫D(η->begin
        x = -(√q0*z0 + (s1-s0)/b*η) / a
        y = -(s0/√q0*z0 + b*η) / √(1-Q)
        H(x) * logH(y)
    end)
end

function GePF_theta(β, q0, qs, s0, s1, Q)
    ∫D(z0->begin
        argGePF_theta(z0, q0, qs, s0, s1, Q) / H(-z0 * √q0 / √(qs - q0))
    end)
end

function argGePF_loglike(z0, hβ, β, q0, qs, s0, s1, Q)
    b = √((Q*q0-s0^2)/q0)
    num = ∫D(η->begin
        x = √q0*z0 + (s1-s0)/b*η
        y = -(s0/√q0*z0 + b*η) / √(1-Q)
        hβ(x) * logH(y)
    end)

    # no interp hβ
    # a = √(qs-q0 - (s1-s0)^2/(Q-s0) *(1 - s0*(q0-s0)/(Q*q0-s0^2)))
    #     b = √((Q*q0-s0^2)/q0)
    #     num = ∫D(η->begin
    #         h = ∫D(z1->begin
    #             x = -(√q0*z0 + (s1-s0)/b*η + √a*z1) / √(1-qs)
    #             H(x)^β
    #         end)
    #         y = -(s0/√q0*z0 + b*η) / √(1-Q)
    #         h * logH(y)
    #     end)

    den = ∫D(z1->begin
        x = -(√q0*z0 + √(qs-q0)*z1) / √(1-qs)
        H(x)^β
    end)
    return num/den
end


function GePF_loglike(β, q0, qs, s0, s1, Q)
    a = √(qs-q0 - (s1-s0)^2/(Q-s0) *(1 - s0*(q0-s0)/(Q*q0-s0^2)))
    x = [-2∞:0.1:2∞;]
    y = Float64[
        ∫D(z1->begin
            H(-(x + √a*z1) / √(1-qs))^β
        end) for x in x]
    hβ = Spline1D(x, y)
    return ∫D(z0->begin
        argGePF_loglike(z0, hβ, β, q0, qs, s0, s1, Q)
    end)

    # no interp hβ
    #     ∫D(z0->begin
    #         argGePF_loglike(z0, β, q0, qs, s0, s1, Q)
    #     end)
end
#### ENTROPIC TERM ##############

function argGsPF_continuous(η, qh0, qh1, sh0, sh1, Sh, Qh)
    a = (sh1-sh0)/(qh0-qh1) * √(2qh0-qh1) + sh0 / √(2qh0-qh1)
    b = √(Qh-sh0^2/(2qh0-qh1))

    ∫D(W->begin
        log(2cosh(a*W + Sh*sign(W) + b*η))
    end)
end

function GsPF_continuous(qh0, qh1, sh0, sh1, Sh, Qh)
    ∫D(η->begin
        argGsPF_continuous(η, qh0, qh1, sh0, sh1, Sh, Qh)
    end)
end

function argGsPF_continuous1(z0, qh0, qh1, sh0, sh1, Sh, Qh)
    ferf = qh1 > qh0 ? erfi : erf
    Δqh = abs(qh1-qh0)
    I0 = √(π/(2Δqh)) * (ferf((Δqh+√qh0*z0)/√(2*Δqh)) + ferf((Δqh-√qh0*z0)/√(2*Δqh)))

    a = √((Qh*qh0*(qh0 - qh1) + qh1*sh0^2 + qh0*sh1*(-2*sh0 + sh1))/
            ((qh0 - qh1)*(-sh0^2 + qh0*(Qh + (sh0 - sh1)^2))))
    b = z0*qh0/(qh0 - qh1) * (sh1-sh0) / √(Qh*qh0 - sh0^2 + qh0*(sh1 - sh0)^2)
    AB = √((Qh*qh0 - sh0^2)/qh0 +(sh1-sh0)^2)

    A = √((Qh*qh0 - sh0^2)/qh0) / AB
    B = (sh1-sh0) / AB
    C = B^2 +(qh0-qh1)*A^2
    aC = abs(C)
    ferfp = C > 0 ? (x,y)->√(π/(2aC)) * (erf(x/√(2aC)/A) - erf(y/√(2aC)/A)) :
                    (x,y)->√(π/(2aC)) * (-erfi(x/√(2aC)/A) + erfi(y/√(2aC)/A))
    Ip = ∫D(η-> begin
        Θ = a*η+b
        D = A*B*Θ+√qh0*z0*A + A*(qh1-qh0)*B*Θ
        return ferfp(A*D+C*B*Θ, A*D+C*(B*Θ-1)) *
                log(2cosh(AB*Θ + Sh + sh0/√qh0*z0))
        end)

    ferfm = C > 0 ? (x,y)->√(π/(2aC)) * (-erf(x/√(2aC)/A) + erf(y/√(2aC)/A)) :
                    (x,y)->√(π/(2aC)) * (erfi(x/√(2aC)/A) - erfi(y/√(2aC)/A))
    Im = ∫D(η-> begin
        Θ = a*η+b
        D = A*B*Θ+√qh0*z0*A + A*(qh1-qh0)*B*Θ
        return ferfm(A*D+C*B*Θ, A*D+C*(B*Θ+1)) *
                log(2cosh(AB*Θ - Sh + sh0/√qh0*z0))
        end)

    return a*(Ip + Im)/I0
end
#
# function argGsPF_continuous1(z0, qh0, qh1, sh0, sh1, Sh, Qh)
#     a = √((Qh*qh0-sh0^2) / qh0)
#     b = sh0 /√qh0
#
#     den = ∫d(w->exp((qh1-qh0)*w^2/2+√qh0*z0*w), -1., 1.)
#
#     num = ∫d(w-> begin
#         c0p = exp((qh1-qh0)*w^2/2+√qh0*z0*w)
#         cp = ∫D(η->log(2cosh((sh1-sh0)*w + Sh + a*η + b*z0)))
#         c0m = exp((qh1-qh0)*w^2/2-√qh0*z0*w)
#         cm = ∫D(η->log(2cosh(-(sh1-sh0)*w - Sh + a*η + b*z0)))
#         return c0p*cp+c0m*cm
#     end, 0., 1.)
#
#     return num/den
# end

function GsPF_continuous1(qh0, qh1, sh0, sh1, Sh, Qh)
    ∫D(z0->begin
        argGsPF_continuous1(z0, qh0, qh1, sh0, sh1, Sh, Qh)
    end)
end


function argGsPF_binary(z0, qh0, qh1, sh0, sh1, Qh)
    b = sqrt((Qh*qh0-sh0^2)/qh0)
    ez0 = exp(√qh0*z0)
    ez0inv = 1/ez0

    ∫D(η->begin
        y = sh0/√qh0*z0 + b*η
        ez0*log(2cosh(y + (sh1-sh0))) + ez0inv *log(2cosh(y - (sh1-sh0)))
    end)
end

function GsPF_binary(qh0, qh1, sh0, sh1, Qh)
    # x = [-∞:0.1:∞;]
    # y = Float64[ argGsPF_binary(x, qh0, qh1, sh0, sh1, Qh) for x in x]
    # f = Spline1D(x, y)
    # return ∫D(z0->begin
    #     f(z0) / (2cosh(z0 * √qh0))
    # end)
    ∫D(z0->begin
        argGsPF_binary(z0, qh0, qh1, sh0, sh1, Qh) / (2cosh(z0 * √qh0))
    end)
end
####

function free_entropyPF(ep, op, vartype, enetype)
    @extract ep : α β qs S
    @extract op : q0 s0 s1 Q qh0 qh1 sh0 sh1 Sh Qh
    ϕ = -0.5*Qh*(1-Q) + sh0*s0 - sh1*s1 - Sh*S
    if vartype == :continuous
        ϕ +=  GsPF_continuous(qh0, qh1, sh0, sh1, Sh, Qh)
    elseif vartype == :continuous1
        ϕ +=  GsPF_continuous1(qh0, qh1, sh0, sh1, Sh, Qh)
    elseif vartype == :binary
        ϕ +=  GsPF_binary(qh0, qh1, sh0, sh1, Qh)
    end
    if enetype == :theta
        ϕ += α*GePF_theta(β, q0, qs, s0, s1, Q)
    elseif enetype == :loglike
        ϕ += α*GePF_loglike(β, q0, qs, s0, s1, Q)
    end
    return ϕ
end


function all_therm_funcPF(ep::ExtParamsPF, op::OrderParamsPF, vartype, enetype)
    ϕ = free_entropyPF(ep, op, vartype, enetype)
    return ThermFuncPF(ϕ)
end

#################

fs0PF_binary(qh0, qh1, sh0, sh1, Qh) = -deriv(GsPF_binary, 3, qh0, qh1, sh0, sh1, Qh)
fs1PF_binary(qh0, qh1, sh0, sh1, Qh) = deriv(GsPF_binary, 4, qh0, qh1, sh0, sh1, Qh)
fQPF_binary(qh0, qh1, sh0, sh1, Qh) = 1-2*deriv(GsPF_binary, 5, qh0, qh1, sh0, sh1, Qh)

fs0PF_continuous(qh0, qh1, sh0, sh1, Sh, Qh) = -deriv(GsPF_continuous, 3, qh0, qh1, sh0, sh1, Sh, Qh)
fs1PF_continuous(qh0, qh1, sh0, sh1, Sh, Qh) = deriv(GsPF_continuous, 4, qh0, qh1, sh0, sh1,  Sh, Qh)
fSPF_continuous(qh0, qh1, sh0, sh1, Sh, Qh) = deriv(GsPF_continuous, 5, qh0, qh1, sh0, sh1, Sh, Qh)
fQPF_continuous(qh0, qh1, sh0, sh1, Sh, Qh) = 1-2*deriv(GsPF_continuous, 6, qh0, qh1, sh0, sh1, Sh, Qh)

fs0PF_continuous1(qh0, qh1, sh0, sh1, Sh, Qh) = -deriv(GsPF_continuous1, 3, qh0, qh1, sh0, sh1, Sh, Qh)
fs1PF_continuous1(qh0, qh1, sh0, sh1, Sh, Qh) = deriv(GsPF_continuous1, 4, qh0, qh1, sh0, sh1,  Sh, Qh)
fSPF_continuous1(qh0, qh1, sh0, sh1, Sh, Qh) = deriv(GsPF_continuous1, 5, qh0, qh1, sh0, sh1, Sh, Qh)
fQPF_continuous1(qh0, qh1, sh0, sh1, Sh, Qh) = 1-2*deriv(GsPF_continuous1, 6, qh0, qh1, sh0, sh1, Sh, Qh)
# fs0PF_continuous1(qh0, qh1, sh0, sh1, Sh, Qh) = -deriv∫D(argGsPF_continuous1, 3, qh0, qh1, sh0, sh1, Sh, Qh)
# fs1PF_continuous1(qh0, qh1, sh0, sh1, Sh, Qh) = deriv∫D(argGsPF_continuous1, 4, qh0, qh1, sh0, sh1,  Sh, Qh)
# fSPF_continuous1(qh0, qh1, sh0, sh1, Sh, Qh) = deriv∫D(argGsPF_continuous1, 5, qh0, qh1, sh0, sh1, Sh, Qh)
# fQPF_continuous1(qh0, qh1, sh0, sh1, Sh, Qh) = 1-2*deriv∫D(argGsPF_continuous1, 6, qh0, qh1, sh0, sh1, Sh, Qh)

function iShPF_continuous(S, qh0, qh1, sh0, sh1, Sh_0, Qh)
    ok, Sh, it, normf0 = findroot(Sh -> S - fSPF_continuous(qh0, qh1, sh0, sh1, Sh, Qh), Sh_0,
                            InterpolationMethod(), atol=1e-7, n=5)
    ok || normf0 < 1e-7 || warn("findroot failed: z=$(Sh_0), it=$it, normf0=$normf0")
    ok = true
    if normf0 > 1e-3
        ok = false
    end
    return ok, Sh
end

function iShPF_continuous1(S, qh0, qh1, sh0, sh1, Sh_0, Qh)
    ok, Sh, it, normf0 = findroot(Sh -> S - fSPF_continuous1(qh0, qh1, sh0, sh1, Sh, Qh), Sh_0,
                            InterpolationMethod(), atol=1e-7, n=5)
    ok || normf0 < 1e-7 || warn("findroot failed: z=$(Sh_0), it=$it, normf0=$normf0")
    ok = true
    if normf0 > 1e-3
        ok = false
    end
    return ok, Sh
end

function ish1PF_binary(s1, qh0, qh1, sh0, sh1_0, Qh)
    ok, sh1, it, normf0 = findroot(sh1 -> s1 - fs1PF_binary(qh0, qh1, sh0, sh1, Qh), sh1_0,
                            InterpolationMethod(), atol=1e-7, n=5)

    ok || normf0 < 1e-7 || warn("findroot failed: z=$sh1, it=$it, normf0=$normf0")
    ok = true
    if normf0 > 1e-3
        ok = false
    end
    return ok, sh1
end

fsh0PF_theta(α, β, q0, qs, s0, s1, Q) = -α*deriv(GePF_theta, 4, β, q0, qs, s0, s1, Q)
fsh1PF_theta(α, β, q0, qs, s0, s1, Q) = α*deriv(GePF_theta, 5, β, q0, qs, s0, s1, Q)
fQhPF_theta(α, β, q0, qs, s0, s1, Q) = -2α*deriv(GePF_theta, 6, β, q0, qs, s0, s1, Q)

fsh0PF_loglike(α, β, q0, qs, s0, s1, Q) = -α*deriv(GePF_loglike, 4, β, q0, qs, s0, s1, Q)
fsh1PF_loglike(α, β, q0, qs, s0, s1, Q) = α*deriv(GePF_loglike, 5, β, q0, qs, s0, s1, Q)
fQhPF_loglike(α, β, q0, qs, s0, s1, Q) = -2α*deriv(GePF_loglike, 6, β, q0, qs, s0, s1, Q)

# fsh1PF_theta(α, q0, qs, s0, s1, Q) = α*deriv(GePF_theta, 4, q0, qs, s0, s1, Q))

###########
function fix!(ep::ExtParamsPF, op::OrderParamsPF)
    @extract ep : qs
    @extract op : q0 s0 s1 Q qh0 qh1 sh0 sh1 Sh Qh
    if Qh*qh0  - sh0^2 < 0
        op.Qh = sh0^2/qh0 + 1e-3
        warn("fix Qh: $Qh -> $(op.Qh)")
    end
    #
    # if s0 > s1
    #     s1, s0 = s0, s1
    #     op.s1 = s1
    #     op.s0 = s0
    #     warn("fix s0 s1: $s0 $s1")
    # end
    d = Q*q0  - s0^2
    if d < 0
        print("!")
        while d < 0
            s0 *=(1-1e-2)
            d = Q*q0  - s0^2
        end
        op.s0 = s0
        warn("fix1 s0: $s0")
    end
    #
    # d = qs-q0 - (s1-s0)^2/(Q-s0) *(1 - s0*(q0-s0)/(Q*q0-s0^2))
    # if d < 0
    #     while d < 0
    #         Q *= (1+1e-3)
    #         d = qs-q0 - (s1-s0)^2/(Q-s0) *(1 - s0*(q0-s0)/(Q*q0-s0^2))
    #     end
    #     op.Q = Q
    #     warn("fix1 Q: $Q")
    # end

    d = qs-q0 - (s1-s0)^2/(Q-s0) *(1 - s0*(q0-s0)/(Q*q0-s0^2))
    if d < 0
        print("!d ")
        # if s1 > s0
        #     while d < 0
        #         s0 *= (1+1e-3)
        #         d = qs-q0 - (s1-s0)^2/(Q-s0) *(1 - s0*(q0-s0)/(Q*q0-s0^2))
        #     end
        #     op.s0 = s0
        #     warn("fix2 s0: $s0")
        # else
            while d < 0
                Q *= (1+1e-3)
                d = qs-q0 - (s1-s0)^2/(Q-s0) *(1 - s0*(q0-s0)/(Q*q0-s0^2))
            end
            op.Q = Q
            warn("fix1 Q: $Q")
        # end
    end


end
############

function convergePF!(ep::ExtParamsPF, op::OrderParamsPF, pars::Params,
                    vartype::Symbol, enetype::Symbol; withSh::Bool=true)
    @extract pars : maxiters verb ϵ ψ

    if vartype == :binary
        op.s1 = ep.S
        op.Sh = 0
        op.qh1 = 0
    end

    # converge central replica
    ep2 = ExtParams(ep.α, ep.β, ep.qs)
    pars2 = deepcopy(pars)
    pars2.verb = 2
    pars2.ψ = 0
    op2 = OrderParams(op.q0, op.qh0, op.qh1)
    converge!(ep2, op2, pars2, vartype, enetype)
    op.q0 = op2.q0
    op.qh0 = op2.qh0
    op.qh1 = op2.qh1

    Δ = Inf
    ok = false
    println(op)
    it = 0
    for it = 1:maxiters
        Δ = 0.0
        oki = true
        verb > 1 && println("it=$it")

        if enetype == :theta
            @update  op.Qh      fQhPF_theta     Δ ψ verb  ep.α ep.β op.q0 ep.qs op.s0 op.s1 op.Q
            @update  op.sh0     fsh0PF_theta    Δ ψ verb  ep.α ep.β op.q0 ep.qs op.s0 op.s1 op.Q
            if vartype != :binary
                @update  op.sh1     fsh1PF_theta    Δ ψ verb  ep.α ep.β op.q0 ep.qs op.s0 op.s1 op.Q
            end
        elseif enetype == :loglike
            @update  op.Qh      fQhPF_loglike     Δ ψ verb  ep.α ep.β op.q0 ep.qs op.s0 op.s1 op.Q
            @update  op.sh0     fsh0PF_loglike    Δ ψ verb  ep.α ep.β op.q0 ep.qs op.s0 op.s1 op.Q
            if vartype != :binary
                @update  op.sh1     fsh1PF_loglike    Δ ψ verb  ep.α ep.β op.q0 ep.qs op.s0 op.s1 op.Q
            end
        end

        fix!(ep, op)

        if vartype == :binary
            @update  op.Q       fQPF_binary        Δ ψ verb  op.qh0 op.qh1 op.sh0 op.sh1 op.Qh
            @update  op.s0      fs0PF_binary       Δ ψ verb  op.qh0 op.qh1 op.sh0 op.sh1 op.Qh
            @updateI op.sh1 oki  ish1PF_binary      Δ ψ verb  op.s1 op.qh0 op.qh1 op.sh0 op.sh1 op.Qh
        elseif vartype == :continuous
            @update  op.Q       fQPF_continuous        Δ ψ verb  op.qh0 op.qh1 op.sh0 op.sh1 op.Sh op.Qh
            @update  op.s0      fs0PF_continuous       Δ ψ verb  op.qh0 op.qh1 op.sh0 op.sh1 op.Sh op.Qh
            @update  op.s1      fs1PF_continuous       Δ ψ verb  op.qh0 op.qh1 op.sh0 op.sh1 op.Sh op.Qh
            if withSh
                @update ep.S    fSPF_continuous       Δ ψ verb op.qh0 op.qh1 op.sh0 op.sh1 op.Sh op.Qh
            else
                @updateI op.Sh oki   iShPF_continuous       Δ ψ verb  ep.S op.qh0 op.qh1 op.sh0 op.sh1 op.Sh op.Qh
            end
        elseif vartype == :continuous1
            @update  op.Q       fQPF_continuous1        Δ ψ verb  op.qh0 op.qh1 op.sh0 op.sh1 op.Sh op.Qh
            @update  op.s0      fs0PF_continuous1       Δ ψ verb  op.qh0 op.qh1 op.sh0 op.sh1 op.Sh op.Qh
            @update  op.s1      fs1PF_continuous1       Δ ψ verb  op.qh0 op.qh1 op.sh0 op.sh1 op.Sh op.Qh
            if withSh
                @update ep.S  fSPF_continuous1       Δ ψ verb  op.qh0 op.qh1 op.sh0 op.sh1 op.Sh op.Qh
            else
                @updateI op.Sh oki iShPF_continuous1       Δ ψ verb  ep.S op.qh0 op.qh1 op.sh0 op.sh1 op.Sh op.Qh
            end
        end

        fix!(ep, op)

        verb > 1 && println(" Δ=$Δ\n")
        ok = Δ < ϵ && oki
        ok && break
    end

    if verb > 0
        println(ok ? "converged" : "failed", " (it=$it Δ=$Δ)")
        println(ep)
        println(op)
    end
    return ok
end

function convergePF(; α = 0.2, β=Inf, qs = 1.,
                    q0=0.13501842754948257,
                    qh0=0.2307230899667445,
                    qh1=-1.0764979782649626,
                    Qh = 0.06460351429903043,
                    sh1 = 0.17240741966211987,
                    sh0 = 0.06819069072915576,
                    Q = 0.07666022140353401,
                    S = 0.15769587024409643,
                    s0 = S,
                    s1 = S,
                    Sh = 0,
                    vartype = :binary, # :binary, :continuous, :continuous1
                    enetype = :theta, # :theta, :loglike
                    ϵ = 1e-5, ψ = 0.0, maxiters = 1_000, verb = 3)

    ep = ExtParamsPF(α, β, qs, S)
    pars = Params(ϵ, ψ, maxiters, verb)
    op = OrderParamsPF(q0, s0, s1, Q, qh0, qh1, sh0, sh1, Sh, Qh)
    convergePF!(ep, op, pars, vartype, enetype)

    tf = all_therm_funcPF(ep, op, vartype, enetype)
    println(tf)

    return ep, op, tf, pars
end

"""
    readparamsPF(file::String, line::Int=-1)

Read order and external params from results file.
Zero or negative line numbers are counted
from the end of the file.
"""
function readparamsPF(file::String, line::Int=0)
    lines = readlines(file)
    l = line > 0 ? line : length(lines) + line
    v = map(x->parse(Float64, x), split(lines[l]))

    i0 = length(fieldnames(ExtParamsPF))
    iend = i0 + length(fieldnames(OrderParamsPF))
    return ExtParamsPF(v[1:i0]...), OrderParamsPF(v[i0+1:iend]...)
end

function spanPF(; q0=0.13,
                qh0=0.23,
                qh1=-1.0,
                Qh = 0.06,
                sh1 = 0.17,
                sh0 = 0.06,
                Q = 0.07,
                s0 = 0.06,
                s1 = 0.1,
                Sh = sh1,
                kws...)

        op = OrderParamsPF(q0, s0, s1, Q, qh0, qh1, sh0, sh1, Sh, Qh)
        spanPF!(op; kws...)
end

function spanPF!(op::OrderParamsPF; β=Inf,
                qslist = [1.],
                Slist = [0.157],
                Shlist = op.Sh,
                withSh = true,
                αlist = [0.1],
                vartype = :binary, # :binary, :continuous, :continuous1
                enetype = :theta, # :theta, :loglike
                ϵ = 1e-5, ψ = 0.0, maxiters = 1_000, verb = 3,
                resfile = "resultsPF_$(vartype)_$(enetype).txt")

    @assert vartype in [:continuous, :continuous1, :binary]
    @assert enetype in [:loglike, :theta]

    lockfile = resfile *".lock"
    if !isfile(resfile)
        open(resfile, "w") do f
            allheadersshow(f, ExtParamsPF, OrderParamsPF, ThermFuncPF)
        end
    end

    ep = ExtParamsPF(first(αlist), β, first(qslist), first(Slist))
    pars = Params(ϵ, ψ, maxiters, verb)
    results = []

    for S in Slist, α in αlist, qs in qslist, Sh in Shlist
        println("\n########  NEW ITER  ########\n")
        if withSh
            op.Sh = Sh
        else
            ep.S = S
        end
        ep.α = α
        ep.qs = qs
        verb > 0 && println(ep)

        ok = convergePF!(ep, op, pars, vartype, enetype, withSh=withSh)
        tf = all_therm_funcPF(ep, op, vartype, enetype)
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
