module Perceptron
using LittleScienceTools.Roots
using ExtractMacro
using Dierckx
using AutoGrad
if VERSION > v"0.5"
    using SpecialFunctions
end
import QuadGK: quadgk

@assert success(`which lockfile`)

function exclusive(f::Function, fn::AbstractString = "lock.tmp")
    run(`lockfile -1 $fn`)
    try
        f()
    finally
        run(`rm -f $fn`)
    end
end

G(x) = exp(-(x^2)/2) / √(2π)
H(x) = erfc(x / √2) / 2
Hi(x) = 0.5*(1-erfi(x / √2)) / 2
function GHapp(x)
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4y2)))
end
GH(x) = ifelse(x > 30.0, GHapp(x), G(x) / H(x))
GH(x::BigFloat) = G(x) / H(x)

logH(x) = x < -35.0 ? G(x) / x :
          x >  35.0 ? -x^2 / 2 - log(2π) / 2 - log(x) :
          log(H(x))

const ∞ = 30.0
const dx = 0.1 #0.005

interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-12, maxevals=10^5)[1]

∫d(f, a, b) = quadgk(f, union([a:dx:b;],b)..., abstol=1e-12, maxevals=10^5)[1]

function deriv(f::Function, i::Integer, x...; δ::Float64 = 1e-5)
    f0 = f(x[1:i-1]..., x[i]-δ, x[i+1:end]...)
    f1 = f(x[1:i-1]..., x[i]+δ, x[i+1:end]...)
    return (f1-f0) / 2δ
    # return grad(f, i)(x...)
end

function grad∫D(f::Function, i::Integer, x...)
    g = grad(f, i+1)
    return ∫D(z->g(z,x...))
end

function shortshow(io::IO, x)
    T = typeof(x)
    print(io, T.name.name, "(", join([string(f, "=", getfield(x, f)) for f in fieldnames(T)], ","), ")")
end

function headershow(io::IO, T::Type, i0 = 0)
    print(io, join([string(i+i0,"=",f) for (i,f) in enumerate(fieldnames(T))], " "))
    return i0 + length(fieldnames(T))
end

function allheadersshow(io::IO, x...)
    i0 = 0
    print(io, "#")
    for y in x
        i0 = headershow(io, y, i0)
        print(io, " ")
    end
    println(io)
end

function plainshow(x)
    T = typeof(x)
    join([getfield(x, f) for f in fieldnames(T)], " ")
end


abstract AbstractParams

type Params <: AbstractParams
    ϵ::Float64
    ψ::Float64
    maxiters::Int
    verb::Int
end

Base.show(io::IO, params::AbstractParams) = shortshow(io, params)


macro update(x, func, Δ, ψ, verb, params...)
    n = string(x.args[2].args[1])
    x = esc(x)
    Δ = esc(Δ)
    ψ = esc(ψ)
    verb = esc(verb)
    params = map(esc, params)
    fcall = Expr(:call, func, params...)
    quote
        oldx = $x
        newx = $fcall
        $Δ = max($Δ, abs(newx - oldx) / ((abs(newx) + abs(oldx)) / 2))
        $x = (1 - $ψ) * newx + $ψ * oldx
        $verb > 1 && println("  ", $n, " = ", $x)
    end
end

macro updateI(x, ok, func, Δ, ψ, verb, params...)
    n = string(x.args[2].args[1])
    x = esc(x)
    ok = esc(ok)
    Δ = esc(Δ)
    ψ = esc(ψ)
    verb = esc(verb)
    params = map(esc, params)
    fcall = Expr(:call, func, params...)
    quote
        oldx = $x
        $ok, newx = $fcall
        abserr = abs(newx - oldx)
        relerr = abserr / ((abs(newx) + abs(oldx)) / 2)
        $Δ = max($Δ, min(abserr, relerr))
        $x = (1 - $ψ) * newx + $ψ * oldx
        $verb > 1 && println("  ", $n, " = ", $x)
    end
end

include("standard_RS.jl")
include("parisi_franz.jl")


end # module
