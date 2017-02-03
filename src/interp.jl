gplfile = joinpath(dirname(@__FILE__),"interpolation.gpl")
function findzero_interp(f::Function, x0::Float64; dx::Float64 = 0.005, maxiter::Int = 10,
                        parallel::Bool = false, verb::Int = 1, ϵ = 1e-7)
    s = x0
    ok = false
    iter = 1
    normf0 = Inf
    while !ok && iter <= maxiter
        verb > 3 && println("# TRIAL $iter for findzero_interp")
        xmax = s + 2*dx
        xmin  = s - 2*dx
        r = collect(xmin:dx:xmax)
        if parallel
            refs = RemoteRef[]
            for i=1:length(r)
                push!(refs, @spawn f(r[i]))
            end

            f0 = [fetch(refs[i]) for i=1:length(r)]
        else
            f0 = [f(r[i]) for i=1:length(r)]
        end
        dummyfile, rf = mktemp()
        for i=1:length(r)
            println(rf, "$(r[i]) $(f0[i])")
        end
        close(rf)
        try
            s = float(readstring(`gnuplot -e "filename='$dummyfile'" $gplfile`))
        catch
            error("ERROR GNUPLOT")
        end
        rm(dummyfile)

        normf0 = abs(f(s))
        if normf0 < ϵ
            verb > 3 && println("# SUCCESS x* = $(s), normf = $normf0")
            ok =true
        else
            verb > 1 && warn("failed: x* = $(s), normf = $normf0")
            verb == 1 && iter > 4  && warn("failed: x* = $(s), normf = $normf0")
            ok =false
        end

        if !ok && f0[1]*f0[end] > 0
            dx *= 2
            verb > 3 && println("# dx=$dx")
        elseif !ok && f0[1]*f0[end] < 0
            dx /= 2
            verb > 3 && println("# dx=$dx")
        end
        iter += 1
    end
    return ok, s, iter, normf0
end
#
# using OffsetArrays
#
# function centeredvec(radius::Int)
#     return OffsetArray(zeros(2radius+1), -radius-1)
# end
#
# type Interpolator{F<:Function}
#     f::F
#     dx::Float64
#     xlim::Float64
#     L::Int
#     v::OffsetArray{Float64,1,Array{Float64,1}}
# end
#
# function Interpolator(f, dx::Float64, L::Int)
#     xlim = (L-2)*dx
#     v = centeredvec(L)
#     for i=-L:L
#         v[i] = f(dx*i)
#     end
#     return Interpolator(f, dx, xlim, L, v)
# end
# # call(fi::Interpolator, x) = fi.f(x)
# function (fi::Interpolator)(x::Float64)
#     if abs(x) < fi.xlim
#         i = Int(fld(x, fi.dx))
#         md = mod(x, fi.dx) / fi.dx
#         return @unsafe (1-md) * fi.v[i] + md * fi.v[i+1]
#     else
#         warn("Outside bound")
#         return fi.f(x)
#     end
# end
#
# using Base.Test
# fi = Interpolator(x->x^2, 0.1, 100)
# @test fi(1.) ≈ 1
# @test fi(-1.) ≈ 1
#
# # end #module
