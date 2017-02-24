qh0=0.20562803659345238
qh1=11.463722205728125
@testset "continuous1" begin
    for z0=-20:1.:20
        # ENTROPIC TERM
        b = √qh0 * z0
        a = qh1 - qh0
        # res = -b^2/(2a) + 0.5*log(π/2) - 0.5*log(a)
        # res2 = res

        res = log(erfi((a-b)/√(2a)) + erfi((a+b)/√(2a)))
        res2 = log1p(erfi((a-b)/√(2a) / erfi((a+b)/√(2a)))) + log(erfi((a+b)/√(2a)))

        println("z0=$z0")
        @test res ≈ res2
    end
end
