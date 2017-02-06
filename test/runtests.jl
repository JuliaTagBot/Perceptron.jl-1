include("../src/Perceptron.jl")
using Perceptron
using Base.Test

resfile, _ = mktemp()

# RS
res = Perceptron.span( αlist=[0.2:0.1:0.6;], enetype=:theta, vartype=:binary, resfile=resfile)
res = Perceptron.span( αlist=[0.2:0.1:0.6;], enetype=:continuous, vartype=:binary, resfile=resfile)


# Parisi Franz
file = joinpath(dirname(@__FILE__), "../data/resultsPF_binary_theta.txt")
ep, op = Perceptron.readparamsPF(file)
res = Perceptron.spanPF!(op;
                    αlist=ep.α,Slist=ep.S, qslist=ep.qs,
                    enetype=:theta, vartype=:binary, resfile=resfile,
                    ψ = 0.3
                    )

file = joinpath(dirname(@__FILE__), "../data/resultsPF_continuous_theta.txt")
ep, op = Perceptron.readparamsPF(file)
res = Perceptron.spanPF!(op;
                    αlist=ep.α,Slist=ep.S, qslist=ep.qs,
                    enetype=:theta, vartype=:continuous, resfile=resfile,
                    ψ = 0.3
                    )

file = joinpath(dirname(@__FILE__), "../data/resultsPF_continuous1_theta.txt")
ep, op = Perceptron.readparamsPF(file)
res = Perceptron.spanPF!(op;
                    αlist=ep.α,Slist=ep.S, qslist=ep.qs,
                    enetype=:theta, vartype=:continuous1, resfile=resfile,
                    ψ = 0.3
                    )
