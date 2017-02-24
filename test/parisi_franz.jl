resfile, _ = mktemp()

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
