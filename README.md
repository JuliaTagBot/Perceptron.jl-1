# Perceptron.jl
Code for replica analysis of the perceptron problem written in Julia.
```julia
Pkg.clone("https://github.com/CarloLucibello/LittleScienceTools.jl")
Pkg.clone("https://github.com/CarloLucibello/Perceptron.jl")
```
Supported `vartype`:
- `:binary`: ±1
- `:continuous`: real values with spherical constraint (`qs`, default to 1)
- `:continuous1`: real values in `[-1,1]` and global spherical norm `qs^2`

Supported `enetype`:
- `:theta`
- `:loglike`

## Entropy calculation
```julia
res = Perceptron.span( αlist=[0.2:0.1:0.6;], enetype=:theta, vartype=:binary, resfile=resfile)
```
## Parisi-Franz
```julia
res = Perceptron.spanPF(
            Slist=[0.92],#distances
            αlist=[0.7],
            ψ=0.3, #dumping
            enetype=:theta,
            vartype=:binary
            q0=0.46257,
            s0=0.47225,
            Q=0.879382384,
            qh0=1.45359134,
            sh0=1.196264792,
            sh1=2.96155611,
            Qh=1.518234)
```
