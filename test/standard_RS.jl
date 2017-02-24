resfile, _ = mktemp()

# RS
res = Perceptron.span( αlist=[0.2:0.1:0.6;], enetype=:theta, vartype=:binary, resfile=resfile)
res = Perceptron.span( αlist=[0.2:0.1:0.6;], enetype=:continuous, vartype=:binary, resfile=resfile)
