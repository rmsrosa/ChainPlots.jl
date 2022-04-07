include("../src/ChainPlots.jl")

using Flux
using Plots
using Test
using .ChainPlots

import .ChainPlots.NeuralNumbers: cold, hot, fneutralize

m = Chain(Conv((2,), 1 => 1))
fm = fneutralize(m)

inp = [1; 0; 0; 0; 0;;;]
@profview m(inp)

inp = [hot; cold; cold; cold; cold;;;]

@test m(inp) == [hot; cold; cold; cold;;;]
@test fm(inp) == [hot; cold; cold; cold;;;]

inp = [cold; hot; cold; cold; cold;;;]

@test m(inp) == [hot; hot; cold; cold;;;]
@test fm(inp) == [hot; hot; cold; cold;;;]

inp = [cold; cold; cold; cold; hot;;;]

@test m(inp) == [cold; cold; cold; hot;;;]
@test fm(inp) == [cold; cold; cold; hot;;;]

@profview m(inp)

m2d = Chain(Conv((2,2), 1 => 1))
fm2d = fneutralize(m2d)

m2d([1 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0;;;;])

inp2d = [hot cold cold;
       cold cold cold;
       cold cold cold;
       cold cold cold;
       cold cold cold;;;;]

m2d(inp2d)
fm2d(inp2d)

@profview ChainPlots.neuron_connections(m2d, inp2d)

@profview plot(m2d, [6 5 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0;;;;])
