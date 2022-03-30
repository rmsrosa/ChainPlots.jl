include("../src/ChainPlot.jl")
include("../src/NeuronNumbers.jl")

using Flux
using Plots
using Test
using .NeuronNumbers
using .ChainPlot

m = Chain(Conv((2,), 1 => 1))
fm = fcooloffneurons(m)

inp = [1; 0; 0; 0; 0;;;]
@profview m(inp)

inp = [hotneuron; coldneuron; coldneuron; coldneuron; coldneuron;;;]

@test m(inp) == [hotneuron; coldneuron; coldneuron; coldneuron;;;]
@test fm(inp) == [hotneuron; coldneuron; coldneuron; coldneuron;;;]

inp = [coldneuron; hotneuron; coldneuron; coldneuron; coldneuron;;;]

@test m(inp) == [hotneuron; hotneuron; coldneuron; coldneuron;;;]
@test fm(inp) == [hotneuron; hotneuron; coldneuron; coldneuron;;;]

inp = [coldneuron; coldneuron; coldneuron; coldneuron; hotneuron;;;]

@test m(inp) == [coldneuron; coldneuron; coldneuron; hotneuron;;;]
@test fm(inp) == [coldneuron; coldneuron; coldneuron; hotneuron;;;]

@profview m(inp)

m2d = Chain(Conv((2,2), 1 => 1))
fm2d = fcooloffneurons(m2d)

m2d([1 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0;;;;])

inp2d = [hotneuron coldneuron coldneuron;
       coldneuron coldneuron coldneuron;
       coldneuron coldneuron coldneuron;
       coldneuron coldneuron coldneuron;
       coldneuron coldneuron coldneuron;;;;]

m2d(inp2d)
fm2d(inp2d)

@profview ChainPlot.neuron_connections(m2d, inp2d)

@profview plot(m2d, [6 5 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0;;;;])
