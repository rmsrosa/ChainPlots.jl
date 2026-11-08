# ChainGraph examples

The function [`chaingraph`](@ref) builds a [`ChainGraph`](@ref) of a neural network, with a vertex for each neuron and an edge for each connection between neurons of consecutive layers. This is what the plots are made of, and it can also be used directly.

```@example graphs
using Flux
using Graphs
using ChainPlots

nnr = Chain(Dense(2 => 5, σ), RNN(5 => 4, relu), LSTM(4 => 4), GRU(4 => 4), Dense(4 => 3))
cg = chaingraph(nnr)
```

If the first layer accepts inputs of any size, an input, or the shape of an input, is needed:

```@example graphs
chaingraph(Chain(Conv((3, 3), 1 => 2), MaxPool((2, 2))), (6, 6, 1, 1))
```

## The Graphs.jl interface

A `ChainGraph` is a `Graphs.AbstractGraph`, so the functions of `Graphs.jl` work on it, as do packages built on it, such as `GraphPlot.jl`:

```@example graphs
nv(cg), ne(cg)
```

```@example graphs
neighbors(cg, 1)
```

## Neuron properties

The properties of a neuron, listed in [`NODE_PROPERTIES`](@ref ChainPlots.NODE_PROPERTIES), are given by [`nodeprops`](@ref):

```@example graphs
nodeprops(cg, 8)
```

and one property at a time by [`nodeprop`](@ref), which can be broadcast over the vertices:

```@example graphs
nodeprop.(Ref(cg), vertices(cg), :layer_number)
```

## Layer properties

The dimensions and names of the layers, starting with the input layer, are given by [`layer_dimensions`](@ref) and [`layer_names`](@ref):

```@example graphs
layer_dimensions(cg)
```

```@example graphs
layer_names(cg)
```

## Plotting

A `ChainGraph` is plotted with `Plots.jl`:

```@example graphs
using Plots
plot(cg, title="$nnr", titlefontsize=7)
```
