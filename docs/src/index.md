# ChainPlots.jl documentation

The aim of `ChainPlots.jl` is to provide a visualization of the topology of a neural network as constructed with `Flux.Chain`. It does that by building a [`ChainGraph`](@ref) of the neural network, with [`chaingraph`](@ref), and by applying a `Plots.jl` recipe to the `ChainGraph`.

```julia
using Flux, Plots, ChainPlots

m = Chain(Dense(2 => 5, σ), RNN(5 => 4, relu), Dense(4 => 3))
cg = chaingraph(m)  # the graph of neurons and connections
plot(cg)            # or, in one step, plotchain(m)
```

If the first layer accepts inputs of any size, such as a convolutional layer, pass an input, or the shape of an input, as in `plotchain(m, (16, 16, 1, 1))`.

Upgrading from ChainPlots 0.1? See [Migrating from 0.1 to 0.2](@ref).
