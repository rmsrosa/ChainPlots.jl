# ChainPlots.jl documentation

The aim of `ChainPlots.jl` is to provide a visualization of the topology of a neural network as constructed with `Flux.Chain`. It does that by building a [`ChainGraph`](@ref) of the neural network, with [`chaingraph`](@ref), and by applying a `Plots.jl` recipe to the `ChainGraph`.

```julia
using Flux, Plots, ChainPlots

m = Chain(Dense(2 => 5, σ), RNN(5 => 4, relu), Dense(4 => 3))
cg = chaingraph(m)  # the graph of neurons and connections
plot(cg)            # or, in one step, plotchain(m)
```

Plotting works with either Plots.jl or Makie.jl, both optional dependencies: the plot recipes and [`plotchain`](@ref) are loaded with them, as package extensions. So run `using Plots`, or load a Makie backend, as in `using CairoMakie`, to plot, while [`chaingraph`](@ref) works without either; see [Plotting with Makie](@ref).

If the first layer accepts inputs of any size, such as a convolutional layer, pass an input, or the shape of an input, as in `plotchain(m, (16, 16, 1, 1))`.

Upgrading from ChainPlots 0.2 or 0.1? See [From 0.2 to 0.3](@ref) and [From 0.1 to 0.2](@ref).
