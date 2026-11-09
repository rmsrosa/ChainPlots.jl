"""
    AbstractPlotBackend

Supertype of the plotting packages [`plotchain`](@ref) can use, [`PlotsBackend`](@ref) and
[`MakieBackend`](@ref). Each is supported by a package extension, loaded with the package.
"""
abstract type AbstractPlotBackend end

"""
    PlotsBackend()

Plot with Plots.jl, loaded with `using Plots`. See [`plotchain`](@ref).
"""
struct PlotsBackend <: AbstractPlotBackend end

"""
    MakieBackend()

Plot with Makie.jl, loaded with one of its backends, such as `using CairoMakie` or
`using GLMakie`. See [`plotchain`](@ref).
"""
struct MakieBackend <: AbstractPlotBackend end

"""
    plotbackend() -> AbstractPlotBackend

Return the backend for the plotting package that is loaded, Plots.jl or Makie.jl.

Throw an `ArgumentError` if neither is loaded, or if both are, in which case pass the
backend to [`plotchain`](@ref) explicitly.
"""
function plotbackend()
    plots = !isnothing(Base.get_extension(@__MODULE__, :ChainPlotsPlotsExt))
    makie = !isnothing(Base.get_extension(@__MODULE__, :ChainPlotsMakieExt))
    plots && makie && throw(ArgumentError("Both Plots.jl and Makie.jl are loaded; choose one with `plotchain(ChainPlots.PlotsBackend(), m, ...)` or `plotchain(ChainPlots.MakieBackend(), m, ...)`."))
    plots && return PlotsBackend()
    makie && return MakieBackend()
    throw(ArgumentError("Plotting needs Plots.jl or Makie.jl: run `using Plots`, or `using CairoMakie` (or another Makie backend), first."))
end

"""
    plotchain([backend,] m, input = nothing; kwargs...)

Plot the topology of the neural network `m`, which is `plot(chaingraph(m, input); kwargs...)`.

It works with Plots.jl and with Makie.jl, whichever is loaded, as in `using Plots` or
`using CairoMakie`. If both are loaded, choose one by passing `ChainPlots.PlotsBackend()`
or `ChainPlots.MakieBackend()` as the first argument.

`m` can be a `Flux.Chain`, a `Lux.Chain`, or a single layer. If the first layer accepts an
input with arbitrary dimensions, such as a convolutional or a functional layer, an `input`
must be given, either as an `Array` or as a `Tuple` with the dimensions of the input.

Any other keyword argument is passed on to `plot`, such as `title` with Plots.jl, or
`axis = (; title = "...")` with Makie.jl, as well as the attributes of the plot recipe for
[`ChainGraph`](@ref):

* `connection_color`: color of the connections;
* `neuron_colors = NEURON_COLORS`: either a `Dict` from layer types to colors, as in
  [`NEURON_COLORS`](@ref), or a color for all neurons;
* `neuron_shape = :auto`: `:auto` for a marker depending on the type of the layer, or a
  marker shape for all neurons.

With Plots.jl, it returns a `Plots.Plot`; with Makie.jl, a `Makie.FigureAxisPlot`.
"""
plotchain(m, input=nothing; kwargs...) = plotchain(plotbackend(), m, input; kwargs...)
plotchain(b::AbstractPlotBackend, m, input=nothing; kwargs...) = _missing_backend(b)

"""
    plotchain!([backend,] [target,] m, input = nothing; kwargs...)

Add the topology of the neural network `m` to the current plot, or to `target`, which is a
`Plots.Plot` with Plots.jl, or an axis, such as a `Makie.Axis`, with Makie.jl.

See [`plotchain`](@ref).
"""
plotchain!(m, input=nothing; kwargs...) = plotchain!(plotbackend(), m, input; kwargs...)
plotchain!(b::AbstractPlotBackend, m, input=nothing; kwargs...) = _missing_backend(b)

_missing_backend(::PlotsBackend) = throw(ArgumentError("`PlotsBackend` needs Plots.jl: run `using Plots` first."))
_missing_backend(::MakieBackend) = throw(ArgumentError("`MakieBackend` needs Makie.jl: run `using CairoMakie` (or another Makie backend) first."))
