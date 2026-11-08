"""
    plotchain(m, input = nothing; kwargs...)

Plot the topology of the neural network `m`, which is `plot(chaingraph(m, input); kwargs...)`.

It needs Plots.jl to be loaded, as in `using Plots`.

`m` can be a `Flux.Chain` or a single layer. If the first layer accepts an input with
arbitrary dimensions, such as a convolutional or a functional layer, an `input` must be
given, either as an `Array` or as a `Tuple` with the dimensions of the input.

Any other keyword argument is passed on to `plot`, such as `title`, as well as the
attributes of the plot recipe for [`ChainGraph`](@ref):

* `connection_color = :gray68`: color of the connections;
* `neuron_colors = NEURON_COLORS`: either a `Dict` from layer types to colors, as in
  [`NEURON_COLORS`](@ref), or a color for all neurons;
* `neuron_shape = :auto`: `:auto` for a marker depending on the type of the layer, or a
  marker shape for all neurons.
"""
function plotchain end

"""
    plotchain!([plt,] m, input = nothing; kwargs...)

Add the topology of the neural network `m` to the current plot, or to `plt`.

It needs Plots.jl to be loaded, as in `using Plots`. See [`plotchain`](@ref).
"""
function plotchain! end

function __init__()
    # `plotchain` only has methods once the Plots extension is loaded
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f in (plotchain, plotchain!) && isnothing(Base.get_extension(@__MODULE__, :ChainPlotsPlotsExt))
            print(io, "\n`", exc.f, "` needs Plots.jl: run `using Plots` first.")
        end
    end
end
