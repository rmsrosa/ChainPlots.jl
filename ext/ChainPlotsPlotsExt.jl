module ChainPlotsPlotsExt

# The plot recipe for `ChainGraph` and the methods of `plotchain` and `plotchain!`,
# loaded with Plots.jl, which the recipe needs for the shapes of the recurrent neurons.

using ChainPlots: ChainPlots, ChainGraph, NEURON_COLORS, chaingraph, nlayers
import ChainPlots: plotchain, plotchain!
using Graphs: edges, vertices
using MetaGraphs: get_prop
using Plots
import Plots: RecipesBase

circle_verts = [(1.41 * sin(2π * n / 20), 1.41 * cos(2π * n / 20)) for n = -10:10]
lrnn_verts = [(1.2 * sin(2π * n / 20), 1.2 * (1 + cos(2π * n / 20))) for n = -10:10]
lstm_verts = vcat([(1.0 * sin(2π * n / 20), 1.2 + 1.0 * cos(2π * n / 20)) for n = -10:10],
    [(1.4 * sin(2π * n / 20), 1.2 + 1.4 * cos(2π * n / 20)) for n = -10:10])
lgru_verts = vcat([(0.5 * sin(2π * n / 20), 1.2 + 1.4 * cos(2π * n / 20)) for n = -10:10],
    [(1.0 * sin(2π * n / 20), 1.2 + 1.0 * cos(2π * n / 20)) for n = -10:10])

"""
    layerplotattributes()

Retrive plot attributes for each specific type of layer.
"""
function layerplotattributes(s::Symbol; neuron_colors = NEURON_COLORS)
    if s == :input_layer
        return (mrkrsize=12, mrkrshape=[Plots.Shape(circle_verts), :rtriangle], mrkrcolor=[false, neuron_colors[s]])
    elseif s == :output_layer
        return (mrkrsize=12, mrkrshape=[:rtriangle], mrkrcolor=[neuron_colors[s]])
    elseif s == :Dense
        return (mrkrsize=12, mrkrshape=[:circle], mrkrcolor=[neuron_colors[s]])
    elseif s == :RNNCell || s == :RNN
        return (mrkrsize=12, mrkrshape=[Plots.Shape(lrnn_verts), :circle], mrkrcolor=[false, neuron_colors[s]])
    elseif s == :LSTMCell || s == :LSTM
        return (mrkrsize=12, mrkrshape=[Plots.Shape(lstm_verts), :circle], mrkrcolor=[false, neuron_colors[s]])
    elseif s == :GRUCell || s in (:GRU, :GRUv3, :GRUv3Cell)
        return (mrkrsize=12, mrkrshape=[Plots.Shape(lgru_verts), :circle], mrkrcolor=[false, neuron_colors[s]])
    elseif s == :Conv
        return (mrkrsize=10, mrkrshape=[:square], mrkrcolor=[neuron_colors[s]])
    else
        return (mrkrsize=12, mrkrshape=[:circle], mrkrcolor=[neuron_colors[:Any]])
    end
end

plotchain(m, input=nothing; kwargs...) = Plots.plot(chaingraph(m, input); kwargs...)

plotchain!(m, input=nothing; kwargs...) = Plots.plot!(chaingraph(m, input); kwargs...)
plotchain!(plt::Plots.Plot, m, input=nothing; kwargs...) = Plots.plot!(plt, chaingraph(m, input); kwargs...)

RecipesBase.@recipe function f(
    cg::ChainGraph;
    connection_color=:gray68,
    neuron_colors=NEURON_COLORS,
    neuron_shape=:auto
)
    mg = cg.graph
    chain_dimensions = cg.layer_dimensions
    max_width, = maximum(chain_dimensions)
    m_len = nlayers(cg)

    framestyle --> :grid
    xrotation --> 60
    xticks --> begin
        ll = 0:m_len
        (ll,
            vcat(
                ["input \nlayer "],
                ["hidden \n   layer $n" for n in 1:m_len-1],
                ["output \nlayer   "]
            )
        )
    end
    xlims --> m_len .* (-0.2, 1.2)
    yticks --> false
    ylims --> (-0.1, 1.2)
    legend --> false

    # draw connections
    @series begin
        seriescolor --> connection_color
        dataseries = [
            ([get_prop(mg, e.src, :loc_x), get_prop(mg, e.dst, :loc_x)],
                [get_prop(mg, e.src, :loc_y), get_prop(mg, e.dst, :loc_y)]) for e in edges(mg)]
        return dataseries
    end

    # draw neurons
    @series begin
        scale_sz(sz, max_width) = min(sz, 7.5 * sz / max_width)
        seriestype --> :scatter
        markersize --> vcat(
            [scale_sz(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrsize, max_width)
             for v in vertices(mg)],
            [scale_sz(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrsize, max_width)
             for v in vertices(mg)
             if length(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape) > 1],
            [scale_sz(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrsize, max_width)
             for v in vertices(mg) if get_prop(mg, v, :layer_number) == m_len]
        )
        markershape --> begin
            neuron_shape == :auto ? vcat(
                [layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape[1]
                 for v in vertices(mg)],
                [layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape[end]
                 for v in vertices(mg)
                 if length(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape) > 1],
                [layerplotattributes(:output_layer).mrkrshape[end] for v in vertices(mg) if get_prop(mg, v, :layer_number) == m_len]
            ) : neuron_shape
        end
        markercolor --> begin
            neuron_colors isa Dict ? vcat(
                [layerplotattributes(get_prop(mg, v, :layer_type); neuron_colors).mrkrcolor[1]
                 for v in vertices(mg)],
                [layerplotattributes(get_prop(mg, v, :layer_type); neuron_colors).mrkrcolor[end]
                 for v in vertices(mg)
                 if length(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape) > 1],
                [layerplotattributes(:output_layer; neuron_colors).mrkrcolor[end]
                 for v in vertices(mg)
                 if get_prop(mg, v, :layer_number) == m_len]
            ) : neuron_colors
        end
        dataseries = vcat(
            [(get_prop(mg, v, :loc_x), get_prop(mg, v, :loc_y)) for v in vertices(mg)],
            [(get_prop(mg, v, :loc_x), get_prop(mg, v, :loc_y)) for v in vertices(mg)
             if length(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape) > 1],
            [(get_prop(mg, v, :loc_x), get_prop(mg, v, :loc_y)) for v in vertices(mg)
             if get_prop(mg, v, :layer_number) == m_len]
        )
        return dataseries
    end

    # display layer type
    for ln in 0:m_len
        @series begin
            nj = chain_dimensions[ln+1]
            series_annotations --> Plots.series_annotations(
                [ln == 0 ? "input" : cg.layer_names[ln+1]], Plots.font(8, rotation=20)
            )
            return [ln], [(nj[1] / 2 + 1 + max_width / 2) / (max_width + 1)]
        end
    end
end

end # module
