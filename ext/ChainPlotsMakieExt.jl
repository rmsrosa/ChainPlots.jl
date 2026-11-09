module ChainPlotsMakieExt

# The Makie recipe for `ChainGraph`, so that `plot(cg)` works with Makie, and the methods
# of `plotchain` and `plotchain!` for `MakieBackend`, loaded with Makie.jl.

using ChainPlots: ChainPlots, ChainGraph, MakieBackend, NEURON_COLORS, chaingraph, nlayers, nodeprop
import ChainPlots: plotchain, plotchain!
using Graphs: edges, vertices
using Makie

"""
    ChainGraphPlot

Makie recipe for a [`ChainGraph`](@ref ChainPlots.ChainGraph), used by `plot(cg)`.
"""
Makie.@recipe ChainGraphPlot (cg,) begin
    "Color of the connections."
    connection_color = :gray68
    "A `Dict` from layer types to colors, as `NEURON_COLORS`, or a color for all neurons."
    neuron_colors = NEURON_COLORS
    "`:auto` for a marker depending on the type of the layer, or a marker for all neurons."
    neuron_shape = :auto
    "Font size of the names of the layers."
    fontsize = 11
end

Makie.plottype(::ChainGraph) = ChainGraphPlot

# the outlines drawn above the recurrent neurons, as marker paths, in units of the marker
# size, with the neuron itself as a circle of radius 1/2 at the origin
function loops(radii; center=0.5)
    commands = Makie.PathCommand[]
    for r in radii, n in 0:40
        θ = 2π * n / 40
        p = Makie.Point2d(r * sin(θ), center + r * cos(θ))
        push!(commands, n == 0 ? Makie.MoveTo(p) : Makie.LineTo(p))
    end
    return Makie.BezierPath(commands)
end

const RECURRENT_LOOPS = Dict(
    :RNN => loops((0.5,)),
    :LSTM => loops((0.4, 0.55)),
    :GRU => loops((0.2, 0.45)),
)

function recurrent_loop(t::Symbol)
    t in (:RNN, :RNNCell) && return RECURRENT_LOOPS[:RNN]
    t in (:LSTM, :LSTMCell) && return RECURRENT_LOOPS[:LSTM]
    t in (:GRU, :GRUCell, :GRUv3, :GRUv3Cell) && return RECURRENT_LOOPS[:GRU]
    return nothing
end

neuron_marker(t::Symbol) = t === :Conv ? :rect : :circle

neuron_fill(colors::AbstractDict, t::Symbol) = get(colors, t, colors[:Any])
neuron_fill(color, ::Symbol) = color

function Makie.plot!(p::ChainGraphPlot)
    cg = p[1][]
    vs = collect(vertices(cg))
    xs = nodeprop.(Ref(cg), vs, :loc_x)
    ys = nodeprop.(Ref(cg), vs, :loc_y)
    types = nodeprop.(Ref(cg), vs, :layer_type)
    layers = nodeprop.(Ref(cg), vs, :layer_number)
    dims = cg.layer_dimensions
    max_width = maximum(first.(dims))
    m_len = nlayers(cg)
    colors = p.neuron_colors[]
    shape = p.neuron_shape[]
    # marker sizes shrink for wide layers, as in the Plots recipe
    sz(s) = 1.4 * min(s, 7.5 * s / max_width)

    # connections
    linesegments!(p, [Makie.Point2d(xs[e.src], ys[e.src]) => Makie.Point2d(xs[e.dst], ys[e.dst]) for e in edges(cg)];
        color=p.connection_color)

    if shape === :auto
        # outlines above the recurrent neurons, and a ring around the input neurons
        for v in eachindex(vs)
            loop = recurrent_loop(types[v])
            loop === nothing && continue
            scatter!(p, [Makie.Point2d(xs[v], ys[v])]; marker=loop, markersize=sz(12),
                color=:transparent, strokecolor=:black, strokewidth=1)
        end
        inputs = findall(==(:input_layer), types)
        scatter!(p, xs[inputs], ys[inputs]; marker=:circle, markersize=sz(17), color=:white,
            strokecolor=:black, strokewidth=1)
        # neurons, then the triangles of the input and output layers
        hidden = findall(!=(:input_layer), types)
        scatter!(p, xs[hidden], ys[hidden]; marker=neuron_marker.(types[hidden]),
            markersize=[types[v] === :Conv ? sz(10) : sz(12) for v in hidden],
            color=[neuron_fill(colors, types[v]) for v in hidden], strokecolor=:black, strokewidth=1)
        outputs = findall(==(m_len), layers)
        scatter!(p, xs[[inputs; outputs]], ys[[inputs; outputs]]; marker=:rtriangle, markersize=sz(10),
            color=[fill(neuron_fill(colors, :input_layer), length(inputs)); fill(neuron_fill(colors, :output_layer), length(outputs))],
            strokecolor=:black, strokewidth=0.5)
    else
        scatter!(p, xs, ys; marker=shape, markersize=sz(12),
            color=[neuron_fill(colors, t) for t in types], strokecolor=:black, strokewidth=1)
    end

    # names of the layers
    for (ln, name) in enumerate(cg.layer_names)
        text!(p, ln - 1, (dims[ln][1] / 2 + 1 + max_width / 2) / (max_width + 1);
            text=ln == 1 ? "input" : name, rotation=π / 9, fontsize=p.fontsize, align=(:center, :bottom))
    end
    return p
end

function chaingraph_axis(cg::ChainGraph)
    m_len = nlayers(cg)
    return (
        xticks=(0:m_len, ["input\nlayer"; ["hidden\nlayer $n" for n in 1:m_len-1]; "output\nlayer"]),
        xticklabelrotation=π / 3,
        limits=(m_len * -0.2, m_len * 1.2, -0.1, 1.2),
        yticksvisible=false, yticklabelsvisible=false, ygridvisible=false,
        leftspinevisible=false, rightspinevisible=false, topspinevisible=false, bottomspinevisible=false,
        xticksvisible=false,
    )
end

Makie.preferred_axis_attributes(::Type{<:Makie.Axis}, p::ChainGraphPlot) = chaingraph_axis(p[1][])

plotchain(::MakieBackend, m, input=nothing; kwargs...) = Makie.plot(chaingraph(m, input); kwargs...)

plotchain!(::MakieBackend, m, input=nothing; kwargs...) = Makie.plot!(chaingraph(m, input); kwargs...)
plotchain!(::MakieBackend, ax::Makie.AbstractAxis, m, input=nothing; kwargs...) = Makie.plot!(ax, chaingraph(m, input); kwargs...)
plotchain!(ax::Makie.AbstractAxis, m, input=nothing; kwargs...) = plotchain!(MakieBackend(), ax, m, input; kwargs...)

end # module
