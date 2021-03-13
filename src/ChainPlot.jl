# See Functors.jl https://github.com/FluxML/Functors.jl
# See `show(Chain)T` (not yet implemented): https://github.com/FluxML/Flux.jl/pull/1467
# See https://docs.juliaplots.org/latest/generated/supported/
# See https://fluxml.ai/Flux.jl/stable/models/layers/

module ChainPlot

using Flux
using RecipesBase
# import Functors: functor

lrnn_verts = [(1.2*sin(2π*n/20), 1.2*(1+cos(2π*n/20))) for n=-10:10]
lstm_verts = vcat([(1.0*sin(2π*n/20), 1.4 + 1.0*cos(2π*n/20)) for n=-10:10],
                  [(1.4*sin(2π*n/20), 1.4 + 1.4*cos(2π*n/20)) for n=-10:10])
lgru_verts = vcat([(0.6*sin(2π*n/20), 1.4 + 1.0*cos(2π*n/20)) for n=-10:10],
                  [(1.4*sin(2π*n/20), 1.4 + 1.4*cos(2π*n/20)) for n=-10:10])

layerplotattributes(::Any) = (ms = :circle, mc = :black)
layerplotattributes(::Flux.Dense) = (ms = :circle, mc = :lightgreen)
layerplotattributes(::Flux.RNNCell) = (ms = [Main.Plots.Shape(lrnn_verts), :circle], mc = [false, :lightblue])
layerplotattributes(::Flux.LSTMCell) = (ms = [Main.Plots.Shape(lstm_verts), :circle], mc = [false, :lightblue])
layerplotattributes(::Flux.GRUCell) = (ms = [Main.Plots.Shape(lgru_verts), :circle], mc = [false, :lightblue])
layerplotattributes(r::Flux.Recur) = layerplotattributes(r.cell)

layeractivationfn(::Any) = ""
layeractivationfn(d::Flux.Dense) = string(Symbol(d.σ))
layeractivationfn(r::Flux.RNNCell) = string(Symbol(r.σ))
layeractivationfn(r::Flux.LSTMCell) = "LSTM"
layeractivationfn(r::Flux.GRUCell) = "GRU"
layeractivationfn(r::Flux.Recur) = layeractivationfn(r.cell)

layerdimensions(::Any) = (1,1)
layerdimensions(l::Flux.Dense) = size(l.W)
layerdimensions(l::Flux.RNNCell) = (size(l.Wh)[2], size(l.Wi)[2])
layerdimensions(l::Flux.LSTMCell) = (size(l.Wh)[2], size(l.Wi)[2])
layerdimensions(l::Flux.GRUCell) = (size(l.Wh)[2], size(l.Wi)[2])
layerdimensions(r::Flux.Recur) = layerdimensions(r.cell)

"""
    plot(m::Flux.Chain)

Plot a Flux.Chain neural network.

☡ Currently only works with Dense and Recurrent layers.
"""
@recipe function plot(m::Flux.Chain)
    max_width = maximum(layerdimensions(l)[1] for l in m.layers)

    axis --> false
    xrotation   --> 60
    xticks --> begin ll = 1:length(m)+1; (ll, vcat(["input \nlayer "], ["hidden \n   layer $n" for n in ll[1:end-2]], ["output \nlayer   "])); end
    yaxis --> nothing
    ylims --> (-0.0,1.1)
    legend --> false
    markersize --> min(12, 100/max_width)
    seriescolor --> :gray

    # draw connections
    for (ln, l) in enumerate(m.layers)
        @series begin
            # arrow --> true
            nj, ni = layerdimensions(l)
            layer_center = [ni,nj]./2
            dataseries = hcat([[i,j] for i in 1:ni for j in 1:nj]...) .- layer_center
            dataseries = map(x -> ((x + max_width/2)/(max_width+1)), dataseries)
            return [ln,ln + 1], dataseries
        end
    end

    # draw input layer cells
    @series begin
        markershape --> :circle
        markercolor --> :yellow
        nj, ni = layerdimensions(m[1])
        layer_center = ni/2
        dataseries = hcat([[i] for i in 1:ni]...) .- layer_center
        dataseries = map(x -> ((x + max_width/2)/(max_width+1)), dataseries)
        return [1], dataseries        
    end

    # draw hidden layer neurons
    for (ln, l) in enumerate(m.layers)
        @series begin
            markershape --> layerplotattributes(l).ms
            markercolor --> ifelse(ln == length(m), [:transparent,:orange], layerplotattributes(l).mc)
            nj, ni = layerdimensions(l)
            layer_center = nj/2
            dataseries = hcat([[j, j] for j in 1:nj]...) .- layer_center
            dataseries = map(x -> ((x + max_width/2)/(max_width+1)), dataseries)
            return [ln+1, ln+1], dataseries
        end
    end

#=     # draw output layer cells
    @series begin
        markershape --> :diamond
        markercolor --> :orange
        nj, ni = size(params(m[end])[1])
        layer_center = nj/2
        # dataseries = hcat([[j] for j in 1:nj]...) .- layer_center
        dataseries = hcat([j for j in 1:nj]...) .- layer_center
        dataseries = map(x -> ((x + max_width/2)/(max_width+1)), dataseries)
        return [length(m)+1], dataseries        
    end =#

    # display activation functions
    for (ln, l) in enumerate(m.layers)
        @series begin
            nj, ni = layerdimensions(l)
            series_annotations --> Main.Plots.series_annotations([layeractivationfn(l)], Main.Plots.font("Sans", 8))
            return [ln+1], [(nj/2 + 1 + max_width/2)/(max_width+1)]
        end
    end

end

@recipe function plot(d::Union{Flux.Dense, Flux.Recur, Flux.RNNCell})
    Flux.Chain(d)
end

end

#= 
Questions:

1) If `m` is a chain, is there any case in which `length(m)` is different from `length(m.layers)`?
2) Similarly, is there any difference between walking through the layers with `for l in m` or `for l in m.layers`?
3) How complicate can `Recur` be? Can I assume there is always one and only one `RNNCell` in it?
=#