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

"""
    layerplotattributes()

Retrive plot attributes for each specific type of layer.
"""                  
layerplotattributes(::Any) = (ms = :circle, mc = :black)
layerplotattributes(::Flux.Dense) = (ms = :circle, mc = :lightgreen)
layerplotattributes(::Flux.RNNCell) = (ms = [Main.Plots.Shape(lrnn_verts), :circle], mc = [false, :lightblue])
layerplotattributes(::Flux.LSTMCell) = (ms = [Main.Plots.Shape(lstm_verts), :circle], mc = [false, :lightblue])
layerplotattributes(::Flux.GRUCell) = (ms = [Main.Plots.Shape(lgru_verts), :circle], mc = [false, :lightblue])
layerplotattributes(r::Flux.Recur) = layerplotattributes(r.cell)

"""
    layeractivationfn()

Retrive activation function of a given layer.
"""
layeractivationfn(::Any) = ""
layeractivationfn(d::Flux.Dense) = string(Symbol(d.σ))
layeractivationfn(r::Flux.RNNCell) = string(Symbol(r.σ))
layeractivationfn(r::Flux.LSTMCell) = "LSTM"
layeractivationfn(r::Flux.GRUCell) = "GRU"
layeractivationfn(r::Flux.Recur) = layeractivationfn(r.cell)

"""
    layerdimensions()

Retrive dimensions of a given layer.
"""
layerdimensions(::Any) = (1,1)
layerdimensions(l::Flux.Dense) = size(l.W)
layerdimensions(l::Flux.RNNCell) = (size(l.Wh)[2], size(l.Wi)[2])
layerdimensions(l::Flux.LSTMCell) = (size(l.Wh)[2], size(l.Wi)[2])
layerdimensions(l::Flux.GRUCell) = (size(l.Wh)[2], size(l.Wi)[2])
layerdimensions(r::Flux.Recur) = layerdimensions(r.cell)

fixed_input_dim_layers = (Flux.Dense, Flux.Recur, Flux.RNNCell, Flux.LSTMCell, Flux.GRUCell) # list of types of layers with fixed input dimensions

"""
    get_chain_dimensions(m::Flux.Chain)

Get input and output dimensions of each layer of a Flux.Chain.

The first layer is required to be a layer with fixed input dimensions,
such as Flux.Dense or a Flux.Recur.
"""
function get_chain_dimensions(m::Flux.Chain)

    if !(m.layers[1] isa Union{fixed_input_dim_layers...})
        throw(ArgumentError("The first layer is required to be a layer with fixed input dimensions, such as Flux.Dense or a Flux.Recur."))
    end

    chain_dimensions = fill(Int16(1),length(m.layers)+1)
    chain_dimensions[1] = layerdimensions(m.layers[1])[2] # input dimensions
    for (ln, l) in enumerate(m.layers)
        chain_dimensions[ln+1] = layerdimensions(l)[1]
    end
    return chain_dimensions
end

"""
    plot(m::Flux.Chain)

Plot a Flux.Chain neural network.

☡ Currently only works with Dense and Recurrent layers.
"""
@recipe function plot(m::Flux.Chain)
    chain_dimensions = get_chain_dimensions(m)
    max_width = maximum(chain_dimensions)

    axis --> false
    xrotation   --> 60
    xticks --> begin ll = 1:length(m)+1; (ll, vcat(["input \nlayer "], ["hidden \n   layer $n" for n in ll[1:end-2]], ["output \nlayer   "])); end
    yaxis --> nothing
    ylims --> (-0.0,1.1)
    legend --> false
    markersize --> min(12, 100/max_width)
    seriescolor --> :gray

    # draw connections
    ni = chain_dimensions[1]
    for (ln, l) in enumerate(m.layers)
        @series begin
            # arrow --> true
            # nj, ni = layerdimensions(l)
            nj = chain_dimensions[ln+1]
            layer_center = [ni,nj]./2
            dataseries = hcat([[i,j] for i in 1:ni for j in 1:nj]...) .- layer_center
            dataseries = map(x -> ((x + max_width/2)/(max_width+1)), dataseries)
            ni=nj
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

    # draw hidden and output layer neurons
    for (ln, l) in enumerate(m.layers)
        @series begin
            markershape --> layerplotattributes(l).ms
            markercolor --> ifelse(ln == length(m), [:transparent,:orange], layerplotattributes(l).mc)
            nj = chain_dimensions[ln+1]
            layer_center = nj/2
            dataseries = hcat([[j, j] for j in 1:nj]...) .- layer_center
            dataseries = map(x -> ((x + max_width/2)/(max_width+1)), dataseries)
            return [ln+1, ln+1], dataseries
        end
    end

    # display activation functions
    for (ln, l) in enumerate(m.layers)
        @series begin
            nj = chain_dimensions[ln+1]
            series_annotations --> Main.Plots.series_annotations([layeractivationfn(l)], Main.Plots.font("Sans", 8))
            return [ln+1], [(nj/2 + 1 + max_width/2)/(max_width+1)]
        end
    end

end

@recipe function plot(d::Union{Flux.Dense, Flux.Recur, Flux.RNNCell, Flux.LSTMCell, Flux.GRUCell})
    Flux.Chain(d)
end

"""
    plot((m,s)::Tuple{Flux.Chain,Tuple{Int}})

Plot a Flux.Chain neural network `m` with input dimensions `s`.

Useful when the neural network starts with a variable-input layer, such as a convolutional layer.
"""
@recipe function plot((m,s)::Tuple{Flux.Chain,Tuple{Int}})

end

"""
    plot((m,a)::Tuple{Flux.Chain,Array})

Plot a Flux.Chain neural network `m` with an input array `a`.

Useful when the neural network starts with a variable-input layer, such as a convolutional layer.
"""
@recipe function plot((m,s)::Tuple{Flux.Chain,Array})
    return (m, size(s))
end

end