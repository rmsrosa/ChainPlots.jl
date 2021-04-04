# See Functors.jl https://github.com/FluxML/Functors.jl
# See `show(Chain)T` (not yet implemented): https://github.com/FluxML/Flux.jl/pull/1467
# See https://docs.juliaplots.org/latest/generated/supported/
# See https://fluxml.ai/Flux.jl/stable/models/layers/
# See colors: http://juliagraphics.github.io/Colors.jl/stable/namedcolors/


lrnn_verts = [(1.2*sin(2π*n/20), 1.2*(1+cos(2π*n/20))) for n=-10:10]
lstm_verts = vcat([(1.0*sin(2π*n/20), 1.2 + 1.0*cos(2π*n/20)) for n=-10:10],
                  [(1.4*sin(2π*n/20), 1.2 + 1.4*cos(2π*n/20)) for n=-10:10])
lgru_verts = vcat([(0.5*sin(2π*n/20), 1.2 + 1.4*cos(2π*n/20)) for n=-10:10],
                  [(1.0*sin(2π*n/20), 1.2 + 1.0*cos(2π*n/20)) for n=-10:10])

"""
    layerplotattributes()

Retrive plot attributes for each specific type of layer.
"""                  
layerplotattributes(::Any) = (mrkrsize = 12, mrkrshape =  :circle, mrkrcolor = :gray)
layerplotattributes(::Flux.Dense) = (mrkrsize = 12, mrkrshape =  :circle, mrkrcolor = :lightgreen)
layerplotattributes(::Flux.RNNCell) = (mrkrsize = 12, mrkrshape =  [Main.Plots.Shape(lrnn_verts), :circle], mrkrcolor = [false, :lightskyblue1])
layerplotattributes(::Flux.LSTMCell) = (mrkrsize = 12, mrkrshape =  [Main.Plots.Shape(lstm_verts), :circle], mrkrcolor = [false, :skyblue2])
layerplotattributes(::Flux.GRUCell) = (mrkrsize = 12, mrkrshape =  [Main.Plots.Shape(lgru_verts), :circle], mrkrcolor = [false, :skyblue3])
layerplotattributes(r::Flux.Recur) = layerplotattributes(r.cell)
layerplotattributes(::Flux.Conv) = (mrkrsize = 10, mrkrshape =  :square, mrkrcolor = :plum)

# layer plot attributes for input and output layers
inputlayerplotattributes = (mrkrsize = 12, mrkrshape =  :rtriangle, mrkrcolor = :yellow)
outputlayerplotattributes = (mrkrsize = 12, mrkrshape =  :rtriangle, mrkrcolor = :orange)

"""
    layeractivationfn()

Retrive activation function name of a given layer.
"""
layeractivationfn(::Any) = ""
layeractivationfn(f::Function) = nameof(f)
layeractivationfn(d::Flux.Dense) = nameof(d.σ)
layeractivationfn(r::Flux.RNNCell) = nameof(r.σ)
layeractivationfn(r::Flux.LSTMCell) = "LSTM"
layeractivationfn(r::Flux.GRUCell) = "GRU"
layeractivationfn(r::Flux.Recur) = layeractivationfn(r.cell)
layeractivationfn(r::Flux.Conv) = "Conv"


"""
project(x, center, max_width, slope)

Transform a CartesianIndex x of a neuron into its y-coordinate for plotting.
"""
project(x, center, max_width, slope=0) = ((x[1] - center + max_width/2)/(max_width + 1))

"""
    plot(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)

Plot a Flux.Chain neural network.
"""
@recipe function plot(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)
    m = f32(m)
    if input_data !== nothing
        input_data = convert.(Float32, input_data)
    end
    chain_dimensions = get_dimensions(m, input_data)
    max_width, = maximum(chain_dimensions)

    axis --> false
    xrotation   --> 60
    xticks --> begin
        ll = 1:length(m)+1
        (ll,
            vcat(
                ["input \nlayer "],
                ["hidden \n   layer $n" for n in ll[1:end-2]],
                ["output \nlayer   "]
            )
        )
    end
    yaxis --> nothing
    ylims --> (-0.1,1.2)
    legend --> false
    seriescolor --> :gray

    # get connections
    connections = get_connections(m, input_data)

    # draw connections
    for (ln, l) in enumerate(m.layers)
        @series begin
            ni, nj = chain_dimensions[ln:ln+1]
            layer_center = [ni[1],nj[1]]./2
            dataseries = hcat([
                hcat([
                    [project(neuron_in, layer_center[1], max_width),
                     project(neuron_out, layer_center[2], max_width)
                    ]
                    for neuron_out in connections[ln][neuron_in]
                ]...)
                for neuron_in in get_cartesians(ni) if length(connections[ln][neuron_in]) > 0
            ]...)
            return [ln,ln + 1], dataseries            
        end
    end

    # draw input layer cells
    @series begin
        #markersize --> min(12, 100/max_width)
        markersize --> begin
            sz = inputlayerplotattributes.mrkrsize
            min(sz, 7.5*sz/max_width)
        end
        markershape --> inputlayerplotattributes.mrkrshape
        markercolor --> inputlayerplotattributes.mrkrcolor
        ni = chain_dimensions[1]
        layer_center = ni[1]/2
        dataseries = reshape([project(neuron, layer_center, max_width) for neuron in get_cartesians(ni)], 1, :)
        return [1], dataseries        
    end

    # draw hidden and output layer neurons
    for (ln, l) in enumerate(m.layers)
        @series begin
            markersize --> begin
                sz = layerplotattributes(l).mrkrsize
                min(sz, 7.5*sz/max_width)
            end
            markershape --> layerplotattributes(l).mrkrshape
            markercolor --> layerplotattributes(l).mrkrcolor
            nj = chain_dimensions[ln+1]
            layer_center = nj[1]/2
            dataseries = reshape([project(neuron, layer_center, max_width) for neuron in get_cartesians(nj)], 1, : ) |> v -> vcat(v,v)
            return [ln+1, ln+1], dataseries
        end
    end

    # finish drawing output layer neurons
    @series begin
        l = m.layers[end]
        ln = length(m.layers)
        markersize --> begin
            sz = outputlayerplotattributes.mrkrsize
            min(sz, 7.5*sz/max_width)
        end
        markershape --> outputlayerplotattributes.mrkrshape
        markercolor --> outputlayerplotattributes.mrkrcolor
        nj = chain_dimensions[end]
        layer_center = nj[1]/2
        dataseries = reshape([project(neuron, layer_center, max_width) for neuron in get_cartesians(nj)], 1, : )
        return [ln+1], dataseries
    end  

    # display activation functions
    for (ln, l) in enumerate(m.layers)
        @series begin
            nj = chain_dimensions[ln+1]
            series_annotations --> Main.Plots.series_annotations([layeractivationfn(l)], Main.Plots.font("Sans", 8))
            return [ln+1], [(nj[1]/2 + 1 + max_width/2)/(max_width+1)]
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
    nothing
end
