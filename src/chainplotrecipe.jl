circle_verts = [(1.41*sin(2π*n/20), 1.41*cos(2π*n/20)) for n=-10:10]
lrnn_verts = [(1.2*sin(2π*n/20), 1.2*(1+cos(2π*n/20))) for n=-10:10]
lstm_verts = vcat([(1.0*sin(2π*n/20), 1.2 + 1.0*cos(2π*n/20)) for n=-10:10],
                  [(1.4*sin(2π*n/20), 1.2 + 1.4*cos(2π*n/20)) for n=-10:10])
lgru_verts = vcat([(0.5*sin(2π*n/20), 1.2 + 1.4*cos(2π*n/20)) for n=-10:10],
                  [(1.0*sin(2π*n/20), 1.2 + 1.0*cos(2π*n/20)) for n=-10:10])

"""
    layerplotattributes()

Retrive plot attributes for each specific type of layer.
"""                  
layerplotattributes(::Any) = (mrkrsize = 12, mrkrshape =  [:circle], mrkrcolor = [:gray])
layerplotattributes(::Flux.Dense) = (mrkrsize = 12, mrkrshape =  [:circle], mrkrcolor = [:lightgreen])
layerplotattributes(::Flux.RNNCell) = (mrkrsize = 12, mrkrshape =  [Main.Plots.Shape(lrnn_verts), :circle], mrkrcolor = [false, :lightskyblue1])
layerplotattributes(::Flux.LSTMCell) = (mrkrsize = 12, mrkrshape =  [Main.Plots.Shape(lstm_verts), :circle], mrkrcolor = [false, :skyblue2])
layerplotattributes(::Flux.GRUCell) = (mrkrsize = 12, mrkrshape =  [Main.Plots.Shape(lgru_verts), :circle], mrkrcolor = [false, :skyblue3])
layerplotattributes(r::Flux.Recur) = layerplotattributes(r.cell)
layerplotattributes(::Flux.Conv) = (mrkrsize = 10, mrkrshape = [:square], mrkrcolor = [:plum])
function layerplotattributes(s::Symbol)
    if s == :input_layer
        return (mrkrsize = 12, mrkrshape = [Main.Plots.Shape(circle_verts), :rtriangle], mrkrcolor = [false, :yellow]) 
    elseif s == :output_layer
        return (mrkrsize = 12, mrkrshape = [:rtriangle], mrkrcolor = [:orange])
    else
        return (mrkrsize = 0, mrkrshape = [:none], mrkrcolor = [:none])
    end
end

"""
projection(x, center, max_width, slope)

Transform a Tuple x of a neuron into its y-coordinate for plotting.
"""
projection(x, center, max_width, slope=0) = ((x[1] - center + max_width/2)/(max_width + 1))

"""
    plot(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)

Plot a Flux.Chain neural network according to recipe.
"""
@recipe function plot(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)
    m = f32(m)
    if input_data !== nothing
        input_data = convert.(Float32, input_data)
        chain_dimensions = get_dimensions(m, input_data)
    else
        chain_dimensions = get_dimensions(m)
    end
    # chain_dimensions = get_dimensions(m, input_data)
    max_width, = maximum(chain_dimensions)
    mg = chaingraph(m, input_data)
    connections = neuron_connections(m, input_data)

    axis --> false
    xrotation --> 60
    xticks --> begin
        ll = 0:length(m)
        (ll,
            vcat(
                ["input \nlayer "],
                ["hidden \n   layer $n" for n in ll[1:end-2]],
                ["output \nlayer   "]
            )
        )
    end
    xlims --> length(m).*(-0.2,1.2)
    yticks --> false
    ylims --> (-0.1,1.2)
    legend --> false

    # draw connections
    @series begin
        seriescolor --> :gray
        dataseries = [([get_prop(mg, e.src, :layer_number), get_prop(mg, e.dst, :layer_number)], [projection(get_prop(mg, e.src, :index_in_layer), get_prop(mg, e.src, :layer_center), max_width), projection(get_prop(mg, e.dst, :index_in_layer), get_prop(mg, e.dst, :layer_center), max_width)]) for e in edges(mg)]
        return dataseries  
    end

    # draw neurons
    @series begin
        scale_sz(sz, max_width) = min(sz, 7.5*sz/max_width)
        seriestype --> :scatter
        markersize --> vcat([scale_sz(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrsize, max_width) for v in vertices(mg)],
                            [scale_sz(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrsize, max_width) for v in vertices(mg) if length(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape)>1],
                            [scale_sz(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrsize, max_width) for v in vertices(mg) if get_prop(mg, v, :layer_number) == length(m)])
        markershape --> vcat([layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape[1] for v in vertices(mg)],
                            [layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape[end] for v in vertices(mg) if length(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape)>1],
                            [layerplotattributes(:output_layer).mrkrshape[end] for v in vertices(mg) if get_prop(mg, v, :layer_number) == length(m)])
        markercolor --> vcat([layerplotattributes(get_prop(mg, v, :layer_type)).mrkrcolor[1] for v in vertices(mg)],
                            [layerplotattributes(get_prop(mg, v, :layer_type)).mrkrcolor[end] for v in vertices(mg) if length(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape)>1],
                            [layerplotattributes(:output_layer).mrkrcolor[end] for v in vertices(mg) if get_prop(mg, v, :layer_number) == length(m)])
        dataseries = vcat([(get_prop(mg, v, :layer_number), projection(get_prop(mg, v, :index_in_layer), get_prop(mg, v, :layer_center), max_width)) for v in vertices(mg)],
                          [(get_prop(mg, v, :layer_number), projection(get_prop(mg, v, :index_in_layer), get_prop(mg, v, :layer_center), max_width)) for v in vertices(mg) if length(layerplotattributes(get_prop(mg, v, :layer_type)).mrkrshape)>1],
                          [(get_prop(mg, v, :layer_number), projection(get_prop(mg, v, :index_in_layer), get_prop(mg, v, :layer_center), max_width)) for v in vertices(mg) if get_prop(mg, v, :layer_number) == length(m)])
        return dataseries
    end

    # display layer type
    for (ln, l) in enumerate(m.layers)
        @series begin
            nj = chain_dimensions[ln+1]
            series_annotations --> Main.Plots.series_annotations([string(l)], Main.Plots.font("Sans", 8, rotation=20))
            return [ln], [(nj[1]/2 + 1 + max_width/2)/(max_width+1)]
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

YET TO BE IMPLEMENTED.
"""
@recipe function plot((m,s)::Tuple{Flux.Chain,Tuple{Int}})
    nothing
end
