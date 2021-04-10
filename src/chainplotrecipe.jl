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
    get_layer_type(m::Flux.Chain, i::Int)

Return `:input_layer` if "layer number" `i` is zero and the layer `m[i]` itself,
otherwise.

For use with [`layerplotattributes`](@ref), for properly retrieving the 
corresponding plot attributes.
"""
get_layer_type(m::Flux.Chain, i::Int) = i == 0 ? :input_layer : m[i]

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
        dataseries = [
            ([get_prop(mg, e.src, :loc_x), get_prop(mg, e.dst, :loc_x)],
             [get_prop(mg, e.src, :loc_y), get_prop(mg, e.dst, :loc_y)]) for e in edges(mg)]
        return dataseries  
    end

    # draw neurons
    @series begin
        scale_sz(sz, max_width) = min(sz, 7.5*sz/max_width)
        seriestype --> :scatter
        markersize --> vcat(
            [scale_sz(layerplotattributes(get_layer_type(m, get_prop(mg, v, :layer_number))).mrkrsize, max_width)
                for v in vertices(mg)],
            [scale_sz(layerplotattributes(get_layer_type(m, get_prop(mg, v, :layer_number))).mrkrsize, max_width)
                for v in vertices(mg)
                    if length(layerplotattributes(get_layer_type(m, get_prop(mg, v, :layer_number))).mrkrshape)>1],
            [scale_sz(layerplotattributes(get_layer_type(m, get_prop(mg, v, :layer_number))).mrkrsize, max_width)
                for v in vertices(mg) if get_prop(mg, v, :layer_number) == length(m)]
        )
        markershape --> vcat(
            [layerplotattributes(get_layer_type(m, get_prop(mg, v, :layer_number))).mrkrshape[1]
                for v in vertices(mg)],
            [layerplotattributes(get_layer_type(m, get_prop(mg, v, :layer_number))).mrkrshape[end]
                for v in vertices(mg)
                    if length(layerplotattributes(get_layer_type(m, get_prop(mg, v, :layer_number))).mrkrshape)>1],
            [layerplotattributes(:output_layer).mrkrshape[end] for v in vertices(mg) if get_prop(mg, v, :layer_number) == length(m)]
        )
        markercolor --> vcat(
            [layerplotattributes(get_layer_type(m, get_prop(mg, v, :layer_number))).mrkrcolor[1]
                for v in vertices(mg)],
            [layerplotattributes(get_layer_type(m, get_prop(mg, v, :layer_number))).mrkrcolor[end]
                for v in vertices(mg)
                    if length(layerplotattributes(get_layer_type(m, get_prop(mg, v, :layer_number))).mrkrshape)>1],
            [layerplotattributes(:output_layer).mrkrcolor[end]
                for v in vertices(mg)
                    if get_prop(mg, v, :layer_number) == length(m)]
        )
        dataseries = vcat(
            [(get_prop(mg, v, :loc_x), get_prop(mg, v, :loc_y)) for v in vertices(mg)],
            [(get_prop(mg, v, :loc_x), get_prop(mg, v, :loc_y)) for v in vertices(mg)
                if length(layerplotattributes(get_layer_type(m, get_prop(mg, v, :layer_number))).mrkrshape)>1],
            [(get_prop(mg, v, :loc_x), get_prop(mg, v, :loc_y)) for v in vertices(mg)
                if get_prop(mg, v, :layer_number) == length(m)]
        )
        return dataseries
    end

    # display layer type
    for ln in 0:length(m)
        @series begin
            nj = chain_dimensions[ln+1]
            series_annotations --> Main.Plots.series_annotations(
                [ln == 0 ? "input" : string(m[ln])], Main.Plots.font("Sans", 8, rotation=20)
            )
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
