# See Functors.jl https://github.com/FluxML/Functors.jl
# See `show(Chain)T` (not yet implemented): https://github.com/FluxML/Flux.jl/pull/1467
# See https://docs.juliaplots.org/latest/generated/supported/
# See https://fluxml.ai/Flux.jl/stable/models/layers/

using Flux
using RecipesBase
# import Functors: functor

lrnn_verts = [(1.2*sin(2π*n/20), 1.2*(1+cos(2π*n/20))) for n=-10:10]
lstm_verts = vcat([(1.0*sin(2π*n/20), 1.4 + 1.0*cos(2π*n/20)) for n=-10:10],
                  [(1.4*sin(2π*n/20), 1.4 + 1.4*cos(2π*n/20)) for n=-10:10])
lgru_verts = vcat([(0.6*sin(2π*n/20), 1.4 + 1.6*cos(2π*n/20)) for n=-10:10],
                  [(1.2*sin(2π*n/20), 1.2 + 1.2*cos(2π*n/20)) for n=-10:10])

"""
    layerplotattributes()

Retrive plot attributes for each specific type of layer.
"""                  
layerplotattributes(::Any) = (ms = :circle, mc = :gray35)
layerplotattributes(::Flux.Dense) = (ms = :circle, mc = :lightgreen)
layerplotattributes(::Flux.RNNCell) = (ms = [Main.Plots.Shape(lrnn_verts), :circle], mc = [false, :lightskyblue1])
layerplotattributes(::Flux.LSTMCell) = (ms = [Main.Plots.Shape(lstm_verts), :circle], mc = [false, :skyblue2])
layerplotattributes(::Flux.GRUCell) = (ms = [Main.Plots.Shape(lgru_verts), :circle], mc = [false, :skyblue3])
layerplotattributes(r::Flux.Recur) = layerplotattributes(r.cell)

# layer plot attributes for input and output layers
inputlayerplotattributes = (ms = :circle, mc = :yellow)
outputlayerplotattributes = (ms = :circle, mc = :orange)

"""
    layeractivationfn()

Retrive activation function name of a given layer.
"""
layeractivationfn(::Any) = ""
layeractivationfn(f::Function) = string(f)
layeractivationfn(d::Flux.Dense) = string(d.σ)
layeractivationfn(r::Flux.RNNCell) = string(r.σ)
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

# list layers with fixed-sized data input
fixed_input_dim_layers = (Flux.Dense, Flux.Recur, Flux.RNNCell, Flux.LSTMCell, Flux.GRUCell) # list of types of layers with fixed input dimensions

"""
    get_dimensions(m::Flux.Chain, input_data = nothing)

Get the dimensions of the input layer and of the output layer of each hidden layer.

If `input_data` is not given, the first layer is required to be a layer
with fixed input dimensions,  such as Flux.Dense or Flux.Recur,
otherwise the given data is used to infer the dimensions of each layer.
"""
function get_dimensions(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)

    if (m.layers[1] isa Union{fixed_input_dim_layers...})
        input_data = rand(layerdimensions(m.layers[1])[2])
    elseif input_data === nothing
        throw(ArgumentError("An `input_data` is required when the first layer accepts variable-dimension input"))
    end

    chain_dimensions = vcat(size(input_data), [size(m[1:nl](input_data)) for nl in 1:length(m.layers)])
    # chain_dimensions = map(x -> x[1], chain_dimensions) # temporary, only for the current 1D structure
    return chain_dimensions
end

# structure for unit vectors in a linear space, for generating a basis to infer the layer connection
struct UnitVector{T} <: AbstractVector{T}
    idx::Int
    length::Int
end

Base.getindex(x::UnitVector{T}, i) where T = x.idx==i ? one(T) : zero(T)
Base.length(x::UnitVector) = x.length
Base.size(x::UnitVector) = (x.length,)

"""
get_cartesians(lsize:Tuple)

"""
function get_cartesians(ldim::Tuple)
    cartesians = Array{CartesianIndex,1}()
    foreach(1:prod(ldim)) do idx
        basis_element = reshape(UnitVector{Float64}(idx, prod(ldim)),ldim...)
        push!(cartesians, CartesianIndex(findfirst(x->x==1, basis_element)))
    end
    return cartesians
end

"""
    get_connections(m::Flux.Chain)

Get all the connections to the next layer of each neuron in each layer.
"""
function get_connections(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)
    chain_dimensions = get_dimensions(m, input_data)
    connections = []

    for (ln, l) in enumerate(m)
        ldim = chain_dimensions[ln]
        layer_connections = Dict{CartesianIndex,Array{CartesianIndex,1}}()
        foreach(1:prod(ldim)) do idx
            affected = Array{CartesianIndex,1}()
            basis_element = reshape(UnitVector{Float64}(idx, prod(ldim)),ldim...)
            for rv in 1000*(rand(10) .- 0.5)
                union!(affected, CartesianIndex.(findall(x -> abs(x) > eps(), l(rv*basis_element))))
            end
            push!(layer_connections, CartesianIndex(findfirst(x->x==1, basis_element)) => affected)
        end
        push!(connections, layer_connections)
    end

    return connections    
end

"""
    get_max_width(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)

Get the maximum display width for the chain.
"""
get_max_width(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing) =
    mapreduce(x->x[1], max, get_dimensions(m,input_data))

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
                for neuron_in in get_cartesians(ni)
            ]...)
            return [ln,ln + 1], dataseries            
        end
    end

    # draw input layer cells
    @series begin
        markersize --> min(12, 100/max_width)
        markershape --> inputlayerplotattributes.ms
        markercolor --> inputlayerplotattributes.mc
        ni = chain_dimensions[1]
        layer_center = ni[1]/2
        dataseries = reshape([project(neuron, layer_center, max_width) for neuron in get_cartesians(ni)], 1, :)
        return [1], dataseries        
    end

    # draw hidden and output layer neurons
    for (ln, l) in enumerate(m.layers)
        @series begin
            markersize --> min(12, 100/max_width)
            markershape --> layerplotattributes(l).ms
            markercolor --> layerplotattributes(l).mc
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
        markersize --> min(8, 66/max_width)
        markershape --> outputlayerplotattributes.ms
        markercolor --> outputlayerplotattributes.mc
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
