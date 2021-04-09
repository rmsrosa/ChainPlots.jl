# Just for testing - remove when it is for real
#= using Flux
using LightGraphs
using MetaGraphs
using RecipesBase
import Base: getindex, length, size

include("chaintools.jl") =#

# Remove the above ↑↑↑ when it is for real

"""
    chaingraph(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)

Return a MetaGraph representing the graph structure of the neural network.

Each node represents a neuron of the Chain `m` and contains the following properties
    `:layer_number`: indicates to each layer it belongs (an `Int` value with `0` indicating 
        the input layer, `1, …, length(m)-1` indicating the hidden layers, 
        and with `length(m)` indicating the output layer)
    `:layer_type`: indicates the layer type it is part of.
    `:index_in_layer`: indicates a `Tuple` with the position of the neuron
        within the layer. The indices cover the size of the layer, which is 
        given by a Tuple, e.g. of the form `(n,)` for `Dense(n,m)` and `RNN(n,m)`,
        or `(n₁, …, nₖ,m,d,b)` for convolutional layers, and so on.
"""
function chaingraph(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)
    m = f32(m)
    if m.layers[1] isa Union{FIXED_INPUT_DIM_LAYERS...}
        input_data = rand(Float32, layerdimensions(m.layers[1])[2]) 
    elseif input_data === nothing
        throw(ArgumentError("An `input_data` is required when the first layer accepts variable-dimension input"))
    else
        input_data = convert.(Float32, input_data)
    end
    chain_dimensions = get_dimensions(m, input_data)
    max_width, = maximum(chain_dimensions)
    connections = neuron_connections(m, input_data)

    neuron_to_node = Dict(
        (nl, c) =>  nl == 0 ? nc : sum([prod(chain_dimensions[nli+1]) for nli=0:nl-1]) +  nc 
            for nl in 0:length(m) for (nc,c) in enumerate(sort(neuron_indices(chain_dimensions[nl+1]))))
    node_to_neuron = Dict{Int, Tuple{Int,Tuple}}(v => k for (k,v) in neuron_to_node)

    mg = MetaGraph(length(node_to_neuron))
    for i in 1:length(node_to_neuron)
        set_props!(
            mg, i, Dict(
                :layer_number => first(node_to_neuron[i]),
                :layer_type => first(node_to_neuron[i]) == 0 ? "input" : string(m[first(node_to_neuron[i])]),
                :index_in_layer => last(node_to_neuron[i])
            )
        )
    end
    for nl in 0:length(m)-1
        for ni in neuron_indices(chain_dimensions[nl+1])
            for nj in connections[nl+1][ni]
                add_edge!(mg, neuron_to_node[(nl,ni)], neuron_to_node[(nl+1,nj)])
            end
        end
    end
    return mg
end

#= dx(x) = x[2:end]-x[1:end-1]
x³(x) = x.^3
m = Chain(x³, dx, LSTM(5,10), Dense(10,5))
input_data = rand(Float32,6)
mg = chaingraph(m, input_data)
@show get_prop(mg, 3, :layer_type)
@show collect(edges(mg)) =#
