"""
    NEURON_COLORS::Dict{Symbol, Symbol}

Specify the color for each type of layer.
"""
NEURON_COLORS = Dict(
    :Any => :gray,
    :Dense => :lightgreen,
    :RNN => :lightskyblue1,
    :RNNCell => :lightskyblue1,
    :LSTM => :skyblue2,
    :LSTMCell => :skyblue2,
    :GRU => :skyblue3,
    :GRUCell => :skyblue3,
    :GRUv3 => :skyblue3,
    :GRUv3Cell => :skyblue3,
    :Conv => :plum,
    :input_layer => :yellow,
    :output_layer => :orange
)
"""
    neuron_color(s; neuron_colors = NEURON_COLORS)

Grab the color for each specific type of neuron.
    
The color depends on the type of layer the neuron belongs to and the colorset given `neuron_colors`, which defaults to [`NEURON_COLORS`](@ref).
"""
neuron_color(::T; neuron_colors = NEURON_COLORS) where T = nameof(T) in keys(neuron_colors) ? neuron_colors[nameof(T)] : neuron_colors[:Any]
neuron_color(s::Symbol; neuron_colors = NEURON_COLORS) = s in keys(neuron_colors) ? neuron_colors[s] : throw(ArgumentError("Color not defined for the given symbol $s."))

"""
    projection(z, center, max_widths, dimensions)

Transform the indexing of a neuron into its x and y coordinates for plotting.
"""
function projection(z::Tuple{T, NTuple{N, T}}, center, max_width, dimensions) where {T, N}
    layer, pos = z
    slope = 0.2
    span = 0.3
    x = N > 2 && dimensions[2] > 1 ? layer + span * ( ( pos[2] - 1 ) / ( dimensions[2] - 1) - 0.5) : layer
    y = ((pos[1] - center + max_width / 2) / (max_width + 1)) + slope * (x - layer)
    return float(x), float(y)
end

"""
    chaingraph(m::Flux.Chain, input_data::Array) -> ChainGraph

Return a [`ChainGraph`](@ref) with the graph structure of the neural network `m`
when applied to `input_data`.

Each vertex is a neuron of the network, and each edge connects a neuron to a neuron of
the next layer that it affects. The properties of each neuron, listed in
[`NODE_PROPERTIES`](@ref), are read with [`nodeprop`](@ref) and [`nodeprops`](@ref).

The recurrent layers `RNN`, `LSTM`, `GRU` and `GRUv3` are drawn folded, as a single
time step, when the input is a vector. If the input is a sequence of size `(in, len)`,
the graph has a neuron for each time step instead, with each time step connected to the
same and the later time steps of the next layer; the plot recipe does not lay the time
steps out separately, though. The recurrent cells (`RNNCell`, …) return a tuple
`(y, state)`, so they can only be the last layer of a chain.

The plot recipe for `ChainGraph` draws it, as in `plot(chaingraph(m, input_data))`;
see also [`plotchain`](@ref).
"""
chaingraph(m::Flux.Chain, input_data::Array) = chaingraph(modellayers(m), input_data)

"""
    chaingraph(ml::ModelLayers, input_data::Array) -> ChainGraph

Build the [`ChainGraph`](@ref) of the model described by `ml`, when applied to `input_data`.

This is the method all the others end up calling, after describing their model as a
[`ModelLayers`](@ref).
"""
function chaingraph(ml::ModelLayers, input_data::Array)
    chain_dimensions = get_dimensions(ml, input_data)
    max_width = mapreduce(x -> x[1], max, chain_dimensions)
    connections = neuron_connections(ml, input_data)

    neuron_to_node = Dict(
        (nl, c) => nl == 0 ? nc : sum([prod(chain_dimensions[nli+1]) for nli = 0:nl-1]) + nc
        for nl in 0:length(ml) for (nc, c) in enumerate(sort(neuron_indices(chain_dimensions[nl+1]))))
    node_to_neuron = Dict{Int,Tuple{Int,Tuple}}(v => k for (k, v) in neuron_to_node)

    layer_names = ["input layer"; ml.names]
    layer_types = [:input_layer; ml.types]
    mg = MetaGraph(length(node_to_neuron))
    for i in 1:length(node_to_neuron)
        center = chain_dimensions[first(node_to_neuron[i])+1][1] / 2
        x, y = projection(node_to_neuron[i], center, max_width, chain_dimensions[first(node_to_neuron[i])+1])
        set_props!(
            mg, i, Dict(
                :layer_number => first(node_to_neuron[i]),
                :layer_name => layer_names[first(node_to_neuron[i])+1],
                :layer_type => layer_types[first(node_to_neuron[i])+1],
                :index_in_layer => last(node_to_neuron[i]),
                :layer_center => center,
                :loc_x => x,
                :loc_y => y,
                :neuron_color => get(NEURON_COLORS, layer_types[first(node_to_neuron[i])+1], NEURON_COLORS[:Any])
            )
        )
    end
    for nl in 0:length(ml)-1
        for ni in neuron_indices(chain_dimensions[nl+1])
            for nj in connections[nl+1][ni]
                add_edge!(mg, neuron_to_node[(nl, ni)], neuron_to_node[(nl + 1, nj)])
            end
        end
    end
    return ChainGraph(mg, chain_dimensions, layer_names)
end

"""
    chaingraph(m::Flux.Chain, ldim::Tuple) -> ChainGraph

Return a [`ChainGraph`](@ref) with the graph structure of the neural network `m` with an input of shape `ldim`.

See [`chaingraph`](@ref) for the properties of each node of the graph.
"""
chaingraph(m::Flux.Chain, ldim::Tuple) = chaingraph(m, rand(Float32, ldim))

"""
    chaingraph(m::Flux.Chain) -> ChainGraph

Return a [`ChainGraph`](@ref) with the graph structure of the neural network `m`.

In this case, the first layer must be a layer with fixed input dimension.

See [`chaingraph`](@ref) for the properties of each node of the graph.
"""
chaingraph(m::Flux.Chain, ::Nothing=nothing) = chaingraph(m, input_array(m, nothing))

"""
    chaingraph(l::Union{FIXED_INPUT_DIM_LAYERS...}) -> ChainGraph

Return a [`ChainGraph`](@ref) with the graph structure of a neural network composed of the single layer `l`.

This covers `Flux.Dense`, the sequence layers `Flux.RNN`, `Flux.LSTM`, `Flux.GRU` and `Flux.GRUv3`,
and their cells.
"""
chaingraph(l::Union{FIXED_INPUT_DIM_LAYERS...}, ::Nothing=nothing) = chaingraph(Flux.Chain(l))

"""
    chaingraph(l, input::Union{Array,Tuple}) -> ChainGraph

Return a [`ChainGraph`](@ref) with the graph structure of a neural network composed of
the single layer `l`, such as a `Flux.Conv`, applied to `input`, or to an input of shape `input`.
"""
chaingraph(l, input::Union{Array,Tuple}) = chaingraph(Flux.Chain(l), input)
