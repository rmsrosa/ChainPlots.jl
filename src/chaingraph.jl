"""
    NEURON_COLORS::Dict{Symbol, Symbol}

Specify the color for each type of layer.
"""
NEURON_COLORS = Dict(
    :Any => :gray,
    :Dense => :lightgreen,
    :RNNCell => :lightskyblue1,
    :LSTMCell => :skyblue2,
    :GRUCell => :skyblue3,
    :Conv => :plum,
    :input_layer => :yellow,
    :output_layer => :orange
)
"""
    neuron_color()

Grab the color for each specific type of neuron.
    
The color depends on the type of layer the neuron belongs to and the colorset given by [`NEURON_COLORS`](@ref).
"""
neuron_color(::T; neuron_colorset = NEURON_COLORS) where T = nameof(T) in keys(neuron_colorset) ? neuron_colorset[nameof(T)] : neuron_colorset[:Any]
neuron_color(r::Flux.Recur; neuron_colorset = NEURON_COLORS) = neuron_color(r.cell; neuron_colorset)
neuron_color(s::Symbol; neuron_colorset = NEURON_COLORS) = s in keys(neuron_colorset) ? neuron_colorset[s] : throw(ArgumentError("Color not defined for the given symbol $s."))

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
    chaingraph(m::Flux.Chain, input_data::Array)

Return a MetaGraph representing the graph structure of the neural network.

Each node represents a neuron of the Chain `m` and contains the following properties:
    `:layer_number`: Int indicating to each layer it belongs (
        with `0` indicating the input layer, `1, …, length(m)-1` 
        indicating the hidden layers, and with `length(m)` indicating
        the output layer);
    `:layer_type`: string indicating the layer type it is part of in the Chain
        (e.g. "Dense()", "Recur()", "Conv()", …);
    `:index_in_layer`: `Tuple` indicating the position of the neuron
        within the layer. The indices cover the size of the layer, which is 
        given by a Tuple, e.g. of the form `(n,)` for `Dense(n,m)` and
        `RNN(n,m)`, or `(n₁, …, nₖ,m,d,b)` for convolutional layers, and so on;
    `:layer_center`: Float64 with the vertical mid-point of the layer it belongs to.
"""
function chaingraph(m::Flux.Chain, input_data::Array)
    # m32 = f32(m) # this is not needed so far but leave it as a comment just in case
    chain_dimensions = get_dimensions(m, input_data)
    #= max_dim = maximum(length.(chain_dimensions))
    full_dimensions = [
        Tuple(length(v) ≥ n ? v[n] : 1 for n in 1:max_dim) for v in chain_dimensions
    ]
    max_widths = [maximum(getindex.(full_dimensions, i)) for i in 1:max_dim] =#
    # max_width, = maximum(chain_dimensions)
    max_width = maximum(getindex.(chain_dimensions, 1))
    connections = neuron_connections(m, input_data)

    neuron_to_node = Dict(
        (nl, c) => nl == 0 ? nc : sum([prod(chain_dimensions[nli+1]) for nli = 0:nl-1]) + nc
        for nl in 0:length(m) for (nc, c) in enumerate(sort(neuron_indices(chain_dimensions[nl+1]))))
    node_to_neuron = Dict{Int,Tuple{Int,Tuple}}(v => k for (k, v) in neuron_to_node)

    mg = MetaGraph(length(node_to_neuron))
    for i in 1:length(node_to_neuron)
        center = chain_dimensions[first(node_to_neuron[i])+1][1] / 2
        x, y = projection(node_to_neuron[i], center, max_width, chain_dimensions[first(node_to_neuron[i])+1])
        set_props!(
            mg, i, Dict(
                :layer_number => first(node_to_neuron[i]),
                :layer_type => first(node_to_neuron[i]) == 0 ? "input layer" : string(m[first(node_to_neuron[i])]),
                :index_in_layer => last(node_to_neuron[i]),
                :layer_center => center,
                :loc_x => x,
                :loc_y => y,
                :neuron_color => neuron_color(first(node_to_neuron[i]) == 0 ? :input_layer : m[first(node_to_neuron[i])])
            )
        )
    end
    for nl in 0:length(m)-1
        for ni in neuron_indices(chain_dimensions[nl+1])
            for nj in connections[nl+1][ni]
                add_edge!(mg, neuron_to_node[(nl, ni)], neuron_to_node[(nl + 1, nj)])
            end
        end
    end
    return mg
end

"""
    chaingraph(m::Flux.Chain, ldim::Tuple)

Return a MetaGraph representing the graph structure of the neural network with an input of shape `ldim`.

See [`chaingraph`](@ref) for the properties of each node of the graph.
"""
chaingraph(m::Flux.Chain, ldim::Tuple) = chaingraph(m, rand(Float32, ldim))

"""
    chaingraph(m::Flux.Chain)

Return a MetaGraph representing the graph structure of the neural network.

In this case, the first layer must be a layer with fixed input dimension.

See [`chaingraph`](@ref) for the properties of each node of the graph.
"""
function chaingraph(m::Flux.Chain, ::Nothing=nothing)
    m.layers[1] isa Union{FIXED_INPUT_DIM_LAYERS...} || throw(ArgumentError("An input data or shape is required when the first layer accepts variable-dimension input"))

    input_data = rand(Float32, layerdimensions(m.layers[1])[2])
    return chaingraph(m, input_data)
end

"""
    chaingraph(l::Union{Flux.Dense,Flux.Recur,Flux.RNNCell,Flux.LSTMCell,Flux.GRUCell})

Return a MetaGraph representing the graph structure of a neural network composed of the single layer `l`.
"""
chaingraph(l::Union{Flux.Dense,Flux.Recur,Flux.RNNCell,Flux.LSTMCell,Flux.GRUCell}) = chaingraph(Flux.Chain(l))
