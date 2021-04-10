"""
    neuron_color()

Define color for each specific type of neuron, depending ot the type of layer it belongs to.
"""                  
neuron_color(::Any) = :gray
neuron_color(::Flux.Dense) = :lightgreen
neuron_color(::Flux.RNNCell) = :lightskyblue1
neuron_color(::Flux.LSTMCell) = :skyblue2
neuron_color(::Flux.GRUCell) = :skyblue3
neuron_color(r::Flux.Recur) = neuron_color(r.cell)
neuron_color(::Flux.Conv) = :plum
neuron_color(s::Symbol) = s == :input_layer ? :yellow : s == :output_layer ? :orange : 
    throw(ArgumentError("Color not defined for the given symbol $s."))

"""
projection(x, center, max_width, slope)

Transform a Tuple x of a neuron into its y-coordinate for plotting.
"""
projection(x, center, max_width, slope=0) = ((x[1] - center + max_width/2)/(max_width + 1))

"""
    chaingraph(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)

Return a MetaGraph representing the graph structure of the neural network.

Each node represents a neuron of the Chain `m` and contains the following
properties:
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
                :layer_type => first(node_to_neuron[i]) == 0 ? "input layer" : string(m[first(node_to_neuron[i])]),
                :index_in_layer => last(node_to_neuron[i]),
                :layer_center => chain_dimensions[first(node_to_neuron[i])+1][1]/2,
                :loc_x => convert(Float64,first(node_to_neuron[i])),
                :loc_y => projection(last(node_to_neuron[i]), chain_dimensions[first(node_to_neuron[i])+1][1]/2, max_width),
                :neuron_color => neuron_color(first(node_to_neuron[i]) == 0 ? :input_layer : m[first(node_to_neuron[i])])
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
