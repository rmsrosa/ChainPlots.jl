"""
    graphchain(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)

Return a MetaGraph representing the graph structure of the neural network.
"""
function graphchain(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)
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
    connections = get_connections(m, input_data)

    neuron_to_node = Dict{Tuple{Int,CartesianIndex}, Int}(
        [(nl,c) =>  nl == 0 ? nc : sum([prod(chain_dimensions[nli+1]) for nli=0:nl-1]) +  nc 
            for nl in 0:length(m) for (nc,c) in enumerate(sort(get_cartesians(chain_dimensions[nl+1])))]
    )
    neuron_to_note = Dict(
        [(nl, c) =>  nl == 0 ? nc : sum([prod(chain_dimensions[nli+1]) for nli=0:nl-1]) +  nc 
            for nl in 0:length(m) for (nc,c) in enumerate(sort(get_cartesians(chain_dimensions[nl+1])))])
    node_to_neuron = Dict{Int, Tuple{Int,CartesianIndex}}(v => k for (k,v) in neuron_to_node)

    mg = MetaGraph(length(node_to_neuron))
    for i in 1:length(node_to_neuron)
        set_props!(mg, i, Dict([:layer => first(node_to_neuron[i])]))
    end
    get_prop(mg, 2, :layer)
    for nl in 0:length(m)-1
        for ni in get_cartesians(chain_dimensions[nl+1])
            for nj in connections[nl+1][ni]
                add_edge!(mg, neuron_to_node[(nl,ni)], neuron_to_node[(nl+1,nj)])
            end
        end
    end
    return mg
end

graphchain(Chain(x³, dx, LSTM(5,10), Dense(10,5)), rand(Float32,6))

# collect(edges(g))

        


