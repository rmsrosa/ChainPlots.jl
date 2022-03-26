"""
    layerdimensions()

Retrive dimensions of a given fixed-input-size layer.
"""
layerdimensions(::Any) = (1,1)
layerdimensions(l::Flux.Dense) = size(l.weight)
layerdimensions(l::Flux.RNNCell) = (size(l.Wh)[2], size(l.Wi)[2])
layerdimensions(l::Flux.LSTMCell) = (size(l.Wh)[2], size(l.Wi)[2])
layerdimensions(l::Flux.GRUCell) = (size(l.Wh)[2], size(l.Wi)[2])
layerdimensions(r::Flux.Recur) = layerdimensions(r.cell)

"""
    FIXED_INPUT_DIM_LAYERS

List of layers with fixed-sized input data
"""
const FIXED_INPUT_DIM_LAYERS = (Flux.Dense, Flux.Recur, Flux.RNNCell, Flux.LSTMCell, Flux.GRUCell) # list of types of layers with fixed input dimensions

"""
    get_dimensions(m::Flux.Chain, input_data = nothing)

Return the dimensions of the input and of the output data of each hidden layer.

If `input_data` is not given, the first layer is required to be a layer
with fixed input dimensions, such as Flux.Dense or Flux.Recur,
otherwise the given data is used to infer the dimensions of each layer.
"""
function get_dimensions(m::Flux.Chain, input_data::Union{Nothing,Array} = nothing)

    if (input_data === nothing) & (m.layers[1] isa Union{FIXED_INPUT_DIM_LAYERS...})
        input_data = rand(Float32, layerdimensions(m.layers[1])[2]) 
    elseif input_data === nothing
        throw(ArgumentError("An `input_data` is required when the first layer accepts variable-dimension input"))
    else
        input_data = convert.(Float32, input_data)
    end

    chain_dimensions = vcat(size(input_data), [size(m[1:nl](input_data)) for nl in 1:length(m.layers)])
    return chain_dimensions
end

function get_dimensions(m::Flux.Chain)
    if m.layers[1] isa Union{FIXED_INPUT_DIM_LAYERS...}
        input_data = rand(Float32, layerdimensions(m.layers[1])[2]) 
    else
        throw(ArgumentError("An `input_data` is required when the first layer accepts variable-dimension input"))
    end
    return get_dimensions(m, input_data)
end

function get_dimensions(m::Flux.Chain, ldim::Tuple)
    input_data = rand(Float32, ldim)
    return get_dimensions(m, input_data)
end


"""
    UnitVector{T}

Structure for unit vectors in a linear space
    
Used for generating a basis to infer the layer connection
"""
struct UnitVector{T} <: AbstractVector{T}
    idx::Int
    length::Int
end

Base.getindex(x::UnitVector{T}, i) where T = x.idx == i ? one(T) : zero(T)
Base.length(x::UnitVector) = x.length
Base.size(x::UnitVector) = (x.length,)

"""
    neuron_indices(ldim:Tuple) -> Vector{NTuple{N, Int}} where N

Return all possible indices for a given Tuple `ldim`.
"""
function neuron_indices(ldim::Tuple)
    return [Tuple(1+mod(div(i,prod(ldim[1:j-1])), ldim[j]) for j in 1:length(ldim)) for i in 0:prod(ldim)-1]
end

"""
    neuron_connections(m::Flux.Chain) -> Vector{Dict{Tuple, Vector{Tuple}}}

Return all the connections from every neuron in each layer to the corresponding neurons in the next layer.
"""
function neuron_connections(morg::Flux.Chain, input_data::Union{Nothing,Array} = nothing)
    chain_dimensions = get_dimensions(morg, input_data)
    m = fcooloffneurons(morg)
    connections = Vector{Dict{Tuple, Vector{Tuple}}}()

    for (ln, l) in enumerate(m)
        ldim = chain_dimensions[ln]
        layer_connections = Dict{Tuple,Array{Tuple,1}}()
        basis_element = fill(coldneuron, ldim)
        for idx in neuron_indices(ldim)
            connected = Array{Tuple,1}()
            basis_element[idx...] = hotneuron
            for j in 1:6 # multiple passes needed to get all the connections in some conv layers; don't know why
                union!(connected, Tuple.(findall(x -> x == hotneuron, l(basis_element))))
            end
            push!(layer_connections, idx => connected)
            basis_element[idx...] = coldneuron
        end
        push!(connections, layer_connections)
    end
    return connections
end

function neuron_connections_alt(morg::Flux.Chain, input_data::Union{Nothing,Array} = nothing)
    chain_dimensions = get_dimensions(morg, input_data)
    m = fcooloffneurons(morg)
    connections = Vector{Dict{Tuple, Vector{Tuple}}}()

    for (ln, l) in enumerate(m)
        ldim = chain_dimensions[ln]
        layer_connections = Dict{Tuple,Array{Tuple,1}}()
        basis_element = fill(coldneuron, ldim)
        for idx in neuron_indices(ldim)
            connected = Array{Tuple,1}()
            basis_element[idx...] = hotneuron
            for j in 1:10 # multiple passes needed to get all the connections in some conv layers
                union!(connected, Tuple.(findall(x -> x == hotneuron, l(basis_element))))
            end
            push!(layer_connections, idx => connected)
            basis_element[idx...] = coldneuron
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
