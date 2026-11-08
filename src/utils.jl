"""
    SEQUENCE_LAYERS

Recurrent layers that process a whole sequence at once (Flux ≥ 0.15).

ChainPlots draws them folded, as a single time step: a vector input of length `in`
is fed to them as a sequence of length one, and their output is taken as a vector
of length `out`.
"""
const SEQUENCE_LAYERS = (Flux.RNN, Flux.LSTM, Flux.GRU, Flux.GRUv3)

"""
    RECURRENT_CELLS

Recurrent cells that process a single time step, returning `(y, state)`.

Since a cell returns a tuple, a `Flux.Chain` cannot continue after it, so a cell is
only accepted on its own or as the last layer of a chain.
"""
const RECURRENT_CELLS = (Flux.RNNCell, Flux.LSTMCell, Flux.GRUCell, Flux.GRUv3Cell)

"""
    FIXED_INPUT_DIM_LAYERS

List of layers with fixed-size input data
"""
const FIXED_INPUT_DIM_LAYERS = (Flux.Dense, SEQUENCE_LAYERS..., RECURRENT_CELLS...)

"""
    layerdimensions(l)

Retrieve the dimensions `(out, in)` of a given fixed-input-size layer.
"""
layerdimensions(l::Flux.Dense) = size(l.weight)
layerdimensions(l::Union{RECURRENT_CELLS...}) = (size(l.Wh, 2), size(l.Wi, 2))
layerdimensions(l::Union{SEQUENCE_LAYERS...}) = layerdimensions(l.cell)

"""
    applylayer(l, x)

Apply layer `l` to `x` the way ChainPlots models it.

Sequence layers given a vector are run on a sequence of length one and their output
is returned as a vector, and recurrent cells return only their output `y`, dropping
the state.
"""
applylayer(l, x) = l(x)
applylayer(l::Union{SEQUENCE_LAYERS...}, x::AbstractVector) = vec(l(reshape(x, :, 1)))
applylayer(l::Union{RECURRENT_CELLS...}, x) = first(l(x))

"""
    check_chain(m::Flux.Chain)

Throw an `ArgumentError` if a recurrent cell appears before the last layer of `m`.
"""
function check_chain(m::Flux.Chain)
    for (n, l) in enumerate(m.layers)
        if l isa Union{RECURRENT_CELLS...} && n < length(m.layers)
            seqlayer = replace(string(nameof(typeof(l))), "Cell" => "")
            throw(ArgumentError("The recurrent cell $l at position $n returns a tuple `(y, state)`, so it can only be the last layer of a Chain. Use the sequence layer `$seqlayer` instead."))
        end
    end
    return nothing
end

"""
    ModelLayers

Internal description of a model as a sequence of layers, which is what [`chaingraph`](@ref)
works with, so that it supports models from different packages.

For each layer, it holds a function applying the layer to an array of `Float32`, used to
infer the dimensions of each layer, a function applying it to an array of `NeuralNumber`s,
used to infer the connections between neurons, and its name and type, such as `:Dense`.
"""
struct ModelLayers
    forward::Vector{Any}
    neutral::Vector{Any}
    names::Vector{String}
    types::Vector{Symbol}
end

Base.length(ml::ModelLayers) = length(ml.forward)

"""
    modellayers(m::Flux.Chain) -> ModelLayers

Describe the Flux chain `m` as a [`ModelLayers`](@ref).
"""
function modellayers(m::Flux.Chain)
    check_chain(m)
    m32 = f32(m)
    mn = fneutralize(m)
    return ModelLayers(
        Any[x -> applylayer(l, x) for l in m32.layers],
        Any[x -> applylayer(l, x) for l in mn.layers],
        [string(l) for l in m.layers],
        [nameof(typeof(l)) for l in m.layers]
    )
end

"""
    input_array(m, input::Union{Nothing,Array,Tuple})

Return an input for the model `m`: `input` itself if it is an `Array`, a random array of
shape `input` if it is a `Tuple`, and, if it is `nothing`, a random input of the size
expected by the first layer, when it has a fixed input size.
"""
input_array(m, input::Array) = input
input_array(m, ldim::Tuple) = rand(Float32, ldim)
function input_array(m::Flux.Chain, ::Nothing)
    m.layers[1] isa Union{FIXED_INPUT_DIM_LAYERS...} || throw(ArgumentError("An input data or shape is required when the first layer accepts variable-dimension input"))
    return rand(Float32, layerdimensions(m.layers[1])[2])
end

"""
    get_dimensions(m, input = nothing)

Return the dimensions of the input and of the output data of each layer of the model `m`,
given as a `Flux.Chain` or a [`ModelLayers`](@ref).

If `input` is not given, the first layer is required to be a layer with fixed input
dimensions, such as `Flux.Dense` or `Flux.RNN`, otherwise the given data, or the shape
given as a `Tuple`, is used to infer the dimensions of each layer.
"""
function get_dimensions(ml::ModelLayers, input_data::Array)
    x = fmap(f32, input_data)
    chain_dimensions = Any[size(x)]
    for f in ml.forward
        x = f(x)
        push!(chain_dimensions, size(x))
    end
    return [chain_dimensions...]
end

get_dimensions(m::Flux.Chain, input=nothing) = get_dimensions(modellayers(m), input_array(m, input))

"""
    UnitVector{T}

Structure for unit vectors in a linear space
    
Used for generating a basis to infer the layer connection
"""
struct UnitVector{T} <: AbstractVector{T}
    idx::Int
    length::Int
end

Base.getindex(x::UnitVector{T}, i) where {T} = x.idx == i ? one(T) : zero(T)
Base.length(x::UnitVector) = x.length
Base.size(x::UnitVector) = (x.length,)

"""
    neuron_indices(ldim:Tuple) -> Vector{NTuple{N, Int}} where N

Return all possible indices for a given Tuple `ldim`.
"""
function neuron_indices(ldim::Tuple)
    return [Tuple(1 + mod(div(i, prod(ldim[1:j-1])), ldim[j]) for j in 1:length(ldim)) for i in 0:prod(ldim)-1]
end

"""
    neuron_connections(m, input = nothing) -> Vector{Dict{Tuple, Vector{Tuple}}}

Return all the connections from every neuron in each layer to the corresponding neurons
in the next layer of the model `m`, given as a `Flux.Chain` or a [`ModelLayers`](@ref).
"""
function neuron_connections(ml::ModelLayers, input_data::Array)
    chain_dimensions = get_dimensions(ml, input_data)

    connections = Vector{Dict{Tuple,Vector{Tuple}}}()

    for (ln, f) in enumerate(ml.neutral)
        ldim = chain_dimensions[ln]
        layer_connections = Dict{Tuple,Vector{Tuple}}()
        basis_element = fill(cold, ldim)
        for idx in neuron_indices(ldim)
            connected = Vector{Tuple}()
            basis_element[idx...] = hot
            union!(connected, Tuple.(findall(x -> x == hot, f(basis_element))))
            push!(layer_connections, idx => connected)
            basis_element[idx...] = cold
        end
        push!(connections, layer_connections)
    end
    return connections
end

neuron_connections(m::Flux.Chain, input=nothing) = neuron_connections(modellayers(m), input_array(m, input))
