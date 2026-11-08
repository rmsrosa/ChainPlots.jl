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
    get_dimensions(m::Flux.Chain, inp::Union{Nothing, Array, Tuple} = nothing)

Return the dimensions of the input and of the output data of each hidden layer.

If `input_data` is not given, the first layer is required to be a layer
with fixed input dimensions, such as `Flux.Dense` or `Flux.RNN`,
otherwise the given data or shape is used to infer the dimensions of each layer.
"""
function get_dimensions(m::Flux.Chain, input_data::Array)
    check_chain(m)
    x = fmap(f32, input_data)
    m32 = f32(m)

    chain_dimensions = Any[size(x)]
    for l in m32.layers
        x = applylayer(l, x)
        push!(chain_dimensions, size(x))
    end
    return [chain_dimensions...]
end

function get_dimensions(m::Flux.Chain, ::Nothing=nothing)
    m.layers[1] isa Union{FIXED_INPUT_DIM_LAYERS...} || throw(ArgumentError("An input data or shape is required when the first layer accepts variable-dimension input"))

    input_data = rand(Float32, layerdimensions(m.layers[1])[2])
    return get_dimensions(m, input_data)
end

get_dimensions(m::Flux.Chain, ldim::Tuple) = get_dimensions(m, rand(Float32, ldim))

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
    neuron_connections(m::Flux.Chain, input_data::Union{Nothing,Array,Tuple} = nothing) -> Vector{Dict{Tuple, Vector{Tuple}}}

Return all the connections from every neuron in each layer to the corresponding neurons in the next layer.
"""
function neuron_connections(m::Flux.Chain, input_data::Union{Nothing,Array,Tuple}=nothing)
    chain_dimensions = get_dimensions(m, input_data)
    mn = fneutralize(m)

    connections = Vector{Dict{Tuple,Vector{Tuple}}}()

    for (ln, l) in enumerate(mn)
        ldim = chain_dimensions[ln]
        layer_connections = Dict{Tuple,Vector{Tuple}}()
        basis_element = fill(cold, ldim)
        for idx in neuron_indices(ldim)
            connected = Vector{Tuple}()
            basis_element[idx...] = hot
            union!(connected, Tuple.(findall(x -> x == hot, applylayer(l, basis_element))))
            push!(layer_connections, idx => connected)
            basis_element[idx...] = cold
        end
        push!(connections, layer_connections)
    end
    return connections
end
