module ChainPlotsLuxExt

# Support for Lux.jl models: `chaingraph` and `plotchain` accept a `Lux.Chain` or a single
# Lux layer. Lux layers keep no parameters, so the extension sets them up itself; since
# the graph only depends on the shapes of the parameters, their values do not matter.

using ChainPlots: ChainPlots, ModelLayers
import ChainPlots: chaingraph, modellayers, input_array
using ChainPlots.NeuralNumbers: fneutralize
using Lux
using Random: Xoshiro

const LUX_CELLS = (Lux.RNNCell, Lux.LSTMCell, Lux.GRUCell)

# the layers of a model, with a `Lux.Chain` as the only container that is unpacked
luxlayers(m::Lux.Chain) = collect(values(m.layers))
luxlayers(l) = Any[l]

# how ChainPlots applies a Lux layer: cells return `(y, state)`, of which only `y` is
# kept, and a `Recurrence` given a vector is run on a sequence of length one, as with
# Flux's sequence layers
luxapply(l, x, ps, st) = first(l(x, ps, st))
luxapply(l::Union{LUX_CELLS...}, x, ps, st) = first(first(l(x, ps, st)))
function luxapply(l::Lux.Recurrence, x::AbstractVector, ps, st)
    y = first(l(reshape(x, :, 1), ps, st))
    # with `return_sequence = true`, the output is a vector with one array per time step
    return vec(y isa AbstractVector{<:AbstractArray} ? only(y) : y)
end

# names and types, with a `Recurrence` named and drawn after its cell
luxname(l) = sprint(show, l)
luxname(l::Lux.Chain) = "Chain(" * join(luxname.(luxlayers(l)), ", ") * ")"
function luxname(l::Lux.Recurrence)
    return "Recurrence(" * luxname(l.cell) * (l.return_sequence == true ? "; return_sequence = true" : "") * ")"
end
luxtype(l) = nameof(typeof(l))
luxtype(l::Lux.Recurrence) = Symbol(replace(string(nameof(typeof(l.cell))), "Cell" => ""))

# input size of the layers with a fixed one
luxinputdims(l) = nothing
luxinputdims(l::Union{Lux.Dense,LUX_CELLS...}) = l.in_dims
luxinputdims(l::Lux.Recurrence) = luxinputdims(l.cell)
luxinputdims(l::Lux.Chain) = luxinputdims(first(luxlayers(l)))

function check_luxchain(layers)
    for (n, l) in enumerate(layers)
        if l isa Union{LUX_CELLS...} && n < length(layers)
            throw(ArgumentError("The recurrent cell $(luxname(l)) at position $n returns a tuple `(y, state)`, so it can only be the last layer of a Chain. Wrap it in `Lux.Recurrence` instead."))
        end
    end
    return nothing
end

function modellayers(m::Lux.AbstractLuxLayer)
    layers = luxlayers(m)
    check_luxchain(layers)
    rng = Xoshiro(0)
    forward = Any[]
    neutral = Any[]
    for l in layers
        ps, st = Lux.setup(rng, l)
        st = Lux.testmode(st)
        nps = fneutralize(ps)
        push!(forward, x -> luxapply(l, x, ps, st))
        push!(neutral, x -> luxapply(l, x, nps, st))
    end
    return ModelLayers(forward, neutral, [luxname(l) for l in layers], [luxtype(l) for l in layers])
end

function input_array(m::Lux.AbstractLuxLayer, ::Nothing)
    dims = luxinputdims(m)
    dims === nothing && throw(ArgumentError("An input data or shape is required when the first layer accepts variable-dimension input"))
    return rand(Float32, dims)
end

chaingraph(m::Lux.AbstractLuxLayer, input::Array) = chaingraph(modellayers(m), input)
chaingraph(m::Lux.AbstractLuxLayer, input::Tuple) = chaingraph(m, input_array(m, input))
chaingraph(m::Lux.AbstractLuxLayer, ::Nothing=nothing) = chaingraph(m, input_array(m, nothing))

end # module
