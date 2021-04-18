# module NeuronState

import Random
import Base: show, isless, ==
import Adapt: adapt, adapt_storage

"""
    NeuronState <: Number

NeuronState encodes the "state" of a neuron as an Int8.
The possible states are:
    * `state = -1` for an off state, meaning it cannot be triggered by a signal
    * `state = 0` for a cold state, meaning it can be triggered by a signal but it has not yet been triggered
    * `state = 1` for a hot state, meaning it has been triggered by a signal.

The aliases are
    * `offneuron = NeuronState(Int8(-1))`
    * `coldneuron = NeuronState(Int8(0))`
    * `hotneuron = NeuronState(Int8(1))`

It subtypes `Number`, but the logic for the arithmetics is different.
"""
struct NeuronState <: Number
    state::Int8
end

const offneuron = NeuronState(Int8(-1))
const coldneuron = NeuronState(Int8(0))
const hotneuron = NeuronState(Int8(1))

Base.show(io::IO, x::NeuronState) = print(io, x == coldneuron ? "cold" : x == hotneuron ? "hot " : "off ")
Base.show(io::IO, ::MIME"text/plain", x::NeuronState) = print(io, "NeuronState:\n  ", x)

NeuronState(x::T) where {T<:Number} = iszero(x) ? offneuron : coldneuron
(::Type{T})(x::NeuronState) where {T<:Number} = x
Base.convert(::Type{NeuronState}, y::Number) = NeuronState(y)
Base.convert(::Type{NeuronState}, y::NeuronState) = y

Base.float(x::Type{NeuronState}) = x

isless(::NeuronState, ::Number) = true
isless(::Number, ::NeuronState) = true
isless(x::NeuronState, y::NeuronState) = isless(x.state, y.state)

==(::NeuronState, ::Number) = false
==(::Number, ::NeuronState) = false
==(x::NeuronState, y::NeuronState) = x.state == y.state

Base.isnan(::NeuronState) = false
Base.isfinite(::NeuronState) = true
Base.typemin(::Type{NeuronState}) = offneuron
Base.typemax(::Type{NeuronState}) = hotneuron

Base.promote_rule(::Type{NeuronState}, ::Type{<:Number}) = NeuronState

Random.rand(rng::Random.AbstractRNG, ::Random.SamplerType{NeuronState}) = 
    rand(rng, (offneuron, coldneuron))

for f in [:copy, :+, :-, :abs, :abs2, :inv, :tanh, :conj]
  @eval Base.$f(x::NeuronState) = x
end

for f in [:one, :oneunit, :exp]
@eval Base.$f(::NeuronState) = coldneuron
end

for f in [:zero, :log, :log1p, :log2, :log10]
    @eval Base.$f(::NeuronState) = offneuron
end

for f in [:+, :-]
  @eval Base.$f(x::NeuronState, y::NeuronState) = max(x,y)
end

for f in [:*, :/, :^, :mod, :div, :rem]
    @eval Base.$f(x::NeuronState, y::NeuronState) = 
        min(x, y) == offneuron ? offneuron : max(x, y)
end

for f in [:+, :-, :*, :/, :^, :mod, :div, :rem]
    @eval Base.$f(x::NeuronState, ::Number) = x
    @eval Base.$f(::Number, y::NeuronState) = y
end

adapt_storage(NeuronState, xs::AbstractArray{<:Number}) = convert.(NeuronState, xs)
"""
    fNeuronState(m)

Convert the `eltype` of model's parameters to `NeuronState`.

In the conversion, 0.0 turns into `offneuron` and any other value, into `coldneuron`.
"""
fNeuronState(m) = fmap(x -> adapt(NeuronState, x), m)

turnneuroncold(x) = x isa Number ? coldneuron : x

# end  # module

m = Chain(Dense(2,3), RNN(3,2))
mst = fmap(x -> adapt(NeuronState, x), m)
params(mst)
mst([coldneuron, hotneuron])
mopen = fmap( x -> turnneuroncold.(x), mst)
mopen([coldneuron, hotneuron])

m = Chain(x -> x[2:end] - x[1:end-1])
mst = fmap(x -> adapt(NeuronState, x), m)
mopen = fmap( x -> turnneuroncold.(x), mst)
mopen([coldneuron, hotneuron, coldneuron, coldneuron, coldneuron])