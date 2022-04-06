module NeuralNumbers

import Random
import Base: isless, ==
import Functors: fmap

"""
    NeuronState <: Real

NeuronState encodes the "state" of a neuron as an Int8.

The possible states are:
    * `state = Int8(0)` for a "cold", or "off", state, meaning it can be triggered by a signal but it has not yet been triggered. It works as a neutral element in any diadic operation.
    * `state = Int8(1)` for a "hot", or "on", state, meaning it has been triggered by a signal.

The aliases are
    * `cold = NeuronState(Int8(0))`
    * `hot = NeuronState(Int8(1))`
"""
struct NeuronState <: Real
    state::Int8
end

const cold = NeuronState(Int8(0))
const hot = NeuronState(Int8(1))

Base.show(io::IO, x::NeuronState) = print(io, x == hot ? "hot" : "cold")
Base.show(io::IO, ::MIME"text/plain", x::NeuronState) = print(io, "NeuronState:\n  ", x)

NeuronState(::Number) = cold
(::Type{NeuronState})(x::NeuronState) = x
Base.convert(::Type{NeuronState}, y::Number) = cold
Base.convert(::Type{NeuronState}, y::NeuronState) = y

Base.float(x::NeuronState) = x

==(::NeuronState, ::Number) = false
==(::Number, ::NeuronState) = false
==(x::NeuronState, y::NeuronState) = x.state == y.state

isless(x::NeuronState, ::Number) = false
isless(::Number, x::NeuronState) = true
isless(x::NeuronState, y::NeuronState) = isless(x.state, y.state)

Base.:<(x::NeuronState, y::NeuronState) = x.state < y.state
Base.:≤(x::NeuronState, y::NeuronState) = x.state ≤ y.state

Base.one(::Type{NeuronState}) = cold
Base.zero(::Type{NeuronState}) = cold
Base.one(x::NeuronState) = x
Base.oneunit(x::NeuronState) = x
Base.zero(::NeuronState) = cold

Base.iszero(x::NeuronState) = x == cold
Base.isnan(::NeuronState) = false
Base.isfinite(::NeuronState) = true
Base.isinf(::NeuronState) = false
Base.typemin(::Type{NeuronState}) = cold
Base.typemax(::Type{NeuronState}) = hot

Base.size(::NeuronState) = ()
Base.size(::NeuronState, d::Integer) = d < 1 ? throw(BoundsError()) : 1
Base.axes(::NeuronState) = ()
Base.axes(::NeuronState, d::Integer) = d < 1 ? throw(BoundsError()) : Base.OneTo(1)
Base.eltype(::Type{NeuronState}) = NeuronState
Base.ndims(x::NeuronState) = 0
Base.ndims(::Type{NeuronState}) = 0
Base.length(x::NeuronState) = 1
Base.firstindex(x::NeuronState) = 1
Base.firstindex(::NeuronState, d::Int) = d < 1 ? throw(BoundsError()) : 1
Base.lastindex(x::NeuronState) = 1
Base.lastindex(::NeuronState, d::Int) = d < 1 ? throw(BoundsError()) : 1
Base.IteratorSize(::Type{NeuronState}) = Base.HasShape{0}()
Base.keys(::NeuronState) = Base.OneTo(1)

Base.getindex(x::NeuronState) = x

@inline Base.getindex(x::NeuronState, i::Integer) = @boundscheck i == 1 ? x : throw(BoundsError())
@inline Base.getindex(x::NeuronState, I::Integer...) = @boundscheck all(isone, I) ? x : throw(BoundsError())

Base.first(x::NeuronState) = x
Base.last(x::NeuronState) = x
Base.copy(x::NeuronState) = x

Base.signbit(x::NeuronState) = x.state < 0
Base.sign(x::NeuronState) = x.state

Base.iterate(x::NeuronState) = (x, nothing)
Base.iterate(::NeuronState, ::Any) = nothing
Base.isempty(x::NeuronState) = false
Base.in(x::NeuronState, y::NeuronState) = x == y

Base.map(f, x::NeuronState, ys::NeuronState...) = f(x, ys...)

Base.big(::NeuronState) = NeuronState

Base.promote_rule(::Type{NeuronState}, ::Type{<:Number}) = NeuronState

Random.rand(rng::Random.AbstractRNG, ::Random.SamplerType{NeuronState}) = cold

for f in [:+, :-, :abs, :abs2, :inv, :tanh, :sqrt,
    :exp, :log, :log1p, :log2, :log10,
    :conj, :transpose, :adjoint, :angle]
    @eval Base.$f(x::NeuronState) = x
end

for f in [:+, :-, :*, :/, :^, :mod, :div, :rem, :widemul]
    @eval Base.$f(x::NeuronState, y::NeuronState) = max(x, y)
end

for f in [:+, :-, :*, :/, :^, :mod, :div, :rem, :widemul]
    # specialize to avoid conflict with Base
    @eval Base.$f(x::NeuronState, ::Integer) = x
    @eval Base.$f(::Integer, y::NeuronState) = y
    @eval Base.$f(x::NeuronState, ::Real) = x
    @eval Base.$f(::Real, y::NeuronState) = y
    @eval Base.$f(x::NeuronState, ::Number) = x
    @eval Base.$f(::Number, y::NeuronState) = y
end

Base.:*(x::NeuronState, b::Bool) = b === true ? x : cold
Base.:*(b::Bool, x::NeuronState) = *(x, b)

Base.clamp(x::NeuronState, y...) = x

"""
    fneutralize(m)

Convert the parameters of a model to `NeuronState` with value `cold`.
"""
fneutralize(m) = fmap(x -> x isa AbstractArray{<:Number} ? fill(cold, size(x)) : x, m)

end # module
