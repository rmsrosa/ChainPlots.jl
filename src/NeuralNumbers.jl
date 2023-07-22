module NeuralNumbers

import Random
import Base: isless, ==
import Functors: fmap

"""
    NeuralNumber <: Real

NeuralNumber encodes the "state" of a neuron as an Int8.

The possible states are:
    * `state = Int8(0)` for a "cold", or "off", state, meaning it can be triggered by a signal but it has not yet been triggered. It works as a neutral element in any diadic operation.
    * `state = Int8(1)` for a "hot", or "on", state, meaning it has been triggered by a signal.

The aliases are
    * `cold = NeuralNumber(Int8(0))`
    * `hot = NeuralNumber(Int8(1))`
"""
struct NeuralNumber <: Real
    state::Int8
end

const cold = NeuralNumber(Int8(0))
const hot = NeuralNumber(Int8(1))

Base.show(io::IO, x::NeuralNumber) = print(io, x == hot ? "hot" : "cold")
Base.show(io::IO, ::MIME"text/plain", x::NeuralNumber) = print(io, "NeuralNumber:\n  ", x)

NeuralNumber(::Number) = cold
(::Type{NeuralNumber})(x::NeuralNumber) = x
Base.convert(::Type{NeuralNumber}, y::Number) = cold
Base.convert(::Type{NeuralNumber}, y::NeuralNumber) = y

Base.float(x::NeuralNumber) = x

==(::NeuralNumber, ::Number) = false
==(::Number, ::NeuralNumber) = false
==(x::NeuralNumber, y::NeuralNumber) = x.state == y.state

isless(x::NeuralNumber, ::Number) = false
isless(::Number, x::NeuralNumber) = true
isless(x::NeuralNumber, y::NeuralNumber) = isless(x.state, y.state)

Base.:<(x::NeuralNumber, y::NeuralNumber) = x.state < y.state
Base.:≤(x::NeuralNumber, y::NeuralNumber) = x.state ≤ y.state

Base.one(::Type{NeuralNumber}) = cold
Base.zero(::Type{NeuralNumber}) = cold
Base.one(x::NeuralNumber) = x
Base.oneunit(x::NeuralNumber) = x
Base.zero(::NeuralNumber) = cold

Base.iszero(x::NeuralNumber) = x == cold
Base.isnan(::NeuralNumber) = false
Base.isfinite(::NeuralNumber) = true
Base.isinf(::NeuralNumber) = false
Base.typemin(::Type{NeuralNumber}) = cold
Base.typemax(::Type{NeuralNumber}) = hot

Base.size(::NeuralNumber) = ()
Base.size(::NeuralNumber, d::Integer) = d < 1 ? throw(BoundsError()) : 1
Base.axes(::NeuralNumber) = ()
Base.axes(::NeuralNumber, d::Integer) = d < 1 ? throw(BoundsError()) : Base.OneTo(1)
Base.eltype(::Type{NeuralNumber}) = NeuralNumber
Base.ndims(x::NeuralNumber) = 0
Base.ndims(::Type{NeuralNumber}) = 0
Base.length(x::NeuralNumber) = 1
Base.firstindex(x::NeuralNumber) = 1
Base.firstindex(::NeuralNumber, d::Int) = d < 1 ? throw(BoundsError()) : 1
Base.lastindex(x::NeuralNumber) = 1
Base.lastindex(::NeuralNumber, d::Int) = d < 1 ? throw(BoundsError()) : 1
Base.IteratorSize(::Type{NeuralNumber}) = Base.HasShape{0}()
Base.keys(::NeuralNumber) = Base.OneTo(1)

Base.getindex(x::NeuralNumber) = x

@inline Base.getindex(x::NeuralNumber, i::Integer) = @boundscheck i == 1 ? x : throw(BoundsError())
@inline Base.getindex(x::NeuralNumber, I::Integer...) = @boundscheck all(isone, I) ? x : throw(BoundsError())

Base.first(x::NeuralNumber) = x
Base.last(x::NeuralNumber) = x
Base.copy(x::NeuralNumber) = x

Base.signbit(x::NeuralNumber) = x.state < 0
Base.sign(x::NeuralNumber) = x.state

Base.iterate(x::NeuralNumber) = (x, nothing)
Base.iterate(::NeuralNumber, ::Any) = nothing
Base.isempty(x::NeuralNumber) = false
Base.in(x::NeuralNumber, y::NeuralNumber) = x == y

Base.map(f, x::NeuralNumber, ys::NeuralNumber...) = f(x, ys...)

Base.big(::NeuralNumber) = NeuralNumber

Base.promote_rule(::Type{NeuralNumber}, ::Type{<:Number}) = NeuralNumber

Random.rand(::Random.AbstractRNG, ::Random.Sampler{NeuralNumber}) = cold

for f in (:log, :exp, :sin, :cos, :tan, :sinh, :cosh, :tanh, :asin,
    :acos, :atan, :asinh, :acosh, :atanh, :cbrt, :sqrt, :log2, :log10,
    :max, :min, :exp2, :exp10, :expm1, :log1p, :sinpi, :cospi, :mod2pi,
    )
    @eval Base.$f(x::NeuralNumber) = x
end

for f in (:+, :-, :abs, :abs2, :inv, :conj, :transpose, :adjoint, :angle)
    @eval Base.$f(x::NeuralNumber) = x
end

for f in (:+, :-, :*, :/, :^, :mod, :div, :rem, :widemul)
    @eval Base.$f(x::NeuralNumber, y::NeuralNumber) = max(x, y)
end

for f in (:+, :-, :*, :/, :^, :mod, :div, :rem, :widemul)
    # specialize to avoid conflict with Base
#=     @eval Base.$f(x::NeuralNumber, ::Bool) = x
    @eval Base.$f(::Bool, y::NeuralNumber) = y =#
    @eval Base.$f(x::NeuralNumber, ::Integer) = x
    @eval Base.$f(::Integer, y::NeuralNumber) = y
    @eval Base.$f(x::NeuralNumber, ::Real) = x
    @eval Base.$f(::Real, y::NeuralNumber) = y
    @eval Base.$f(x::NeuralNumber, ::Number) = x
    @eval Base.$f(::Number, y::NeuralNumber) = y
end

Base.:*(x::NeuralNumber, b::Bool) = b === true ? x : cold
Base.:*(b::Bool, x::NeuralNumber) = *(x, b)

Base.clamp(x::NeuralNumber, y...) = x

"""
    fneutralize(m)

Convert the parameters of a model to `NeuralNumber` with value `cold`.
"""
fneutralize(m) = fmap(x -> x isa AbstractArray{<:Number} ? convert.(NeuralNumber, x) : x, m)
# do not force it to be a matrix, try with the same type
# fneutralize(m) = fmap(x -> x isa AbstractArray{<:Number} ? fill(cold, size(x)) : x, m)

end # module
