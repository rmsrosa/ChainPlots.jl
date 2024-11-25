using Flux, ChainPlots, Flux.NNlib, Plots
# To concatenate layers
struct Concat{T}
    catted::T
end
Concat(xs...) = Concat(xs)

Flux.@functor Concat

function (C::Concat)(x)
    mapreduce((f, x) -> f(x), vcat, C.catted, x)
end

wdt = 16 # width of hidden layers
ϕ = Chain(
        Concat(
            Dense(1 => wdt, swish),
            Dense(1 => wdt, swish)
        ),
        Dense(2 * wdt => wdt, swish),
        Dense(wdt => 1)
)
input = [ones(1),ones(1)] .|> Vector{Float32}
ϕ(input) 
# 1-element Vector{Float32}:
# 0.05928603

chaingraph(ϕ, input)

plot(ϕ, input) 