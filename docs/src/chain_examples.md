# Chain Plots

Here, we consider some examples of building both a MetaGraph and some plots.

First we load the necessary packages:

```@example chainplots

using Flux
using Plots
using ChainPlots
```

```@example chainplots
nnr = Chain(Dense(2, 5, σ), RNN(5, 4, relu), LSTM(4, 4), GRU(4, 4), Dense(4, 3))

plot(nnr)
```

## Single layer networks with fixed-size input


For illustrative purposes, we start with some simple, single-layer networks:

```@example chainplots
dl = Dense(2, 3)
plot(dl, title="$dl", titlefontsize=12)
```

```@example chainplots
rl = RNN(3, 5)
plot(rl, title="$rl", titlefontsize=12)
```

```@example chainplots
llstm = LSTM(4, 7)
plot(llstm, title="$llstm", titlefontsize=12)
```

```@example chainplots
lgru = GRU(5, 7)
plot(lgru, title="$lgru", titlefontsize=12)
```

## Single-layer with variable input

Some layers accept input with varied size. In this case, we need to provide either an input, in the form of a `Vector` or `Array`, or the size of the input, in the form of a `Tuple`.

```@example chainplots
lvar = Conv((2,), 1 => 1)
plot(lvar, rand(5, 1, 1))
```

```@example chainplots
plot(lvar, (8, 1, 1))
```

```@example chainplots
nnc = Conv((3,3), 1=>2)
plot(nnc, (6, 5, 1, 1), title="$nnc", titlefontsize=10)
```

## Multilayer networks

```@example chainplots
nnd = Chain(Dense(2, 5), Dense(5, 7, σ), Dense(7, 2, relu), Dense(2, 3))
plot(nnd, title="$nnd", titlefontsize=10, xaxis=nothing)
```

```@example chainplots
nnr = Chain(Dense(2, 5, σ), RNN(5, 4, relu), LSTM(4, 4), GRU(4, 4), Dense(4, 3))
plot(nnr, title="$nnr", titlefontsize=7)
```

```@example chainplots
x³(x) = x .^ 3
dx(x) = x[2:end] - x[1:end-1]
nna = Chain(Dense(2, 5, σ), dx, RNN(4, 6, relu), x³, LSTM(6, 4), GRU(4, 4), Dense(4, 3))
plot(nna, title="$nna", titlefontsize=7)
```

```@example chainplots
nnx = Chain(x³, dx, LSTM(5, 10), Dense(10, 5))
input_data = rand(6)
plot(nnx, input_data, title="$nnx", titlefontsize=9)
```

or just passing the dimensions:

```@example chainplots
nnx = Chain(x³, dx, LSTM(5, 10), Dense(10, 5))
plot(nnx, (6,), title="$nnx", titlefontsize=9)
```

```@example chainplots
nnrlwide = Chain(Dense(5, 8), RNN(8, 20), LSTM(20, 10), Dense(10, 7))
plot(nnrlwide, title="$nnrlwide", titlefontsize=9)
```

```@example chainplots
reshape6x1x1(a) = reshape(a, 6, 1, 1)
nnrs = Chain(x³, Dense(3, 6), reshape6x1x1, Conv((2,), 1 => 1), vec, Dense(5, 4))
plot(nnrs, rand(Float32, 3), title="$nnrs", titlefontsize=9)
```

```@example chainplots
N = 4
reshapeNxNx1x1(a) = reshape(a, N, N, 1, 1)
nnrs2d = Chain(x³, Dense(4, N^2), reshapeNxNx1x1, Conv((2, 2), 1 => 1), vec)
plot(nnrs2d, (4,), title="$nnrs2d", titlefontsize=9)
```

```@example chainplots
nncg = Chain(Conv((3,3), 1=>4, leakyrelu, pad = 1),GroupNorm(4,2))
plot(nncg, (6,6,1,1), title="$nncg", titlefontsize=10)
```

```@example chainplots
nncp = Chain(
    Conv((3, 3), 1=>2, pad=(1,1), bias=false),
    MaxPool((2,2)),
    Conv((3, 3), 2=>4, pad=SamePad(), relu),
    AdaptiveMaxPool((4,4)),
    Conv((3, 3), 4=>4, relu),
    GlobalMaxPool()
)
plot(nncp, (16, 16, 1, 1), title="Chain with convolutional and pooling layers", titlefontsize=10)
```
