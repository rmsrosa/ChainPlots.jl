# Migrating

## From 0.2 to 0.3

Plots.jl is now an optional dependency, and the plot recipe and [`plotchain`](@ref) are loaded with it, as a package extension. So add Plots to your environment and run `using Plots` before plotting; otherwise `plotchain` throws an error that says so. Since 0.3.1, Makie.jl can be used instead of Plots.jl; see [Plotting with Makie](@ref). [`chaingraph`](@ref) works without Plots.

## From 0.1 to 0.2

ChainPlots 0.2 supports Flux 0.16 only and changes what [`chaingraph`](@ref) returns.

### `chaingraph` returns a `ChainGraph`

[`chaingraph`](@ref) used to return a `MetaGraphs.MetaGraph`. It now returns a [`ChainGraph`](@ref), which keeps the graph backend private, so that it can change without breaking your code. A `ChainGraph` is a `Graphs.AbstractGraph`, so the functions of `Graphs.jl` still work on it.

| 0.1 | 0.2 |
| :-- | :-- |
| `get_prop(mg, v, :loc_x)` | `nodeprop(cg, v, :loc_x)` |
| `get_prop.(Ref(mg), vertices(mg), :loc_x)` | `nodeprop.(Ref(cg), vertices(cg), :loc_x)` |
| `props(mg, v)`, `mg.vprops[v]` | `nodeprops(cg, v)`, a `NamedTuple` |
| `nv(mg)`, `ne(mg)`, `neighbors(mg, v)`, … | unchanged: `nv(cg)`, `ne(cg)`, `neighbors(cg, v)`, … |

The neuron properties are the same as before: `:layer_number`, `:layer_name`, `:layer_type`, `:index_in_layer`, `:layer_center`, `:loc_x`, `:loc_y` and `:neuron_color`.

### Plotting

The `Plots.jl` recipes for `Flux.Chain` and for single Flux layers were removed, since defining them was type piracy: ChainPlots owns neither `plot` nor the Flux types. Plot the `ChainGraph` instead, or use [`plotchain`](@ref):

| 0.1 | 0.2 |
| :-- | :-- |
| `plot(m)` | `plotchain(m)` or `plot(chaingraph(m))` |
| `plot(m, input; kwargs...)` | `plotchain(m, input; kwargs...)` or `plot(chaingraph(m, input); kwargs...)` |
| `plot!(m, ...)` | `plotchain!(m, ...)` or `plot!(chaingraph(m, ...))` |

### Flux 0.16

Flux 0.15 removed `Recur` and turned `RNN`, `LSTM` and `GRU` into layers that process a whole sequence, and Flux 0.16 made the recurrent cells return `(y, state)`. As a result:

* the layers are built with `in => out`, e.g. `RNN(5 => 4, relu)` instead of `RNN(5, 4, relu)`;
* `:layer_type` is the type of the layer itself, e.g. `:RNN`, `:LSTM` or `:GRU`, instead of the type of its cell, `:RNNCell`, `:LSTMCell` or `:GRUCell`, and `:layer_name` is, e.g., `"RNN(5 => 4, relu)"` instead of `"Recur(RNNCell(5 => 4, relu))"`;
* with a vector input, `RNN`, `LSTM` and `GRU` are drawn folded, as a single time step, as before; with an `(in, len)` sequence as input, the graph has a neuron for each time step, but the plot does not lay the time steps out separately yet;
* the cells `RNNCell`, `LSTMCell` and `GRUCell` can only be drawn on their own or as the last layer of a chain.
