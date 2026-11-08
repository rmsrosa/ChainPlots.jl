# Changelog

## Unreleased

This release is breaking, so it will be v0.3.0.

### Breaking

- **Plots.jl is an optional dependency**, loaded as a package extension ([#11](https://github.com/rmsrosa/ChainPlots.jl/issues/11)): run `using Plots` to get the plot recipe and `plotchain`, after adding Plots to your environment. Before that, `plotchain` throws a `MethodError` with a hint to load Plots. `chaingraph` works without Plots, and `using ChainPlots` is faster. RecipesBase is no longer a dependency.

### Added

- A [Custom layers](https://rmsrosa.github.io/ChainPlots.jl/dev/custom_layers/) page in the docs, with tests: custom layers that take and return a single array work, as do `Parallel` and `SkipConnection`, and a model with several inputs can be drawn by writing it to split a single array ([#14](https://github.com/rmsrosa/ChainPlots.jl/issues/14)).

### Fixed

- `neuron_colors` sets the color of layer types that have no marker of their own, such as custom layers; they were always drawn with the `:Any` color.

## v0.2.0

This release has breaking changes. See [Migrating from 0.1 to 0.2](https://rmsrosa.github.io/ChainPlots.jl/dev/migration/) for how to update your code.

### Breaking

- **Flux 0.16 only.** ChainPlots now requires Flux 0.16 and Julia 1.10 or later, and drops support for Flux 0.12–0.14. Recurrent layers follow the Flux 0.16 API: `Recur` is gone, `RNN`, `LSTM` and `GRU` are sequence layers, and the cells `RNNCell`, `LSTMCell` and `GRUCell` return `(y, state)`.
  - `RNN`, `LSTM`, `GRU` and `GRUv3` are drawn folded, as a single time step, when the input is a vector.
  - The cells can only be drawn on their own or as the last layer of a chain; elsewhere `chaingraph` throws an `ArgumentError`.
  - The `:layer_type` of a recurrent layer is the type of the layer itself, e.g. `:RNN`, instead of that of its cell, `:RNNCell`, and its `:layer_name` is, e.g., `"RNN(5 => 4, relu)"`.
- **`chaingraph` returns a `ChainGraph`** instead of a `MetaGraphs.MetaGraph`. A `ChainGraph` is a `Graphs.AbstractGraph`, so the functions of Graphs.jl keep working. Neuron properties are read with `nodeprop(cg, v, prop)` and `nodeprops(cg, v)` instead of `get_prop` and `props`, and layer properties with `layer_dimensions(cg)` and `layer_names(cg)`. The graph backend is now internal.
- **`plot(::Flux.Chain)` is removed**, along with the plot recipes for single Flux layers, which were type piracy. Use `plotchain(m, input; kwargs...)`, or `plot(chaingraph(m, input); kwargs...)`, and `plotchain!` instead of `plot!`.

### Added

- `ChainGraph` type, with its plot recipe, and `nodeprop`, `nodeprops`, `layer_dimensions`, `layer_names` and `plotchain!`.
- Support for `GRUv3` and `GRUv3Cell`.
- `chaingraph(layer, input)` for a single layer with variable-size input, such as `Conv`.

### Fixed

- `chaingraph` works with the `gelu_erf` activation, through a package extension for SpecialFunctions.
- Plots made with current Plots.jl versions show the layer labels on the x axis again, and annotate the layers in a sans-serif font.
