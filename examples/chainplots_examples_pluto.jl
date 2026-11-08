### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# ╔═╡ 43e0f6e0-64bc-5fbb-a7e2-748873396dce
begin
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.develop(path = dirname(@__DIR__))
    Pkg.add(["Flux", "Plots", "Graphs", "GraphPlot", "Colors"])
    using Flux, Plots, Graphs, GraphPlot, Colors, ChainPlots
end

# ╔═╡ 3ebb158c-7517-5c94-8e84-269d1165a895
md"""
# ChainPlots.jl examples

This notebook shows how [ChainPlots.jl](https://github.com/rmsrosa/ChainPlots.jl) visualizes the topology of neural networks built with `Flux.Chain`: it builds a `ChainGraph` of the network, with `chaingraph`, and plots it with its `Plots.jl` recipe, or does both at once with `plotchain`.

It follows [`examples/examples.jl`](https://github.com/rmsrosa/ChainPlots.jl/blob/main/examples/examples.jl), and was inspired by the Pluto notebook contributed in [issue #9](https://github.com/rmsrosa/ChainPlots.jl/issues/9).
"""

# ╔═╡ e0bbc3d3-4604-5961-a0a0-562b10886652
md"""
## Packages

The first cell uses the ChainPlots.jl checkout this notebook lives in, so that the notebook always matches the code next to it. To use a registered version instead, replace `Pkg.develop(...)` with `Pkg.add("ChainPlots")`.
"""

# ╔═╡ f5cab2cb-e3c6-579d-941a-3218ca63d28c
md"""
## From a Chain to a ChainGraph

Here is a neural network with several types of layers:
"""

# ╔═╡ 9b0bffb3-8d3d-5524-80b4-07b728b34087
nnr = Chain(Dense(2 => 5, σ), RNN(5 => 4, relu), LSTM(4 => 4), GRU(4 => 4), Dense(4 => 3))

# ╔═╡ 133c5932-66ca-5c39-b62a-007f2ba88644
md"""
Its `ChainGraph` has a vertex for each neuron and an edge for each connection between neurons of consecutive layers:
"""

# ╔═╡ 30a28c03-1bad-5b9c-93cb-1c59ec4a4ab5
cg = chaingraph(nnr)

# ╔═╡ 94535b41-0439-5ad5-84c4-36467bfa264e
md"""
A `ChainGraph` is a `Graphs.AbstractGraph`, so the functions of `Graphs.jl` work on it:
"""

# ╔═╡ 5a653bf5-3533-5d9a-8b02-90be226ef9a6
nv(cg), ne(cg), neighbors(cg, 1)

# ╔═╡ 2289e712-5f6b-5a98-ba45-ad26e378c109
md"""
The properties of a neuron are given by `nodeprops`, and one property at a time by `nodeprop`:
"""

# ╔═╡ e5a54416-efb5-503c-8710-3e4ff9b832f1
nodeprops(cg, 8)

# ╔═╡ 96ddd90e-dc98-5233-ab55-18d191bad128
layer_dimensions(cg), layer_names(cg)

# ╔═╡ 5b6c0f29-3ce0-5641-b56c-9f3afc5a8c46
md"""
The properties `loc_x`, `loc_y` and `neuron_color` are enough to draw the network with `GraphPlot.jl`:
"""

# ╔═╡ 1c5ed885-207a-50e9-a14e-625c2e63fd24
gplot(
    cg,
    nodeprop.(Ref(cg), vertices(cg), :loc_x),
    nodeprop.(Ref(cg), vertices(cg), :loc_y),
    nodefillc = [parse(Colorant, nodeprop(cg, v, :neuron_color)) for v in vertices(cg)]
)

# ╔═╡ df75117c-f698-5692-b82a-8bd6da323729
md"""
## Plotting with Plots.jl

It is easier, however, to use the `Plots.jl` recipe for `ChainGraph`:
"""

# ╔═╡ 3c84598d-fcf0-5dab-bfca-fcfc0e5ad1ec
plot(cg, title = "$nnr", titlefontsize = 7)

# ╔═╡ 889b76dd-42a4-57d9-af75-cdef4ce1e65d
md"""
or, in one step, `plotchain(m)`, which is the same as `plot(chaingraph(m))`.

### Single layers with fixed-size input

The recurrent layers are drawn folded, as a single time step, with a marker that hints at the recurrence.
"""

# ╔═╡ 2b5e1458-9c16-5d9a-b56e-8b8e7e4da29d
plotchain(Dense(2 => 3), title = "Dense(2 => 3)", titlefontsize = 12)

# ╔═╡ 81bd16d7-48ba-5f32-bbd2-7c048b271d59
plotchain(RNN(3 => 5), title = "RNN(3 => 5)", titlefontsize = 12)

# ╔═╡ 1b944de8-836f-56fa-adb5-378aaf8daf7a
plotchain(LSTM(4 => 7), title = "LSTM(4 => 7)", titlefontsize = 12)

# ╔═╡ e8f6e766-9053-5800-844e-dae4a5d6c661
plotchain(GRU(5 => 7), title = "GRU(5 => 7)", titlefontsize = 12)

# ╔═╡ 66a50400-207f-5f11-8576-6ca39c8f58e0
md"""
### Single layers with variable-size input

Some layers accept inputs of any size. In this case, pass either an input, as an `Array`, or the shape of an input, as a `Tuple`:
"""

# ╔═╡ 58622162-0c7b-5628-8598-08f1367b4055
plotchain(Conv((2,), 1 => 1), rand(Float32, 5, 1, 1))

# ╔═╡ 4984c4fc-5319-53d9-8988-9e2e1ff2013e
plotchain(Conv((2,), 1 => 1), (8, 1, 1))

# ╔═╡ b3096e05-e9b5-5bba-959b-580c8367d0a4
plotchain(Conv((3, 3), 1 => 2), (6, 5, 1, 1), title = "Conv((3, 3), 1 => 2)", titlefontsize = 10)

# ╔═╡ 5147553a-87f1-52f3-ac5d-b0bf2e9a2fcb
md"""
## Multilayer networks
"""

# ╔═╡ 2e817b3d-68c3-522f-b753-11f896e47bd5
nnd = Chain(Dense(2 => 5), Dense(5 => 7, σ), Dense(7 => 2, relu), Dense(2 => 3))

# ╔═╡ 1bd677af-2ce3-5214-a500-d36910981746
plotchain(nnd, title = "$nnd", titlefontsize = 10)

# ╔═╡ c0862503-a0cb-5de1-97bb-e374f22918a2
md"""
Functional layers, such as plain Julia functions, are accepted as well:
"""

# ╔═╡ a4f37328-424f-5d7e-87c0-4bc0da930489
x³(x) = x .^ 3

# ╔═╡ a2dd4d5f-8942-57e6-968c-d5bf75bf96a2
dx(x) = x[2:end] - x[1:end-1]

# ╔═╡ 34284a33-3714-5847-a4a5-b0f8047e8525
nna = Chain(Dense(2 => 5, σ), dx, RNN(4 => 6, relu), x³, LSTM(6 => 4), GRU(4 => 4), Dense(4 => 3))

# ╔═╡ f2300786-9648-5d3a-b9a6-93aeeaeec30e
plotchain(nna, title = "$nna", titlefontsize = 7)

# ╔═╡ a9c56689-54dd-57d7-b5d3-7ee11e5a73d4
md"""
If the first layer accepts inputs of any size, an input, or the shape of an input, is needed:
"""

# ╔═╡ 900660bc-d02b-59e3-aebc-8580b385978e
nnx = Chain(x³, dx, LSTM(5 => 10), Dense(10 => 5))

# ╔═╡ d676f5c3-1441-5a2c-97a0-8ebbc4b29770
plotchain(nnx, (6,), title = "$nnx", titlefontsize = 9)

# ╔═╡ e66ad52f-6a76-5ee9-b621-de843dd84234
nnrlwide = Chain(Dense(5 => 8), RNN(8 => 20), LSTM(20 => 10), Dense(10 => 7))

# ╔═╡ 60609da6-2c90-513a-86da-0ed4eca3731f
plotchain(nnrlwide, title = "$nnrlwide", titlefontsize = 9)

# ╔═╡ 02f39f81-7dd8-51bf-a184-b6e80234074a
md"""
## Convolutional networks

A network with a one-dimensional convolution:
"""

# ╔═╡ 1fc6de8a-6c32-5b58-962d-b4e87f1d0fd7
reshape6x1x1(a) = reshape(a, 6, 1, 1)

# ╔═╡ 38e5fe8f-bd76-544f-86da-c6d6175abccf
nnrs = Chain(x³, Dense(3 => 6), reshape6x1x1, Conv((2,), 1 => 1), vec, Dense(5 => 4))

# ╔═╡ fce68744-647d-524b-92de-6b71462a4d08
plotchain(nnrs, (3,), title = "$nnrs", titlefontsize = 9)

# ╔═╡ c2a79c06-5c96-5bb4-8245-428b990aac96
md"""
Now with a two-dimensional convolution:
"""

# ╔═╡ 758ec6bc-a6fc-5ed6-87d1-c02b5b65bc98
reshape4x4x1x1(a) = reshape(a, 4, 4, 1, 1)

# ╔═╡ beac1806-9ec3-5e38-8026-398c4861000e
nnrs2d = Chain(x³, Dense(4 => 16), reshape4x4x1x1, Conv((2, 2), 1 => 1), vec)

# ╔═╡ 36cee6f0-58c8-5000-a4eb-2735d46cec64
plotchain(nnrs2d, (4,), title = "$nnrs2d", titlefontsize = 9)

# ╔═╡ 345d6135-a2f4-5123-98e6-03a5c5b9a80f
nncg = Chain(Conv((3, 3), 1 => 4, leakyrelu, pad = 1), GroupNorm(4, 2))

# ╔═╡ 11a303aa-cf88-5b28-b0cd-787379ddbd45
plotchain(nncg, (6, 6, 1, 1), title = "$nncg", titlefontsize = 10)

# ╔═╡ 8387b699-9320-5b4b-990f-d6440bf689cb
md"""
With convolutional and pooling layers:
"""

# ╔═╡ 645819d4-76e9-53c0-ac8a-c690096ead4d
nncp = Chain(
    Conv((3, 3), 1 => 2, pad = (1, 1), bias = false),
    MaxPool((2, 2)),
    Conv((3, 3), 2 => 4, pad = SamePad(), relu),
    AdaptiveMaxPool((4, 4)),
    Conv((3, 3), 4 => 4, relu),
    GlobalMaxPool()
)

# ╔═╡ 6129efa3-2ef3-547a-ad8d-49a92b2e06cf
plotchain(nncp, (16, 16, 1, 1), title = "Chain with convolutional and pooling layers", titlefontsize = 10)

# ╔═╡ 46066772-e34f-5cf4-a6c3-8297ad5fda78
md"""
## Plot attributes and themes

Other `Plots.jl` attributes are accepted, as well as the attributes of the recipe, `connection_color`, `neuron_colors` and `neuron_shape`:
"""

# ╔═╡ a2fe4003-03b1-5f3f-becb-6df03d48be29
plotchain(nnr, title = "ChainPlots.jl", titlefontsize = 24, connection_color = :lightblue, size = (720, 380))

# ╔═╡ 5f5b3065-7d2a-582d-a005-6aa5a8132ea8
md"""
The `Plots.jl` themes work too. Pick one of `:default`, `:dark`, `:ggplot2`, `:juno`, `:lime`, `:orange`, `:sand`, `:solarized`, `:solarized_light`, `:wong`, `:wong2`, `:gruvbox_dark` or `:gruvbox_light`:
"""

# ╔═╡ 533e7504-aa31-5f03-b7a3-2bf5ea86fdca
plot_theme = :dark

# ╔═╡ 27c2a134-743f-5454-ad82-c38cc7bcc110
let
    theme(plot_theme)
    plt = plotchain(nnr, title = "With theme $plot_theme", titlefontsize = 10)
    theme(:default)
    plt
end

# ╔═╡ Cell order:
# ╟─3ebb158c-7517-5c94-8e84-269d1165a895
# ╟─e0bbc3d3-4604-5961-a0a0-562b10886652
# ╠═43e0f6e0-64bc-5fbb-a7e2-748873396dce
# ╟─f5cab2cb-e3c6-579d-941a-3218ca63d28c
# ╠═9b0bffb3-8d3d-5524-80b4-07b728b34087
# ╟─133c5932-66ca-5c39-b62a-007f2ba88644
# ╠═30a28c03-1bad-5b9c-93cb-1c59ec4a4ab5
# ╟─94535b41-0439-5ad5-84c4-36467bfa264e
# ╠═5a653bf5-3533-5d9a-8b02-90be226ef9a6
# ╟─2289e712-5f6b-5a98-ba45-ad26e378c109
# ╠═e5a54416-efb5-503c-8710-3e4ff9b832f1
# ╠═96ddd90e-dc98-5233-ab55-18d191bad128
# ╟─5b6c0f29-3ce0-5641-b56c-9f3afc5a8c46
# ╠═1c5ed885-207a-50e9-a14e-625c2e63fd24
# ╟─df75117c-f698-5692-b82a-8bd6da323729
# ╠═3c84598d-fcf0-5dab-bfca-fcfc0e5ad1ec
# ╟─889b76dd-42a4-57d9-af75-cdef4ce1e65d
# ╠═2b5e1458-9c16-5d9a-b56e-8b8e7e4da29d
# ╠═81bd16d7-48ba-5f32-bbd2-7c048b271d59
# ╠═1b944de8-836f-56fa-adb5-378aaf8daf7a
# ╠═e8f6e766-9053-5800-844e-dae4a5d6c661
# ╟─66a50400-207f-5f11-8576-6ca39c8f58e0
# ╠═58622162-0c7b-5628-8598-08f1367b4055
# ╠═4984c4fc-5319-53d9-8988-9e2e1ff2013e
# ╠═b3096e05-e9b5-5bba-959b-580c8367d0a4
# ╟─5147553a-87f1-52f3-ac5d-b0bf2e9a2fcb
# ╠═2e817b3d-68c3-522f-b753-11f896e47bd5
# ╠═1bd677af-2ce3-5214-a500-d36910981746
# ╟─c0862503-a0cb-5de1-97bb-e374f22918a2
# ╠═a4f37328-424f-5d7e-87c0-4bc0da930489
# ╠═a2dd4d5f-8942-57e6-968c-d5bf75bf96a2
# ╠═34284a33-3714-5847-a4a5-b0f8047e8525
# ╠═f2300786-9648-5d3a-b9a6-93aeeaeec30e
# ╟─a9c56689-54dd-57d7-b5d3-7ee11e5a73d4
# ╠═900660bc-d02b-59e3-aebc-8580b385978e
# ╠═d676f5c3-1441-5a2c-97a0-8ebbc4b29770
# ╠═e66ad52f-6a76-5ee9-b621-de843dd84234
# ╠═60609da6-2c90-513a-86da-0ed4eca3731f
# ╟─02f39f81-7dd8-51bf-a184-b6e80234074a
# ╠═1fc6de8a-6c32-5b58-962d-b4e87f1d0fd7
# ╠═38e5fe8f-bd76-544f-86da-c6d6175abccf
# ╠═fce68744-647d-524b-92de-6b71462a4d08
# ╟─c2a79c06-5c96-5bb4-8245-428b990aac96
# ╠═758ec6bc-a6fc-5ed6-87d1-c02b5b65bc98
# ╠═beac1806-9ec3-5e38-8026-398c4861000e
# ╠═36cee6f0-58c8-5000-a4eb-2735d46cec64
# ╠═345d6135-a2f4-5123-98e6-03a5c5b9a80f
# ╠═11a303aa-cf88-5b28-b0cd-787379ddbd45
# ╟─8387b699-9320-5b4b-990f-d6440bf689cb
# ╠═645819d4-76e9-53c0-ac8a-c690096ead4d
# ╠═6129efa3-2ef3-547a-ad8d-49a92b2e06cf
# ╟─46066772-e34f-5cf4-a6c3-8297ad5fda78
# ╠═a2fe4003-03b1-5f3f-becb-6df03d48be29
# ╟─5f5b3065-7d2a-582d-a005-6aa5a8132ea8
# ╠═533e7504-aa31-5f03-b7a3-2bf5ea86fdca
# ╠═27c2a134-743f-5454-ad82-c38cc7bcc110
