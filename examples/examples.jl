# # Examples
#
# The aim of `ChainPlots.jl` is to provide a visualization of the topology of a neural network constructed with `Flux.Chain`. It does that by applying a `Plots.jl` recipe. The recipe is built by first constructing a `MetaGraph`.
#
# Here, we consider some examples of building both a MetaGraph and some plots.
#
# ## Load Packages
#
# First we load some packages:

using Flux
using Plots
using Random
using Colors
using Cairo
using Compose
using Graphs
using MetaGraphs
using GraphPlot

# and, of course, `ChainPlots`:
include("../src/ChainPlots.jl")

using .ChainPlots

# We play with different themes for `Plots.jl`, so we collect them here:

themes = [
    :default
    :dark
    :ggplot2
    :juno
    :lime
    :orange
    :sand
    :solarized
    :solarized_light
    :wong
    :wong2
    :gruvbox_dark
    :gruvbox_light
]

# We start with the default one:

gr()
theme(:default)

# ## Chain to MetaGraph

# We first consider a neural network with several different types of layers and building a MetaGraph from it.

# Here is the neural network:
nnr = Chain(Dense(2, 5, σ), RNN(5, 4, relu), LSTM(4, 4), GRU(4, 4), Dense(4, 3))
#
# The MetaGraph is built with the function `ChainPlots.chaingraph`:
#
mg_nnr = ChainPlots.chaingraph(nnr)
#
# We can already see the number of vertices and edges in this graph.
#
# We can also look at the first vertice for all the attributes it contains:
mg_nnr.vprops[1]

# The two attributes `locs_x` and `locs_y` give the coordinates of each vertice, while `neuron_color` gives the color based on the type of layer, whether dense, recurrent, convolutional, etc.
#
# We can get the list of those attributes for all the vertices and use them to plot the topology of the network with `GraphPlot`:

locs_x = [get_prop(mg_nnr, v, :loc_x) for v in vertices(mg_nnr)]
locs_y = [get_prop(mg_nnr, v, :loc_y) for v in vertices(mg_nnr)]
nodefillc = [parse(Colorant, get_prop(mg_nnr, v, :neuron_color)) for v in vertices(mg_nnr)] 
plt = gplot(mg_nnr, locs_x, locs_y, nodefillc=nodefillc)
#
draw(PNG("img/mg_nnr.png", 600, 400), plt)

# ## Now with Plots.jl

# It is easier, however, to use a `Plots.jl` recipe, as we show here.

# ### Single layer networks with fixed-size input
#
# For illustrative purposes, we start with some simple, single-layer networks:

dl = Dense(2, 3)
plot(dl, title="$dl", titlefontsize=12)
#-

savefig("img/dl.png")

#-

rl = RNN(3, 5)
plot(rl, title="$rl", titlefontsize=12)

#-
savefig("img/rl.png")

#

llstm = LSTM(4, 7)
plot(llstm, title="$llstm", titlefontsize=12)
#
savefig("img/llstm.png")

#

lgru = GRU(5, 7)
plot(lgru, title="$lgru", titlefontsize=12)
#
savefig("img/lgru.png")

# ## Single-layer with variable input

# Some layers accept input with varied size. In this case, we need to provide either an input, in the form of a `Vector` or `Array`, or the size of the input, in the form of a `Tuple`.

lvar = Conv((2,), 1 => 1)
plot(lvar, rand(5, 1, 1))
#
savefig("img/lvar5.png")

#
plot(lvar, (8, 1, 1))
#
savefig("img/lvar8.png")

#

nnc = Conv((3,3), 1=>2)
plot(nnc, (6, 5, 1, 1), title="$nnc", titlefontsize=10)
#
savefig("img/nnc.png")

# ## Multilayer networks

nnd = Chain(Dense(2, 5), Dense(5, 7, σ), Dense(7, 2, relu), Dense(2, 3))
plot(nnd, title="$nnd", titlefontsize=10, xaxis=nothing)
#
savefig("img/nnd.png")

#

nnr = Chain(Dense(2, 5, σ), RNN(5, 4, relu), LSTM(4, 4), GRU(4, 4), Dense(4, 3))
plot(nnr, title="$nnr", titlefontsize=7)
#
savefig("img/nnr.png")

#

x³(x) = x .^ 3
dx(x) = x[2:end] - x[1:end-1]
nna = Chain(Dense(2, 5, σ), dx, RNN(4, 6, relu), x³, LSTM(6, 4), GRU(4, 4), Dense(4, 3))
plot(nna, title="$nna", titlefontsize=7)
#
savefig("img/nna.png")

#

nnx = Chain(x³, dx, LSTM(5, 10), Dense(10, 5))
input_data = rand(6)
plot(nnx, input_data, title="$nnx", titlefontsize=9)
#
savefig("img/nnx.png")

# or just passing the dimensions:
nnx = Chain(x³, dx, LSTM(5, 10), Dense(10, 5))
plot(nnx, (6,), title="$nnx", titlefontsize=9)
#
savefig("img/nnx_ldim.png")
#

nnrlwide = Chain(Dense(5, 8), RNN(8, 20), LSTM(20, 10), Dense(10, 7))
plot(nnrlwide, title="$nnrlwide", titlefontsize=9)
#
savefig("img/nnrlwide.png")

#

reshape6x1x1(a) = reshape(a, 6, 1, 1)
nnrs = Chain(x³, Dense(3, 6), reshape6x1x1, Conv((2,), 1 => 1), vec, Dense(5, 4))
plot(nnrs, rand(Float32, 3), title="$nnrs", titlefontsize=9)
#
savefig("img/nnrs.png")

#

N = 4
reshapeNxNx1x1(a) = reshape(a, N, N, 1, 1)
nnrs2d = Chain(x³, Dense(4, N^2), reshapeNxNx1x1, Conv((2, 2), 1 => 1), vec)
plot(nnrs2d, (4,), title="$nnrs2d", titlefontsize=9)
#
savefig("img/nnrs2d.png")

#

nncg = Chain(Conv((3,3), 1=>4, leakyrelu, pad = 1),GroupNorm(4,2))
plot(nncg, (6,6,1,1), title="$nncg", titlefontsize=10)
#
savefig("img/nncg.png")

#
nncp = Chain(
    Conv((3, 3), 1=>2, pad=(1,1), bias=false),
    MaxPool((2,2)),
    Conv((3, 3), 2=>4, pad=SamePad(), relu),
    AdaptiveMaxPool((4,4)),
    Conv((3, 3), 4=>4, relu),
    GlobalMaxPool()
)
plot(nncp, (16, 16, 1, 1), title="Chain with convolutional and pooling layers", titlefontsize=10)
#
savefig("img/nncp.png")

#

hdf5()
plot(nnr, title="$nnr with HDF5", titlefontsize=7)
Plots.hdf5plot_write("img/nnrhdf5.hdf5")
gr()
plthdf5_read = Plots.hdf5plot_read("img/nnrhdf5.hdf5")

# ### Plotting with different themes

gr()
plts = []
for t in themes
    theme(t)
    try
        push!(plts, plot(nnr, title="With theme $t", titlefontsize=10, show=true))
        display(plts[end])
        ##
        savefig("img/nnr_$t.png")
    catch err
        @warn "Error in chain plot with theme $t: $err"
    end
end

# ![](img/nnr_default.png)
# ![](img/nnr_dark.png)
# ![](img/nnr_ggplot2.png)
# ![](img/nnr_juno.png)
# ![](img/nnr_lime.png)
# ![](img/nnr_orange.png)
# ![](img/nnr_sand.png)
# ![](img/nnr_solarized.png)
# ![](img/nnr_solarized_light.png)
# ![](img/nnr_wong.png)
# ![](img/nnr_wong2.png)
# ![](img/nnr_gruvbox_dark.png)
# ![](img/nnr_gruvbox_light.png)

# Back to `:default` theme.

theme(:default)

# ### Plotting with extra arguments

# One is free to use (most of) other arguments accepted by `Plots.jl`. For instance, the following plot is used for the repo's social preview:

plot(nnr, title="ChainPlots.jl", topmargin=6mm, bottommargin=4mm, size=(720,380), titlefont=(42, "Bookman Demi"))

#

savefig("img/nnr_chainplots_socialpreview.png")

# Some attributes, like `annotationfontsize` and `markersize`, however, which are explicitly set in the Plots, recipe cannot be currently changed, but there are plans to make this more flexible.

# ---

# *This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl), with `Literate.markdown("examples.jl", "build", execute = true, flavor = Literate.CommonMarkFlavor(), postprocess = s -> replace(s, r"(?!\")!\[\]\(img\/" => "![](../img/"), credit=false)`.*
