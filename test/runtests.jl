# Usually, running tests work with either
# 1) inside VSCode with ⌘⇧T (command+shift+T) (but plots will show up externally)
# 2) in shell=>Julia with `]test` (which work the same as above)
# 3) by launching the VSCode REPL and then `]test` (but now no plots are displayed)
#
# To overcome the plotting issue and have the plots displayed in the plots panel:
# - I added all the dependencies for the Package in the Package.toml of the test package
# - Replaced `using ChainPlot` with `include("../src/ChainPlot.jl")`
# - Open up `runtests.jl` in VSCode
# - "Julia: Change to This Directory"
# - "Julia: Activate This Environment" (dismiss popup warning if it appears)
# - "Julia: Execute File in REPL"
# voilá!

# using ChainPlot
using Flux
using Plots

include("../src/ChainPlot.jl")

gr()
theme(:default)

themes = [
    :default
    :dark
    :ggplot2 # somehow throwing error
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

dl = Dense(2,3)
display(plot(dl, title="$dl", titlefontsize=12))
savefig("img/dl.png")

rl = RNN(3,5)
display(plot(rl, title="$rl", titlefontsize=12))
savefig("img/rl.png")

llstm = LSTM(4,7)
display(plot(llstm, title="$llstm", titlefontsize=12))
savefig("img/llstm.png")

lgru = GRU(5,7)
display(plot(lgru, title="$lgru", titlefontsize=12))
savefig("img/lgru.png")

nnd = Chain(Dense(2,5), Dense(5,7,σ), Dense(7,2,relu),Dense(2,3))
display(plot(nnd, title="$nnd", titlefontsize=10, xaxis=nothing))
savefig("img/nnd.png")

nnr = Chain(Dense(2,5,σ), RNN(5,4,relu), LSTM(4,4), GRU(4,4), Dense(4,3))
display(plot(nnr, title="$nnr", titlefontsize=7))
savefig("img/nnr.png")

dx(x) = x[2:end]-x[1:end-1]
x³(x) = x.^3
nna = Chain(Dense(2,5,σ), dx, RNN(4,6,relu), x³, LSTM(6,4), GRU(4,4), Dense(4,3))
display(plot(nna, title="$nna", titlefontsize=7))
savefig("img/nna.png")

nnx = Chain(x³, dx, LSTM(5,10), Dense(10,5))
input_data = rand(6)
display(plot(nnx, input_data, title="$nnx", titlefontsize=9))
savefig("img/nnx.png")

nnrlwide = Chain(Dense(5,8), RNN(8,20), LSTM(20,10), Dense(10,7))
display(plot(nnrlwide, title="$nnrlwide", titlefontsize=9))
savefig("img/nnrlwide.png")

reshape6x1x1(a) = reshape(a, 6,  1, 1)
slice(a) = a[:,1,1]
nnrs = Chain(x³, Dense(3,6), reshape6x1x1, Conv((2,), 1=>1), slice, Dense(5,4))
display(plot(nnrs, Float32.(rand(3)), title="$nnrs", titlefontsize=9))
savefig("img/nnrs.png")

reshape3x3x1x1(a) = reshape(a, 3, 3, 1, 1)
nnrs2d = Chain(x³, Dense(4,9), reshape3x3x1x1, Conv((2,2), 1=>1), slice)
display(plot(nnrs2d, Float32.(rand(4)), title="$nnrs2d", titlefontsize=9))
savefig("img/nnrs2d.png")

#= nncg = Chain(Conv((3,3), 1=>8, leakyrelu, pad = 1),GroupNorm(8,4))
display(plot(nncg, Float32.(rand(6,6,1,1)), title="$nncg", titlefontsize=10))
savefig("img/nncg.png") =#

hdf5()
plot(nnr, title="$nnr with HDF5", titlefontsize=7)
Plots.hdf5plot_write("img/nnrhdf5.hdf5")
gr()
plthdf5_read = Plots.hdf5plot_read("img/nnrhdf5.hdf5")
display(plthdf5_read)

gr()
for t in themes
    theme(t)
    try        
        display(plot(nnr, title="With theme $t", titlefontsize=10))
        savefig("img/nnr_$t.png")
    catch err
        println("Error in chain plot with theme $t: $err")
    end
end

theme(:default)

nothing
