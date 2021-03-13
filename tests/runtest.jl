using ChainPlot
using Flux
using Plots

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
savefig("tests/img/dl.png")

rl = RNN(3,5)
display(plot(rl, title="$rl", titlefontsize=12))
savefig("tests/img/rl.png")

llstm = LSTM(4,7)
display(plot(llstm, title="$llstm", titlefontsize=12))
savefig("tests/img/llstm.png")

lgru = GRU(5,7)
display(plot(lgru, title="$lgru", titlefontsize=12))
savefig("tests/img/lgru.png")

nnd = Chain(Dense(2,5), Dense(5,7,σ), Dense(7,2,relu),Dense(2,3))
display(plot(nnd, title="$nnd", titlefontsize=10, xaxis=nothing))
savefig("tests/img/nnd.png")

nnr = Chain(Dense(2,5,σ), RNN(5,4,relu), LSTM(4,4), GRU(4,4), Dense(4,3))
display(plot(nnr, title="$nnr", titlefontsize=7))
savefig("tests/img/nnr.png")

nnrlwide = Chain(Dense(5,8), RNN(8,20), LSTM(20,10), Dense(10,7))
display(plot(nnrlwide, title="$nnrlwide", titlefontsize=9))
savefig("tests/img/nnrlwide.png")

for t in themes
    theme(t)
    try        
        display(plot(nnr, title="With theme $t", titlefontsize=10))
        savefig("tests/img/nnr_$t.png")
    catch err
        println("Error in chain plot with theme $t: $err")
    end
end

theme(:default)

nothing
