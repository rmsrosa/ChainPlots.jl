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

dr = RNN(3,5)
display(plot(dr, title="$dr", titlefontsize=12))
savefig("tests/img/dr.png")

drc = Flux.RNNCell(3,5)
display(plot(drc, title="$drc", titlefontsize=12))
savefig("tests/img/drc.png")

drlstm = Flux.LSTM(4,7)
display(plot(drlstm, title="$drlstm", titlefontsize=12))
savefig("tests/img/drlstm.png")

nn = Chain(Dense(2,5), Dense(5,7,σ), Dense(7,2,relu),Dense(2,3))
display(plot(nn, title="$nn", titlefontsize=10, xlabel="layer", xaxis=nothing))
savefig("tests/img/nn.png")

nnr = Chain(Dense(2,5,σ),RNN(5,4,relu), LSTM(4,4), Dense(4,3))
display(plot(nnr, title="$nnr", titlefontsize=10, yaxis=true))
savefig("tests/img/nnr.png")

nnrl = Chain(Dense(5,8), RNN(8,20), LSTM(20,10), Dense(10,7))
display(plot(nnrl, title="$nnrl", titlefontsize=12, yaxis=true))
savefig("tests/img/nnrl.png")

for t in themes
    theme(t)
    try        
        display(plot(nnr, title="$nnr with theme $t", titlefontsize=8, yaxis=true))
        savefig("tests/img/nnr_$t.png")
    catch err
        println("Error in chain plot with theme $t: $err")
    end
end

theme(:default)

nothing
