using Flux
using Graphs
using MetaGraphs
using Random
using Test

using ChainPlot
include("../src/NeuronNumbers.jl")
using .NeuronNumbers

#= 
@testset "nnr" begin
    nnr = Chain(Dense(2, 5, σ), RNN(5, 4, relu), LSTM(4, 4), GRU(4, 4), Dense(4, 3))
    mg = ChainPlot.chaingraph(nnr)
    @test nv(mg) == 2 + 5 + 4 + 4 + 4 + 3 == 22
    @test ne(mg) == 2 * 5 + 5 * 4 + 4 * 4 + 4 * 4 + 4 * 3 == 74
    @test all(==("input layer"), get_prop.(Ref(mg), 1:2, :layer_type))
    @test all(==("Dense(2, 5, σ)"), get_prop.(Ref(mg), 3:7, :layer_type))
    @test all(==("Recur(RNNCell(5, 4, relu))"), get_prop.(Ref(mg), 8:11, :layer_type))
    @test all(==("Recur(LSTMCell(4, 4))"), get_prop.(Ref(mg), 12:15, :layer_type))
    @test all(==("Recur(GRUCell(4, 4))"), get_prop.(Ref(mg), 16:19, :layer_type))
    @test all(==("Dense(4, 3)"), get_prop.(Ref(mg), 20:22, :layer_type))
    @test get_prop(mg, 1, :index_in_layer) == (1,)
    @test get_prop(mg, 2, :index_in_layer) == (2,)
    @test get_prop(mg, 3, :index_in_layer) == (1,)
    @test get_prop(mg, 7, :index_in_layer) == (5,)
    @test get_prop(mg, 3, :neuron_color) == ChainPlot.neuron_color(Dense(2, 5, σ))
    @test get_prop(mg, 8, :neuron_color) == ChainPlot.neuron_color(RNN(5, 4, relu))
    @test neighbors(mg, 1) == neighbors(mg, 2) == (3:7)
    @test all(==([(1:2); (8:11)]), neighbors.(Ref(mg), 3:7))
    @test all(==([(3:7); (12:15)]), neighbors.(Ref(mg), 8:11))
    @test all(==([(8:11); (16:19)]), neighbors.(Ref(mg), 12:15))
    @test all(==([(12:15); (20:22)]), neighbors.(Ref(mg), 16:19))
    @test all(==((16:19)), neighbors.(Ref(mg), 20:22))
    @test length(get_prop(mg, v, :loc_x) for v in vertices(mg)) == 22
    @test length(get_prop(mg, v, :loc_y) for v in vertices(mg)) == 22
    @test unique(get_prop(mg, v, :loc_x) for v in vertices(mg)) == (0.0:length(nnr))
end

@testset "specific neurons" begin
    m = Chain(Dense(2, 3), RNN(3, 2))
    fm = fcooloffneurons(m)
    @test fm([coldneuron, hotneuron]) == [hotneuron, hotneuron]

    m = Chain(x -> x[2:end] - x[1:end-1])
    fm = fcooloffneurons(m)
    inp = [coldneuron, hotneuron, coldneuron, coldneuron, coldneuron]
    @test fm(inp) == [hotneuron, hotneuron, coldneuron, coldneuron]

    m = Chain(Conv((2,), 1 => 1))
    inp = [hotneuron; coldneuron; coldneuron; coldneuron; coldneuron;;;]
    @test m(inp) == [hotneuron; coldneuron; coldneuron; coldneuron;;;]
    inp = [coldneuron; hotneuron; coldneuron; coldneuron; coldneuron;;;]
    @test m(inp) == [hotneuron; hotneuron; coldneuron; coldneuron;;;]
    inp = [coldneuron; coldneuron; hotneuron; coldneuron; coldneuron;;;]
    @test m(inp) == [coldneuron; hotneuron; hotneuron; coldneuron;;;]
    inp = [coldneuron; coldneuron; coldneuron; hotneuron; coldneuron;;;]
    @test m(inp) == [coldneuron; coldneuron; hotneuron; hotneuron;;;]
    inp = [coldneuron; coldneuron; coldneuron; coldneuron; hotneuron;;;]
    @test m(inp) == [coldneuron; coldneuron; coldneuron; hotneuron;;;]
end

@testset "layers" begin
    for (l, ni, no) in (
        (:Dense, 2, 3),
        (:RNN, 3, 5),
        (:LSTM, 7, 4),
        (:GRU, 8, 5)
    )
        @testset "$l" begin
            @eval m = Flux.$l($ni, $no)
            mg = ChainPlot.chaingraph(m)
            @test nv(mg) == ni + no
            @test ne(mg) == ni * no
            @test get_prop(mg, 1, :layer_type) == get_prop(mg, ni, :layer_type) == "input layer"
            @test occursin("$l", get_prop(mg, ni + 1, :layer_type))
            @test occursin("$l", get_prop(mg, ni + no, :layer_type))
            @test get_prop(mg, 1, :layer_number) == 0
            @test get_prop(mg, ni + 1, :layer_number) == 1
            @test get_prop(mg, 1, :layer_center) == ni / 2
            @test get_prop(mg, ni + 1, :layer_center) == no / 2
            @test get_prop(mg, 1, :index_in_layer) == get_prop(mg, ni + 1, :index_in_layer) == (1,)
            @test get_prop(mg, ni, :index_in_layer) == (ni,)
            @test get_prop(mg, ni + no, :index_in_layer) == (no,)
            @test neighbors(mg, 1) == neighbors(mg, ni) == (ni+1:ni+no)
            @test neighbors(mg, ni + 1) == neighbors(mg, ni + no) == (1:ni)
            @test length(get_prop.(Ref(mg), vertices(mg), :loc_x)) == length(get_prop.(Ref(mg), vertices(mg), :loc_y)) == ni + no
            @test unique(get_prop(mg, v, :loc_x) for v in vertices(mg)) == (0.0:1.0)
        end
    end
end

@testset "chains" begin
    for (m, num_vert, num_edges) in (
        (Chain(Dense(2, 5), Dense(5, 7, σ), Dense(7, 2, relu), Dense(2, 3)), 19, 65),
        (Chain(Dense(2, 5, σ), RNN(5, 4, relu), LSTM(4, 4), GRU(4, 4), Dense(4, 3)), 22, 74),
        (Chain(Dense(5, 8), RNN(8, 20), LSTM(20, 10), Dense(10, 7)), 50, 470)

    )
        mg = ChainPlot.chaingraph(m)
        @test nv(mg) == num_vert
        @test ne(mg) == num_edges
    end
end

@testset "functional" begin
    x³(x) = x .^ 3
    dx(x) = x[2:end] - x[1:end-1]
    nna = Chain(Dense(2, 5, σ), dx, RNN(4, 6, relu), x³, LSTM(6, 4), GRU(4, 4), Dense(4, 3))
    mg = ChainPlot.chaingraph(nna)
    @test nv(mg) == 2 + 5 + 4 + 6 + 6 + 4 + 4 + 3 == 34
    @test ne(mg) == 2 * 5 + 2 * 4 + 4 * 6 + 1 * 6  + 6 * 4 + 4 * 4 + 4 * 3 == 100
end =#

@testset "variable input" begin
    x³(x) = x .^ 3
    dx(x) = x[2:end] - x[1:end-1]
    
    m = Chain(x³, dx, LSTM(5, 4), Dense(4, 5))
    input_data = rand(Float32, 6)
    mg = ChainPlot.chaingraph(m, input_data)
    @test nv(mg) == 6 + 6 + 5 + 4 + 5 == 26
    @test ne(mg) == 6 + 2 * 5 + 5 * 4 + 4 * 5 == 56

    m = Chain(x³, dx, LSTM(10, 4), Dense(4, 2))
    input_data = rand(Float32, 11)
    mg = ChainPlot.chaingraph(m, input_data)
    @test nv(mg) == 38
    @test ne(mg) == 79
end

# Convolution tests work in REPL but not in `]test`...
@testset "Convolutions" begin
    x³(x) = x .^ 3
    reshape6x1x1(a) = reshape(a, 6, 1, 1)
    reshape3x3x1x1(a) = reshape(a, 3, 3, 1, 1)

    m = Chain(Conv((2,), 1 => 1))
    input_data = rand(Float32, 5, 1, 1)
    @test_skip mg = ChainPlot.chaingraph(m, input_data)
    @test_skip nv(mg) == 5 + 4 == 9
    @test_skip ne(mg) == 2 * 4 == 8

    m = Chain(x³, Dense(3, 6), reshape6x1x1, Conv((2,), 1 => 1), vec, Dense(5, 4))
    input_data = rand(Float32, 3)
    @test_skip mg = ChainPlot.chaingraph(m, input_data)

    m = Chain(x³, Dense(4, 9), reshape3x3x1x1, Conv((2, 2), 1 => 1), vec)
    input_data = rand(Float32, 4)
    @test_skip mg = ChainPlot.chaingraph(m, input_data)
    @test_skip nv(mg) == 4 + 4 + 9 + 9 + 4 + 4 == 34
    @test_skip ne(mg) == 69

    m = Chain(Conv((3, 3), 1 => 8, leakyrelu, pad=1), GroupNorm(8, 4))
    input_data = rand(6,5,1,1)
    @test_skip mg = ChainPlot.chaingraph(m, input_data)
end
