using Flux
using Graphs
using MetaGraphs
using Random
using Test

using ChainPlots

import ChainPlots.NeuralNumbers: cold, hot, fneutralize

@testset "neurons" begin
    m = Chain(Dense(2, 3), RNN(3, 2))
    fm = fneutralize(m)
    @test fm([cold, hot]) == [hot, hot]

    m = Chain(x -> x[2:end] - x[1:end-1])
    fm = fneutralize(m)
    inp = [cold, hot, cold, cold, cold]
    @test fm(inp) == [hot, hot, cold, cold]

    m = Chain(Conv((2,), 1 => 1))
    fm = fneutralize(m)
    inp = fill(cold, 5, 1, 1)
    inp[1] = hot
    @test fm(inp) == reshape([hot; cold; cold; cold;;;], 4, 1, 1)
    inp[1:2] .= [cold, hot]
    @test fm(inp) == reshape([hot; hot; cold; cold;;;], 4, 1, 1)
    inp[2:3] .= [cold, hot]
    @test fm(inp) == reshape([cold; hot; hot; cold;;;], 4, 1, 1)
    inp[3:4] .= [cold, hot]
    @test fm(inp) == reshape([cold; cold; hot; hot;;;], 4, 1, 1)
    inp[4:5] .= [cold, hot]
    @test fm(inp) == reshape([cold; cold; cold; hot;;;], 4, 1, 1)
end

@testset "activations" begin
    for fn in Flux.NNlib.ACTIVATIONS
        for x in (cold, hot)
            @test eval(fn)(x) == x
        end
    end
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
            mg = chaingraph(m)
            @test nv(mg) == ni + no
            @test ne(mg) == ni * no
            @test get_prop(mg, 1, :layer_name) == get_prop(mg, ni, :layer_name) == "input layer"
            @test occursin("$l", get_prop(mg, ni + 1, :layer_name))
            @test occursin("$l", get_prop(mg, ni + no, :layer_name))
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
        mg = chaingraph(m)
        @test nv(mg) == num_vert
        @test ne(mg) == num_edges
    end
end

@testset "nnr" begin
    nnr = Chain(Dense(2, 5, σ), RNN(5, 4, relu), LSTM(4, 4), GRU(4, 4), Dense(4, 3))
    mg = chaingraph(nnr)
    @test nv(mg) == 2 + 5 + 4 + 4 + 4 + 3 == 22
    @test ne(mg) == 2 * 5 + 5 * 4 + 4 * 4 + 4 * 4 + 4 * 3 == 74
    @test all(==("input layer"), get_prop.(Ref(mg), 1:2, :layer_name))
    @test all(s -> occursin("Dense", s), get_prop.(Ref(mg), 3:7, :layer_name))
    @test all(s -> occursin("RNNCell", s), get_prop.(Ref(mg), 8:11, :layer_name))
    @test all(s -> occursin("LSTMCell", s), get_prop.(Ref(mg), 12:15, :layer_name))
    @test all(s -> occursin("GRUCell", s), get_prop.(Ref(mg), 16:19, :layer_name))
    @test all(s -> occursin("Dense", s), get_prop.(Ref(mg), 20:22, :layer_name))
    @test get_prop(mg, 1, :index_in_layer) == (1,)
    @test get_prop(mg, 2, :index_in_layer) == (2,)
    @test get_prop(mg, 3, :index_in_layer) == (1,)
    @test get_prop(mg, 7, :index_in_layer) == (5,)
    @test get_prop(mg, 3, :neuron_color) == ChainPlots.neuron_color(Dense(2, 5, σ))
    @test get_prop(mg, 8, :neuron_color) == ChainPlots.neuron_color(RNN(5, 4, relu))
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

@testset "functional" begin
    x³(x) = x .^ 3
    dx(x) = x[2:end] - x[1:end-1]
    nna = Chain(Dense(2, 5, σ), dx, RNN(4, 6, relu), x³, LSTM(6, 4), GRU(4, 4), Dense(4, 3))
    mg = chaingraph(nna)
    @test nv(mg) == 2 + 5 + 4 + 6 + 6 + 4 + 4 + 3 == 34
    @test ne(mg) == 2 * 5 + 2 * 4 + 4 * 6 + 1 * 6  + 6 * 4 + 4 * 4 + 4 * 3 == 100
end

@testset "variable input" begin
    x³(x) = x .^ 3
    dx(x) = x[2:end] - x[1:end-1]
    
    m = Chain(x³, dx, LSTM(5, 4), Dense(4, 5))
    input_data = rand(Float32, 6)
    mg = chaingraph(m, input_data)
    @test nv(mg) == 6 + 6 + 5 + 4 + 5 == 26
    @test ne(mg) == 6 + 2 * 5 + 5 * 4 + 4 * 5 == 56
    mg_ldim = chaingraph(m, size(input_data))
    @test mg_ldim == mg

    m = Chain(x³, dx, LSTM(10, 4), Dense(4, 2))
    input_data = rand(Float32, 11)
    mg = chaingraph(m, input_data)
    @test nv(mg) == 38
    @test ne(mg) == 79
    mg_ldim = chaingraph(m, size(input_data))
    @test mg_ldim == mg
end

@testset "Convolutions" begin
    x³(x) = x .^ 3
    reshape6x1x1(a) = reshape(a, 6, 1, 1)
    reshape4x4x1x1(a) = reshape(a, 4, 4, 1, 1)

    m = Chain(Conv((2,), 1 => 1))
    input_data = rand(Float32, 5, 1, 1)
    mg = chaingraph(m, input_data)
    nv(mg) == 5 + 4 == 9
    ne(mg) == 2 * 4 == 8
    mg_ldim = chaingraph(m, size(input_data))
    @test mg_ldim == mg

    m = Chain(x³, Dense(3, 6), reshape6x1x1, Conv((2,), 1 => 1), vec, Dense(5, 4))
    input_data = rand(Float32, 3)
    mg = chaingraph(m, input_data)
    @test nv(mg) == 3 + 3 + 6 + 6 + 5 + 5 + 4 == 32
    @test ne(mg) == 3 + 3 * 6 + 6 + 2 * 5 + 5 + 5 * 4 == 62
    mg_ldim = chaingraph(m, size(input_data))
    @test mg_ldim == mg

    m = Chain(x³, Dense(4, 16), reshape4x4x1x1, Conv((2, 2), 1 => 1), vec)
    input_data = rand(Float32, 4)
    mg = chaingraph(m, input_data)
    @test nv(mg) == 4 + 4 + 16 + 16 + 9 + 9 == 58
    @test ne(mg) == 4 + 4 * 16 + 16 + 9 * 4 + 9 == 129
    mg_ldim = chaingraph(m, size(input_data))
    @test mg_ldim == mg

    m = Chain(Conv((3,3), 1=>2))
    input_data = rand(Float32, 10, 10, 1, 1)
    mg = chaingraph(m, input_data)
    @test nv(mg) == 10 * 10 + 8 * 8 * 2 == 228
    @test ne(mg) == 8 * 8 * 9 * 2 == 1152
    mg_ldim = chaingraph(m, size(input_data))
    @test mg_ldim == mg

    m = Chain(Conv((3, 3), 1 => 4, leakyrelu, pad=1), GroupNorm(4, 2))
    input_data = rand(6,5,1,1)
    mg = chaingraph(m, input_data)
    @test nv(mg) == 6 * 5 + 6 * 5 * 4 * 2 == 270
    @test ne(mg) == (4 * 3 * 9 + ( 2 * 4 + 2 * 3 ) * 6 + 4 * 4 ) * 4 + ( 6 * 5 )^2 * 2 * 4 == 8032

    m = Chain(
        Conv((3, 3), 1=>2, pad=(1,1), bias=false),
        MaxPool((2,2)),
        Conv((3, 3), 2=>4, pad=SamePad(), relu),
        AdaptiveMaxPool((4,4)),
        Conv((3, 3), 4=>4, relu),
        GlobalMaxPool()
    )
    input_size = (16, 16, 1, 1)
    mg = chaingraph(m, input_size)
    @test nv(mg) == 16^2 + 16^2 * 2 + 8^2 * 2 + 8^2 * 4 + 4^2 * 4 + 2^2 * 4 + 1^2 * 4 == 1236
end
