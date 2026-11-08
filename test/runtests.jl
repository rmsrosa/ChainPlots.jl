using Aqua
using Flux
using Graphs
using Random
using Test

using ChainPlots

import ChainPlots.NeuralNumbers: cold, hot, fneutralize

@testset "neurons" begin
    m = Chain(Dense(2 => 3), RNN(3 => 2))
    fm = fneutralize(m)
    # sequence layers take an `(in, len)` input
    @test fm(reshape([cold, hot], 2, 1)) == reshape([hot, hot], 2, 1)
    # a signal at time step `t` reaches every later time step, but no earlier one
    @test fm([cold cold hot; cold cold cold]) == [cold cold hot; cold cold hot]
    @test fm([cold hot cold; cold cold cold]) == [cold hot hot; cold hot hot]
    # and ChainPlots runs it as a single time step on a vector input
    @test foldl((x, l) -> ChainPlots.applylayer(l, x), fm.layers; init = [cold, hot]) == [hot, hot]

    for c in (RNNCell(2 => 3), LSTMCell(2 => 3), GRUCell(2 => 3), GRUv3Cell(2 => 3))
        @test ChainPlots.applylayer(fneutralize(c), [hot, cold]) == [hot, hot, hot]
    end

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
        (:GRU, 8, 5),
        (:GRUv3, 4, 3),
        (:RNNCell, 3, 5),
        (:LSTMCell, 7, 4),
        (:GRUCell, 8, 5),
        (:GRUv3Cell, 4, 3)
    )
        @testset "$l" begin
            m = getfield(Flux, l)(ni => no)
            cg = chaingraph(m)
            @test nv(cg) == ni + no
            @test ne(cg) == ni * no
            @test nodeprop(cg, 1, :layer_name) == nodeprop(cg, ni, :layer_name) == "input layer"
            @test occursin("$l", nodeprop(cg, ni + 1, :layer_name))
            @test occursin("$l", nodeprop(cg, ni + no, :layer_name))
            @test nodeprop(cg, 1, :layer_number) == 0
            @test nodeprop(cg, ni + 1, :layer_number) == 1
            @test nodeprop(cg, 1, :layer_center) == ni / 2
            @test nodeprop(cg, ni + 1, :layer_center) == no / 2
            @test nodeprop(cg, 1, :index_in_layer) == nodeprop(cg, ni + 1, :index_in_layer) == (1,)
            @test nodeprop(cg, ni, :index_in_layer) == (ni,)
            @test nodeprop(cg, ni + no, :index_in_layer) == (no,)
            @test neighbors(cg, 1) == neighbors(cg, ni) == (ni+1:ni+no)
            @test neighbors(cg, ni + 1) == neighbors(cg, ni + no) == (1:ni)
            @test length(nodeprop.(Ref(cg), vertices(cg), :loc_x)) == length(nodeprop.(Ref(cg), vertices(cg), :loc_y)) == ni + no
            @test unique(nodeprop(cg, v, :loc_x) for v in vertices(cg)) == (0.0:1.0)
            @test nodeprop(cg, ni + 1, :layer_type) == l
            @test nodeprop(cg, ni + 1, :neuron_color) == ChainPlots.neuron_color(m)
        end
    end
end

@testset "chains" begin
    for (m, num_vert, num_edges) in (
        (Chain(Dense(2 => 5), Dense(5 => 7, σ), Dense(7 => 2, relu), Dense(2 => 3)), 19, 65),
        (Chain(Dense(2 => 5, σ), RNN(5 => 4, relu), LSTM(4 => 4), GRU(4 => 4), Dense(4 => 3)), 22, 74),
        (Chain(Dense(5 => 8), RNN(8 => 20), LSTM(20 => 10), Dense(10 => 7)), 50, 470),
        (Chain(Dense(2 => 3), GRUCell(3 => 4)), 9, 18)
    )
        cg = chaingraph(m)
        @test nv(cg) == num_vert
        @test ne(cg) == num_edges
    end
end

@testset "nnr" begin
    nnr = Chain(Dense(2 => 5, σ), RNN(5 => 4, relu), LSTM(4 => 4), GRU(4 => 4), Dense(4 => 3))
    cg = chaingraph(nnr)
    @test nv(cg) == 2 + 5 + 4 + 4 + 4 + 3 == 22
    @test ne(cg) == 2 * 5 + 5 * 4 + 4 * 4 + 4 * 4 + 4 * 3 == 74
    @test all(==("input layer"), nodeprop.(Ref(cg), 1:2, :layer_name))
    @test all(s -> occursin("Dense", s), nodeprop.(Ref(cg), 3:7, :layer_name))
    @test all(s -> startswith(s, "RNN("), nodeprop.(Ref(cg), 8:11, :layer_name))
    @test all(s -> startswith(s, "LSTM("), nodeprop.(Ref(cg), 12:15, :layer_name))
    @test all(s -> startswith(s, "GRU("), nodeprop.(Ref(cg), 16:19, :layer_name))
    @test nodeprop.(Ref(cg), [3, 8, 12, 16, 20], :layer_type) == [:Dense, :RNN, :LSTM, :GRU, :Dense]
    @test all(s -> occursin("Dense", s), nodeprop.(Ref(cg), 20:22, :layer_name))
    @test nodeprop(cg, 1, :index_in_layer) == (1,)
    @test nodeprop(cg, 2, :index_in_layer) == (2,)
    @test nodeprop(cg, 3, :index_in_layer) == (1,)
    @test nodeprop(cg, 7, :index_in_layer) == (5,)
    @test nodeprop(cg, 3, :neuron_color) == ChainPlots.neuron_color(Dense(2 => 5, σ))
    @test nodeprop(cg, 8, :neuron_color) == ChainPlots.neuron_color(RNN(5 => 4, relu))
    @test neighbors(cg, 1) == neighbors(cg, 2) == (3:7)
    @test all(==([(1:2); (8:11)]), neighbors.(Ref(cg), 3:7))
    @test all(==([(3:7); (12:15)]), neighbors.(Ref(cg), 8:11))
    @test all(==([(8:11); (16:19)]), neighbors.(Ref(cg), 12:15))
    @test all(==([(12:15); (20:22)]), neighbors.(Ref(cg), 16:19))
    @test all(==((16:19)), neighbors.(Ref(cg), 20:22))
    @test length(nodeprop(cg, v, :loc_x) for v in vertices(cg)) == 22
    @test length(nodeprop(cg, v, :loc_y) for v in vertices(cg)) == 22
    @test unique(nodeprop(cg, v, :loc_x) for v in vertices(cg)) == (0.0:length(nnr))
end

@testset "functional" begin
    x³(x) = x .^ 3
    dx(x) = x[2:end] - x[1:end-1]
    nna = Chain(Dense(2 => 5, σ), dx, RNN(4 => 6, relu), x³, LSTM(6 => 4), GRU(4 => 4), Dense(4 => 3))
    cg = chaingraph(nna)
    @test nv(cg) == 2 + 5 + 4 + 6 + 6 + 4 + 4 + 3 == 34
    @test ne(cg) == 2 * 5 + 2 * 4 + 4 * 6 + 1 * 6  + 6 * 4 + 4 * 4 + 4 * 3 == 100
end

@testset "variable input" begin
    x³(x) = x .^ 3
    dx(x) = x[2:end] - x[1:end-1]
    
    m = Chain(x³, dx, LSTM(5 => 4), Dense(4 => 5))
    input_data = rand(Float32, 6)
    cg = chaingraph(m, input_data)
    @test nv(cg) == 6 + 6 + 5 + 4 + 5 == 26
    @test ne(cg) == 6 + 2 * 5 + 5 * 4 + 4 * 5 == 56
    cg_ldim = chaingraph(m, size(input_data))
    @test cg_ldim == cg

    m = Chain(x³, dx, LSTM(10 => 4), Dense(4 => 2))
    input_data = rand(Float32, 11)
    cg = chaingraph(m, input_data)
    @test nv(cg) == 38
    @test ne(cg) == 79
    cg_ldim = chaingraph(m, size(input_data))
    @test cg_ldim == cg
end

@testset "Convolutions" begin
    x³(x) = x .^ 3
    reshape6x1x1(a) = reshape(a, 6, 1, 1)
    reshape4x4x1x1(a) = reshape(a, 4, 4, 1, 1)

    m = Chain(Conv((2,), 1 => 1))
    input_data = rand(Float32, 5, 1, 1)
    cg = chaingraph(m, input_data)
    @test nv(cg) == 5 + 4 == 9
    @test ne(cg) == 2 * 4 == 8
    cg_ldim = chaingraph(m, size(input_data))
    @test cg_ldim == cg

    m = Chain(x³, Dense(3 => 6), reshape6x1x1, Conv((2,), 1 => 1), vec, Dense(5 => 4))
    input_data = rand(Float32, 3)
    cg = chaingraph(m, input_data)
    @test nv(cg) == 3 + 3 + 6 + 6 + 5 + 5 + 4 == 32
    @test ne(cg) == 3 + 3 * 6 + 6 + 2 * 5 + 5 + 5 * 4 == 62
    cg_ldim = chaingraph(m, size(input_data))
    @test cg_ldim == cg

    m = Chain(x³, Dense(4 => 16), reshape4x4x1x1, Conv((2, 2), 1 => 1), vec)
    input_data = rand(Float32, 4)
    cg = chaingraph(m, input_data)
    @test nv(cg) == 4 + 4 + 16 + 16 + 9 + 9 == 58
    @test ne(cg) == 4 + 4 * 16 + 16 + 9 * 4 + 9 == 129
    cg_ldim = chaingraph(m, size(input_data))
    @test cg_ldim == cg

    m = Chain(Conv((3,3), 1=>2))
    input_data = rand(Float32, 10, 10, 1, 1)
    cg = chaingraph(m, input_data)
    @test nv(cg) == 10 * 10 + 8 * 8 * 2 == 228
    @test ne(cg) == 8 * 8 * 9 * 2 == 1152
    cg_ldim = chaingraph(m, size(input_data))
    @test cg_ldim == cg

    m = Chain(Conv((3, 3), 1 => 4, leakyrelu, pad=1), GroupNorm(4, 2))
    input_data = rand(6,5,1,1)
    cg = chaingraph(m, input_data)
    @test nv(cg) == 6 * 5 + 6 * 5 * 4 * 2 == 270
    @test ne(cg) == (4 * 3 * 9 + ( 2 * 4 + 2 * 3 ) * 6 + 4 * 4 ) * 4 + ( 6 * 5 )^2 * 2 * 4 == 8032

    m = Chain(
        Conv((3, 3), 1=>2, pad=(1,1), bias=false),
        MaxPool((2,2)),
        Conv((3, 3), 2=>4, pad=SamePad(), relu),
        AdaptiveMaxPool((4,4)),
        Conv((3, 3), 4=>4, relu),
        GlobalMaxPool()
    )
    input_size = (16, 16, 1, 1)
    cg = chaingraph(m, input_size)
    @test nv(cg) == 16^2 + 16^2 * 2 + 8^2 * 2 + 8^2 * 4 + 4^2 * 4 + 2^2 * 4 + 1^2 * 4 == 1236
end

@testset "recurrent" begin
    # a cell can only be the last layer of a chain, since it returns `(y, state)`
    for c in (RNNCell(3 => 4), LSTMCell(3 => 4), GRUCell(3 => 4), GRUv3Cell(3 => 4))
        @test_throws ArgumentError chaingraph(Chain(c, Dense(4 => 2)))
    end
    cg = chaingraph(Chain(Dense(2 => 3), LSTMCell(3 => 4)))
    @test nv(cg) == 2 + 3 + 4
    @test ne(cg) == 2 * 3 + 3 * 4
    @test nodeprop(cg, 9, :layer_type) == :LSTMCell

    # a vector input is drawn folded, as a single time step
    @test chaingraph(Chain(RNN(3 => 2)), (3,)) == chaingraph(RNN(3 => 2))

    # an `(in, len)` sequence input is unrolled in time, and the step `t` of
    # the input reaches all the steps `t′ ≥ t` of the output
    for (L, ni, no, len) in ((RNN, 3, 2, 4), (LSTM, 2, 3, 3), (GRU, 2, 2, 5))
        cg = chaingraph(Chain(L(ni => no)), (ni, len))
        @test nv(cg) == (ni + no) * len
        @test ne(cg) == ni * no * len * (len + 1) ÷ 2
    end
end

# custom layers, defined at the top level, as structs must be
struct Scale{T}
    w::T
end
Flux.@layer Scale
(s::Scale)(x) = s.w .* x

# a two-input model, wrapped so that it takes one array and splits it
struct Split{T}
    parts::T
end
Flux.@layer Split
(s::Split)(x) = vcat(s.parts[1](x[1:1]), s.parts[2](x[2:2]))

@testset "custom layers" begin
    # an array-to-array custom layer, in the middle, first, or alone
    cg = chaingraph(Chain(Dense(2 => 3), Scale(rand(Float32, 3))))
    @test nv(cg) == 2 + 3 + 3
    @test ne(cg) == 2 * 3 + 3
    @test nodeprop(cg, 8, :layer_type) == :Scale
    @test nodeprop(cg, 8, :neuron_color) == ChainPlots.NEURON_COLORS[:Any]
    cg = chaingraph(Chain(Scale(rand(Float32, 3)), Dense(3 => 2)), (3,))
    @test nv(cg) == 3 + 3 + 2 && ne(cg) == 3 + 3 * 2
    cg = chaingraph(Scale(rand(Float32, 3)), (3,))
    @test nv(cg) == 3 + 3 && ne(cg) == 3

    # Flux's container layers
    cg = chaingraph(Chain(Parallel(vcat, Dense(2 => 3), Dense(2 => 4)), Dense(7 => 1)), (2,))
    @test nv(cg) == 2 + 7 + 1
    @test ne(cg) == 2 * 7 + 7
    cg = chaingraph(Chain(SkipConnection(Dense(3 => 3), +), Dense(3 => 1)), (3,))
    @test nv(cg) == 3 + 3 + 1
    @test ne(cg) == 3 * 3 + 3

    # a two-input model, taking one array instead of a vector of arrays:
    # each half of the input only reaches its own `Dense`
    cg = chaingraph(Chain(Split((Dense(1 => 4, swish), Dense(1 => 4, swish))), Dense(8 => 1)), (2,))
    @test nv(cg) == 2 + 8 + 1
    @test ne(cg) == 4 + 4 + 8
    @test neighbors(cg, 1) == 3:6
    @test neighbors(cg, 2) == 7:10

    # a vector of arrays as input is not supported
    @test_throws Exception chaingraph(Chain(Dense(1 => 2)), [ones(Float32, 1), ones(Float32, 1)])
end

@testset "ChainGraph" begin
    nnr = Chain(Dense(2 => 5, σ), RNN(5 => 4, relu), LSTM(4 => 4), GRU(4 => 4), Dense(4 => 3))
    cg = chaingraph(nnr)
    @test cg isa ChainGraph
    @test cg isa Graphs.AbstractGraph{Int}
    @test !is_directed(cg)
    @test nv(cg) == 22 && ne(cg) == 74
    @test has_vertex(cg, 22) && !has_vertex(cg, 23)
    @test has_edge(cg, 1, 3) && has_edge(cg, 3, 1) && !has_edge(cg, 1, 2)
    @test length(collect(edges(cg))) == 74
    @test ChainPlots.nlayers(cg) == length(nnr) == 5
    @test layer_dimensions(cg) == [(2,), (5,), (4,), (4,), (4,), (3,)]
    @test layer_names(cg) == ["input layer"; [string(l) for l in nnr.layers]]
    @test cg == chaingraph(nnr, (2,))

    props = nodeprops(cg, 8)
    @test keys(props) == ChainPlots.NODE_PROPERTIES
    @test props.layer_number == 2
    @test props.layer_type == :RNN
    @test props.layer_name == "RNN(5 => 4, relu)"
    @test props.index_in_layer == (1,)
    @test all(p -> nodeprop(cg, 8, p) == props[p], ChainPlots.NODE_PROPERTIES)
    @test_throws ArgumentError nodeprop(cg, 1, :not_a_property)
    @test_throws BoundsError nodeprop(cg, 23, :layer_number)

    @test sprint(show, cg) == "ChainGraph with 22 neurons, 74 connections and 5 layers"
    @test occursin("2: RNN(5 => 4, relu) (4,)", sprint(show, MIME("text/plain"), cg))

    # single layers, with and without input
    @test chaingraph(Dense(2 => 3)) == chaingraph(Chain(Dense(2 => 3)))
    @test chaingraph(Conv((2,), 1 => 1), (5, 1, 1)) == chaingraph(Chain(Conv((2,), 1 => 1)), (5, 1, 1))
    @test_throws ArgumentError chaingraph(Chain(x -> x .^ 3, Dense(3 => 2)))
end

@testset "plotchain without Plots" begin
    # the Plots extension is not loaded yet
    @test isnothing(Base.get_extension(ChainPlots, :ChainPlotsPlotsExt))
    err = try plotchain(Dense(2 => 3)) catch e; e end
    @test err isa MethodError
    @test occursin("using Plots", sprint(showerror, err))
end

using Plots

@testset "plot recipe" begin
    @test !isnothing(Base.get_extension(ChainPlots, :ChainPlotsPlotsExt))
    nnr = Chain(Dense(2 => 5, σ), RNN(5 => 4, relu), Dense(4 => 3))
    cg = chaingraph(nnr)
    plt = Plots.plot(cg)
    @test plt isa Plots.Plot
    @test length(plotchain(nnr).series_list) == length(plt.series_list)

    # a layer type without its own marker takes its color from `neuron_colors`, if there
    ext = Base.get_extension(ChainPlots, :ChainPlotsPlotsExt)
    @test only(ext.layerplotattributes(:Scale).mrkrcolor) == ChainPlots.NEURON_COLORS[:Any]
    @test only(ext.layerplotattributes(:Scale; neuron_colors = merge(ChainPlots.NEURON_COLORS, Dict(:Scale => :tomato))).mrkrcolor) == :tomato
    @test plotchain(Dense(2 => 3)) isa Plots.Plot
    @test plotchain(Conv((2,), 1 => 1), (5, 1, 1)) isa Plots.Plot
    @test plotchain(Chain(x -> x .^ 3, Dense(3 => 2)), (3,); title = "t") isa Plots.Plot

    p = Plots.plot(layout = 2)
    plotchain!(p, Dense(2 => 3); subplot = 1)
    plotchain!(p, GRU(2 => 3); subplot = 2)
    @test length(p.series_list) ==
        length(Plots.plot(chaingraph(Dense(2 => 3))).series_list) + length(Plots.plot(chaingraph(GRU(2 => 3))).series_list)

    # no recipe for types ChainPlots does not own
    @test_throws Exception Plots.plot(nnr)
end

@testset "Aqua" begin
    # NeuralNumbers has method ambiguities with Base and ForwardDiff numbers,
    # unrelated to type piracy, so ambiguities are not checked here
    Aqua.test_all(ChainPlots; ambiguities = false)
    Aqua.test_piracies(ChainPlots)
end
