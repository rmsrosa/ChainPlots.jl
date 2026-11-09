"""
    NODE_PROPERTIES

Names of the properties stored for each neuron (node) of a [`ChainGraph`](@ref).

* `:layer_number`: `Int` with the layer the neuron belongs to, with `0` for the input
  layer, `1, …, n-1` for the hidden layers and `n` for the output layer of a chain
  with `n` layers;
* `:layer_name`: `String` with the name of the layer, e.g. `"Dense(2 => 5, σ)"`, or
  `"input layer"`;
* `:layer_type`: `Symbol` with the type of the layer, e.g. `:Dense`, `:LSTM`,
  `:GRUCell`, `:Conv`, or `:input_layer`;
* `:index_in_layer`: `Tuple` with the position of the neuron within its layer, e.g.
  `(i,)` for the output of `Dense(n => m)` or `RNN(n => m)`, or `(i, j, c)` for a
  two-dimensional convolutional layer, and so on;
* `:layer_center`: `Float64` with the vertical mid-point of the layer;
* `:loc_x`, `:loc_y`: `Float64` with the coordinates of the neuron in the plot;
* `:neuron_color`: `Symbol` with the default color of the neuron, see [`NEURON_COLORS`](@ref).
"""
const NODE_PROPERTIES = (
    :layer_number, :layer_name, :layer_type, :index_in_layer,
    :layer_center, :loc_x, :loc_y, :neuron_color
)

"""
    ChainGraph <: Graphs.AbstractGraph{Int}

Graph of the neurons of a neural network, as built by [`chaingraph`](@ref), with one
vertex per neuron and one edge per connection between neurons of consecutive layers.

A `ChainGraph` implements the `Graphs.jl` interface, so functions such as `nv`, `ne`,
`vertices`, `edges` and `neighbors` work on it, and so do packages built on that
interface, such as `GraphPlot.jl`. It is plotted with `plot(cg)`, once Plots.jl or
Makie.jl is loaded; see [`plotchain`](@ref) for the attributes of the plot recipes.

Use [`nodeprop`](@ref) and [`nodeprops`](@ref) to read the properties of a neuron
(see [`NODE_PROPERTIES`](@ref)), and [`layer_dimensions`](@ref) and
[`layer_names`](@ref) for the properties of each layer.

The fields of a `ChainGraph` are internal: the graph backend may change in a future
release without notice.
"""
struct ChainGraph <: Graphs.AbstractGraph{Int}
    graph::MetaGraph{Int,Float64}
    layer_dimensions::Vector{Tuple}
    layer_names::Vector{String}
end

# Graphs.jl interface, forwarded to the backend
Graphs.is_directed(::Type{ChainGraph}) = false
Graphs.is_directed(::ChainGraph) = false
Graphs.edgetype(cg::ChainGraph) = Graphs.edgetype(cg.graph)
Graphs.nv(cg::ChainGraph) = Graphs.nv(cg.graph)
Graphs.ne(cg::ChainGraph) = Graphs.ne(cg.graph)
Graphs.vertices(cg::ChainGraph) = Graphs.vertices(cg.graph)
Graphs.edges(cg::ChainGraph) = Graphs.edges(cg.graph)
Graphs.has_vertex(cg::ChainGraph, v::Integer) = Graphs.has_vertex(cg.graph, v)
Graphs.has_edge(cg::ChainGraph, s::Integer, d::Integer) = Graphs.has_edge(cg.graph, s, d)
Graphs.inneighbors(cg::ChainGraph, v::Integer) = Graphs.inneighbors(cg.graph, v)
Graphs.outneighbors(cg::ChainGraph, v::Integer) = Graphs.outneighbors(cg.graph, v)

Base.:(==)(a::ChainGraph, b::ChainGraph) =
    a.graph == b.graph && a.layer_dimensions == b.layer_dimensions && a.layer_names == b.layer_names

"""
    nodeprop(cg::ChainGraph, v::Integer, prop::Symbol)

Return the property `prop` of the neuron `v` of `cg`.

See [`NODE_PROPERTIES`](@ref) for the available properties. To get one property of
all the neurons, broadcast over the vertices, e.g.
`nodeprop.(Ref(cg), vertices(cg), :loc_x)`.
"""
function nodeprop(cg::ChainGraph, v::Integer, prop::Symbol)
    prop in NODE_PROPERTIES || throw(ArgumentError("Unknown neuron property `:$prop`; the properties are $NODE_PROPERTIES."))
    Graphs.has_vertex(cg, v) || throw(BoundsError(cg, v))
    return get_prop(cg.graph, v, prop)
end

"""
    nodeprops(cg::ChainGraph, v::Integer) -> NamedTuple

Return all the properties of the neuron `v` of `cg` as a `NamedTuple`.

See [`NODE_PROPERTIES`](@ref) for the available properties.
"""
nodeprops(cg::ChainGraph, v::Integer) = NamedTuple{NODE_PROPERTIES}(map(p -> nodeprop(cg, v, p), NODE_PROPERTIES))

"""
    layer_dimensions(cg::ChainGraph) -> Vector{Tuple}

Return the dimensions of each layer of `cg`, starting with the input layer.
"""
layer_dimensions(cg::ChainGraph) = copy(cg.layer_dimensions)

"""
    layer_names(cg::ChainGraph) -> Vector{String}

Return the name of each layer of `cg`, starting with `"input layer"`.
"""
layer_names(cg::ChainGraph) = copy(cg.layer_names)

"""
    nlayers(cg::ChainGraph)

Return the number of layers of the network, not counting the input layer.
"""
nlayers(cg::ChainGraph) = length(cg.layer_dimensions) - 1

function Base.show(io::IO, cg::ChainGraph)
    print(io, "ChainGraph with ", Graphs.nv(cg), " neurons, ", Graphs.ne(cg), " connections and ", nlayers(cg), nlayers(cg) == 1 ? " layer" : " layers")
end

function Base.show(io::IO, ::MIME"text/plain", cg::ChainGraph)
    show(io, cg)
    print(io, ":")
    for (n, (name, dims)) in enumerate(zip(cg.layer_names, cg.layer_dimensions))
        print(io, "\n  ", n - 1, ": ", name, " ", dims)
    end
end
