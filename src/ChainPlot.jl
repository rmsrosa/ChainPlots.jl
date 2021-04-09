module ChainPlot

using Flux
using LightGraphs
using MetaGraphs
using RecipesBase
import Base: getindex, length, size
# import Functors: functor

include("chaintools.jl")
include("chaingraph.jl")
include("chainplotrecipe.jl")

end