module ChainPlot

using Flux
using RecipesBase
import Base: getindex, length, size
# import Functors: functor

include("chaintools.jl")
include("chainplotrecipe.jl")

end