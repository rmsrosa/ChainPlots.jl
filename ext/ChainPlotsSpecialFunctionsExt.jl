module ChainPlotsSpecialFunctionsExt

import SpecialFunctions
using ChainPlots.NeuralNumbers: NeuralNumber

# Activations such as `NNlib.gelu_erf` call `erf`, which, like any other
# unary function, should leave the state of a neuron unchanged.
for f in (:erf, :erfc, :erfinv, :erfcinv, :erfcx, :erfi)
    @eval SpecialFunctions.$f(x::NeuralNumber) = x
end

end # module
