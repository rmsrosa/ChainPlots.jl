# Attempting to debug the call to https://github.com/FluxML/NNlib.jl/blob/master/src/impl/conv_direct.jl

using Flux
import Flux.NNlib: check_dims, input_size, calc_padding_regions, channels_in, kernel_size, channels_out, padding, dilation, stride, flipkernel

include("../src/ChainPlots.jl")

using .ChainPlots
import ChainPlots.NeuralNumbers: cold, hot, fneutralize

m = Chain(Conv((2,), 1 => 1))

fm = fneutralize(m)

inp = [hot; cold; cold; cold; cold;;;]

w = fm[1].weight
w = [w[:,:,1];;;;;]
x = [inp;;;;;]
y = zeros(eltype(w), (4, 1, 1, 1, 1))
cdims = Flux.DenseConvDims(x, w)

#

kproj(k, _, ::Val{true}) = k
kproj(k, M, ::Val{false}) = M - k + 1

project(idx, stride, pad) = (idx - 1)*stride - pad + 1

#

Flux.NNlib.check_dims(size(x), size(w), size(y), cdims)

width, height, depth = input_size(cdims)
kernel_w, kernel_h, kernel_d = kernel_size(cdims)
pad_w_lo, _, pad_h_lo, _, pad_d_lo, _ = padding(cdims)
dil_w, dil_h, dil_d = dilation(cdims)
stride_w, stride_h, stride_d = stride(cdims)

C = channels_out(cdims)
yT = eltype(y)
fk = Val(flipkernel(cdims))
alpha = yT(0)
beta = false

padded_regions, central_region = calc_padding_regions(cdims)

#

w_region, h_region, d_region = central_region

@inbounds for batch in 1:size(x, 5),
    c_out in 1:C,
    d_idx in d_region,
    h_idx in h_region,
    w_idx in w_region

    # Since we're in the central region, we don't need to worry about clamping
    dotprod = yT(0)
    for c_in in 1:channels_in(cdims),
        kd in 1:kernel_d,
        kh in 1:kernel_h,
        kw in 1:kernel_w

        # Hoist me, you coward.
        x_d = project(d_idx, stride_d, pad_d_lo) + (kd - 1)*dil_d
        x_h = project(h_idx, stride_h, pad_h_lo) + (kh - 1)*dil_h
        x_w = project(w_idx, stride_w, pad_w_lo) + (kw - 1)*dil_w

        x_val = x[x_w, x_h, x_d, c_in, batch]
        w_val = w[kproj(kw, kernel_w, fk),
                kproj(kh, kernel_h, fk),
                kproj(kd, kernel_d, fk),
                c_in, c_out]
        @info "x_val = $x_val; w_val = $w_val"
        dotprod = muladd(x_val, w_val, dotprod)
        @info "dotprod = $dotprod"
    end
    @info "alpha = $alpha; beta = $beta; y = $(y[w_idx, h_idx, d_idx, c_out, batch]); beta * y = $(beta * y[w_idx, h_idx, d_idx, c_out, batch])"
    y[w_idx, h_idx, d_idx, c_out, batch] = alpha*dotprod + beta*y[w_idx, h_idx, d_idx, c_out, batch]
    @info "y = $(y[w_idx, h_idx, d_idx, c_out, batch])"
end