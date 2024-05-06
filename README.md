# Notes from Doug

I did some basic profiling and playing here. 
It seems like we are now only 2x faster than a pytorch linear stacked set of layers which is very impressive!

It looks like the long leg now is b-splines in the forward pass

The backprop process seems to take only 6 seconds on my mac compared to the 12 for the forward passes. Which is wild!

This makes adopting the fourier transform here: https://github.com/GistNoesis/FourierKAN/blob/main/fftKAN.py

Suddenly sound very appealing as that should be much more performant. However, the author there does note there is some degradation in core performance.

I still don't fully understand this algorithm, so I think I might need to do that before I go adopt the fourier version. But it is starting to seem possible that we might get to near MLP performance!

5475    0.023    0.000   12.007    0.002 /Users/douglasschonholtz/Documents/Research/efficient-kan/src/efficient_kan/kan.py:269(forward)
    10950    0.374    0.000   11.936    0.001 /Users/douglasschonholtz/Documents/Research/efficient-kan/src/efficient_kan/kan.py:153(forward)
   377385   10.126    0.000   10.126    0.000 {method 'to' of 'torch._C.TensorBase' objects}
    10952    9.623    0.001    9.934    0.001 /Users/douglasschonholtz/Documents/Research/efficient-kan/src/efficient_kan/kan.py:78(b_splines)

# An Efficient Implementation of Kolmogorov-Arnold Network

This repository contains an efficient implementation of Kolmogorov-Arnold Network (KAN).
The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).

The performance issue of the original implementation is mostly because it needs to expand all intermediate variables to perform the different activation functions.
For a layer with `in_features` input and `out_features` output, the original implementation needs to expand the input to a tensor with shape `(batch_size, out_features, in_features)` to perform the activation functions.
However, all activation functions are linear combination of a fixed set of basis functions which are B-splines; given that, we can reformulate the computation as activate the input with different basis functions and then combine them linearly.
This reformulation can significantly reduce the memory cost and make the computation a straightforward matrix multiplication, and works with both forward and backward pass naturally.

The problem is in the **sparsification** which is claimed to be critical to KAN's interpretability.
The authors proposed a L1 regularization defined on the input samples, which requires non-linear operations on the `(batch_size, out_features, in_features)` tensor, and is thus not compatible with the reformulation.
I instead replace the L1 regularization with a L1 regularization on the weights, which is more common in neural networks and is compatible with the reformulation.
The author's implementation indeed include this kind of regularization alongside the one described in the paper as well, so I think it might help.
More experiments are needed to verify this; but at least the original approach is infeasible if efficiency is wanted.

Another difference is that, beside the learnable activation functions (B-splines), the original implementation also includes a learnable scale on each activation function.
I provided an option `enable_standalone_scale_spline` that defaults to `True` to include this feature; disable it will make the model more efficient, but potentially hurts results.
It needs more experiments.

2024-05-04 Update: @xiaol hinted that the constant initialization of `base_weight` parameters can be a problem on MNIST.
For now I've changed both the `base_weight` and `spline_scaler` matrices to be initialized with `kaiming_uniform_`, following `nn.Linear`'s initialization.
It seems to work much much better on MNIST (~20% to ~97%), but I'm not sure if it's a good idea in general.
