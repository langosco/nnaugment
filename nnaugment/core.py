from jax.typing import ArrayLike
import os
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from meta_transformer import torch_utils, module_path
import flax.linen as nn
from einops import rearrange

# just linear layers for now
# IMPORTANT ASSUMPTION: the layers are numbered consecutively and ordered

# BEWARE: pytorch and flax naming schemes are different. not only are weights
# and biases named differently, but also the ordering of the dimensions is
# different. pytorch: (out, in), flax: (in, out).

# I will assume flax conventions for now

# def make_layer(w: ArrayLike, b: ArrayLike, naming_scheme: str = "pytorch") -> dict:
#     """Returns a dictionary with weights and biases."""
#     if naming_scheme == "pytorch":
#         return {"w": w, "b": b}
#     elif naming_scheme == "flax":
#         return {"kernel": w, "bias": b}
#     else:
#         raise ValueError(f"Unknown naming scheme: {naming_scheme}")


def permute_linear_layer(
        layer: dict, 
        permutation: ArrayLike,
        mode: str = "output"):
    """Permute the weights and biases of a linear layer.
    Modes:
        - output: permute the rows of the weight matrix. this
                changes the ordering of the output.
        - input: permute the columns of the weight matrix
    """
    assert permutation is not None

    w = layer["kernel"]
    b = layer["bias"]

    if mode == "output":
        assert len(permutation) == w.shape[1]
        assert len(permutation) == len(b)
        w_perm = w[:, permutation]
        b_perm = b[permutation]
    elif mode == "input":
        assert len(permutation) == w.shape[0], f"{len(permutation)}, {w.shape[0]}"
        w_perm = w[permutation, :]
        b_perm = b
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return {"kernel": w_perm, "bias": b_perm}


def permute_conv_layer(layer: dict, 
                       permutation: ArrayLike,
                       mode: str = "output"):
    assert permutation is not None

    w = layer["kernel"]
    b = layer["bias"]

    if mode == "output":
        assert len(permutation) == w.shape[-1]
        assert len(permutation) == len(b)
        w_perm = w[..., permutation]
        b_perm = b[permutation]
    elif mode == "input":
        assert len(permutation) == w.shape[-2]
        w_perm = w[:, :, permutation, :]
        b_perm = b
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return {"kernel": w_perm, "bias": b_perm}


def get_linear_permutation_from_conv(
        permutation: ArrayLike, 
        next_layer: dict, 
        convention: str = "pytorch"
    ):
    """Get permutation to use for a linear layer following a conv layer.
    - In Flax, the conv convention is (h, w, in, out), and similarly for the
    activations. So the output is flattened first. Thus to match the permutation
     of the output, the flat vector of length (n * n * output_channels) 
     is permuted following "[p, p, ..., p] (n*n times)".
    
    - In Pytorch, the convention is (out, in, h, w), so the output is flattened
    last. This means the flattened activation vector is different.
    """
    next_layer_input_size = next_layer["kernel"].shape[0]
    assert next_layer_input_size % len(permutation) == 0
    factor = next_layer_input_size // len(permutation)

    if convention == "flax":
        out = np.concatenate(
            [permutation + len(permutation) * i for i in range(factor)])
    elif convention == "pytorch":
        out = np.arange(len(permutation)*factor)  # correct total perm length
        # split into n = len(permutation) blocks:
        out = rearrange(out, "(n f) -> n f", n=len(permutation), f=factor)
        out = out[permutation]  # permute
        out = np.concatenate(out)
    return out


def permute_layer_and_next(layer: dict, 
                           next_layer: dict, 
                           name: str, 
                           next_name: str,
                           permutation: ArrayLike,
                           convention: str = "pytorch"):
    """Permute the weights of layer (effectively re-ordering the output),
    then permute the weights of next_layer (re-ordering the input to match).
    
    Currently this requires an awkward case distinction between conv
    and dense layers (that's why this function also takes the layer names).
    """
    if 'Conv' in name and 'Conv' in next_name:
        layer_perm = permute_conv_layer(layer, permutation, mode="output")
        next_layer_perm = permute_conv_layer(next_layer, permutation, mode="input")
    elif 'Dense' in name and 'Dense' in next_name:
        layer_perm = permute_linear_layer(layer, permutation, mode="output")
        next_layer_perm = permute_linear_layer(next_layer, permutation, mode="input")
    elif 'Conv' in name and 'Dense' in next_name:
        layer_perm = permute_conv_layer(layer, permutation, mode="output")
        next_permutation = get_linear_permutation_from_conv(
            permutation, next_layer, convention)
        next_layer_perm = permute_linear_layer(
            next_layer, next_permutation, mode="input")
    else:
        raise NotImplementedError(f"Not implemented for layers: {name}, {next_name}")
    return layer_perm, next_layer_perm


def random_permutation(params: dict, 
                       layers_to_permute: list,
                       convention: str = "pytorch") -> dict:
    """
    Permute all layers in layers_to_permute with a random permutation.
    Important assumption: the layers are numbered consecutively and ordered.

    Convention: unfortunately, need to distinguish between pytorch and flax
    even though I standardize weights to flax convention. This is because
    the output of a conv + flatten layer is different in pytorch and flax,
    since channels vs features are flattened in a different order.
    """
    layer_names = list(params.keys())

    p_params = params.copy()
    for i, k in enumerate(layer_names):
        if k in layers_to_permute:
            next_k = layer_names[i+1]
            permutation = np.random.permutation(p_params[k]["kernel"].shape[-1])
            p_params[k], p_params[next_k] = permute_layer_and_next(
                p_params[k], p_params[next_k], k, next_k, permutation, convention)
    return p_params