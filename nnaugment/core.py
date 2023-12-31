from einops import rearrange
from jax.typing import ArrayLike
from jax import random
import jax.numpy as jnp
from nnaugment.conventions import is_sorted_and_numbered, sort_layers

# just linear layers for now
# IMPORTANT ASSUMPTION: the layers are numbered consecutively and ordered

# BEWARE: pytorch and flax naming schemes are different. not only are weights
# and biases named differently, but also the ordering of the dimensions is
# different. pytorch: (out, in), flax: (in, out). See conventions.py


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


def permute_batchnorm_layer(layer: dict,
                            permutation: ArrayLike):
    assert permutation is not None
    assert len(permutation) == len(layer["kernel"])
    w_perm = layer["kernel"][permutation]
    b_perm = layer["bias"][permutation]
    out = {"kernel": w_perm, "bias": b_perm}
    try:
        m_perm = layer["mean"][permutation]
        v_perm = layer["var"][permutation]
        out.update({"mean": m_perm, "var": v_perm})
    except KeyError as e:
        pass
    return out


def permute_layernorm_layer(layer: dict,
                            permutation: ArrayLike):
    assert permutation is not None
    assert len(permutation) == len(layer["scale"])
    b_perm = layer["bias"][permutation]
    s_perm = layer["scale"][permutation]
    return {"bias": b_perm, "scale": s_perm}



def get_linear_permutation_from_conv(
        permutation: ArrayLike, 
        next_layer: dict, 
        convention: str = "flax"
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
    assert next_layer_input_size % len(permutation) == 0, (
        f"Input size of layer is not divisible by len(permutation)."
        f"\nlayer shape: {next_layer['kernel'].shape}, "
        f"\nlen(permutation): {len(permutation)}"
    )
    factor = next_layer_input_size // len(permutation)

    if convention == "flax":
        out = jnp.concatenate(
            [permutation + len(permutation) * i for i in range(factor)])
    elif convention == "pytorch":
        out = jnp.arange(len(permutation)*factor)  # correct total perm length
        # split into n = len(permutation) blocks:
        out = rearrange(out, "(n f) -> n f", n=len(permutation), f=factor)
        out = out[permutation]  # permute
        out = jnp.concatenate(out)
    return out


def permute_layer_and_next(layers: list[dict],
                           names: list[str],
                           permutation: ArrayLike,
                           convention: str = "flax"):
    """Permute the weights of layer (effectively re-ordering the output),
    then permute the weights of next_layer (re-ordering the input to match).

    This function expects len(layers) == len(names) == 2, except when the second 
    layer is a {batch,layer}norm layer, in which case len(layers) == 3.
    
    Currently this requires an awkward case distinction between conv
    and dense layers (that's why this function also takes the layer names).
    """
    assert len(layers) == len(names)
    assert len(layers) in (1, 2, 3)

    if len(layers) == 1:
        # assume last layer
        if 'Conv' in names[0]:
            layer_perm = permute_conv_layer(layers[0], permutation, mode="output")
        elif 'Dense' in names[0]:
            layer_perm = permute_linear_layer(layers[0], permutation, mode="output")
        else:
            raise NotImplementedError(f"Not implemented for layers: {names}")
        return [layer_perm]

    if 'Conv' in names[0] and 'Conv' in names[-1]:
        layer_perm = permute_conv_layer(layers[0], permutation, mode="output")
        next_layer_perm = permute_conv_layer(layers[-1], permutation, mode="input")
    elif 'Dense' in names[0] and 'Dense' in names[-1]:
        layer_perm = permute_linear_layer(layers[0], permutation, mode="output")
        next_layer_perm = permute_linear_layer(layers[-1], permutation, mode="input")
    elif 'Conv' in names[0] and 'Dense' in names[-1]:
        layer_perm = permute_conv_layer(layers[0], permutation, mode="output")
        next_permutation = get_linear_permutation_from_conv(
            permutation, layers[-1], convention)
        next_layer_perm = permute_linear_layer(
            layers[-1], next_permutation, mode="input")
    else:
        raise NotImplementedError(f"Not implemented for layers: {names}")

    if len(layers) == 2:
        out_layers = [layer_perm, next_layer_perm]
    elif len(layers) == 3:
        if 'BatchNorm' in names[1]:
            norm_layer_permuted = permute_batchnorm_layer(layers[1], permutation)
        elif 'LayerNorm' in names[1]:
            norm_layer_permuted = permute_layernorm_layer(layers[1], permutation)
        else:
            raise ValueError(f"Unknown middle layer: {names[1]}")
        out_layers = [layer_perm, norm_layer_permuted, next_layer_perm]
    else:
        raise ValueError(f"Number of layers must be 1, 2, or 3, received {len(layers)}.")
    return out_layers


def random_permutation(rng: random.PRNGKey,
                       params: dict, 
                       layers_to_permute: list,
                       convention: str = "flax",
                       sort=False,
                       allow_permute_last=False) -> dict:
    """
    Permute layers specified in layers_to_permute with a random permutation.
    Important assumption: the layers (keys of params dict) are numbered 
    consecutively and ordered from first (input) to last (output) layer.
    Only supports pure feedforward networks (no skip connections).

    - convention: unfortunately, need to distinguish between pytorch and flax
    even though I standardize weights to flax convention. This is because
    the output of a conv + flatten layer is different in pytorch and flax,
    since channels vs features are flattened in a different order.
    """
    if sort:
        params = sort_layers(params)

    assert is_sorted_and_numbered(params), ("Parameter dictionary must be "
            "numbered and ordered consistently. Received layer names: "
            f"{params.keys()}.")
    # TODO I think I can get rid of the need for 'convention' by permuting 
    # linear layers following a flatten layer when standardizing weight representations
    # to flax convention.
    layer_names = list(params.keys())
    n_layers = len(layer_names)
    if layers_to_permute is None:
        raise ValueError("layers_to_permute must be specified (received None).")

    p_params = params.copy()
    for i, k in enumerate(layer_names):
        if k in layers_to_permute:
            assert 'Conv' in k or 'Dense' in k, f"Layer {k} is not a conv or dense layer."

            # check if last layer
            if i == n_layers - 1:
                if not allow_permute_last:
                    raise ValueError("Cannot permute last layer.")
                else:
                    layer_group = [k]
            else:
                # check if next layer is batchnorm (if so, need permute three 
                # layers total: current, batchnorm, and next)
                if 'Norm' in layer_names[i+1]:
                    layer_group = [k, layer_names[i+1], layer_names[i+2]]
                else:
                    layer_group = [k, layer_names[i+1]]

            try:
                permutation = random.permutation(
                    rng, p_params[k]["kernel"].shape[-1])
            except KeyError as e:
                raise KeyError(f"Caught KeyError: {e}. "
                            f"Keys in p_params[{k}]: {p_params[k].keys()}.")

            permuted_layers = permute_layer_and_next(
                layers=[p_params[name] for name in layer_group], 
                names=layer_group, 
                permutation=permutation, 
                convention=convention)

            for name, layer in zip(layer_group, permuted_layers):
                p_params[name] = layer

    return p_params
