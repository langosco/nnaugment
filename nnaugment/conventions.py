from einops import rearrange


def conv_from_pytorch(conv_layer: dict) -> dict:
    w, b = conv_layer["w"], conv_layer["b"]
    w = rearrange(w, "out in h w -> h w in out")
    return {"kernel": w, "bias": b}


def linear_from_pytorch(linear_layer: dict) -> dict:
    w, b = linear_layer["w"], linear_layer["b"]
    w = rearrange(w, "out in -> in out")
    return {"kernel": w, "bias": b}


def batchnorm_from_pytorch(batchnorm_layer: dict) -> dict:
    w, b = batchnorm_layer["w"], batchnorm_layer["b"]
    m, v = batchnorm_layer["m"], batchnorm_layer["v"]
    return {"kernel": w, "bias": b, "mean": m, "var": v}


def linear_to_pytorch(linear_layer: dict) -> dict:
    w, b = linear_layer["kernel"], linear_layer["bias"]
    w = rearrange(w, "in out -> out in")
    return {"w": w, "b": b}


def conv_to_pytorch(conv_layer: dict) -> dict:
    w, b = conv_layer["kernel"], conv_layer["bias"]
    w = rearrange(w, "h w in out -> out in h w")
    return {"w": w, "b": b}


def batchnorm_to_pytorch(batchnorm_layer: dict) -> dict:
    w, b = batchnorm_layer["kernel"], batchnorm_layer["bias"]
    m, v = batchnorm_layer["mean"], batchnorm_layer["var"]
    return {"w": w, "b": b, "m": m, "v": v}


def is_ordered_and_numbered(params: dict):
    """Check if the keys of params are ordered and numbered,
    eg [Conv_0, Conv_1, Dense_2, Dense_3]"""
    def get_suffix(s):
        return int(s.split("_")[-1])
    return all([get_suffix(k) == i for i, k in enumerate(params.keys())])


def import_params(params_dict, from_naming_scheme="flax"):
    """Converts a dictionary of parameters from pytorch to flax conventions.
    """
    assert is_ordered_and_numbered(params_dict)

    if from_naming_scheme == "flax":
        return params_dict
    elif from_naming_scheme == "pytorch":
        conv_fn = conv_from_pytorch
        linear_fn = linear_from_pytorch
        batchnorm_fn = batchnorm_from_pytorch
    else:
        raise ValueError(f"Unknown naming scheme: {from_naming_scheme}")

    new_params = {}
    for k, v in params_dict.items():
        if "Conv" in k:
            new_params[k] = conv_fn(v)
        elif "Linear" in k or "Dense" in k:
            new_params["Dense_" + k.split("_")[-1]] = linear_fn(v)
        elif "BatchNorm" in k:
            new_params[k] = batchnorm_fn(v)
        else:
            raise ValueError(f"Unknown layer type: {k}.")
    return new_params


def export_params(params_dict, to_naming_scheme="flax"):
    """Converts a dictionary of parameters from flax to pytorch conventions.
    """
    if to_naming_scheme == "flax":
        return params_dict
    elif to_naming_scheme == "pytorch":
        conv_fn = conv_to_pytorch
        linear_fn = linear_to_pytorch
        batchnorm_fn = batchnorm_to_pytorch
    else:
        raise ValueError(f"Unknown naming scheme: {to_naming_scheme}")

    new_params = {}
    for k, v in params_dict.items():
        if "Conv" in k:
            new_params[k] = conv_fn(v)
        elif "Dense" in k:
            new_params["Linear_" + k.split("_")[-1]] = linear_fn(v)
        elif "BatchNorm" in k:
            new_params[k] = batchnorm_fn(v)
        else:
            raise ValueError(f"Unknown layer type: {k}.")
    return new_params