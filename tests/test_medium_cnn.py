import pytest
import os
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from meta_transformer import torch_utils, module_path, on_cluster
import flax.linen as nn
from jax import random
import nnaugment
from nnaugment.conventions import import_params, export_params
import torch


if not on_cluster:
    # dpath = os.path.join(module_path, "data/david_backdoors")  # local
    # use for testing with small dataset sizes (only works if rds storage is mounted):
    dpath = os.path.join(module_path, "/home/lauro/rds/model-zoo/")
else:
    dpath = "/rds/user/lsl38/rds-dsk-lab-eWkDxBhxBrQ/model-zoo/"  

model_dataset_dir = os.path.join(dpath, "cifar10")

inputs, targets, get_pytorch_model = torch_utils.load_input_and_target_weights(
    model=torch_utils.CNNMedium(),
    num_models=50,
    data_dir=model_dataset_dir,
    inputs_dirname="poison_noL1",
    targets_dirname="clean",
)

all_params = np.concatenate([inputs, targets], axis=0)


batch_size = 5
input_tensor = torch.randn(batch_size, 3, 32, 32)
rng = np.random.default_rng(0)
layers_to_permute=[f'Conv2d_{i}' for i in range(6)] + ['Dense_6']


@pytest.mark.parametrize("params", all_params)
def test_weight_augmentation(params):
    params = import_params(params, from_naming_scheme="pytorch")
    augmented_params = nnaugment.random_permutation(
        params,
        layers_to_permute=layers_to_permute,
        rng=rng,
    )

    # Check non-equality of augmented model's parameters against the original
    for name, layer in augmented_params.items():
        if name in layers_to_permute:
            assert not np.allclose(layer["kernel"], params[name]["kernel"], rtol=5e-2), \
                f"Parameters {name} are almost identical after augmentation."


    model = get_pytorch_model(export_params(params))
    model.eval()
    perm_model = get_pytorch_model(export_params(augmented_params))
    perm_model.eval()

    # Feed the tensor to the model
    with torch.no_grad():
        output = model(input_tensor)
        perm_output = perm_model(input_tensor)
        assert torch.allclose(output, perm_output, rtol=5e-2), \
            ("Outputs differ after weight augmentation. "
             f"Differences: {torch.abs(output - perm_output)}")



@pytest.mark.parametrize("params", all_params)
def test_determinism(params):
    """Test that the augmentation is deterministic, ie I can
    get the same permutation by setting the seed."""
    params = import_params(params, from_naming_scheme="pytorch")

    rng_state = rng.bit_generator.state

    rng_0 = np.random.default_rng()
    rng_1 = np.random.default_rng()

    rng_0.bit_generator.state = rng_state
    rng_1.bit_generator.state = rng_state

    augmented_params_0 = nnaugment.random_permutation(
        params,
        layers_to_permute=layers_to_permute,
        rng=rng_0,
    )


    augmented_params_1 = nnaugment.random_permutation(
        params,
        layers_to_permute=layers_to_permute,
        rng=rng_1,
    )

    # Check non-equality of augmented model's parameters against the original
    for name, layer in augmented_params_0.items():
        comparison_layer = augmented_params_1[name]
        assert np.allclose(layer["kernel"], comparison_layer["kernel"], rtol=1e-3), \
            f"Parameters in {name} differ despite same seed."

        assert np.allclose(layer["bias"], comparison_layer["bias"], rtol=1e-3), \
            f"Parameters in {name} differ despite same seed."