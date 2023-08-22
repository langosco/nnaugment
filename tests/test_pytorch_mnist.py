import pytest
import torch
from flax import linen as nn
import jax.numpy as jnp
import jax.random as random
import nnaugment
import os
from meta_transformer import torch_utils, module_path
from nnaugment.conventions import import_params, export_params

# load nets
dpath = os.path.join(module_path, "data/david_backdoors")  # local

model_dataset_paths = {
    "mnist": "mnist-cnns",
    #"mnist": "mnist/models",  # old mnist checkpoints
    "cifar10": "cifar10",
    "svhn": "svhn",
}

model_dataset_paths = {
    k: os.path.join(dpath, v) for k, v in model_dataset_paths.items()
}

inputs, targets, get_pytorch_model = torch_utils.load_input_and_target_weights(
    model=torch_utils.CNNSmall(),
    num_models=10,
    data_dir=model_dataset_paths["mnist"],
    inputs_dirname="poison",
    targets_dirname="clean",
)


@pytest.mark.parametrize("params", inputs)
def test_weight_augmentation(params):
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    params = import_params(params, from_naming_scheme="pytorch")
    augmented_params = nnaugment.random_permutation(
        params,
        layers_to_permute=['Conv2d_0', 'Conv2d_1', 'Linear_2'],
    )

    model = get_pytorch_model(export_params(params))
    perm_model = get_pytorch_model(export_params(augmented_params))

    # Feed the tensor to the model
    output = model(input_tensor)
    perm_output = perm_model(input_tensor)
    assert torch.allclose(output, perm_output, atol=1e-6, rtol=1e-3), "Outputs differ after weight augmentation."