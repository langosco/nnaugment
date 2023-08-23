import pytest
import numpy as np
import torch
from flax import linen as nn
import jax.numpy as jnp
import jax.random as random
import nnaugment
import os
from meta_transformer import torch_utils, module_path, on_cluster
from nnaugment.conventions import import_params, export_params

# load nets
if not on_cluster:
    dpath = os.path.join(module_path, "data/david_backdoors")  # local
else:
    dpath = "/rds/user/lsl38/rds-dsk-lab-eWkDxBhxBrQ/model-zoo/"  
model_dataset_dir = os.path.join(dpath, "mnist-cnns")

inputs, targets, get_pytorch_model = torch_utils.load_input_and_target_weights(
    model=torch_utils.CNNSmall(),
    num_models=500,
    data_dir=model_dataset_dir,
    inputs_dirname="poison",
    targets_dirname="clean",
)


all_params = np.concatenate([inputs, targets], axis=0)
batch_size = 5
input_tensor = torch.randn(batch_size, 1, 28, 28)


@pytest.mark.parametrize("params", all_params)
def test_weight_augmentation(params):
    params = import_params(params, from_naming_scheme="pytorch")
    augmented_params = nnaugment.random_permutation(
        params,
        layers_to_permute=['Conv2d_0', 'Conv2d_1', 'Dense_2'],
    )

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
