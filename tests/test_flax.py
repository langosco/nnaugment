import pytest
from flax import linen as nn
import jax.numpy as jnp
import jax.random as random
import nnaugment
import numpy as np


bias_init = nn.initializers.normal()

class SimpleMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32, bias_init=bias_init)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=50, bias_init=bias_init)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=40, bias_init=bias_init)(x)
        return x


class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), bias_init=bias_init)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), bias_init=bias_init)(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.flatten()
        x = nn.Dense(features=15, name="Dense_2", bias_init=bias_init)(x)
        x = nn.Dense(features=10, name="Dense_3", bias_init=bias_init)(x)
        return x


def conv_block(x, features, index: int):
    x = nn.Conv(features=features, kernel_size=(3, 3), padding="SAME",
                name=f"Conv_{index}", bias_init=bias_init)(x)
    x = nn.LayerNorm(name=f"LayerNorm_{index+0.5}")(x)
    x = nn.relu(x)

    x = nn.Conv(features=features, kernel_size=(3, 3), padding="SAME", 
                name=f"Conv_{index+1}", bias_init=bias_init)(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.LayerNorm(name=f"LayerNorm_{index+1.5}")(x)
    x = nn.relu(x)
    return x


class CNN(nn.Module):
    """CNN for CIFAR-10 and SVHN."""

    @nn.compact
    def __call__(self, x):
        x = conv_block(x, features=16, index=0)
        x = conv_block(x, features=32, index=2)
        x = conv_block(x, features=64, index=4)

        x = jnp.max(x, axis=(-3, -2))  # GlobalMaxPool (x had shape (b h w c))
        x = nn.Dense(features=10, name="Dense_6")(x)
        return x


@pytest.mark.parametrize("seed, model, input_shape", [
    (0, SimpleMLP(), (32,)),
    (1, SimpleMLP(), (32,)),
    (2, SimpleCNN(), (28, 28, 1)),
    (3, SimpleCNN(), (32, 32, 3)),
    (4, CNN(), (32, 32, 3)),
    (5, CNN(), (32, 32, 3)),
    (7, CNN(), (32, 32, 3)),
    (8, CNN(), (32, 32, 3)),
])
def test_weight_augmentation(seed, 
                             model, 
                             input_shape):
    rng = random.PRNGKey(seed)
    subrng, rng = random.split(rng)

    # Initialize the weights
    x = jnp.ones(input_shape)
    variables = model.init(subrng, x)
    initial_output = model.apply(variables, x)


    # Augment weights
    if isinstance(model, SimpleMLP):
        layers_to_permute = ["Dense_0", "Dense_1"]
    elif isinstance(model, SimpleCNN):
        layers_to_permute = ["Conv_0", "Conv_1", "Dense_2"]
    elif isinstance(model, CNN):
        layers_to_permute = ["Conv_0", "Conv_1", "Conv_2", "Conv_3", "Conv_4", "Conv_5"]
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    augmented_params = nnaugment.random_permutation(
        rng,
        variables['params'], 
        layers_to_permute=layers_to_permute,
        convention="flax")
    augmented_variables = {'params': augmented_params}

    # Check non-equality of augmented model's parameters against the original
    for name, layer in augmented_params.items():
        if name in layers_to_permute:
            assert not np.allclose(layer["kernel"], variables['params'][name]["kernel"], rtol=5e-2), \
                f"Kernel parameters of model {type(model).__name__}, layer {name} are almost identical after augmentation."
            assert not np.allclose(layer["bias"], variables['params'][name]["bias"], rtol=5e-2), \
                f"Bias parameters of model {type(model).__name__}, layer {name} are almost identical after augmentation."

            assert not np.allclose(layer["kernel"], variables['params'][name]["kernel"], rtol=0.2), \
                f"Kernel parameters of model {type(model).__name__}, layer {name} are within +-20% size of each other after augmentation."
            assert not np.allclose(layer["bias"], variables['params'][name]["bias"], rtol=0.2), \
                f"Bias parameters of model {type(model).__name__}, layer {name} are within +-20% size of each other after augmentation."

    # Check for unchanged output
    augmented_output = model.apply(augmented_variables, x)
    assert jnp.allclose(initial_output, augmented_output, atol=1e-6), (
        "Outputs differ after weight augmentation."
    )
