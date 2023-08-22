import pytest
from flax import linen as nn
import jax.numpy as jnp
import jax.random as random
import nnaugment


class SimpleMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32, bias_init=nn.initializers.normal())(x)
        x = nn.gelu(x)
        x = nn.Dense(features=50, bias_init=nn.initializers.normal())(x)
        x = nn.gelu(x)
        x = nn.Dense(features=40)(x)
        return x


class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.flatten()
        x = nn.Dense(features=15, name="Dense_2")(x)
        x = nn.Dense(features=10, name="Dense_3")(x)
        return x


@pytest.mark.parametrize("seed, model, input_shape", [
    (0, SimpleMLP(), (32,)),
    (1, SimpleMLP(), (32,)),
    (2, SimpleCNN(), (28, 28, 1)),
    (3, SimpleCNN(), (32, 32, 3)),
])
def test_weight_augmentation(seed, 
                             model, 
                             input_shape):
    rng = random.PRNGKey(seed)

    # Initialize the weights
    x = jnp.ones(input_shape)
    variables = model.init(rng, x)
    initial_output = model.apply(variables, x)

    # Augment weights
    if isinstance(model, SimpleMLP):
        layers_to_permute = ["Dense_0", "Dense_1"]
    elif isinstance(model, SimpleCNN):
        layers_to_permute = ["Conv_0", "Conv_1", "Dense_2"]
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    augmented_params = nnaugment.random_permutation(
        variables['params'], 
        layers_to_permute=layers_to_permute,
        convention="flax")
    augmented_variables = {'params': augmented_params}

    # Check for unchanged output
    augmented_output = model.apply(augmented_variables, x)
    assert jnp.allclose(initial_output, augmented_output, atol=1e-6), "Outputs differ after weight augmentation."