import operator
from functools import reduce

import jax
from jax import tree_util, numpy as jnp, random as jr

T = lambda X: jnp.swapaxes(X, axis1=-1, axis2=-2)

symmetrize = lambda X: (X + T(X)) / 2.0

normalize = lambda x: x / jnp.sum(x, axis=-1, keepdims=True)

outer = lambda x, y: x[..., :, None] * y[..., None, :]

identity = lambda x: x

sigmoid = lambda x: 1.0 / (1.0 + jnp.exp(-x))

relu = lambda x: jnp.maximum(x, 0.0)

log1pexp = lambda x: jnp.log1p(jnp.exp(x))

normalize = lambda x: x / jnp.sum(x, axis=-1, keepdims=True)

softmax = lambda x: normalize(jnp.exp(x - jnp.max(x, axis=-1, keepdims=True)))

unbox = lambda x: tree_util.tree_map(jax.lax.stop_gradient, x)

flat = lambda x: jax.flatten_util.ravel_pytree(x)[0]


def isarray(x):
    return isinstance(x, jnp.ndarray)


def compose(funcs):
    """Compose a sequence of functions."""

    def composition(x):
        for f in funcs:
            x = f(x)
        return x

    return composition


def inner(a, b):
    """Compute the inner product of two arrays."""
    return jnp.dot(jnp.ravel(a), jnp.ravel(b))


def add(a, b):
    """Element-wise addition of two arrays or nested structures."""
    return tree_util.tree_map(operator.add, a, b)


def sub(a, b):
    """Element-wise subtraction of two arrays or nested structures."""
    return tree_util.tree_map(operator.sub, a, b)


def mul(a, b):
    """Element-wise multiplication of two arrays or nested structures."""
    return tree_util.tree_map(operator.mul, a, b)


def div(a, b):
    """Element-wise division of two arrays or nested structures."""
    return tree_util.tree_map(operator.truediv, a, b)


def sqrt(a):
    """Element-wise square root of an array or nested structure."""
    return tree_util.tree_map(jnp.sqrt, a)


def square(a):
    """Element-wise square of an array or nested structure."""
    return tree_util.tree_map(lambda x: x**2, a)


def zeros_like(a):
    """Create an array of zeros with the same shape as the input."""
    return tree_util.tree_map(lambda x: jnp.zeros_like(x), a)


def randn_like(key, a):
    """Create a random normal array with the same shape as the input."""

    def generate_random(x):
        nonlocal key
        subkey, key = jax.random.split(key)
        return jax.random.normal(subkey, shape=x.shape)

    return tree_util.tree_map(generate_random, a)


def norm(x):
    """Compute the norm of an array or nested structure."""
    return jnp.sqrt(inner(x, x))


def scale(a, scalar):
    """Scale an array or nested structure by a scalar."""
    return tree_util.tree_map(lambda x: x * scalar, a)


def add_scalar(a, scalar):
    """Add a scalar to an array or nested structure."""
    return tree_util.tree_map(lambda x: x + scalar, a)


def flatten(a):
    """Flatten an array or nested structure into a 1D array."""
    leaves, _ = tree_util.tree_flatten(a)
    return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])


def shape(a):
    """Get the shape of an array or nested structure."""
    return tree_util.tree_map(lambda x: x.shape, a)


def allclose(a, b):
    """Check if two arrays or nested structures are element-wise close."""
    return tree_util.tree_reduce(
        lambda x, y: x and y,
        tree_util.tree_map(lambda x, y: jnp.allclose(x, y), a, b),
        True,
    )


def contract(a, b):
    """Contract two arrays or nested structures."""
    return tree_util.tree_reduce(
        lambda x, y: x + y,
        tree_util.tree_map(lambda x, y: jnp.sum(x * y), a, b),
        0.0,
    )


def get_num_datapoints(x):
    if isinstance(x, jnp.ndarray):
        return x.shape[0]
    elif isinstance(x, list):
        return sum(get_num_datapoints(item) for item in x)


def split_into_batches(key, data, batch_size):
    batch_key, key = jr.split(key)
    num_datapoints = get_num_datapoints(data)
    num_batches = num_datapoints // batch_size
    indices = jnp.arange(num_datapoints)
    shuffled_indices = jr.permutation(batch_key, indices)
    batches = [data[shuffled_indices[i * batch_size : (i + 1) * batch_size]] for i in range(num_batches)]
    return batches, num_batches
