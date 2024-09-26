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

stop_gradient = lambda x: tree_util.tree_map(jax.lax.stop_gradient, x)

flat = lambda x: jax.flatten_util.ravel_pytree(x)[0]


def isarray(x):
    return isinstance(x, jnp.ndarray)


def compose(funcs):
    def composition(x):
        for f in funcs:
            x = f(x)
        return x

    return composition


def get_num_datapoints(x):
    if isinstance(x, jnp.ndarray):
        return x.shape[0]
    elif isinstance(x, tuple):
        return get_num_datapoints(x[0])


def split_into_batches(key, data, batch_size):
    batch_key, key = jr.split(key)
    num_datapoints = get_num_datapoints(data)
    num_batches = num_datapoints // batch_size
    indices = jnp.arange(num_datapoints)
    shuffled_indices = jr.permutation(batch_key, indices)

    def batch_data(single_data):
        return [single_data[shuffled_indices[i * batch_size : (i + 1) * batch_size]] for i in range(num_batches)]

    if isinstance(data, tuple):
        batches = tuple(batch_data(d) for d in data)
    else:
        batches = batch_data(data)

    return batches, num_batches
