from functools import partial


import jax
from jax import numpy as jnp, random as jr
from jax.scipy.special import logsumexp


identity = lambda x: x


def init_layer(key, d_in, d_out):
    key_W, key_b = jr.split(key)
    scale = jnp.sqrt(2.0 / (d_in + d_out))
    W = scale * jr.normal(key_W, shape=(d_in, d_out))
    b = jnp.zeros(d_out)
    return W, b


def mlp_forward(params, inputs, activations):
    x = inputs
    for (W, b), act in zip(params, activations):
        x = act(jnp.dot(x, W) + b)
    return x


def init_mlp(key, d_in, layer_specs):
    num_layers = len(layer_specs)
    keys = jr.split(key, num_layers)
    params = []
    sizes = [d_in] + [size for size, _ in layer_specs]
    activations = [act for _, act in layer_specs]

    for i in range(num_layers):
        W, b = init_layer(keys[i], sizes[i], sizes[i + 1])
        params.append((W, b))

    def mlp_fn(params, inputs):
        return mlp_forward(params, inputs, activations)

    return mlp_fn, params


def binary_cross_entropy(ouput, targets):
    output = jnp.clip(ouput, 1e-8, 1.0 - 1e-8)

    bce_loss = -(targets * jnp.log(ouput) + (1 - targets) * jnp.log(1 - ouput))
    return jnp.sum(bce_loss)


def make_loglike(mlp):

    def loglike(params, inputs, targets):
        outputs = mlp(params, inputs)
        return -binary_cross_entropy(outputs, targets)

    return loglike


def gumbel_softmax(logits, key, temperature=1.0):
    uniforms = jnp.clip(jr.uniform(key, logits.shape), a_min=1e-10, a_max=1.0)
    gumbels = -jnp.log(-jnp.log(uniforms))
    scores = (logits + gumbels) / temperature
    return jax.nn.softmax(scores, axis=-1)


def onehot_sample(logits, key):
    B, N, K = logits.shape
    reshaped_logits = logits.reshape(-1, K)
    sampled_indices = jr.categorical(key, reshaped_logits, axis=-1)
    sampled_indices = sampled_indices.reshape(B, N)
    return jax.nn.one_hot(sampled_indices, K)
