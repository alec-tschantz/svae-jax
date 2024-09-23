import jax
import jax.numpy as jnp
from jax import random
from functools import partial

from utils import identity, log1pexp, relu


def logits(x):
    return identity(x)


def gaussian_mean(x, sigmoid_mean=False):
    mu_input, sigmasq_input = jnp.split(x, 2, axis=-1)
    if sigmoid_mean:
        mu = sigmoid(mu_input)
    else:
        mu = mu_input
    sigmasq = log1pexp(sigmasq_input)
    return mu, sigmasq


# def init_layer(key, d_in, d_out):
#     key_W, key_b = random.split(key)
#     W = rand_partial_isometry(key_W, d_in, d_out)
#     b = random.normal(key_b, shape=(d_out,))
#     return W, b


def init_layer(key, d_in, d_out, scale=1e-2):
    key_W, key_b = random.split(key)
    W = scale * random.normal(key_W, shape=(d_in, d_out))
    b = scale * random.normal(key_b, shape=(d_out,))
    return W, b


def rand_partial_isometry(key, m, n):
    d = max(m, n)
    key_matrix = key
    A = random.normal(key_matrix, shape=(d, d))
    Q, _ = jnp.linalg.qr(A)
    return Q[:m, :n]


def mlp_forward(params, inputs, activations):
    x = inputs
    for (W, b), act in zip(params[:-1], activations[:-1]):
        x = act(jnp.dot(x, W) + b)
    W_out, b_out = params[-1]
    x = jnp.dot(x, W_out) + b_out
    return activations[-1](x)


def init_mlp(key, d_in, layer_specs):
    num_layers = len(layer_specs)
    keys = random.split(key, num_layers)
    params = []
    sizes = [d_in] + [size for size, _ in layer_specs]

    for i in range(num_layers):
        W, b = init_layer(keys[i], sizes[i], sizes[i + 1])
        params.append((W, b))

    activations = [act for _, act in layer_specs]

    def mlp_fn(params, inputs):
        return mlp_forward(params, inputs, activations)

    return mlp_fn, params


def gaussian_loglike(targets, mu, sigmasq):
    return -0.5 * jnp.sum(((targets - mu) ** 2) / sigmasq + jnp.log(sigmasq) + jnp.log(2 * jnp.pi))


def make_loglike(mlp):
    def loglike(params, inputs, targets):
        outputs = mlp(params, inputs)
        mu, sigmasq = outputs
        return gaussian_loglike(targets, mu, sigmasq)

    return loglike
