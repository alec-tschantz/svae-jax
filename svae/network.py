from jax import numpy as jnp, random as jr
from functools import partial
from collections import defaultdict


from svae.utils import compose, sigmoid, relu, identity, log1pexp, isarray


def rand_partial_isometry(key, m, n):
    d = max(m, n)
    key_matrix = key
    A = jr.normal(key_matrix, shape=(d, d))
    Q, _ = jnp.linalg.qr(A)
    return Q[:m, :n]


def _make_ravelers(input_shape):
    ravel = lambda inputs: jnp.reshape(inputs, (-1, input_shape[-1]))
    unravel = lambda outputs: jnp.reshape(outputs, input_shape[:-1] + (-1,))
    return ravel, unravel


def layer(nonlin, W, b):
    """Create a layer function given nonlinearity, weights, and biases."""

    def layer_function(inputs):
        return nonlin(jnp.matmul(inputs, W) + b)

    return layer_function


def init_layer_random(key, d_in, d_out, scale=1e-2):
    """Initialize a layer with random weights and biases."""
    key_W, key_b = jr.split(key)
    W = scale * jr.normal(key_W, shape=(d_in, d_out))
    b = scale * jr.normal(key_b, shape=(d_out,))
    return W, b


def init_layer_partial_isometry(key, d_in, d_out):
    """Initialize a layer with a partial isometry matrix."""
    key_W, key_b = jr.split(key)
    W = rand_partial_isometry(key_W, d_in, d_out)
    b = jr.normal(key_b, shape=(d_out,))
    return W, b


def init_layer(key, d_in, d_out, init_fn=None):
    """Initialize a layer using the specified initialization function."""
    if init_fn is None:
        return init_layer_random(key, d_in, d_out)
    else:
        return init_fn(key, d_in, d_out)


def gaussian_mean(inputs, sigmoid_mean=False):
    """Compute Gaussian mean and variance from inputs."""
    mu_input, sigmasq_input = jnp.split(inputs, 2, axis=-1)
    if sigmoid_mean:
        mu = sigmoid(mu_input)
    else:
        mu = mu_input
    sigmasq = log1pexp(sigmasq_input)
    return mu, sigmasq


def gaussian_info(inputs):
    """Compute Gaussian information parameters from inputs."""
    J_input, h = jnp.split(inputs, 2, axis=-1)
    J = -0.5 * log1pexp(J_input)
    return J, h


def _mlp(nonlinearities):
    """Create an MLP function given nonlinearities."""

    def mlp_function(params, inputs):
        ravel, unravel = _make_ravelers(inputs.shape)
        x = ravel(inputs)
        for nonlin, (W, b) in zip(nonlinearities, params):
            x = nonlin(jnp.matmul(x, W) + b)
        out = x
        if isarray(out):
            return unravel(out)
        else:
            return tuple(unravel(o) for o in out)

    return mlp_function


def init_mlp(key, d_in, layer_specs):
    """Initialize an MLP with specified layer specifications."""
    dims = [d_in] + [l[0] for l in layer_specs]
    nonlinearities = [l[1] for l in layer_specs]
    num_layers = len(layer_specs)
    keys = jr.split(key, num_layers)
    params = []
    for i in range(num_layers):
        d_in_i = dims[i]
        d_out_i = dims[i + 1]
        spec = layer_specs[i]
        init_fn = None
        if len(spec) > 2:
            init_fn = spec[2]
        params.append(init_layer(keys[i], d_in_i, d_out_i, init_fn))
    mlp_fn = _mlp(nonlinearities)
    return mlp_fn, params


def _diagonal_gaussian_loglike(x, mu, sigmasq):
    mu = mu if mu.ndim == 3 else mu[:, None, :]
    T, K, p = mu.shape
    assert x.shape == (T, p)
    return (
        -T * p / 2.0 * jnp.log(2 * jnp.pi)
        + (-0.5 * jnp.sum((x[:, None, :] - mu) ** 2 / sigmasq) - 0.5 * jnp.sum(jnp.log(sigmasq))) / K
    )


def make_loglike(gaussian_mlp):
    """Create a log-likelihood function from a Gaussian MLP."""

    def loglike(params, inputs, targets):
        outputs = gaussian_mlp(params, inputs)
        return _diagonal_gaussian_loglike(targets, *outputs)

    return loglike


gaussian_mlp_types = {gaussian_mean: "mean", gaussian_info: "info"}


def gaussian_mlp_type(layer_specs):
    """Determine the type of Gaussian MLP ('mean' or 'info')."""
    return gaussian_mlp_types[layer_specs[-1][1]]


def _gresnet(mlp_type, mlp):
    """Create a Gaussian ResNet function."""

    def gresnet_function(params, inputs):
        mlp_params, (W, b1, b2) = params
        ravel, unravel = _make_ravelers(inputs.shape)
        if mlp_type == "mean":
            mu_mlp, sigmasq_mlp = mlp(mlp_params, inputs)
            mu_res = unravel(jnp.dot(ravel(inputs), W) + b1)
            sigmasq_res = log1pexp(b2)
            return (mu_mlp + mu_res, sigmasq_mlp + sigmasq_res)
        else:
            J_mlp, h_mlp = mlp(mlp_params, inputs)
            J_res = -0.5 * log1pexp(b2)
            h_res = unravel(jnp.dot(ravel(inputs), W) + b1)
            return (J_mlp + J_res, h_mlp + h_res)

    return gresnet_function


def init_gresnet(key, d_in, layer_specs):
    """Initialize a Gaussian ResNet."""
    d_out = layer_specs[-1][0] // 2
    key_res, key_mlp = jr.split(key)
    W_res = rand_partial_isometry(key_res, d_in, d_out)
    b1_res = jnp.zeros(d_out)
    b2_res = jnp.zeros(d_out)
    res_params = (W_res, b1_res, b2_res)
    mlp, mlp_params = init_mlp(key_mlp, d_in, layer_specs)
    mlp_type = gaussian_mlp_type(layer_specs)
    gresnet_fn = _gresnet(mlp_type, mlp)
    params = (mlp_params, res_params)
    return gresnet_fn, params
