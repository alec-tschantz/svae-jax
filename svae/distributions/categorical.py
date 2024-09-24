from jax import numpy as jnp
from jax.nn import softmax
from jax.scipy.special import logsumexp

expected_stats = softmax


def log_partition(nat_param):
    return jnp.sum(logsumexp(nat_param, axis=-1))
