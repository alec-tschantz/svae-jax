from jax import numpy as jnp
from jax.scipy.special import digamma, gammaln


def expected_stats(nat_param):
    alpha = nat_param + 1
    return digamma(alpha) - digamma(jnp.sum(alpha, -1, keepdims=True))


def log_partition(nat_param):
    alpha = nat_param + 1
    return jnp.sum(jnp.sum(gammaln(alpha), -1) - gammaln(jnp.sum(alpha, -1)))
