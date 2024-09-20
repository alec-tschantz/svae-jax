from jax import numpy as jnp
from jax.scipy.special import multigammaln, digamma

from utils import symmetrize, outer
from gaussian import pack_dense, unpack_dense


def expected_stats(nat_param, eps=1e-8):
    S, m, kappa, nu = natural_to_standard(nat_param)
    d = m.shape[-1]

    E_J = nu[..., None, None] * symmetrize(jnp.linalg.inv(S)) + eps * jnp.eye(d)
    E_h = jnp.matmul(E_J, m[..., None])[..., 0]
    E_hTJinvh = d / kappa + jnp.matmul(m[..., None, :], E_h[..., None])[..., 0, 0]
    E_logdetJ = (
        jnp.sum(digamma((nu[..., None] - jnp.arange(d)[None, ...]) / 2.0), -1) + d * jnp.log(2.0)
    ) - jnp.linalg.slogdet(S)[1]

    return pack_dense(-1.0 / 2 * E_J, E_h, -1.0 / 2 * E_hTJinvh, 1.0 / 2 * E_logdetJ)


def log_partition(nat_param):
    S, m, kappa, nu = natural_to_standard(nat_param)
    d = m.shape[-1]
    return jnp.sum(
        d * nu / 2.0 * jnp.log(2.0)
        + multigammaln(nu / 2.0, d)
        - nu / 2.0 * jnp.linalg.slogdet(S)[1]
        - d / 2.0 * jnp.log(kappa)
    )


def natural_to_standard(nat_param):
    A, b, kappa, nu = unpack_dense(nat_param)
    m = b / jnp.expand_dims(kappa, -1)
    S = A - outer(b, m)
    return S, m, kappa, nu


def standard_to_natural(S, m, kappa, nu):
    b = jnp.expand_dims(kappa, -1) * m
    A = S + outer(b, m)
    return pack_dense(A, b, kappa, nu)


def expected_standard_params(natparam):
    S, m, kappa, nu = natural_to_standard(natparam)
    expected_mean = m
    expected_covariance = S / kappa
    return expected_mean, expected_covariance
