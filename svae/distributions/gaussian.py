from functools import partial
from jax import numpy as jnp, random as jr


from svae.utils import T


def expected_stats(nat_param):
    neghalfJ, h, _, _ = unpack_dense(nat_param)
    J = -2 * neghalfJ
    Ex = jnp.linalg.solve(J, h)
    ExxT = jnp.linalg.inv(J) + Ex[..., None] * Ex[..., None, :]
    En = jnp.ones(J.shape[0]) if J.ndim == 3 else 1.0
    return pack_dense(ExxT, Ex, En, En)


def log_partition(nat_param):
    neghalfJ, h, a, b = unpack_dense(nat_param)
    J = -2 * neghalfJ
    L = jnp.linalg.cholesky(J)
    return (
        1.0 / 2 * jnp.sum(h * jnp.linalg.solve(J, h))
        - jnp.sum(jnp.log(jnp.diagonal(L, axis1=-1, axis2=-2)))
        + jnp.sum(a + b)
    )


def natural_sample(key, nat_param, num_samples):
    neghalfJ, h, _, _ = unpack_dense(nat_param)
    sample_shape = h.shape + (num_samples,)
    J = -2 * neghalfJ
    L = jnp.linalg.cholesky(J)
    subkey, key = jr.split(key)
    noise = jnp.linalg.solve(T(L), jr.normal(subkey, sample_shape))
    return jnp.linalg.solve(J, h)[..., None, :] + T(noise)


vs = partial(jnp.concatenate, axis=-2)
hs = partial(jnp.concatenate, axis=-1)


def pack_dense(A, b, *args):
    """Pack Gaussian natural parameters and statistics into a dense ndarray."""
    leading_dim, N = b.shape[:-1], b.shape[-1]
    z1 = jnp.zeros(leading_dim + (N, 1))
    z2 = jnp.zeros(leading_dim + (1, 1))
    c, d = args if args else (z2, z2)

    if A.ndim == b.ndim:
        A = A[..., None] * jnp.eye(N)[None, ...]
    b = b[..., None]
    c = jnp.reshape(c, leading_dim + (1, 1))
    d = jnp.reshape(d, leading_dim + (1, 1))

    packed = vs(
        (
            hs((A, b, z1)),
            hs((T(z1), c, z2)),
            hs((T(z1), z2, d)),
        )
    )
    return packed


def unpack_dense(arr):
    N = arr.shape[-1] - 2
    return (
        arr[..., :N, :N],
        arr[..., :N, N],
        arr[..., N, N],
        arr[..., N + 1, N + 1],
    )
