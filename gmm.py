from jax import numpy as jnp, random as jr

import gaussian, niw, dirichlet, categorical
from utils import unbox, normalize, flat


def run_inference(key, prior_natparam, global_natparam, nn_potentials, num_samples):
    meanfield_key, key = jr.split(key)
    _, stats, local_natparam, local_kl = local_meanfield(meanfield_key, global_natparam, nn_potentials)
    sample_key, key = jr.split(key)
    samples = gaussian.natural_sample(sample_key, local_natparam[1], num_samples)
    global_kl = prior_kl(global_natparam, prior_natparam)
    return samples, unbox(stats), global_kl, local_kl


def init_pgm_param(key, K, N, alpha, niw_conc=10.0, random_scale=0.0):
    def init_niw_natparam(subkey, N):
        nu, S, m, kappa = N + niw_conc, (N + niw_conc) * jnp.eye(N), jnp.zeros(N), niw_conc
        m = jnp.zeros(N) + random_scale * jr.normal(subkey, shape=m.shape)
        return niw.standard_to_natural(S, m, kappa, nu)

    dirichlet_key, key = jr.split(key)
    dirichlet_natparam = alpha * jr.uniform(dirichlet_key, shape=(K,))

    niw_key, key = jr.split(key)
    subkeys = jr.split(niw_key, K)
    niw_natparam = jnp.stack([init_niw_natparam(subkeys[i], N) for i in range(K)])

    return dirichlet_natparam, niw_natparam


def make_encoder_decoder(key, encode, decode):
    def encode_mean(data, nat_param, encoder_params):
        nn_potentials = encode(encoder_params, data)
        (_, gaussian_stats), _, _, _ = local_meanfield(key, nat_param, nn_potentials)
        _, Ex, _, _ = gaussian.unpack_dense(gaussian_stats)
        return Ex

    def decode_mean(z, phi):
        mu, _ = decode(z, phi)
        return mu.mean(axis=1)

    return encode_mean, decode_mean


def local_meanfield(key, global_natparam, node_potentials):
    dirichlet_natparam, niw_natparams = global_natparam
    node_potentials = gaussian.pack_dense(*node_potentials)

    label_global = dirichlet.expected_stats(dirichlet_natparam)
    gaussian_globals = niw.expected_stats(niw_natparams)

    label_stats = meanfield_fixed_point(key, label_global, gaussian_globals, unbox(node_potentials))

    gaussian_natparam, gaussian_stats, gaussian_kl = gaussian_meanfield(gaussian_globals, node_potentials, label_stats)
    label_natparam, label_stats, label_kl = label_meanfield(label_global, gaussian_globals, gaussian_stats)

    dirichlet_stats = jnp.sum(label_stats, 0)
    niw_stats = jnp.tensordot(label_stats, gaussian_stats, [0, 0])

    local_stats = label_stats, gaussian_stats
    prior_stats = dirichlet_stats, niw_stats
    natparam = label_natparam, gaussian_natparam
    kl = label_kl + gaussian_kl

    return local_stats, prior_stats, natparam, kl


def meanfield_fixed_point(key, label_global, gaussian_globals, node_potentials, tol=1e-3, max_iter=100):
    kl = jnp.inf
    label_stats = initialize_meanfield(key, label_global, node_potentials)
    for _ in range(max_iter):
        gaussian_natparam, gaussian_stats, gaussian_kl = gaussian_meanfield(
            gaussian_globals, node_potentials, label_stats
        )
        label_natparam, label_stats, label_kl = label_meanfield(label_global, gaussian_globals, gaussian_stats)

        gaussian_global_potentials = jnp.tensordot(label_stats, gaussian_globals, [1, 0])
        linear_difference = gaussian_natparam - gaussian_global_potentials - node_potentials
        gaussian_kl = gaussian_kl + jnp.tensordot(linear_difference, gaussian_stats, 3)

        kl, prev_kl = label_kl + gaussian_kl, kl
        if abs(kl - prev_kl) < tol:
            break

    return label_stats


def gaussian_meanfield(gaussian_globals, node_potentials, label_stats):
    global_potentials = jnp.tensordot(label_stats, gaussian_globals, [1, 0])
    nat_param = node_potentials + global_potentials
    stats = gaussian.expected_stats(nat_param)
    kl = jnp.tensordot(node_potentials, stats, 3) - gaussian.log_partition(nat_param)
    return nat_param, stats, kl


def label_meanfield(label_global, gaussian_globals, gaussian_stats):
    node_potentials = jnp.tensordot(gaussian_stats, gaussian_globals, [[1, 2], [1, 2]])
    nat_param = node_potentials + label_global
    stats = categorical.expected_stats(nat_param)
    kl = jnp.tensordot(stats, node_potentials) - categorical.log_partition(nat_param)
    return nat_param, stats, kl


def initialize_meanfield(key, label_global, node_potentials):
    meanfield_key, key = jr.split(key)
    T, K = node_potentials.shape[0], label_global.shape[0]
    random_values = jr.uniform(meanfield_key, shape=(T, K))
    return normalize(random_values)


def prior_kl(global_natparam, prior_natparam):
    expected_stats = flat(prior_expected_stats(global_natparam))
    natparam_difference = flat(global_natparam) - flat(prior_natparam)
    log_partition_difference = prior_log_partition(global_natparam) - prior_log_partition(prior_natparam)
    return jnp.dot(natparam_difference, expected_stats) - log_partition_difference


def prior_expected_stats(gmm_natparam):
    dirichlet_natparam, niw_natparams = gmm_natparam
    dirichlet_expectedstats = dirichlet.expected_stats(dirichlet_natparam)
    niw_expectedstats = niw.expected_stats(niw_natparams)
    return dirichlet_expectedstats, niw_expectedstats


def prior_log_partition(gmm_natparam):
    dirichlet_natparam, niw_natparams = gmm_natparam
    return dirichlet.log_partition(dirichlet_natparam) + niw.log_partition(niw_natparams)
