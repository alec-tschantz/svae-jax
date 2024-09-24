import jax
from jax.scipy.special import logsumexp
from jax import numpy as jnp, random as jr

from svae.utils import flat, unbox
from svae.distributions import dirichlet


def run_inference(key, prior_natparam, global_natparam, nn_potentials, num_samples=1):
    sample_key, key = jr.split(key)
    stats, local_natparam, local_kl = local_meanfield(global_natparam, nn_potentials)
    samples = gumbel_softmax(local_natparam, sample_key)
    global_kl = prior_kl(global_natparam, prior_natparam)
    return samples, unbox(stats), global_kl, local_kl


def local_meanfield(global_natparams, node_potentials):
    init_params, trans_params = prior_expected_stats(global_natparams)

    alpha = forward_filter(init_params, trans_params, node_potentials)
    beta = backward_smooth(trans_params, node_potentials)
    log_post, expected_states, expected_transitions, log_normalizer = expected_statistics(
        init_params, trans_params, node_potentials, alpha, beta
    )
    stats = (expected_states[0], expected_transitions)
    return stats, log_post, log_normalizer


def forward_filter(init_params, trans_params, node_potentials):
    B, T, N = node_potentials.shape
    log_alpha = jnp.zeros((B, T, N))
    log_alpha = log_alpha.at[:, 0, :].set(init_params[None, :] + node_potentials[:, 0, :])
    for t in range(1, T):
        trans_probs = log_alpha[:, t - 1, :, None] + trans_params[None, :, :]
        log_alpha = log_alpha.at[:, t, :].set(node_potentials[:, t, :] + logsumexp(trans_probs, axis=1))
    return log_alpha


def backward_smooth(trans_params, node_potentials):
    B, T, N = node_potentials.shape
    log_beta = jnp.zeros((B, T, N))
    log_beta = log_beta.at[:, T - 1, :].set(0.0)
    for t in reversed(range(T - 1)):
        probs = trans_params[None, :, :] + node_potentials[:, t + 1, None, :] + log_beta[:, t + 1, None, :]
        log_beta = log_beta.at[:, t, :].set(logsumexp(probs, axis=2))
    return log_beta


def expected_statistics(init_params, trans_params, node_potentials, log_alpha, log_beta):
    B, T, N = node_potentials.shape
    log_normalizer = logsumexp(log_alpha[:, T - 1, :], axis=-1)
    log_posterior = log_alpha + log_beta - log_normalizer[:, None, None]
    expected_states = jnp.exp(log_posterior)
    log_transitions = (
        log_alpha[:, :-1, :, None]
        + trans_params[None, None, :, :]
        + node_potentials[:, 1:, None, :]
        + log_beta[:, 1:, None, :]
        - log_normalizer[:, None, None, None]
    )
    expected_transitions = jnp.exp(log_transitions)
    expected_transitions_total = expected_transitions.sum(axis=(0, 1))
    expected_states = expected_states.sum(0)
    log_normalizer = logsumexp(log_alpha[:, T - 1, :])
    return log_posterior, expected_states, expected_transitions_total, log_normalizer


def init_pgm_param(key, N, alpha):
    transition_key, key = jr.split(key)
    transition_natparam = alpha * jnp.eye(N) + jr.uniform(transition_key, shape=(N, N))

    initial_key, key = jr.split(key)
    initial_natparam = jnp.full(N, 1.0 / N)
    return initial_natparam, transition_natparam


def prior_kl(global_natparam, prior_natparam):
    expected_stats = flat(prior_expected_stats(global_natparam))
    natparam_difference = flat(global_natparam) - flat(prior_natparam)
    log_partition_difference = prior_log_partition(global_natparam) - prior_log_partition(prior_natparam)
    return jnp.dot(natparam_difference, expected_stats) - log_partition_difference


def prior_expected_stats(natparam):
    init_natparam, trans_natparam = natparam
    trans_stats = jax.vmap(dirichlet.expected_stats, in_axes=1)(trans_natparam)
    return dirichlet.expected_stats(init_natparam), trans_stats


def prior_log_partition(natparam):
    init_natparam, trans_natparam = natparam
    return dirichlet.log_partition(init_natparam) + sum(map(dirichlet.log_partition, trans_natparam))


def gumbel_softmax(logits, key, temperature=1.0):
    uniforms = jnp.clip(jr.uniform(key, logits.shape), a_min=1e-10, a_max=1.0)
    gumbels = -jnp.log(-jnp.log(uniforms))
    scores = (logits + gumbels) / temperature
    return jax.nn.softmax(scores, axis=-1)
