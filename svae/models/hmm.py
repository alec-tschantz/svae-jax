import jax
from jax.scipy.special import logsumexp
from jax import numpy as jnp, random as jr, lax, jit, tree_util

from svae.utils import flat, stop_gradient
from svae.distributions import dirichlet


@jit
def run_inference(key, prior_natparam, global_natparam, nn_potentials, num_samples, actions):
    sample_key, key = jr.split(key)
    stats, local_natparam, local_kl = local_meanfield(global_natparam, nn_potentials, actions)
    samples = gumbel_softmax(local_natparam, sample_key)
    global_kl = prior_kl(global_natparam, prior_natparam)
    return samples, stop_gradient(stats), global_kl, local_kl


def local_meanfield(global_natparams, node_potentials, actions):
    init_params, trans_params = prior_expected_stats(global_natparams)
    alpha = forward_filter(init_params, trans_params, node_potentials, actions)
    beta = backward_smooth(trans_params, node_potentials, actions)
    stats, log_posterior, log_normalizer = expected_statistics(
        init_params, trans_params, node_potentials, alpha, beta, actions
    )
    return stats, log_posterior, log_normalizer


def rollout(natparams, node_potential, actions):
    init_params, trans_params = prior_expected_stats(natparams)

    def scan_fn(logits_prev, action_t):
        next_logits = logsumexp(logits_prev[:, None] + trans_params[action_t], axis=0)
        return next_logits, next_logits

    logits_0 = node_potential + init_params
    _, logits_seq = lax.scan(scan_fn, logits_0, actions[:-1])
    return jnp.concatenate([logits_0[None, :], logits_seq], axis=0)


def forward_filter(init_params, trans_params, node_potentials, actions):
    B, T, N = node_potentials.shape

    def scan_fn(log_alpha_prev, t):
        sum_terms = log_alpha_prev[:, :, None] + trans_params[actions[:, t - 1]]
        log_alpha_t = logsumexp(sum_terms, axis=1) + node_potentials[:, t, :]
        return log_alpha_t, log_alpha_t

    log_alpha_0 = init_params + node_potentials[:, 0, :]
    _, log_alpha_seq = lax.scan(scan_fn, log_alpha_0, jnp.arange(1, T))
    log_alpha = jnp.concatenate([log_alpha_0[None, :, :], log_alpha_seq], axis=0)
    return log_alpha.swapaxes(0, 1)


def backward_smooth(trans_params, node_potentials, actions):
    B, T, N = node_potentials.shape

    def scan_fn(log_beta_next, t):
        sum_terms = trans_params[actions[:, t]] + node_potentials[:, t + 1, None, :] + log_beta_next[:, None, :]
        log_beta_t = logsumexp(sum_terms, axis=2)
        return log_beta_t, log_beta_t

    log_beta_T = jnp.zeros((B, N))
    _, log_beta_seq = lax.scan(scan_fn, log_beta_T, jnp.arange(T - 2, -1, -1))
    log_beta = jnp.concatenate([log_beta_seq[::-1], log_beta_T[None, :, :]], axis=0)
    return log_beta.swapaxes(0, 1)


def expected_statistics(init_params, trans_params, node_potentials, log_alpha, log_beta, actions):
    B, T, N = node_potentials.shape
    A = trans_params.shape[0]

    log_normalizer = logsumexp(log_alpha[:, -1, :], axis=-1)
    log_posterior = log_alpha + log_beta - log_normalizer[:, None, None]

    log_xi = (
        log_alpha[:, :-1, :, None]
        + trans_params[actions[:, :-1], :, :]
        + node_potentials[:, 1:, None, :]
        + log_beta[:, 1:, None, :]
        - log_normalizer[:, None, None, None]
    )

    posterior = jnp.exp(log_posterior)
    expected_initial = posterior[:, 0, :].sum(0)

    actions_flat = actions[:, :-1].reshape(-1)
    expected_transitions_flat = jnp.exp(log_xi).reshape(-1, N, N)

    expected_transitions = jnp.zeros((A, N, N))
    expected_transitions = expected_transitions.at[actions_flat, :, :].add(expected_transitions_flat)
    return (expected_initial, expected_transitions), log_posterior, log_normalizer.mean()


def init_pgm_param(key, N, A, alpha):
    initial_natparam = jnp.full(N, 1.0 / N)
    transition_natparam = alpha * jnp.ones((A, N, N))
    return initial_natparam, transition_natparam


def prior_kl(global_natparam, prior_natparam):
    expected_stats = flat(prior_expected_stats(global_natparam))
    natparam_difference = flat(global_natparam) - flat(prior_natparam)
    log_partition_difference = prior_log_partition(global_natparam) - prior_log_partition(prior_natparam)
    return jnp.dot(natparam_difference, expected_stats) - log_partition_difference


def prior_expected_stats(natparam):
    init_natparam, trans_natparam = natparam
    init_stats = dirichlet.expected_stats(init_natparam)
    trans_stats = jax.vmap(lambda a: jax.vmap(dirichlet.expected_stats)(a))(trans_natparam)
    return init_stats, trans_stats


def prior_log_partition(natparam):
    init_natparam, trans_natparam = natparam
    log_partition_init = dirichlet.log_partition(init_natparam)
    log_partition_trans = jax.vmap(lambda a: dirichlet.log_partition(a).sum())(trans_natparam).sum()
    return log_partition_init + log_partition_trans


def gumbel_softmax(logits, key, temperature=1.0):
    uniforms = jnp.clip(jr.uniform(key, logits.shape), a_min=1e-10, a_max=1.0)
    gumbels = -jnp.log(-jnp.log(uniforms))
    scores = (logits + gumbels) / temperature
    return jax.nn.softmax(scores, axis=-1)


def onehot_sample(logits, key):
    B, K = logits.shape
    sampled_indices = jr.categorical(key, logits, axis=-1)
    return jax.nn.one_hot(sampled_indices, K)
