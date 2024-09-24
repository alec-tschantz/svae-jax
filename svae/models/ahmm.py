import jax
from jax.scipy.special import logsumexp
from jax import numpy as jnp, random as jr

from svae.utils import flat, unbox
from svae.distributions import dirichlet


def run_inference(key, prior_natparam, global_natparam, nn_potentials, actions, num_samples=1):
    sample_key, key = jr.split(key)
    stats, local_natparam, local_kl = local_meanfield(global_natparam, nn_potentials, actions)
    samples = gumbel_softmax(local_natparam, sample_key)
    global_kl = prior_kl(global_natparam, prior_natparam)
    return samples, unbox(stats), global_kl, local_kl


# def rollout(natparams, node_potential, actions):
#     init_params, trans_params = prior_expected_stats(natparams)
#     T = actions.shape[0]

#     logits = [node_potential + init_params]
#     for t in range(T - 1):
#         trans_t = trans_params[actions[t + 1]]
#         next_logits = logsumexp(logits[-1][:, None] + trans_t, axis=0)
#         logits.append(next_logits)
#     return jnp.stack(logits)


def rollout(natparams, node_potential, actions):
    T = actions.shape[0]
    # check implementation of prior expected stats
    init_params, trans_params = prior_expected_stats(natparams)

    init_probs = jax.nn.softmax(init_params)
    node_potential_probs = jax.nn.softmax(node_potential)
    trans_probs = jax.nn.softmax(trans_params, -1)
    print("trans_probs")
    print(trans_probs[0])
    print(trans_probs[1])
    print(trans_probs[2])
    print(trans_probs[3])

    probs = [jax.nn.softmax(node_potential_probs * init_probs * 100)]

    for t in range(T - 1):
        trans_t = trans_probs[actions[t + 1]]
        next_probs = probs[-1] @ trans_t
        next_probs = jax.nn.softmax(next_probs * 100)
        probs.append(next_probs)

    return jnp.stack(probs)


def local_meanfield(global_natparams, node_potentials, actions):
    init_params, trans_params = prior_expected_stats(global_natparams)
    alpha = forward_filter(init_params, trans_params, node_potentials, actions)
    beta = backward_smooth(trans_params, node_potentials, actions)
    log_post, expected_states, expected_transitions, log_normalizer = expected_statistics(
        init_params, trans_params, node_potentials, alpha, beta, actions
    )
    stats = (expected_states[0], expected_transitions)
    return stats, log_post, log_normalizer


def forward_filter(init_params, trans_params, node_potentials, actions):
    B, T, N = node_potentials.shape
    log_alpha = jnp.zeros((B, T, N))
    log_alpha = log_alpha.at[:, 0, :].set(init_params[None, :] + node_potentials[:, 0, :])
    for t in range(T - 1):
        trans_t = trans_params[actions[:, t + 1]]
        trans_probs = log_alpha[:, t, :, None] + trans_t
        trans_probs = trans_probs + node_potentials[:, t + 1, None, :]
        log_alpha = log_alpha.at[:, t + 1, :].set(logsumexp(trans_probs, axis=1))
    return log_alpha


def backward_smooth(trans_params, node_potentials, actions):
    B, T, N = node_potentials.shape
    log_beta = jnp.zeros((B, T, N))
    log_beta = log_beta.at[:, T - 1, :].set(0.0)
    for t in reversed(range(T - 1)):
        trans_t = trans_params[actions[:, t + 1]]
        probs = trans_t + node_potentials[:, t + 1, None, :] + log_beta[:, t + 1, None, :]
        log_beta = log_beta.at[:, t, :].set(logsumexp(probs, axis=2))
    return log_beta


def expected_statistics(init_params, trans_params, node_potentials, log_alpha, log_beta, actions):
    B, T, N = node_potentials.shape
    A = trans_params.shape[0]

    log_normalizer = logsumexp(log_alpha[:, T - 1, :], axis=-1)
    log_posterior = log_alpha + log_beta - log_normalizer[:, None, None]
    expected_states = jnp.exp(log_posterior)

    log_transitions = (
        log_alpha[:, :-1, :, None]
        + trans_params[actions[:, 1:], :, :]
        + node_potentials[:, 1:, None, :]
        + log_beta[:, 1:, None, :]
        - log_normalizer[:, None, None, None]
    )
    expected_transitions = jnp.exp(log_transitions)
    expected_transitions_flat = expected_transitions.reshape(-1, N, N)
    actions_flat = actions[:, 1:].reshape(-1)
    expected_transitions_total = jnp.zeros((A, N, N))
    expected_transitions_total = expected_transitions_total.at[actions_flat, :, :].add(expected_transitions_flat)

    expected_states = expected_states.sum(0)
    log_normalizer = logsumexp(log_alpha[:, T - 1, :])

    return log_posterior, expected_states, expected_transitions_total, log_normalizer


def init_pgm_param(key, N, A, alpha):
    transition_key, key = jr.split(key)
    # transition_natparam = alpha * jnp.eye(N)[None, :, :] + jr.uniform(transition_key, shape=(A, N, N))
    transition_natparam = alpha * jnp.eye(N)[None, :, :] + jnp.ones((A, N, N)) + 1e-9
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
    init_stats = dirichlet.expected_stats(init_natparam)
    expected_stats = lambda trans_natparam_a: jax.vmap(dirichlet.expected_stats)(trans_natparam_a)
    expected_stats_trans = jax.vmap(expected_stats)(trans_natparam)
    return init_stats, expected_stats_trans


def prior_log_partition(natparam):
    init_natparam, trans_natparam = natparam
    log_partition_init = dirichlet.log_partition(init_natparam)
    log_partition = lambda trans_natparam_a: dirichlet.log_partition(trans_natparam_a).sum()
    log_partition_trans = jax.vmap(log_partition)(trans_natparam).sum()
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
