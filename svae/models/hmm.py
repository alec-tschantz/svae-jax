import jax
from jax.scipy.special import logsumexp
from jax import numpy as jnp, random as jr

from svae.utils import flat, unbox
from svae.distributions import dirichlet


def run_inference(key, prior_natparam, global_natparam, nn_potentials, num_samples):
    stats, local_natparam, local_kl = local_meanfield(global_natparam, nn_potentials)
    sample_key, key = jr.split(key)
    samples = gumbel_softmax(sample_key, local_natparam, temperature=1.0, hard=False, num_samples=num_samples)
    global_kl = prior_kl(global_natparam, prior_natparam)
    return samples, unbox(stats), global_kl, local_kl


def rollout(key, global_natparam, nn_potential, num_steps):
    N = global_natparam.shape[0]

    global_params = dirichlet.expected_stats(global_natparam)

    log_transition_probs = jax.nn.log_softmax(global_params, axis=-1)
    log_q_t = jax.nn.log_softmax(nn_potential)
    logits = [log_q_t]

    for t in range(1, num_steps):
        log_q_prev_plus_trans = log_q_t[:, None] + log_transition_probs
        log_q_t_new = logsumexp(log_q_prev_plus_trans, axis=0)
        log_q_t_new = log_q_t_new - logsumexp(log_q_t_new)
        logits.append(log_q_t_new)

    logits = jnp.stack(logits, axis=0)
    return logits


def local_meanfield(global_natparams, node_potentials):
    B, T, N = node_potentials.shape

    global_params = dirichlet.expected_stats(global_natparams)
    log_transition_probs = jax.nn.log_softmax(global_params, axis=-1)
    log_transition_probs = jnp.expand_dims(log_transition_probs, axis=0)
    log_transition_probs = jnp.repeat(log_transition_probs, B, axis=0)

    log_pi = jnp.full((B, N), -jnp.log(N))
    log_alpha = forward_filter(log_transition_probs, log_pi, node_potentials)

    log_beta_init = jnp.zeros((B, N))
    log_beta = backward_smooth(log_transition_probs, node_potentials, log_beta_init)

    log_q_z = log_alpha + log_beta
    log_q_z = log_q_z - logsumexp(log_q_z, axis=-1, keepdims=True)

    E_q_zzT = compute_local_stats(log_alpha, log_beta, log_transition_probs, node_potentials)
    local_kl = compute_local_kl(log_q_z, log_transition_probs, node_potentials)

    return E_q_zzT, log_q_z, local_kl


def forward_filter(log_transition_probs, log_pi, node_potentials):
    B, T, N = node_potentials.shape
    log_alpha = jnp.zeros((B, T, N))
    log_alpha = log_alpha.at[:, 0, :].set(log_pi + node_potentials[:, 0, :])

    def step_forward(t, log_alpha):
        log_alpha_prev = log_alpha[:, t - 1, :]
        log_sum = logsumexp(log_alpha_prev[:, :, None] + log_transition_probs, axis=1)
        log_alpha = log_alpha.at[:, t, :].set(node_potentials[:, t, :] + log_sum)
        return log_alpha

    log_alpha = jax.lax.fori_loop(1, T, step_forward, log_alpha)
    return log_alpha


def backward_smooth(log_transition_probs, node_potentials, log_beta_init):
    B, T, N = node_potentials.shape
    log_beta = jnp.zeros((B, T, N))
    log_beta = log_beta.at[:, T - 1, :].set(log_beta_init)

    def step_backward(t, log_beta):
        log_beta_next = log_beta[:, t + 1, :]
        log_beta_expanded = log_beta_next[:, None, :]
        node_potentials_expanded = node_potentials[:, t + 1, :][:, None, :]
        log_sum = logsumexp(log_beta_expanded + log_transition_probs + node_potentials_expanded, axis=2)
        log_beta = log_beta.at[:, t, :].set(log_sum)
        return log_beta

    log_beta = jax.lax.fori_loop(T - 2, -1, lambda t, lb: step_backward(t, lb), log_beta)
    return log_beta


def compute_local_stats(log_alpha, log_beta, log_transition_probs, node_potentials):
    B, T, N = log_alpha.shape

    def compute_joint(t, E_q_zzT):
        log_alpha_prev = log_alpha[:, t - 1, :]
        log_beta_t = log_beta[:, t, :]
        node_potentials_t = node_potentials[:, t, :]

        log_alpha_prev_expanded = log_alpha_prev[:, :, None]
        log_beta_t_expanded = log_beta_t[:, None, :]
        node_potentials_t_expanded = node_potentials_t[:, None, :]

        joint_log = log_alpha_prev_expanded + log_transition_probs + node_potentials_t_expanded + log_beta_t_expanded
        joint_log -= logsumexp(joint_log, axis=(1, 2), keepdims=True)
        joint = jnp.exp(joint_log)

        E_q_zzT = E_q_zzT.at[:, t - 1, :, :].set(joint)
        return E_q_zzT

    E_q_zzT = jnp.zeros((B, T - 1, N, N))
    E_q_zzT = jax.lax.fori_loop(1, T, compute_joint, E_q_zzT)
    return E_q_zzT.sum((0, 1))


def compute_local_kl(log_q_z, log_transition_probs, node_potentials):
    B, T, N = node_potentials.shape

    log_p_z_t = jnp.full((B, N), -jnp.log(N)) + node_potentials[:, 0, :]

    def forward(t, log_p_z_t):
        log_p_z_t = logsumexp(log_p_z_t[:, :, None] + log_transition_probs, axis=1) + node_potentials[:, t, :]
        return log_p_z_t

    log_p_z_t = jax.lax.fori_loop(1, T, forward, log_p_z_t)
    log_p_z = log_p_z_t - logsumexp(log_p_z_t, axis=-1, keepdims=True)
    log_p_z_expanded = log_p_z[:, None, :]

    kl_per_element = jnp.exp(log_q_z) * (log_q_z - log_p_z_expanded)
    return jnp.sum(kl_per_element)


def init_pgm_param(key, N, alpha):
    dirichlet_natparam = alpha * jnp.eye(N) + jr.uniform(key, shape=(N, N))
    return dirichlet_natparam


def prior_kl(global_natparam, prior_natparam):
    expected_stats = flat(prior_expected_stats(global_natparam))
    natparam_difference = flat(global_natparam) - flat(prior_natparam)
    log_partition_difference = prior_log_partition(global_natparam) - prior_log_partition(prior_natparam)
    return jnp.dot(natparam_difference, expected_stats) - log_partition_difference


def prior_expected_stats(natparam):
    return dirichlet.expected_stats(natparam)


def prior_log_partition(natparam):
    return dirichlet.log_partition(natparam)
