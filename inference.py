from functools import partial

import jax
from jax.flatten_util import ravel_pytree
from jax import numpy as jnp, random as jr, value_and_grad, tree_util

import gaussian
from utils import split_into_batches, get_num_datapoints, flat


def sgd(key, gradfun, init_params, num_iters, step_size):
    params = init_params
    for i in range(num_iters):
        grads = gradfun(key, params, i)
        params = tree_util.tree_map(lambda p, g: p - step_size * g, params, grads)
    return params


def make_gradfun(
    key, run_inference, encoder, decoder, pgm_prior, data, batch_size, num_samples, natgrad_scale=1.0, callback=None
):

    _, unflat = ravel_pytree(pgm_prior)
    data_key, key = jr.split(key)
    num_datapoints = get_num_datapoints(data)
    data_batches, num_batches = split_into_batches(data_key, data, batch_size)
    get_batch = lambda i: data_batches[i % num_batches]
    saved = lambda: None

    def mc_elbo(key, pgm_params, decoder_params, encoder_params, i):
        infer_key, key = jr.split(key)

        nn_potential = encoder(encoder_params, get_batch(i))
        samples, saved.stats, global_kl, local_kl = run_inference(
            infer_key, pgm_prior, pgm_params, nn_potential, num_samples
        )

        return (
            num_batches * decoder(decoder_params, samples, get_batch(i)) - global_kl - num_batches * local_kl
        ) / num_datapoints

    def gradfun(key, params, i):
        pgm_params, decoder_params, encoder_params = params
        objective = lambda decoder_params, encoder_params: -mc_elbo(key, pgm_params, decoder_params, encoder_params, i)
        elbo, (decoder_grad, encoder_grad) = value_and_grad(objective, argnums=(0, 1))(decoder_params, encoder_params)

        pgm_natgrad = (
            -natgrad_scale / num_datapoints * (flat(pgm_prior) + num_batches * flat(saved.stats) - flat(pgm_params))
        )

        grad = (unflat(pgm_natgrad), decoder_grad, encoder_grad)

        if callback:
            callback(i, elbo, params, grad)
        return grad

    return gradfun
