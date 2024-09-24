from functools import partial

import jax
from jax.flatten_util import ravel_pytree
from jax import numpy as jnp, random as jr, value_and_grad, tree_util

from svae.utils import split_into_batches, get_num_datapoints, flat


def make_gradfun(
    key, run_inference, encoder, loglike, pgm_prior, num_samples, num_datapoints, num_batches, natgrad_scale=1.0
):
    _, unflat = ravel_pytree(pgm_prior)
    saved = lambda: None

    def mc_elbo(key, pgm_params, decoder_params, encoder_params, batch):
        infer_key, key = jr.split(key)

        nn_potential = encoder(encoder_params, batch)
        samples, saved.stats, global_kl, local_kl = run_inference(
            infer_key, pgm_prior, pgm_params, nn_potential, num_samples
        )
        return (
            num_batches * loglike(decoder_params, samples, batch) - global_kl - num_batches * local_kl
        ) / num_datapoints

    def gradfun(params, batch):
        pgm_params, decoder_params, encoder_params = params
        objective = lambda decoder_params, encoder_params: -mc_elbo(
            key, pgm_params, decoder_params, encoder_params, batch
        )
        elbo, (decoder_grad, encoder_grad) = value_and_grad(objective, argnums=(0, 1))(decoder_params, encoder_params)

        pgm_natgrad = (
            -natgrad_scale / num_datapoints * (flat(pgm_prior) + num_batches * flat(saved.stats) - flat(pgm_params))
        )
        return unflat(pgm_natgrad), decoder_grad, encoder_grad

    return gradfun
