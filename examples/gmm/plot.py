import jax
from jax import numpy as jnp
from jax.scipy.stats import norm

import numpy as np

import matplotlib.pyplot as plt

from svae.utils import normalize
from svae.distributions import gaussian, niw, dirichlet
from svae.models.gmm import make_encoder_decoder


def make_plotter_2d(key, encode, decode, data, num_clusters, params):
    data_np = np.array(data)
    encode_mean, decode_mean = make_encoder_decoder(key, encode, decode)

    def plot_encoded_means(ax, params):
        pgm_params, loglike_params, recogn_params = params
        encoded_means = encode_mean(data, pgm_params, recogn_params)
        encoded_means_np = np.array(encoded_means)
        ax.plot(encoded_means_np[:, 0], encoded_means_np[:, 1], color="r", marker=".", linestyle="", alpha=0.5)

    def plot_ellipse(ax, alpha, mean, cov):
        alpha = alpha.item()
        t = np.linspace(0, 2 * np.pi, 100) % (2 * np.pi)
        circle = np.vstack((np.sin(t), np.cos(t)))

        mean_np = np.array(mean)
        cov_np = np.array(cov)

        ellipse = 2.0 * np.dot(np.linalg.cholesky(cov_np), circle) + mean_np[:, None]
        ax.plot(ellipse[0], ellipse[1], alpha=alpha, linestyle="-", linewidth=2)

    def get_component(niw_natparam):
        neghalfJ, h, _, _ = gaussian.unpack_dense(niw_natparam)
        J = -2 * neghalfJ
        mean = jnp.linalg.solve(J, h)
        cov = jnp.linalg.inv(J)
        return mean, cov

    def plot_components(ax, params):
        pgm_params, loglike_params, recogn_params = params
        dirichlet_natparams, niw_natparams = pgm_params

        normalize = lambda arr: jnp.minimum(1.0, arr / jnp.sum(arr) * num_clusters)
        weights = normalize(jnp.exp(dirichlet.expected_stats(dirichlet_natparams)))
        components = list(map(get_component, niw.expected_stats(niw_natparams)))

        for weight, (mu, Sigma) in zip(weights, components):
            weight_np = np.array(weight)
            mu_np = np.array(mu)
            Sigma_np = np.array(Sigma)
            plot_ellipse(ax, weight_np, mu_np, Sigma_np)

    def plot(params):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.axis("off")
        plot_encoded_means(ax, params)
        plot_components(ax, params)

        fig.tight_layout()
        plt.show()

    return plot
