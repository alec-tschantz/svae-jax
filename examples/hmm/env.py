from typing import Tuple, Optional

import jax
from jax import numpy as jnp
import numpy as np

from matplotlib import pyplot as plt


class Env:
    def __init__(self, initial_position: Tuple[int, int] = (3, 3)):
        self.grid_size = 10
        self.square_size = 2
        self.actions = ["up", "down", "left", "right"]
        self.action_mapping = {"up": 0, "down": 1, "left": 2, "right": 3}

        x, y = initial_position
        self.initial_position = jnp.array(initial_position)

        self.move_deltas = jnp.array(
            [
                [0, -1],
                [0, 1],
                [-1, 0],
                [1, 0],
            ],
            dtype=jnp.int32,
        )

    def sample(self, key, num_samples: int, length: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        keys = jax.random.split(key, num_samples * length)
        actions = jax.vmap(lambda k: jax.random.randint(k, (), 0, 4))(keys)
        actions = actions.reshape((num_samples, length))

        def generate_sample(actions_seq):
            def step(carry, action):
                pos = carry
                delta = self.move_deltas[action]
                new_pos = pos + delta
                new_x = jnp.clip(new_pos[0], 0, self.grid_size - self.square_size)
                new_y = jnp.clip(new_pos[1], 0, self.grid_size - self.square_size)
                updated_pos = jnp.array([new_x, new_y])
                return updated_pos, updated_pos

            _, positions = jax.lax.scan(step, self.initial_position, actions_seq)
            images = jax.vmap(self._create_image)(positions)
            return images, actions_seq

        generate_sample_vmap = jax.vmap(generate_sample, in_axes=0, out_axes=(0, 0))
        images, actions = generate_sample_vmap(actions)
        return images, actions

    def _create_image(self, pos: jnp.ndarray) -> jnp.ndarray:
        x, y = pos
        xs = jnp.arange(self.grid_size)
        ys = jnp.arange(self.grid_size)
        grid_x, grid_y = jnp.meshgrid(xs, ys)

        mask = (grid_x >= x) & (grid_x < x + self.square_size) & (grid_y >= y) & (grid_y < y + self.square_size)
        image = jnp.where(mask, 1.0, 0.0)
        return image

    def plot_sample(self, images: jnp.ndarray, actions: Optional[jnp.ndarray] = None):
        length = images.shape[0]
        fig, axes = plt.subplots(1, length, figsize=(length * 2, 2))
        if length == 1:
            axes = [axes]
        for i in range(length):
            ax = axes[i]
            ax.imshow(np.array(images[i]), cmap="gray", vmin=0, vmax=1)
            if actions is not None:
                action_label = self.actions[actions[i]]
                ax.set_title(f"{action_label}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
