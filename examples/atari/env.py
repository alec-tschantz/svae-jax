import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize


def sample_data(env_name, num_samples, length, crop=(0, 0, 210, 160), resize_shape=(84, 84)):
    env = gym.make(env_name)

    trajectories, actions = [], []

    for _ in range(num_samples):
        single_trajectory, single_actions = [], []

        env.reset()
        for _ in range(length):
            action = env.action_space.sample()
            obs, reward, _, _, _ = env.step(action)

            cropped_obs = obs[crop[0] : crop[2], crop[1] : crop[3]]
            gray_obs = rgb2gray(cropped_obs)
            binary_obs = np.where(gray_obs > 0.5, 1.0, 0.0)
            resized_obs = resize(binary_obs, resize_shape, anti_aliasing=True)

            single_actions.append(action)
            single_trajectory.append(resized_obs)

        trajectories.append(single_trajectory)
        actions.append(single_actions)

    return np.array(trajectories), np.array(actions)


def plot_data(sample):
    length = sample.shape[0]
    plt.figure(figsize=(15, length * 3))

    for i in range(length):
        plt.subplot(1, length, i + 1)
        plt.imshow(sample[i], cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
