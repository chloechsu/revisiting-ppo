'''Environments used in simulation.'''
import gym
from gym.spaces.box import Box as Continuous
from gym.spaces.discrete import Discrete
import numpy as np


class DiscreteBanditEnv(gym.Env):
    '''
    A simple single-state environment with discrete actions.
    '''
    def __init__(self, avg_rewards, noise_std=1e-3, **kwargs):
        self.action_dim = len(avg_rewards)
        self.avg_rewards = avg_rewards
        self.obs_dim = 1
        self.noise_std = noise_std
        self.action_space = Discrete(self.action_dim)
        self.observation_space = Continuous(
                low=np.zeros(self.obs_dim, dtype=np.float32),
                high=np.ones(self.obs_dim, dtype=np.float32),
                dtype=np.float32) 
        self.reset()

    def reset(self):
        obs = np.random.rand(self.obs_dim)
        return obs

    def step(self, action):
        obs = np.random.rand(self.obs_dim)
        reward = self.avg_rewards[int(action)] + self.noise_std * np.random.normal()
        done = True
        return obs, reward, done, {}


class SingleSmallPeakEnv(gym.Env):
    '''
    A simple single-state environment with continuous actions.
    Reward is 1.0 for a in (-1, -0.8), and 0 otherwise in each dimension.
    '''
    def __init__(self, noise_std=1e-1, action_dim=1):
        self.action_dim = action_dim
        self.obs_dim = 1
        self.noise_std = noise_std
        low = np.array([-1.5] * action_dim)
        high = np.array([1.5] * action_dim)
        self.action_space = Continuous(low=low, high=high, dtype=np.float32) 
        self.observation_space = Continuous(
                low=np.zeros(self.obs_dim, dtype=np.float32),
                high=np.ones(self.obs_dim, dtype=np.float32),
                dtype=np.float32) 
        self.reset()

    def reset(self):
        obs = np.random.rand(self.obs_dim)
        return obs

    def step(self, action):
        obs = np.random.rand(self.obs_dim)
        reward = self.noise_std * np.random.normal()
        action = np.array(action)
        assert len(action) == self.action_dim
        # Highest reward possible is 1.
        reward += ((action > -1.0) & (action < -0.8)).sum() / self.action_dim
        done = True
        return obs, reward, done, {}


class TwoPeakEnv(gym.Env):
    '''
    A simple single-state environment with 1D continuous actions.
    Reward function has two peaks at -2 and 1.
    '''
    def __init__(self, noise_std=1e-1):
        self.action_dim = 1
        self.obs_dim = 1
        self.noise_std = noise_std
        low = np.array([-5])
        high = np.array([5])
        self.action_space = Continuous(low=low, high=high, dtype=np.float32) 
        self.observation_space = Continuous(
                low=np.zeros(self.obs_dim, dtype=np.float32),
                high=np.ones(self.obs_dim, dtype=np.float32),
                dtype=np.float32) 
        self.reset()

    def reset(self):
        obs = np.random.rand(self.obs_dim)
        return obs

    def step(self, action):
        obs = np.random.rand(self.obs_dim)
        reward = 1.1 * np.exp(-1.2 * np.power(action-(-2), 2))
        reward += 0.9 * np.exp(-0.9 * np.power(action-(1), 2))
        reward = reward.sum()
        done = True
        return obs, reward, done, {}
