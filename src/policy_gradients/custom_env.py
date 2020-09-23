from collections import OrderedDict
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box as Continuous
from gym.wrappers import ClipAction
import gym
from .torch_utils import RunningStat, ZFilter, Identity, StateWithTime, RewardFilter, ConstantFilter
from .torch_utils import add_gaussian_noise, add_uniform_noise, add_sparsity_noise

def convert_state_to_array(obs):
    '''Converts DMC style ordered dict observation to array.'''
    if isinstance(obs, OrderedDict):
        return np.concatenate([o.reshape(-1) for o in obs.values()])
    else:
        return obs

class Env:
    '''
    A wrapper around the OpenAI gym environment that adds support for the following:
    - Rewards normalization
    - State normalization
    - Adding timestep as a feature with a particular horizon T
    - Add gaussian noise to reward.
    - Add uniform noise to reward.
    Also provides utility functions/properties for:
    - Whether the env is discrete or continuous
    - Size of feature space
    - Size of action space
    Provides the same API (init, step, reset) as the OpenAI gym
    '''
    def __init__(self, game, norm_states, norm_rewards, params,
            add_t_with_horizon=None, clip_obs=None, clip_rew=None):

        self.env = gym.make(game)
        clip_obs = None if clip_obs < 0 else clip_obs
        clip_rew = None if clip_rew < 0 else clip_rew

        # Environment type
        self.is_discrete = type(self.env.action_space) == Discrete
        assert self.is_discrete or type(self.env.action_space) == Continuous

        # Number of actions
        action_shape = self.env.action_space.shape
        assert len(action_shape) <= 1 # scalar or vector actions
        self.num_actions = self.env.action_space.n if self.is_discrete else 0 \
                            if len(action_shape) == 0 else action_shape[0]
        
        if self.env.observation_space.shape is None:
            self.num_features = convert_state_to_array(
                    self.env.reset()).shape[0]
        else:
            assert len(self.env.observation_space.shape) == 1
            # Number of features
            self.num_features = self.env.reset().shape[0]

        # Support for state normalization or using time as a feature
        self.state_filter = Identity()
        if norm_states:
            self.state_filter = ZFilter(self.state_filter, shape=[self.num_features], \
                                            clip=clip_obs)
        if add_t_with_horizon is not None:
            self.state_filter = StateWithTime(self.state_filter, horizon=add_t_with_horizon)
        
        # Support for rewards normalization
        self.reward_filter = Identity()
        if norm_rewards == "rewards":
            self.reward_filter = ZFilter(self.reward_filter, shape=(), center=False, clip=clip_rew)
        elif norm_rewards == "returns":
            self.reward_filter = RewardFilter(self.reward_filter, shape=(), gamma=params.GAMMA, clip=clip_rew)
        elif norm_rewards == "constant":
            self.reward_filter = ConstantFilter(self.reward_filter, constant=0.05)

        self.reward_gaussian_noise = params.REWARD_GAUSSIAN_NOISE
        self.reward_uniform_noise = params.REWARD_UNIFORM_NOISE
        self.reward_sparsity = params.REWARD_SPARSITY

        # Running total reward (set to 0.0 at resets)
        self.total_true_reward = 0.0

        # OpenAI baselines use clipped actions wrapper, following their clipping
        if params.CLIP_ACTION:
            self.env = ClipAction(self.env) 
        self.strict_action_bounds = params.STRICT_ACTION_BOUNDS

    def reset(self):
        # Reset the state, and the running total reward
        start_state = convert_state_to_array(self.env.reset())
        self.total_true_reward = 0.0
        self.counter = 0.0
        self.state_filter.reset()
        return self.state_filter(start_state, reset=True)

    def step(self, action):
        state, reward, is_done, info = self.env.step(action)

        if self.strict_action_bounds:
            if not self.env.action_space.contains(action):
                reward = -10.0

        state = convert_state_to_array(state)
        state = self.state_filter(state)
        self.total_true_reward += reward
        self.counter += 1
        _reward = add_gaussian_noise(reward, self.reward_gaussian_noise)
        _reward = add_uniform_noise(_reward, self.reward_uniform_noise)
        _reward = add_sparsity_noise(_reward, self.reward_sparsity)
        _reward = self.reward_filter(_reward)
        if is_done:
            info['done'] = (self.counter, self.total_true_reward)
        return state, _reward, is_done, info
