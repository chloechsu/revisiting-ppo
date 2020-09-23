import os
import sys

import gym
import numpy as np
import pandas as pd
import torch

module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
from policy_gradients.steps import ppo_step
from policy_gradients.models import DiscPolicy, CtsPolicy, CtsBetaPolicy
from policy_gradients.torch_utils import Parameters


new_to_old_params = {'kl_penalty_direction': 'new_to_old', 'kl_penalty_coeff': 3., 'clip_eps': None}
old_to_new_params = {'kl_penalty_direction': 'old_to_new', 'kl_penalty_coeff': 3., 'clip_eps': None}
clip_params = {'kl_penalty_coeff': 0., 'clip_eps': 0.2}
plain_params = {'kl_penalty_coeff': 0., 'clip_eps': None}


default_comparisons = {
    'Reverse KL': new_to_old_params,
    'Forward KL': old_to_new_params,
    'Clipping': clip_params,
    'Unregularized': plain_params
}


base_params = {
    'ppo_epochs': 10,
    'num_minibatches': 1,
    'clip_advantages': None,
    'sign_advantages': False,
    'norm_advantages': True,
    'kl_penalty_direction': 'new_to_old',
    'kl_closed_form': True,
    'kl_npg_form': False,
    'entropy_coeff': 0.0,
    'share_weights': False,
    'clip_grad_norm': -1,
    'batch_size': 2,
    'max_episode_len': 1,
    'lr': 1e-2,
}


########### Directly parameterized policy classes without neural networks ###


class SimpleDiscPolicy(DiscPolicy):
    '''A simpler version of disc policy without neural network.'''

    def __init__(self, action_dim, init=None, **kwargs):
        super(DiscPolicy, self).__init__()
        self.action_dim = action_dim
        if init is None:
            init = torch.ones((1, action_dim))
        else:
            init = torch.Tensor(init.reshape(1, action_dim))
        self.logits = torch.nn.Parameter(init)

    def forward(self, x):
        return torch.nn.functional.softmax(self.logits, dim=-1).repeat(
                x.shape[0], 1)

    def calc_kl(self, p, q, **kwargs):
        return super(SimpleDiscPolicy, self).calc_kl(p, q)



class SimpleCtsPolicy(CtsPolicy):
    '''A simpler version of Gaussian policy without neural network.'''

    def __init__(self, action_dim, init=None, **kwargs):
        super(CtsPolicy, self).__init__()
        self.action_dim = action_dim
        if init is None:
            mean_init = torch.zeros((1, action_dim))
            log_std_init = torch.zeros(action_dim)
        else:
            mean_init = torch.Tensor(init['mean'].reshape(1, action_dim))
            log_std_init = torch.Tensor(init['log_std'].reshape(action_dim))
        self.mean = torch.nn.Parameter(mean_init)
        self.log_stdev = torch.nn.Parameter(log_std_init)

    def forward(self, x):
        return self.mean.repeat(x.shape[0], 1), torch.exp(
                self.log_stdev)


class SimpleCtsBetaPolicy(CtsBetaPolicy):
    '''A simpler version of Beta policy without neural network.'''

    def __init__(self, action_dim, action_space_low, action_space_high, init=None):
        super(CtsBetaPolicy, self).__init__()
        self.action_dim = action_dim
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        # Initialize so that the initial Beta policy is close to B(1,1) uniform
        if init is None:
            pre_softplus_a_init = -4. * torch.ones((1, action_dim))
            pre_softplus_b_init = -4. * torch.ones((1, action_dim))
        else:
            pre_softplus_a_init = torch.log(torch.Tensor(init['a'].reshape(1,
                action_dim)) - 1.0)
            pre_softplus_b_init = torch.log(torch.Tensor(init['b'].reshape(1,
                action_dim)) - 1.0)
        self.pre_softplus_a = torch.nn.Parameter(pre_softplus_a_init)
        self.pre_softplus_b = torch.nn.Parameter(pre_softplus_b_init)

    def forward(self, x):
        softplus = torch.nn.Softplus()
        a = torch.add(softplus(self.pre_softplus_a), 1.)
        b = torch.add(softplus(self.pre_softplus_b), 1.)
        return a.repeat(x.shape[0], 1), b.repeat(x.shape[0], 1)


##### Utils for running PPO training ########

def sample_trajectory(env, policy, max_episode_length):
    """Samples one trajectories using the given policy.

    Parameters
    ==========
    env: A gym environment.
    policy: An instance of Policy.
    max_episode_length: Cap on max length for each episode.

    Returns
    =======
    A  dictionary, each dictionary maps keys `observation`, `action`, `reward`,
    `next_observation`, `terminal` to numpy arrays of size episode length.
    """
    # initialize env for the beginning of a new rollout
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    steps = 0
    while True:
        # use the most recent ob to decide what to do
        obs.append(ob)
        action_pd = policy(torch.Tensor(ob[None, :]))
        ac = policy.sample(action_pd).numpy().flatten()
        acs.append(ac)
        # take that action and record results
        ob, rew, done, _ = env.step(ac)
        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)
        # End the rollout if the rollout ended
        # Note that the rollout can end due to done, or due to max_episode_length
        if done or steps >= max_episode_length:
            rollout_done = 1
        else:
            rollout_done = 0
        terminals.append(rollout_done)
        if rollout_done:
            break
    return wrap_trajectory(obs, acs, rewards, next_obs, terminals)


def wrap_trajectory(obs, acs, rewards, next_obs, terminals):
    return {"observation" : np.array(obs, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def sample_trajectories_by_batch_size(env, policy, min_timesteps_per_batch, max_episode_length):
    """Samples multiple trajectories using the given policy to achieve total number of steps.

    Parameters
    ==========
    env: A gym environment.
    policy: An instance of Policy.
    min_timesteps_per_batch: Desired number of timesteps in all trajectories combined.
    max_episode_length: Cap on max length for each episode.

    Returns
    =======
    A list of n dictionaries, each dictionary maps keys `observation`, `action`,
    `reward`, `next_observation`, `terminal` to numpy arrays of size episode length.
    """
    timesteps_this_batch = 0
    trajectories = []
    while timesteps_this_batch < min_timesteps_per_batch:
        trajectory = sample_trajectory(env, policy, max_episode_length)
        trajectories.append(trajectory)
        timesteps_this_batch += len(trajectory['action'])
    return trajectories, timesteps_this_batch


def get_reward_to_go(rewards, gamma=0.99):
    """Helper function to compute discounted reward to go.

    Parameters
    ==========
    rewards: A list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single
    rollout of length T.

    Returns
    =======
    A numpy array where the entry at index t is sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}.
    """
    all_discounted_cumsums = []
    # for loop over steps (t) of the given rollout
    for start_time_index in range(len(rewards)):
        indices = np.arange(start_time_index, len(rewards))
        discounts = gamma**(indices - start_time_index)
        all_discounted_cumsums.append(sum(discounts * rewards[start_time_index:]))
    return np.array(all_discounted_cumsums)
 

def get_obs_acs_rewards_advs(trajectories):
    obs = np.concatenate([tau["observation"] for tau in trajectories], axis=0)
    acs = np.concatenate([tau["action"] for tau in trajectories], axis=0)
    rewards = np.concatenate([tau["reward"] for tau in trajectories])
    reward_to_go = np.concatenate(
        [get_reward_to_go(tau["reward"]) for tau in trajectories])
    advs = reward_to_go - np.mean(reward_to_go)
    return obs, acs, rewards, advs


def step(env, policy, params):
    trajs, steps_in_trajs = sample_trajectories_by_batch_size(
        env, policy, params.batch_size, params.max_episode_len)
    obs, acs, rewards, advs = get_obs_acs_rewards_advs(trajs)
    obs = torch.Tensor(obs)
    acs = torch.Tensor(acs)
    advs = torch.Tensor(advs)
    with torch.no_grad():
        old_pds = policy(obs)
        old_log_ps = policy.get_loglikelihood(old_pds, acs)
    loss = ppo_step(obs, acs, old_log_ps, None, None, None, advs, policy, params, None, None)
    return_dict = {
        'loss': loss.detach().item(),
        'mean_reward': np.mean(rewards)
    }
    if isinstance(old_pds, tuple):
        for i, pd_param in enumerate(old_pds):
            return_dict[f'pd_param_{i}_mean'] = np.mean(pd_param.numpy())
    else:
        for i in range(old_pds.shape[1]):
            return_dict[f'pd_param_{i}_mean'] = np.mean(old_pds[:, i].numpy())
    return return_dict


def train(env, policy, params, n_steps):
    df = pd.DataFrame()
    for i in range(n_steps):
        return_dict = step(env, policy, params)
        return_dict['iter'] = i
        df = df.append(return_dict, ignore_index=True)
    return df


def compare(env, policy_type, n_steps=20, repeats=10, seed=0, policy_init=None,
        comparisons=None, SGD=False, **kwargs):
    np.random.seed(seed)
    # Handle different policy classes.
    policy_type = policy_type.lower()
    policy_cls_map = {
            'gaussian': SimpleCtsPolicy,
            'beta': SimpleCtsBetaPolicy,
            'discrete': SimpleDiscPolicy}
    policy_cls = policy_cls_map[policy_type]
    base_params_ = base_params.copy()
    base_params_.update(**kwargs)
    if comparisons is None:
        comparisons = default_comparisons
    data = None
    for i, name in enumerate(comparisons.keys()):
        params = base_params_.copy()
        params.update(comparisons[name])
        for j in range(repeats):
            if policy_type != 'discrete':
                policy = policy_cls(env.action_dim, init=policy_init,
                        action_space_low=env.action_space.low,
                        action_space_high=env.action_space.high)
            else:
                policy = policy_cls(env.action_dim, init=policy_init)
            if SGD:
                params['policy_adam'] = None
                params['ppo_lr'] = params['lr']
            else:
                params['policy_adam'] = torch.optim.Adam(policy.parameters(),
                        lr=params['lr'])
            data_this_run = train(env, policy, Parameters(params), n_steps)
            data_this_run['method'] = name
            if data is None:
                data = data_this_run
            else:
                data = pd.concat([data, data_this_run])
    return data
