import torch.nn as nn
import math
import torch as ch
import torch.nn.functional as F
from .torch_utils import *

'''
Neural network models for estimating value and policy functions
Contains:
- Initialization utilities
- Value Network(s)
- Policy Network(s)
- Retrieval Function
'''

########################
### INITIALIZATION UTILITY FUNCTIONS:
# initialize_weights
########################

HIDDEN_SIZES = (64, 64)
ACTIVATION = nn.Tanh
STD = 2**0.5

def initialize_weights(mod, initialization_type, scale=STD):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")
            

########################
### INITIALIZATION UTILITY FUNCTIONS:
# Generic Value network, Value network MLP
########################

class ValueDenseNet(nn.Module):
    '''
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 128-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    '''
    def __init__(self, state_dim, init, init_scale=1.0, hidden_sizes=(64, 64)):
        '''
        Initializes the value network.
        Inputs:
        - state_dim, the input dimension of the network (i.e dimension of state)
        - hidden_sizes, an iterable of integers, each of which represents the size
        of a hidden layer in the neural network.
        Returns: Initialized Value network
        '''
        super().__init__()
        self.activation = ACTIVATION()
        self.affine_layers = nn.ModuleList()

        prev = state_dim
        for h in hidden_sizes:
            l = nn.Linear(prev, h)
            initialize_weights(l, init, scale=STD)
            self.affine_layers.append(l)
            prev = h

        self.final = nn.Linear(prev, 1)
        initialize_weights(self.final, init, scale=init_scale)

    def forward(self, x):
        '''
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        '''
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        value = self.final(x)
        return value

    def get_value(self, x):
        return self(x)

########################
### POLICY NETWORKS
# Discrete and Continuous Policy Examples
########################

'''
A policy network can be any class which is initialized 
with a state_dim and action_dim, as well as optional named arguments.
Must provide:
- A __call__ override (or forward, for nn.Module): 
    * returns a tensor parameterizing a distribution, given a 
    BATCH_SIZE x state_dim tensor representing shape
- A function calc_kl(p, q): 
    * takes in two batches tensors which parameterize probability 
    distributions (of the same form as the output from __call__), 
    and returns the KL(p||q) tensor of length BATCH_SIZE
- A function entropies(p):
    * takes in a batch of tensors parameterizing distributions in 
    the same way and returns the entropy of each element in the 
    batch as a tensor
- A function sample(p): 
    * takes in a batch of tensors parameterizing distributions in
    the same way as above and returns a batch of actions to be 
    performed
- A function get_likelihoods(p, actions):
    * takes in a batch of parameterizing tensors (as above) and an 
    equal-length batch of actions, and returns a batch of probabilities
    indicating how likely each action was according to p.
'''

class DiscPolicy(nn.Module):
    '''
    A discrete policy using a fully connected neural network.
    The parameterizing tensor is a categorical distribution over actions
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES,
            time_in_state=False, share_weights=False, **unused_args):
        '''
        Initializes the network with the state dimensionality and # actions
        Inputs:
        - state_dim, dimensionality of the state vector
        - action_dim, # of possible discrete actions
        - hidden_sizes, an iterable of length #layers,
            hidden_sizes[i] = number of neurons in layer i
        - time_in_state, a boolean indicating whether the time is 
            encoded in the state vector
        '''
        super().__init__()
        self.activation = ACTIVATION()
        self.time_in_state = time_in_state

        self.discrete = True
        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        self.final = nn.Linear(prev_size, action_dim)

        # For the case where we want to share parameters 
        # between the policy and value networks
        self.share_weights = share_weights
        if share_weights:
            if time_in_state:
                self.final_value = nn.Linear(prev_size + 1, 1)
            else:
                self.final_value = nn.Linear(prev_size, 1)

            initialize_weights(self.final_value, init, scale=1.0)

    def forward(self, x):
        '''
        Outputs the categorical distribution (via softmax)
        by feeding the state through the neural network
        '''
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        
        probs = F.softmax(self.final(x), dim=-1)
        return probs

    def calc_kl(self, p, q, get_mean=True):
        '''
        Calculates E KL(p||q):
        E[sum p(x) log(p(x)/q(x))]
        Inputs:
        - p, first probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        - q, second probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        - get_mean, whether to return mean or a list
        Returns:
        - Empirical KL from p to q
        '''
        p, q = p.squeeze(), q.squeeze()
        assert shape_equal_cmp(p, q)
        kl = (p * (ch.log(p + 1e-10) - ch.log(q + 1e-10))).sum(-1)
        if get_mean:
            return kl.mean()
        return kl

    def entropies(self, p):
        '''
        p is probs of shape (batch_size, action_space). return mean entropy
        across the batch of states
        '''
        entropies = (p * ch.log(p)).sum(dim=1)
        return entropies

    def get_loglikelihood(self, p, actions):
        '''
        Inputs:
        - p, batch of probability tensors
        - actions, the actions taken
        '''
        try:
            dist = ch.distributions.categorical.Categorical(p)
            actions = actions.squeeze()
            return dist.log_prob(actions)
        except Exception as e:
            raise ValueError("Numerical error")
    
    def sample(self, probs):
        '''
        given probs, return: actions sampled from P(.|s_i), and their
        probabilities
        - s: (batch_size, state_dim)
        Returns actions:
        - actions: shape (batch_size,)
        '''
        dist = ch.distributions.categorical.Categorical(probs)
        actions = dist.sample()
        return actions.long()

    def get_value(self, x):
        # If the time is in the state, discard it
        assert self.share_weights, "Must be sharing weights to use get_value"
        t = None
        if self.time_in_state:
            t = x[...,-1:]
            x = x[...,:-1]

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        if self.time_in_state:
            return self.final_value(ch.cat((x, t), -1))
        else:
            return self.final_value(x)


class CtsPolicy(nn.Module):
    '''
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES,
                 time_in_state=False, share_weights=False,
                 action_space_low=None, action_space_high=None,
                 adjust_init_std=False):
        super().__init__()
        self.activation = ACTIVATION()
        self.action_dim = action_dim
        self.discrete = False
        self.time_in_state = time_in_state

        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        self.final_mean = nn.Linear(prev_size, action_dim)
        initialize_weights(self.final_mean, init, scale=0.01)
        
        # For the case where we want to share parameters 
        # between the policy and value networks
        self.share_weights = share_weights
        if share_weights:
            if time_in_state:
                self.final_value = nn.Linear(prev_size + 1, 1)
            else:
                self.final_value = nn.Linear(prev_size, 1)

            initialize_weights(self.final_value, init, scale=1.0)

        if adjust_init_std: 
            assert action_space_low is not None
            assert action_space_high is not None
            assert np.all(-np.inf < action_space_low)
            assert np.all(np.inf > action_space_high)
            # symmetric action spaces
            # initializing final mean to be around 0 only makes sense if symmetric
            assert (np.mean(action_space_low + action_space_high).round(2) == 0.0).all()
            # initialize std such that high and low are at ~ 2 STD
            stdev_init = (action_space_high - action_space_low) / 4
            log_stdev_init = ch.Tensor(np.log(stdev_init))
        else:
            log_stdev_init = ch.Tensor(ch.zeros(action_dim))
        self.log_stdev = ch.nn.Parameter(log_stdev_init)

    def forward(self, x):
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        
        means = self.final_mean(x)
        std = ch.exp(self.log_stdev)

        return means, std 

    def get_value(self, x):
        assert self.share_weights, "Must be sharing weights to use get_value"

        # If the time is in the state, discard it
        t = None
        if self.time_in_state:
            t = x[...,-1:]
            x = x[...,:-1]

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        if self.time_in_state:
            return self.final_value(ch.cat((x, t), -1))
        else:
            return self.final_value(x)

    def sample(self, p):
        '''
        Given prob dist (mean, var), return: actions sampled from p_i, and their
        probabilities. p is tuple (means, var). means shape 
        (batch_size, action_space), var (action_space,), here are batch_size many
        prboability distributions you're sampling from

        Returns tuple (actions, probs):
        - actions: shape (batch_size, action_dim)
        - probs: shape (batch_size, action_dim)
        '''
        means, std = p
        return (means + ch.randn_like(means)*std).detach()

    def get_loglikelihood(self, p, actions):
        try:    
            mean, std = p
            nll =  0.5 * ((actions - mean).pow(2) / std.pow(2)).sum(-1) \
                   + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] \
                   + self.log_stdev.sum(-1)
            return -nll
        except Exception as e:
            raise ValueError("Numerical error")

    def calc_kl(self, p, q, npg_approx=False, get_mean=True):
        '''
        Get the expected KL distance between two sets of gaussians over states -
        gaussians p and q where p and q are each tuples (mean, var)
        - In other words calculates E KL(p||q): E[sum p(x) log(p(x)/q(x))]
        - From https://stats.stackexchange.com/a/60699
        '''
        p_mean, p_std = p
        q_mean, q_std = q
        p_var = p_std.pow(2) + 1e-10
        q_var = q_std.pow(2) + 1e-10
        assert shape_equal([-1, self.action_dim], p_mean, q_mean)
        assert shape_equal([self.action_dim], p_var, q_var)

        d = q_mean.shape[1]
        # Add 1e-10 to variances to avoid nans.
        logdetp = log_determinant(p_var)
        logdetq = log_determinant(q_var)
        diff = q_mean - p_mean

        log_quot_frac = logdetq - logdetp
        tr = (p_var / q_var).sum()
        quadratic = ((diff / q_var) * diff).sum(dim=1)

        if npg_approx:
            kl_sum = 0.5 * quadratic + 0.25 * (p_var / q_var - 1.).pow(2).sum()
        else:
            kl_sum = 0.5 * (log_quot_frac - d + tr + quadratic)
        assert kl_sum.shape == (p_mean.shape[0],)
        if get_mean:
            return kl_sum.mean()
        return kl_sum

    def entropies(self, p):
        '''
        Get entropies over the probability distributions given by p
        p_i = (mean, var), p mean is shape (batch_size, action_space),
        p var is shape (action_space,)
        '''
        _, std = p
        var = std.pow(2) + 1e-10
        # Add 1e-10 to variance to avoid nans.
        logdetp = log_determinant(var)
        d = var.shape[0]
        entropies = 0.5 * (logdetp + d * (1. + math.log(2 * math.pi)))
        return entropies


class CtsBetaPolicy(nn.Module):
    '''
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is an alpha and beta vector
    which parameterize a beta distribution.
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES,
                 time_in_state=False, share_weights=False,
                 action_space_low=None, action_space_high=None, **unused_args):
        super().__init__()
        assert action_space_low is not None
        assert action_space_high is not None
        assert np.all(-np.inf < action_space_low)
        assert np.all(np.inf > action_space_high)
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        self.activation = ACTIVATION()
        self.action_dim = action_dim
        self.discrete = False
        self.time_in_state = time_in_state

        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        self.alpha_pre_softplus = nn.Linear(prev_size, action_dim)
        initialize_weights(self.alpha_pre_softplus, init, scale=0.01)
        self.beta_pre_softplus = nn.Linear(prev_size, action_dim)
        initialize_weights(self.beta_pre_softplus, init, scale=0.01)
        self.softplus = ch.nn.Softplus()
        
        # For the case where we want to share parameters 
        # between the policy and value networks
        self.share_weights = share_weights
        if share_weights:
            if time_in_state:
                self.final_value = nn.Linear(prev_size + 1, 1)
            else:
                self.final_value = nn.Linear(prev_size, 1)

            initialize_weights(self.final_value, init, scale=1.0)

    def forward(self, x):
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        # Use alpha and beta >= 1 according to [Chou et. al, 2017]
        alpha = ch.add(self.softplus(self.alpha_pre_softplus(x)), 1.)
        beta = ch.add(self.softplus(self.beta_pre_softplus(x)), 1.)
        return alpha, beta

    def get_value(self, x):
        assert self.share_weights, "Must be sharing weights to use get_value"

        # If the time is in the state, discard it
        t = None
        if self.time_in_state:
            t = x[...,-1:]
            x = x[...,:-1]

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        if self.time_in_state:
            return self.final_value(ch.cat((x, t), -1))
        else:
            return self.final_value(x)

    def scale_by_action_bounds(self, beta_dist_samples):
        # Scale [0, 1] back to action space.
        return beta_dist_samples * (self.action_space_high -
                self.action_space_low) + self.action_space_low

    def inv_scale_by_action_bounds(self, actions):
        # Scale action space to [0, 1].
        return (actions - self.action_space_low) / (self.action_space_high -
                self.action_space_low)


    def sample(self, p):
        '''
        Given prob dist (alpha, beta), return: actions sampled from p_i, and their
        probabilities. p is tuple (alpha, beta). means shape 
        (batch_size, action_space), var (action_space,), here are batch_size many
        prboability distributions you're sampling from

        Returns tuple (actions, probs):
        - actions: shape (batch_size, action_dim)
        - probs: shape (batch_size, action_dim)
        '''
        alpha, beta = p
        dist = ch.distributions.beta.Beta(alpha, beta)
        samples = dist.sample()
        assert shape_equal([-1, self.action_dim], samples, alpha, beta)
        return self.scale_by_action_bounds(samples)

    def get_loglikelihood(self, p, actions):
        alpha, beta = p
        dist = ch.distributions.beta.Beta(alpha, beta)
        log_probs = dist.log_prob(self.inv_scale_by_action_bounds(actions))
        assert shape_equal([-1, self.action_dim], log_probs, alpha, beta)
        return ch.sum(log_probs, dim=1)

    def lbeta(self, alpha, beta):
        '''The log beta function.'''
        return ch.lgamma(alpha) + ch.lgamma(beta) - ch.lgamma(alpha+beta)

    def calc_kl(self, p, q, npg_approx=False, get_mean=True):
        '''
        Get the expected KL distance between beta distributions.
        '''
        assert not npg_approx
        p_alpha, p_beta = p
        q_alpha, q_beta = q
        assert shape_equal([-1, self.action_dim], p_alpha, p_beta, q_alpha,
                q_beta)

        # Expectation of log x under p.
        e_log_x = ch.digamma(p_alpha) - ch.digamma(p_alpha + p_beta)
        # Expectation of log (1-x) under p.
        e_log_1_m_x = ch.digamma(p_beta) - ch.digamma(p_alpha + p_beta)
        kl_per_action_dim = (p_alpha - q_alpha) * e_log_x
        kl_per_action_dim += (p_beta - q_beta) * e_log_1_m_x
        kl_per_action_dim -= self.lbeta(p_alpha, p_beta)
        kl_per_action_dim += self.lbeta(q_alpha, q_beta)
        # By chain rule on KL divergence.
        kl_joint = ch.sum(kl_per_action_dim, dim=1)
        if get_mean:
            return kl_joint.mean()
        return kl_joint

    def entropies(self, p):
        '''
        Get entropies over the probability distributions given by p
        p_i = (alpha, beta), p mean is shape (batch_size, action_space),
        p var is shape (action_space,)
        '''
        alpha, beta = p
        entropies = self.lbeta(alpha, beta)
        entropies -= (alpha - 1) * ch.digamma(alpha)
        entropies -= (beta - 1) * ch.digamma(beta)
        entropies += (alpha + beta - 2) * ch.digamma(alpha + beta)
        return ch.sum(entropies, dim=1)


## Retrieving networks
# Make sure to add newly created networks to these dictionaries!

POLICY_NETS = {
    "DiscPolicy": DiscPolicy,
    "CtsPolicy": CtsPolicy,
    "CtsBetaPolicy": CtsBetaPolicy,
}

VALUE_NETS = {
    "ValueNet": ValueDenseNet,
}

def policy_net_with_name(name):
    return POLICY_NETS[name]

def value_net_with_name(name):
    return VALUE_NETS[name]

