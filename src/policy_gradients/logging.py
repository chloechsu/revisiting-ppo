import torch as ch
import numpy as np
from .torch_utils import *
from torch.nn.utils import parameters_to_vector as flatten
from torch.nn.utils import vector_to_parameters as assign
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances


#####
# Understanding TRPO approximations for KL constraint
#####

def paper_constraints_logging(agent, saps, old_pds, table):
    new_pds = agent.policy_model(saps.states)
    new_log_ps = agent.policy_model.get_loglikelihood(new_pds,
                                                    saps.actions)

    ratios = ch.exp(new_log_ps - saps.action_log_probs)
    max_rat = ratios.max()
    avg_rat = ratios.mean()
    avg_rat_minus_one_sq = ch.sub(ratios, 1.).pow(2).mean()

    avg_kl_old_to_new = agent.policy_model.calc_kl(old_pds, new_pds)
    avg_kl_new_to_old = agent.policy_model.calc_kl(new_pds, old_pds)

    row = {
        'avg_kl_old_to_new':avg_kl_old_to_new,
        'avg_kl_new_to_old':avg_kl_new_to_old,
        'max_ratio':max_rat,
        'avg_ratio':avg_rat,
        'avg_rat_minus_one_sq': avg_rat_minus_one_sq,
        'opt_step':agent.n_steps,
    }

    # Hacky way to identify gaussian policy
    if len(old_pds) == 2 and old_pds[0].shape != old_pds[1].shape:
        old_pd_means = ch.mean(old_pds[0].detach(), dim=0)
        old_pd_means_std = ch.std(old_pds[0].detach(), dim=0)
        if old_pd_means.dim() == 0:
            try:
                row[f'mean_0'] = old_pd_means.item()
                row[f'mean_std_0'] = old_pd_means_std.item()
            except:
                row[f'mean_0'] = np.nan
                row[f'mean_std_0'] = np.nan
        else:
            for d in range(old_pd_means.shape[0]):
                row[f'mean_{d}'] = old_pd_means[d]
                row[f'mean_std_{d}'] = old_pd_means_std[d]

    for k in row:
        if k != 'opt_step':
            row[k] = float(row[k])

    agent.store.log_table_and_tb(table, row)
