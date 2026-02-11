import copepodTCR as cpp
import numpy as np
import pandas as pd
import random

import pymc as pm
import pytensor
import pytensor.tensor as pt
#pytensor.config.blas__ldflags = '-framework Accelerate'
import arviz as az

import argparse

def activation_model4(obs, n_pools, inds, neg_control = None, neg_share = None, cores=1):

    """
    Takes a list with observed data (obs), number of pools (n_pools), and indices for the observed data if there were mutiple replicas.
    Returns model fit and a dataframe with probabilities of each pool being drawn from negative or positive distributions.
    """
    
    coords = dict(pool=range(n_pools), component=("positive", "negative"))
    if neg_share is None:
        neg_share = 0.5
    if neg_control is None:
        neg_control = sorted(obs)[:inds.count(0)]
    if np.min(neg_control) > np.max(obs):
        obs = obs/np.max(neg_control)
        neg_control = neg_control/np.max(neg_control)
    else:
        neg_control = neg_control/np.max(obs)
        obs = obs/np.max(obs)

    with pm.Model(coords=coords) as model:
    
        negative = pm.TruncatedNormal(
            "negative",
            mu=0,
            sigma=1,
            lower=0.0,
            upper=1.0,
        )

        negative_obs = pm.TruncatedNormal(
            "negative_obs",
            mu=negative,
            sigma=0.1,
            lower=0.0,
            upper=1.0,
            observed=neg_control,
        )

        # Offset such that negative + offset <= 1
        offset_proportion = pm.Beta("offset_proportion", alpha=5, beta=2)
        offset = pm.Deterministic("offset", (1 - negative) * offset_proportion)
        #offset = pm.TruncatedNormal('offset', mu = 0.6, sigma = 0.1, upper = 1, lower = 0)

        #positive = pm.Deterministic("positive", negative + offset, upper = 0, lower = 1)
        positive = pm.Deterministic("positive", negative + offset)
    
        p = pm.Beta("p", alpha=neg_share * 100, beta=(1 - neg_share) * 100)
        component = pm.Bernoulli("assign", p, dims="pool")

        mu_pool = negative * component + positive * (1 - component)
    
        sigma_neg = pm.HalfNormal("sigma_neg", 0.5)
        #sigma_delta = pm.Exponential("sigma_delta", 0.5)
        sigma_pos = pm.HalfNormal("sigm_pos", 0.2)
        sigma_pool = sigma_pos * (1 - component) + sigma_neg * component
        
        pool_dist = pm.TruncatedNormal(
            "pool_dist",
            mu=mu_pool,
            sigma = sigma_pool,
            lower=0.0,
            upper=1.0,
            dims="pool",
        )
    
        # Likelihood, where the data indices pick out the relevant pool from pool
        sigma_data = pm.Exponential("sigma_data", 1.0)
        pm.TruncatedNormal(
            "lik", mu=pool_dist[inds], sigma=sigma_data, observed=obs, lower=0.0, upper=1.0
        )

        idata_alt = pm.sample(cores = cores)

    with model:
        posterior_predictive = pm.sample_posterior_predictive(idata_alt)

    ax = az.plot_ppc(posterior_predictive, num_pp_samples=100, colors = ['#015396', '#FFA500', '#000000'])

    posterior = az.extract(idata_alt)
    n_mean = float(posterior["negative"].mean(dim="sample"))
    p_mean = float(posterior["offset"].mean(dim="sample"))

    posterior_p_mean = posterior["p"].mean(dim="sample").item()
    print(f"Posterior mean of p: {posterior_p_mean:.3f}")

    #return ax, posterior["assign"].mean(dim="sample").to_dataframe(), neg_control, [p_mean, n_mean]
    return model, ax, posterior["assign"].mean(dim="sample").to_dataframe(), neg_control, idata_alt, [p_mean, n_mean]

parser = argparse.ArgumentParser(description='Evaluate Data')
parser.add_argument('-method', type=str)
parser.add_argument('-scheme', type=str)
parser.add_argument('-data', type=str)
parser.add_argument('-sim_params', type=str)
parser.add_argument('-output', type=str)
parser.add_argument('-len_lst', type=int)
parser.add_argument('-pep_length', type=int)
parser.add_argument('-overlap', type=int)
parser.add_argument('-ep_length', type=int)
parser.add_argument('-mu_off', type=float)
parser.add_argument('-sigma_off', type=float)
parser.add_argument('-mu_n', type=float)
parser.add_argument('-sigma_n', type=float)
parser.add_argument('-r', type=int)
parser.add_argument('-sigma_p_r', type=float)
parser.add_argument('-sigma_n_r', type=float)
parser.add_argument('-low_offset', type=float)
parser.add_argument('-error', type=int)
args = parser.parse_args()

scheme = str(args.scheme)
data = str(args.data)
output = str(args.output)

cells = pd.read_csv(data, sep = "\t")
sim_params = pd.read_csv(args.sim_params, sep = "\t")
n_control = list(sim_params['n_control'])
error_pools = sim_params['error_pools'].iloc[0]
error_pools = str(error_pools).split(';')
if error_pools != [''] and error_pools != ['nan']:
    error_pools = [int(x) for x in error_pools]
else:
    error_pools = []
check_results = pd.read_csv(scheme, sep = "\t")
n_pools = check_results['n_pools'].iloc[0]
iters = check_results['iters'].iloc[0]

inds = list(cells['Pool'])
obs = list(cells['Percentage'])

if args.method == 'copepodTCR':
    all_lst = list(set(check_results['Peptide']))
    c, _ = cpp.how_many_peptides(all_lst, args.ep_length)
    normal = max(c, key=c.get)
    neg_share = 1 - (args.iters + normal -1)/n_pools
else:
    neg_share = (n_pools - iters)/n_pools

model4, fig4, probs4, n_c4, pp4, parameters4 = activation_model4(obs, n_pools, inds, n_control, neg_share, cores = 1)
model4_n, fig4_n, probs4_n, n_c4_n, pp4_n, parameters4_n = activation_model4(obs = obs, n_pools = n_pools, inds = inds, neg_share = neg_share, cores = 1)

peptide_probs4 = cpp.peptide_probabilities(check_results, probs4)
len_act, notification, lst1, lst2 = cpp.results_analysis(peptide_probs4, probs4, check_results)

peptide_probs4_n = cpp.peptide_probabilities(check_results, probs4_n)
len_act_n, notification_n, lst1_n, lst2_n = cpp.results_analysis(peptide_probs4_n, probs4_n, check_results)

cognate = check_results['Act Pools'][check_results['Cognate'] == True].iloc[0]
cognate_peptides = set(check_results['Peptide'][check_results['Cognate'] == True])
cognate = [int(x) for x in cognate[1:-1].split(', ')]
for i in range(len(error_pools)):
    cognate.remove(error_pools[i])

if args.error == 100:
    cognate = []

act_pools_model4 = list(probs4.index[probs4['assign'] < 0.5])
act_pools_model4_n = list(probs4_n.index[probs4_n['assign'] < 0.5])

def calculate_tp_tn_fp_fn(cognate, act_pools, n_pools):
    tp = []
    tn = []
    fp = []
    fn = []

    for i in range(n_pools):
        if i in cognate and i in act_pools:
            tp.append(i)
        elif i not in cognate and i not in act_pools:
            tn.append(i)
        elif i not in cognate and i in act_pools:
            fp.append(i)
        elif i in cognate and i not in act_pools:
            fn.append(i)

    return tp, tn, fp, fn

tp4, tn4, fp4, fn4 = calculate_tp_tn_fp_fn(cognate, act_pools_model4, n_pools)
tp4_n, tn4_n, fp4_n, fn4_n = calculate_tp_tn_fp_fn(cognate, act_pools_model4_n, n_pools)

results_row = dict()
results_row['method'] = args.method
results_row['n_pools'] = n_pools
results_row['iters'] = iters
results_row['len_lst'] = args.len_lst
results_row['pep_length'] = args.pep_length
results_row['shift'] = args.overlap
results_row['ep_length'] = args.ep_length
results_row['mu_off'] = args.mu_off
results_row['sigma_off'] = args.sigma_off
results_row['mu_n'] = args.mu_n
results_row['sigma_n'] = args.sigma_n
results_row['sigma_p_r'] = args.sigma_p_r
results_row['sigma_n_r'] = args.sigma_n_r
results_row['low_offset'] = args.low_offset
results_row['r'] = args.r
results_row['error'] = args.error
results_row['error_pools'] = error_pools

results_row['# true act'] = len(cognate)
results_row['true_pools'] = cognate
results_row['true_peptides'] = ', '.join(cognate_peptides)
if args.error == 100:
    results_row['# true act'] = 0
    results_row['true_pools'] = []
    results_row['true_peptides'] = ''

results_row['# act 4'] = len(act_pools_model4)
results_row['model4_pools'] = act_pools_model4

results_row['# act 4 n'] = len(act_pools_model4_n)
results_row['model4_pools_n'] = act_pools_model4_n

results_row['TruePositive_4'] = len(tp4_n)
results_row['TrueNegative_4'] = len(tn4_n)
results_row['FalsePositive_4'] = len(fp4_n)
results_row['FalseNegative_4'] = len(fn4_n)

results_row['TruePositive_4_n'] = len(tp4)
results_row['TrueNegative_4_n'] = len(tn4)
results_row['FalsePositive_4_n'] = len(fp4)
results_row['FalseNegative_4_n'] = len(fn4)

results_row['predicted'] = ', '.join(lst1)
results_row['possible'] = ', '.join(lst2)
results_row['conclusion_cognate'] = set(cognate_peptides) == set(lst1)
results_row['conclusion_possible'] = all(elem in lst2 for elem in cognate_peptides)

results_row['predicted_n'] = ', '.join(lst1_n)
results_row['possible_n'] = ', '.join(lst2_n)
results_row['conclusion_cognate_n'] = set(cognate_peptides) == set(lst1_n)
results_row['conclusion_possible_n'] = all(elem in lst2 for elem in cognate_peptides)

results_row['pools_indices'] = ', '.join([str(x) for x in inds])
results_row['pools_results'] = ', '.join([str(x) for x in obs])
results_row['pools_var'] = np.var(obs)
results_row['negative_model4'] = parameters4[1]
results_row['positive_model4'] = parameters4[0]

results_row['negative_model4_n'] = parameters4_n[1]
results_row['positive_model4_n'] = parameters4_n[0]

results_row['neg_control'] = sim_params['n_control'].iloc[0]/np.max(obs)
results_row['neg_control_n'] = np.mean(sorted(obs)[:inds.count(0)])/np.max(obs)
results_row['neg_share'] = neg_share
results_row['positive_sim_norm'] = float(sim_params['positive_sim'].iloc[0])/np.max(obs)
results_row['negative_sim_norm'] = float(sim_params['negative_sim'].iloc[0])/np.max(obs)
results_row['positive_sim'] = float(sim_params['positive_sim'].iloc[0])
results_row['negative_sim'] = float(sim_params['negative_sim'].iloc[0])

results_row = pd.DataFrame([results_row])

results_row.to_csv(output, sep = "\t", index = None)