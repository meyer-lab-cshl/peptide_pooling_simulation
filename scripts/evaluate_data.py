import copepodTCR as cpp
import numpy as np
import pandas as pd
import random

import pymc as pm
import arviz as az

import argparse

def activation_model(obs, n_pools, inds, neg_control = None, cores=1):

    """
    Takes a list with observed data (obs), number of pools (n_pools), and indices for the observed data if there were mutiple replicas.
    Returns model fit and a dataframe with probabilities of each pool being drawn from negative or positive distributions.
    """
    
    coords = dict(pool=range(n_pools), component=("positive", "negative"))

    if neg_control is not None:
        neg_control = np.sum(obs <= neg_control)/len(obs)
        neg_control = 0.5 + 0.2*neg_control
    else:
        neg_control = 0.5

    obs = obs/np.max(obs)
    

    with pm.Model(coords=coords) as alternative_model:
        # Define the offset
        offset = pm.Normal("offset", mu=0.6, sigma=0.1)

        # Negative component remains the same
        source_negative = pm.TruncatedNormal(
            "negative",
            mu=0,
            sigma=0.01,
            lower=0,
            upper=1,
            )
        # Adjusted positive distribution with offset
        source_positive = pm.Deterministic("positive", source_negative + offset)

        # Combine the source components
        source = pm.math.stack([source_positive, source_negative], axis=0)
        #source_sigma = pm.math.stack([pm.Exponential("pos_sigma", 0.2), pm.Exponential("neg_sigma", 0.01)], axis=0)

        # Each pool is assigned a 0/1
        # Probability of assigning depends on number of pools with activation signal higher than negative control
        component = pm.Bernoulli("assign", neg_control, dims="pool")

        # Each pool has a normally distributed response whose mu comes from either the
        # postive or negative source distribution
        pool_dist = pm.TruncatedNormal(
            "pool_dist",
            mu=source[component],
            sigma=source[component],
            lower=0,
            upper=1,
            dims="pool",
            )

        # Likelihood, where the data indices pick out the relevant pool from pool
        pm.TruncatedNormal(
            "lik",
            mu=pool_dist[inds],
            sigma=pm.Exponential("sigma_data", 1),
            observed=obs,
            lower=0,
            upper=1,
            )

        idata_alt = pm.sample(cores = cores)
        
    with alternative_model:
        posterior_predictive = pm.sample_posterior_predictive(idata_alt)

    ax = az.plot_ppc(posterior_predictive, num_pp_samples=100, colors = ['#015396', '#FFA500', '#000000'])

    posterior = az.extract(idata_alt)
    n_mean = float(posterior["negative"].mean(dim="sample"))
    p_mean = float(posterior["positive"].mean(dim="sample"))

    return ax, posterior["assign"].mean(dim="sample").to_dataframe(), neg_control, [p_mean, n_mean]

parser = argparse.ArgumentParser(description='Evaluate Data')
parser.add_argument('-scheme', type=str)
parser.add_argument('-data', type=str)
parser.add_argument('-sim_params', type=str)
parser.add_argument('-output', type=str)
parser.add_argument('-n_pools', type=int)
parser.add_argument('-iters', type=int)
parser.add_argument('-len_lst', type=int)
parser.add_argument('-pep_length', type=int)
parser.add_argument('-overlap', type=int)
parser.add_argument('-ep_length', type=int)
parser.add_argument('-n_proteins', type=int)
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
n_control = sim_params['n_control'].iloc[0]
check_results = pd.read_csv(scheme, sep = "\t")

inds = list(cells['Pool'])
obs = list(cells['Percentage'])
fig, probs, neg_ratio, parameters = activation_model(obs, args.n_pools, inds, n_control, cores = 1)
peptide_probs = cpp.peptide_probabilities(check_results, probs)
len_act, notification, lst1, lst2 = cpp.results_analysis(peptide_probs, probs, check_results)
cognate = list(check_results['Peptide'][check_results['Cognate'] == True])

results_row = dict()
results_row['n_pools'] = args.n_pools
results_row['iters'] = args.iters
results_row['len_lst'] = args.len_lst
results_row['pep_length'] = args.pep_length
results_row['shift'] = args.overlap
results_row['ep_length'] = args.ep_length
results_row['n_proteins'] = args.n_proteins
results_row['mu_off'] = args.mu_off
results_row['sigma_off'] = args.sigma_off
results_row['mu_n'] = args.mu_n
results_row['sigma_n'] = args.sigma_n
results_row['sigma_p_r'] = args.sigma_p_r
results_row['sigma_n_r'] = args.sigma_n_r
results_row['low_offset'] = args.low_offset
results_row['r'] = args.r
results_row['error'] = args.error
results_row['# act'] = len_act
if notification == 'All pools were activated':
	results_row['notification'] = 'all activated'
elif notification == 'Zero pools were activated':
    results_row['notification'] = '0 activated'
elif notification == 'No drop-outs were detected':
	results_row['notification'] = '0 drop-outs'
elif notification == 'Cognate peptide is located at one of the ends of the list':
	results_row['notification'] = 'end peptide'
elif notification == 'Cognate peptides are not found':
	results_row['notification'] = 'not found'
elif notification == 'Drop-out was detected':
    results_row['notification'] = 'drop-out'
elif notification == 'False positive was detected':
	results_row['notification'] = 'false positive'
elif notification == 'Analysis error':
    results_row['notification'] = 'error'
results_row['cognate'] = ', '.join(cognate)
results_row['predicted'] = ', '.join(lst1)
results_row['possible'] = ', '.join(lst2)
results_row['conclusion_cognate'] = set(cognate) == set(lst1)
results_row['conclusion_possible'] = all(elem in cognate for elem in lst2)
results_row['negative_model'] = parameters[1]
results_row['positive_model'] = parameters[0]
results_row['neg_control'] = sim_params['n_control'].iloc[0]/np.max(obs)
results_row['pools <= neg_control'] = neg_ratio
results_row['positive_sim'] = sim_params['positive_sim'].iloc[0]
results_row['negative_sim'] = sim_params['negative_sim'].iloc[0]

results_row = pd.DataFrame([results_row])

results_row.to_csv(output, sep = "\t", index = None)