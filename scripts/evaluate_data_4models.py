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

def activation_model1(obs, n_pools, inds, neg_control = None, neg_share = None, cores=1):

    """
    Takes a list with observed data (obs), number of pools (n_pools), and indices for the observed data if there were mutiple replicas.
    Returns model fit and a dataframe with probabilities of each pool being drawn from negative or positive distributions.
    """
    
    coords = dict(pool=range(n_pools), component=("positive", "negative"))
    if neg_share is None:
        neg_share = 0.5
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
        offset_proportion = pm.Beta("offset_proportion", alpha=2, beta=2)
        offset = pm.Deterministic("offset", (1 - negative) * offset_proportion)
        #offset = pm.TruncatedNormal('offset', mu = 0.6, sigma = 0.1, upper = 1, lower = 0)

        #positive = pm.Deterministic("positive", negative + offset, upper = 0, lower = 1)
        positive = pm.Deterministic("positive", negative + offset)
    
        p = pm.Beta("p", alpha=neg_share * 100, beta=(1 - neg_share) * 100)
        component = pm.Bernoulli("assign", p, dims="pool")

        mu_pool = negative * component + positive * (1 - component)
    
        sigma_neg = pm.Exponential("sigma_neg", 0.2)
        #sigma_delta = pm.Exponential("sigma_delta", 0.5)
        sigma_pos = pm.Exponential("sigm_pos", 0.2)
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
    p_mean = float(posterior["positive"].mean(dim="sample"))

    posterior_p_mean = posterior["p"].mean(dim="sample").item()
    print(f"Posterior mean of p: {posterior_p_mean:.3f}")

    #return ax, posterior["assign"].mean(dim="sample").to_dataframe(), neg_control, [p_mean, n_mean]
    return model, ax, posterior["assign"].mean(dim="sample").to_dataframe(), neg_control, idata_alt, [p_mean, n_mean]

def activation_model2(obs, n_pools, inds, neg_control = None, neg_share = None, cores=1):

    """
    Takes a list with observed data (obs), number of pools (n_pools), and indices for the observed data if there were mutiple replicas.
    Returns model fit and a dataframe with probabilities of each pool being drawn from negative or positive distributions.
    """
    
    coords = dict(pool=range(n_pools), component=("positive", "negative"))
    if neg_share is None:
        neg_share = 0.5
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
        offset_proportion = pm.Beta("offset_proportion", alpha=2, beta=2)
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
    p_mean = float(posterior["positive"].mean(dim="sample"))

    posterior_p_mean = posterior["p"].mean(dim="sample").item()
    print(f"Posterior mean of p: {posterior_p_mean:.3f}")

    #return ax, posterior["assign"].mean(dim="sample").to_dataframe(), neg_control, [p_mean, n_mean]
    return model, ax, posterior["assign"].mean(dim="sample").to_dataframe(), neg_control, idata_alt, [p_mean, n_mean]

def activation_model3(obs, n_pools, inds, neg_control = None, neg_share = None, cores=1):

    """
    Takes a list with observed data (obs), number of pools (n_pools), and indices for the observed data if there were mutiple replicas.
    Returns model fit and a dataframe with probabilities of each pool being drawn from negative or positive distributions.
    """
    
    coords = dict(pool=range(n_pools), component=("positive", "negative"))
    if neg_share is None:
        neg_share = 0.5
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
    
        sigma_neg = pm.Exponential("sigma_neg", 0.2)
        #sigma_delta = pm.Exponential("sigma_delta", 0.5)
        sigma_pos = pm.Exponential("sigm_pos", 0.2)
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
    p_mean = float(posterior["positive"].mean(dim="sample"))

    posterior_p_mean = posterior["p"].mean(dim="sample").item()
    print(f"Posterior mean of p: {posterior_p_mean:.3f}")

    #return ax, posterior["assign"].mean(dim="sample").to_dataframe(), neg_control, [p_mean, n_mean]
    return model, ax, posterior["assign"].mean(dim="sample").to_dataframe(), neg_control, idata_alt, [p_mean, n_mean]

def activation_model4(obs, n_pools, inds, neg_control = None, neg_share = None, cores=1):

    """
    Takes a list with observed data (obs), number of pools (n_pools), and indices for the observed data if there were mutiple replicas.
    Returns model fit and a dataframe with probabilities of each pool being drawn from negative or positive distributions.
    """
    
    coords = dict(pool=range(n_pools), component=("positive", "negative"))
    if neg_share is None:
        neg_share = 0.5
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
    p_mean = float(posterior["positive"].mean(dim="sample"))

    posterior_p_mean = posterior["p"].mean(dim="sample").item()
    print(f"Posterior mean of p: {posterior_p_mean:.3f}")

    #return ax, posterior["assign"].mean(dim="sample").to_dataframe(), neg_control, [p_mean, n_mean]
    return model, ax, posterior["assign"].mean(dim="sample").to_dataframe(), neg_control, idata_alt, [p_mean, n_mean]

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
n_control = list(sim_params['n_control'])
check_results = pd.read_csv(scheme, sep = "\t")

inds = list(cells['Pool'])
obs = list(cells['Percentage'])

all_lst = list(set(check_results['Peptide']))
c, _ = cpp.how_many_peptides(all_lst, args.ep_length)
normal = max(c, key=c.get)
neg_share = 1 - (args.iters + normal -1)/args.n_pools
model1, fig1, probs1, n_c1, pp1, parameters1 = activation_model1(obs, args.n_pools, inds, n_control, neg_share, cores = 1)
model2, fig2, probs2, n_c2, pp2, parameters2 = activation_model2(obs, args.n_pools, inds, n_control, neg_share, cores = 1)
model3, fig3, probs3, n_c3, pp3, parameters3 = activation_model3(obs, args.n_pools, inds, n_control, neg_share, cores = 1)
model4, fig4, probs4, n_c4, pp4, parameters4 = activation_model4(obs, args.n_pools, inds, n_control, neg_share, cores = 1)
#peptide_probs = cpp.peptide_probabilities(check_results, probs)
#len_act, notification, lst1, lst2 = cpp.results_analysis(peptide_probs, probs, check_results)
cognate = check_results['Act Pools'][check_results['Cognate'] == True].iloc[0]
cognate = [int(x) for x in cognate[1:-1].split(', ')]
act_pools_model1 = list(probs1.index[probs1['assign'] < 0.5])
act_pools_model2 = list(probs2.index[probs2['assign'] < 0.5])
act_pools_model3 = list(probs3.index[probs3['assign'] < 0.5])
act_pools_model4 = list(probs4.index[probs4['assign'] < 0.5])

def calculate_tp_tn_fp_fn(cognate, act_pools):
    tp = []
    tn = []
    fp = []
    fn = []

    for i in set(inds):
        if i in cognate and i in act_pools:
            tp.append(i)
        elif i not in cognate and i not in act_pools:
            tn.append(i)
        elif i not in cognate and i in act_pools:
            fp.append(i)
        elif i in cognate and i not in act_pools:
            fn.append(i)

    return tp, tn, fp, fn

tp1, tn1, fp1, fn1 = calculate_tp_tn_fp_fn(cognate, act_pools_model1)
tp2, tn2, fp2, fn2 = calculate_tp_tn_fp_fn(cognate, act_pools_model2)
tp3, tn3, fp3, fn3 = calculate_tp_tn_fp_fn(cognate, act_pools_model3)
tp4, tn4, fp4, fn4 = calculate_tp_tn_fp_fn(cognate, act_pools_model4)


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

results_row['# true act'] = len(cognate)

results_row['# act 1'] = len(act_pools_model1)
results_row['# act 2'] = len(act_pools_model2)
results_row['# act 3'] = len(act_pools_model3)
results_row['# act 4'] = len(act_pools_model4)

#if notification == 'All pools were activated':
#   results_row['notification'] = 'all activated'
#elif notification == 'Zero pools were activated':
#    results_row['notification'] = '0 activated'
#elif notification == 'No drop-outs were detected':
#   results_row['notification'] = '0 drop-outs'
#elif notification == 'Cognate peptide is located at one of the ends of the list':
#   results_row['notification'] = 'end peptide'
#elif notification == 'Cognate peptides are not found':
#   results_row['notification'] = 'not found'
#elif notification == 'Drop-out was detected':
#    results_row['notification'] = 'drop-out'
#elif notification == 'False positive was detected':
#   results_row['notification'] = 'false positive'
#elif notification == 'Analysis error':
#    results_row['notification'] = 'error'

results_row['true_pools'] = cognate

results_row['model1_pools'] = act_pools_model1
results_row['model2_pools'] = act_pools_model2
results_row['model3_pools'] = act_pools_model3
results_row['model4_pools'] = act_pools_model4

results_row['TruePositive_1'] = len(tp1)
results_row['TrueNegative_1'] = len(tn1)
results_row['FalsePositive_1'] = len(fp1)
results_row['FalseNegative_1'] = len(fn1)

results_row['TruePositive_2'] = len(tp2)
results_row['TrueNegative_2'] = len(tn2)
results_row['FalsePositive_2'] = len(fp2)
results_row['FalseNegative_2'] = len(fn2)

results_row['TruePositive_3'] = len(tp3)
results_row['TrueNegative_3'] = len(tn3)
results_row['FalsePositive_3'] = len(fp3)
results_row['FalseNegative_3'] = len(fn3)

results_row['TruePositive_4'] = len(tp4)
results_row['TrueNegative_4'] = len(tn4)
results_row['FalsePositive_4'] = len(fp4)
results_row['FalseNegative_4'] = len(fn4)

#results_row['predicted'] = ', '.join(lst1)
#results_row['possible'] = ', '.join(lst2)
#results_row['conclusion_cognate'] = set(cognate) == set(lst1)
#results_row['conclusion_possible'] = all(elem in cognate for elem in lst2)

results_row['negative_model1'] = parameters1[1]
results_row['positive_model1'] = parameters1[0]
results_row['negative_model2'] = parameters2[1]
results_row['positive_model2'] = parameters2[0]
results_row['negative_model3'] = parameters3[1]
results_row['positive_model3'] = parameters3[0]
results_row['negative_model4'] = parameters4[1]
results_row['positive_model4'] = parameters4[0]

results_row['neg_control'] = sim_params['n_control'].iloc[0]/np.max(obs)
results_row['neg_share'] = neg_share
results_row['positive_sim'] = float(sim_params['positive_sim'].iloc[0])/np.max(obs)
results_row['negative_sim'] = float(sim_params['negative_sim'].iloc[0])/np.max(obs)

results_row = pd.DataFrame([results_row])

results_row.to_csv(output, sep = "\t", index = None)