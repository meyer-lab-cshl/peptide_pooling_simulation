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
    neg_share = 1 - (iters + normal -1)/n_pools
else:
    neg_share = (n_pools - iters)/n_pools

model, fig, probs, n_c, pp, parameters = cpp.activation_model(obs, n_pools, inds, n_control, neg_share, cores = 1)
model_n, fig_n, probs_n, n_c_n, pp_n, parameters_n = cpp.activation_model(obs = obs, n_pools = n_pools, inds = inds, neg_share = neg_share, cores = 1)

peptide_probs = cpp.peptide_probabilities(check_results, probs)
len_act, notification, lst1, lst2 = cpp.results_analysis(peptide_probs, probs, check_results)

peptide_probs_n = cpp.peptide_probabilities(check_results, probs_n)
len_act_n, notification_n, lst1_n, lst2_n = cpp.results_analysis(peptide_probs_n, probs_n, check_results)

cognate = check_results['Act Pools'][check_results['Cognate'] == True].iloc[0]
cognate_peptides = set(check_results['Peptide'][check_results['Cognate'] == True])
cognate = [int(x) for x in cognate[1:-1].split(', ')]
for i in range(len(error_pools)):
    cognate.remove(error_pools[i])

if args.error == 100:
    cognate = []

act_pools_model = list(probs.index[probs['assign'] < 0.5])
act_pools_model_n = list(probs_n.index[probs_n['assign'] < 0.5])

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

tp, tn, fp, fn = calculate_tp_tn_fp_fn(cognate, act_pools_model, n_pools)
tp_n, tn_n, fp_n, fn_n = calculate_tp_tn_fp_fn(cognate, act_pools_model_n, n_pools)

results_row = dict()
results_row['method'] = args.method
results_row['balance_var'] = check_results['balance_var'].iloc[0]
results_row['balance_range'] = check_results['balance_range'].iloc[0]
results_row['balance_iqr'] = check_results['balance_iqr'].iloc[0]
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

results_row['# act'] = len(act_pools_model)
results_row['model_pools'] = act_pools_model

results_row['# act n'] = len(act_pools_model_n)
results_row['model_pools_n'] = act_pools_model_n

results_row['TruePositive'] = len(tp_n)
results_row['TrueNegative'] = len(tn_n)
results_row['FalsePositive'] = len(fp_n)
results_row['FalseNegative'] = len(fn_n)

results_row['TruePositive_n'] = len(tp)
results_row['TrueNegative_n'] = len(tn)
results_row['FalsePositive_n'] = len(fp)
results_row['FalseNegative_n'] = len(fn)

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
results_row['negative_model'] = parameters[1]
results_row['positive_model'] = parameters[0]

results_row['negative_model_n'] = parameters_n[1]
results_row['positive_model_n'] = parameters_n[0]

results_row['neg_control'] = sim_params['n_control'].iloc[0]/np.max(obs)
results_row['neg_control_n'] = np.mean(sorted(obs)[:inds.count(0)])/np.max(obs)
results_row['neg_share'] = neg_share
results_row['positive_sim_norm'] = float(sim_params['positive_sim'].iloc[0])/np.max(obs)
results_row['negative_sim_norm'] = float(sim_params['negative_sim'].iloc[0])/np.max(obs)
results_row['positive_sim'] = float(sim_params['positive_sim'].iloc[0])
results_row['negative_sim'] = float(sim_params['negative_sim'].iloc[0])

results_row = pd.DataFrame([results_row])

results_row.to_csv(output, sep = "\t", index = None)