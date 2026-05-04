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
parser.add_argument('-output', type=str)
parser.add_argument('-len_lst', type=int)
parser.add_argument('-pep_length', type=int)
parser.add_argument('-overlap', type=int)
parser.add_argument('-ep_length', type=int)
parser.add_argument('-mu_off_strong', type=float)
parser.add_argument('-mu_off_weak', type=float)
parser.add_argument('-sigma_off', type=float)
parser.add_argument('-mu_n', type=float)
parser.add_argument('-sigma_n', type=float)
parser.add_argument('-r', type=int)
parser.add_argument('-sigma_p_r', type=float)
parser.add_argument('-sigma_n_r', type=float)
args = parser.parse_args()

scheme = str(args.scheme)
data = str(args.data)
output = str(args.output)

cells = pd.read_csv(data, sep = "\t")

check_results = pd.read_csv(scheme, sep = "\t")
n_pools = check_results['n_pools'].iloc[0]
iters = check_results['iters'].iloc[0]

inds = list(cells['Pool'])
obs = list(cells['Percentage'])


all_lst = list(check_results['Peptide'].unique())
c, _ = cpp.how_many_peptides(all_lst, args.ep_length)
normal = max(c, key=c.get)
neg_share = 1 - (iters + normal -1)/n_pools

model, fig, probs, n_c, pp, parameters = cpp.activation_model(obs = obs, n_pools = n_pools, inds = inds, neg_share = neg_share, cores = 1)

peptide_probs = cpp.peptide_probabilities(check_results, probs)
len_act, notification, lst1, lst2 = cpp.results_analysis(peptide_probs, probs, check_results)


cognate_strong = check_results['Act Pools'][check_results['Cognate'] == 'strong'].iloc[0]
cognate_peptides_strong = list(check_results['Peptide'][check_results['Cognate'] == 'strong'])
cognate_weak = check_results['Act Pools'][check_results['Cognate'] == 'weak'].iloc[0]
cognate_peptides_weak = list(check_results['Peptide'][check_results['Cognate'] == 'weak'])

pool_sum_strong = check_results['pool_sum'][check_results['Cognate'] == 'strong'].iloc[0]
pool_sum_weak = check_results['pool_sum'][check_results['Cognate'] == 'weak'].iloc[0]

cognate_strong_pools = [int(x) for x in cognate_strong[1:-1].split(', ')]
cognate_weak_pools = [int(x) for x in cognate_weak[1:-1].split(', ')]
cognate_pools = sorted(set(cognate_strong_pools) | set(cognate_weak_pools))

act_pools_model = list(probs.index[probs['assign'] < 0.5])

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

tp, tn, fp, fn = calculate_tp_tn_fp_fn(cognate_pools, act_pools_model, n_pools)
tp_strong, tn_strong, fp_strong, fn_strong = calculate_tp_tn_fp_fn(cognate_strong_pools, act_pools_model, n_pools)
tp_weak, tn_weak, fp_weak, fn_weak = calculate_tp_tn_fp_fn(cognate_weak_pools, act_pools_model, n_pools)

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

results_row['mu_off_strong'] = args.mu_off_strong
results_row['mu_off_weak'] = args.mu_off_weak

results_row['sigma_off'] = args.sigma_off
results_row['mu_n'] = args.mu_n
results_row['sigma_n'] = args.sigma_n
results_row['sigma_p_r'] = args.sigma_p_r
results_row['sigma_n_r'] = args.sigma_n_r
results_row['low_offset'] = 1
results_row['r'] = args.r

results_row['pool_sum_strong'] = pool_sum_strong
results_row['pool_sum_weak'] = pool_sum_weak

results_row['# true act'] = len(cognate_pools)
results_row['true_pools'] = cognate_pools
results_row['true_peptides_strong'] = ', '.join(cognate_peptides_strong)
results_row['true_peptides_weak'] = ', '.join(cognate_peptides_weak)

results_row['# act'] = len(act_pools_model)
results_row['model_pools'] = act_pools_model

results_row['TruePositive'] = len(tp)
results_row['TrueNegative'] = len(tn)
results_row['FalsePositive'] = len(fp)
results_row['FalseNegative'] = len(fn)

results_row['TruePositive_strong'] = len(tp_strong)
results_row['TrueNegative_strong'] = len(tn_strong)
results_row['FalsePositive_strong'] = len(fp_strong)
results_row['FalseNegative_strong'] = len(fn_strong)

results_row['TruePositive_weak'] = len(tp_weak)
results_row['TrueNegative_weak'] = len(tn_weak)
results_row['FalsePositive_weak'] = len(fp_weak)
results_row['FalseNegative_weak'] = len(fn_weak)

results_row['predicted'] = ', '.join(lst1)
results_row['possible'] = ', '.join(lst2)

results_row['conclusion_cognate_strong'] = all(elem in lst1 for elem in cognate_peptides_strong)
results_row['conclusion_cognate_weak'] = all(elem in lst1 for elem in cognate_peptides_weak)
results_row['conclusion_possible_strong'] = all(elem in lst2 for elem in cognate_peptides_strong)
results_row['conclusion_possible_weak'] = all(elem in lst2 for elem in cognate_peptides_weak)

results_row['pools_indices'] = ', '.join([str(x) for x in inds])
results_row['pools_results'] = ', '.join([str(x) for x in obs])

results_row['neg_share'] = neg_share

results_row = pd.DataFrame([results_row])

results_row.to_csv(output, sep = "\t", index = None)