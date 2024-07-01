import copepodTCR as cpp
import numpy as np
import pandas as pd
import random

import argparse

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
fig, probs, parameters = cpp.activation_model(obs, args.n_pools, inds, n_control, cores = 1)
peptide_probs = cpp.peptide_probabilities(check_results, probs)
notification, lst1, lst2 = cpp.results_analysis(peptide_probs, probs, check_results)
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
if notification == 'Zero pools were activated':
	results_row['notification'] = '0 activated'
elif notification == 'No drop-outs were detected':
	results_row['notification'] = '0 drop-outs'
elif notification == 'Drop-out was detected':
	results_row['notification'] = 'Drop-out'
elif notification == 'False positive was detected':
	results_row['notification'] = 'False positive'
elif notification == 'Not found':
	results_row['notification'] = 'None'
results_row['cognate'] = ', '.join(cognate)
results_row['predicted'] = ', '.join(lst1)
results_row['possible'] = ', '.join(lst1)
results_row['conclusion_cognate'] = set(cognate) == set(lst1)
results_row['conclusion_possible'] = all(elem in cognate for elem in lst2)
results_row['negative_model'] = parameters[1]
results_row['positive_model'] = parameters[0]
results_row['neg_control'] = sim_params['n_control'].iloc[0]
results_row['positive_sim'] = sim_params['positive_sim'].iloc[0]
results_row['negative_sim'] = sim_params['negative_sim'].iloc[0]

results_row = pd.DataFrame([results_row])

results_row.to_csv(output, sep = "\t", index = None)