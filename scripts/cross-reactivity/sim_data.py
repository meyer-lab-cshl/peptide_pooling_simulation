import copepodTCR as cpp
import random
import pandas as pd
import numpy as np
import argparse
from collections import Counter
import pymc as pm

parser = argparse.ArgumentParser(description='Data Simulation')
parser.add_argument('-check_results', type=str)
parser.add_argument('-output', type=str)
parser.add_argument('-mu_off_strong', type=float)
parser.add_argument('-mu_off_weak', type=float)
parser.add_argument('-sigma_off', type=float)
parser.add_argument('-mu_n', type=float)
parser.add_argument('-sigma_n', type=float)
parser.add_argument('-r', type=int)
parser.add_argument('-sigma_p_r', type=float)
parser.add_argument('-sigma_n_r', type=float)
args = parser.parse_args()

check_results = pd.read_csv(args.check_results, sep='\t')
n_pools = int(check_results['n_pools'].iloc[0])
lst = list(set(check_results['Peptide']))

sigma_off = args.sigma_off
mu_n = args.mu_n
sigma_n = args.sigma_n
sigma_p_r = args.sigma_p_r
sigma_n_r = args.sigma_n_r
r = args.r

## strong and weak
mu_off_weak = args.mu_off_weak
mu_off_strong = mu_off_weak + args.mu_off_strong

## low offset
pl_shape = 0
low_offset = 1

## strong
inds_p_check = check_results[check_results['Cognate'] == 'strong']['Act Pools'].values[0]

inds_p_check = [int(x) for x in inds_p_check[1:-1].split(', ')]
inds_n_check = []
for item in range(n_pools):
    if item not in inds_p_check:
        inds_n_check.append(item)
p_shape = len(inds_p_check)

p_results, pl_results, n_results, n_control, parameters = cpp.simulation(mu_off_strong, sigma_off, mu_n, sigma_n, r, sigma_p_r, sigma_n_r, n_pools, p_shape, pl_shape, low_offset)
cells_strong = pd.DataFrame({'Pool': list(np.repeat(inds_p_check, r)) + list(np.repeat(inds_n_check, r)), 'Percentage_strong': p_results + n_results})
cells_strong['replicate'] = cells_strong.groupby('Pool').cumcount()

## weak
inds_p_check = check_results[check_results['Cognate'] == 'weak']['Act Pools'].values[0]

inds_p_check = [int(x) for x in inds_p_check[1:-1].split(', ')]
inds_n_check = []
for item in range(n_pools):
    if item not in inds_p_check:
        inds_n_check.append(item)
p_shape = len(inds_p_check)

p_results, pl_results, n_results, n_control, parameters = cpp.simulation(mu_off_weak, sigma_off, mu_n, sigma_n, r, sigma_p_r, sigma_n_r, n_pools, p_shape, pl_shape, low_offset)
cells_weak = pd.DataFrame({'Pool': list(np.repeat(inds_p_check, r)) + list(np.repeat(inds_n_check, r)), 'Percentage_weak': p_results + n_results})
cells_weak['replicate'] = cells_weak.groupby('Pool').cumcount()

cells = cells_strong.merge(cells_weak, on=['Pool', 'replicate'], how='outer')

cells[['Percentage_strong', 'Percentage_weak']] = cells[['Percentage_strong', 'Percentage_weak']].fillna(0)

cells['Percentage'] = cells['Percentage_strong'] + cells['Percentage_weak']

cells = cells[['Pool', 'Percentage']]

## Writing output
cells.to_csv(args.output, sep="\t", index=None)