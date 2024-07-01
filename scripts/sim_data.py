import copepodTCR as cpp
import random
import pandas as pd
import argparse
from collections import Counter
import pymc as pm

parser = argparse.ArgumentParser(description='Data Simulation')
parser.add_argument('-check_results', type=str)
parser.add_argument('-output', type=str)
parser.add_argument('-output_params', type=str)
parser.add_argument('-mu_off', type=float)
parser.add_argument('-sigma_off', type=float)
parser.add_argument('-mu_n', type=float)
parser.add_argument('-sigma_n', type=float)
parser.add_argument('-r', type=int)
parser.add_argument('-sigma_p_r', type=float)
parser.add_argument('-sigma_n_r', type=float)
parser.add_argument('-low_offset', type=float)
parser.add_argument('-n_pools', type=int)
parser.add_argument('-error', type=int)
args = parser.parse_args()

check_results = pd.read_csv(args.check_results, sep='\t')
lst = list(set(check_results['Peptide']))

inds_p_check = check_results[check_results['Cognate'] == True]['Act Pools'].values[0]
inds_p_check = [int(x) for x in inds_p_check[1:-1].split(', ')]

for i in range(args.error):
    inds_p_check.pop(random.randint(0, len(inds_p_check)-1))

ads = []
for item in list(check_results[check_results['Cognate']]['Address']):
    item = item[1:-1].split(', ')
    ads.append(item)
inds_pl_check = []
for i in range(len(ads)-1):
    inds_pl_check.append(set(ads[i]).difference(set(ads[i+1])))
inds_pl_check = [int(x) for xs in inds_pl_check for x in xs]
inds_p_check = list(set(inds_p_check) - set(inds_pl_check))

pl_shape = len(inds_pl_check)
p_shape = len(inds_p_check)
n_shape = args.n_pools - p_shape - pl_shape

inds_n_check = [item for item in range(args.n_pools) if item not in inds_p_check and item not in inds_pl_check]

p_results, pl_results, n_results, n_control, parameters = cpp.simulation(args.mu_off, args.sigma_off, args.mu_n, args.sigma_n, args.r,
    args.sigma_p_r, args.sigma_n_r, args.n_pools, p_shape, pl_shape, args.low_offset)

cells = pd.DataFrame({
    'Pool': inds_p_check * args.r + inds_pl_check * args.r + inds_n_check * args.r,
    'Percentage': p_results + pl_results + n_results
})

sim_parameters = pd.DataFrame({'n_control':n_control[0], 'negative_sim': parameters[1], 'positive_sim': parameters[0]}, index = [0])

# Writing output
cells.to_csv(args.output, sep="\t", index=None)
sim_parameters.to_csv(args.output_params, sep="\t", index=None)