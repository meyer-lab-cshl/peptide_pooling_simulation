import copepodTCR as cpp
import pandas as pd
import argparse
from collections import Counter
import pymc as pm

parser = argparse.ArgumentParser(description='Data Simulation')
parser.add_argument('-check_results', type=str)
parser.add_argument('-output', type=str)
parser.add_argument('-mu_off', type=float)
parser.add_argument('-sigma_off', type=float)
parser.add_argument('-mu_n', type=float)
parser.add_argument('-sigma_n', type=float)
parser.add_argument('-r', type=int)
parser.add_argument('-n_pools', type=int)
parser.add_argument('-iters', type=int)
parser.add_argument('-ep_length', type=int)
args = parser.parse_args()

check_results = pd.read_csv(args.check_results, sep='\t')
lst = list(set(check_results['Peptide']))
p_shape = len(list(check_results['Act Pools'][check_results['Cognate'] == True])[0][1:-1].split(', '))

n_shape = args.n_pools - p_shape

inds_p_check = check_results[check_results['Cognate'] == True]['Act Pools'].values[0]
inds_p_check = [int(x) for x in inds_p_check[1:-1].split(', ')]

inds_n_check = [item for item in range(args.n_pools) if item not in inds_p_check]

p_results, n_results = cpp.simulation(args.mu_off, args.sigma_off, args.mu_n, args.sigma_n, args.r, args.n_pools, p_shape, cores = 1)

cells = pd.DataFrame({
    'Pool': inds_p_check * args.r + inds_n_check * args.r,
    'Percentage': p_results + n_results
})

# Writing output
cells.to_csv(args.output, sep="\t", index=None)


