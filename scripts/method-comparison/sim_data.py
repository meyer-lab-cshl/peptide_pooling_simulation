import copepodTCR as cpp
import random
import pandas as pd
import argparse
from collections import Counter
import pymc as pm

def simulation(mu_off, sigma_off, mu_n, sigma_n, r, sigma_p_r, sigma_n_r, n_pools, p_shape,
               pl_shape, low_offset, cores=1):

    n_shape = n_pools-p_shape-pl_shape
    with pm.Model() as simulation:
        # offset
        offset = pm.TruncatedNormal("offset", mu=mu_off, sigma=sigma_off, lower=0, upper=100)
    
        # Negative
        n = pm.TruncatedNormal('n', mu=mu_n, sigma=sigma_n, lower=0, upper=100)
        # Positive
        raw_p = n + offset
        p = pm.Deterministic("p", pm.math.clip(raw_p, 0, 100))
        # Low positive
        p_low = pm.Deterministic("p_low", p*low_offset)

        # Negative pools
        n_pools = pm.TruncatedNormal('n_pools', mu=n, sigma=sigma_n, lower=0, upper=100, shape = n_shape)
        inds_n = list(range(n_shape))*r
        n_shape_r = n_shape*r

        # Positive pools
        p_pools = pm.TruncatedNormal('p_pools', mu=p, sigma=sigma_off, lower=0, upper=100, shape = p_shape)
        inds_p = list(range(p_shape))*r
        p_shape_r = p_shape*r

        # Low positive pools
        pl_pools = pm.TruncatedNormal('pl_pools', mu=p_low, sigma=sigma_off, lower=0, upper=100, shape = pl_shape)
        inds_pl = list(range(pl_shape))*r
        pl_shape_r = pl_shape*r

        # With replicas
        p_pools_r = pm.TruncatedNormal('p_pools_r', mu=p_pools[inds_p], sigma=sigma_p_r, lower=0, upper=100, shape=p_shape_r)
        pl_pools_r = pm.TruncatedNormal('pl_pools_r', mu=pl_pools[inds_pl], sigma=sigma_p_r, lower=0, upper=100, shape=pl_shape_r)
        n_pools_r = pm.TruncatedNormal('n_pools_r', mu=n_pools[inds_n], sigma=sigma_n_r, lower=0, upper=100, shape=n_shape_r)

        # negative control
        n_control = pm.TruncatedNormal('n_control', mu=n, sigma=sigma_n, lower=0, upper=100, shape=r)

        trace = pm.sample(draws=1, cores = cores)
        
    p_results = trace.posterior.p_pools_r.mean(dim="chain").values.tolist()[0]
    pl_results = trace.posterior.pl_pools_r.mean(dim="chain").values.tolist()[0]
    n_results = trace.posterior.n_pools_r.mean(dim="chain").values.tolist()[0]
    n_control = trace.posterior.n_control.mean(dim="chain").values.tolist()[0]

    n_mean = float(trace.posterior.n.mean())
    p_mean = float(trace.posterior.offset.mean())

    return p_results, pl_results, n_results, n_control, [p_mean, n_mean]

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
parser.add_argument('-error', type=int)
args = parser.parse_args()

check_results = pd.read_csv(args.check_results, sep='\t')
n_pools = int(check_results['n_pools'].iloc[0])
lst = list(set(check_results['Peptide']))

inds_p_check = check_results[check_results['Cognate'] == True]['Act Pools'].values[0]
inds_p_check = [int(x) for x in inds_p_check[1:-1].split(', ')]

if args.error != 100:

    error_pools = []

    for i in range(args.error):
        error_pool = random.randint(0, len(inds_p_check)-1)
        error_pools.append(inds_p_check[error_pool])
        inds_p_check.pop(error_pool)

    ads = []
    for item in list(check_results[check_results['Cognate']]['Address']):
        item = item[1:-1].split(', ')
        ads.append(item)
    inds_pl_check = []
    for i in range(len(ads)-1):
        inds_pl_check.append(set(ads[i]).difference(set(ads[i+1])))
    inds_pl_check = [int(x) for xs in inds_pl_check for x in xs]
    inds_p_check = list(set(inds_p_check) - set(inds_pl_check))

else:
    error_pools = []
    inds_p_check = []
    inds_pl_check = []

pl_shape = len(inds_pl_check)
p_shape = len(inds_p_check)
n_shape = n_pools - p_shape - pl_shape

inds_n_check = [item for item in range(n_pools) if item not in inds_p_check and item not in inds_pl_check]

p_results, pl_results, n_results, n_control, parameters = simulation(args.mu_off, args.sigma_off, args.mu_n, args.sigma_n, args.r,
    args.sigma_p_r, args.sigma_n_r, n_pools, p_shape, pl_shape, args.low_offset)

cells = pd.DataFrame({
    'Pool': inds_p_check * args.r + inds_pl_check * args.r + inds_n_check * args.r,
    'Percentage': p_results + pl_results + n_results
})

sim_parameters = pd.DataFrame({'n_control':n_control[0], 'negative_sim': parameters[1],
    'positive_sim': parameters[0], 'error_pools': ';'.join(map(str, error_pools))}, index = [0])

# Writing output
cells.to_csv(args.output, sep="\t", index=None)
sim_parameters.to_csv(args.output_params, sep="\t", index=None)