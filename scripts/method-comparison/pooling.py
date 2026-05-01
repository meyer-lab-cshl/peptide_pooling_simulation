import copepodTCR as cpp
import codepub as cdp
import math
import itertools
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='pooling')

parser.add_argument('-method', type=str)
parser.add_argument('-n_pools', type=int)
parser.add_argument('-iters', type=int)
parser.add_argument('-len_lst', type=int)
parser.add_argument('-overlap', type=int)
parser.add_argument('-ep_length', type=int)
parser.add_argument('-pep_length', type=int)
parser.add_argument('-output', type=str)
args = parser.parse_args()

def select_matrix(n):
    rows = int(math.sqrt(n))
    cols = math.ceil(n / rows)

    return rows, cols

def matrix_pooling(n, n_rows, n_cols):
    flat = np.full(n_rows*n_cols, 0)
    flat[:n] = np.arange(1, n+1)
    arranged = flat.reshape(n_rows, n_cols)
    lines = []

    for item in flat[:n]:
        item_idx = np.where(arranged == item)
        lines.append([int(item_idx[0][0]), int(item_idx[1][0])+n_rows])

    b = cdp.item_per_pool(lines, n_rows+n_cols)

    return b, lines

def combinational_pooling(m, r, n):

    lines = list(itertools.combinations(range(m), r))[:n]
    b = cdp.item_per_pool(lines, m)
    return b, lines

method = str(args.method)
m = int(args.n_pools)
r = int(args.iters)
len_lst = int(args.len_lst)
overlap = int(args.overlap)
ep_length = int(args.ep_length)
pep_length = int(args.pep_length)
output_path = str(args.output)

### Peptides
lst_all = []

length = overlap*len_lst + (100-overlap*len_lst%100)
sequence = cpp.random_amino_acid_sequence(length)
for i in range(0, len(sequence), overlap):
    ps = sequence[i:i+pep_length]
    if len(ps) == pep_length:
        lst_all.append(ps)
lst = lst_all[:len_lst]

### copepodTCR
if method == 'copepodTCR':

    b, lines = cdp.bba(m=m, r=r, n=len_lst)


### basic combinatorial pooling
elif method == 'basic':

    b, lines = combinational_pooling(m = m, r = r, n = len_lst)

### matrix pooling
elif method == 'matrix':

    n_rows, n_cols = select_matrix(len_lst)
    b, lines = matrix_pooling(len_lst, n_rows, n_cols)

    m = n_rows + n_cols
    r = 2

b_stat_var = np.var(b)
b_stat_range= np.max(b) - np.min(b)
b_stat_iqr = np.percentile(b, 75) - np.percentile(b, 25)

pools, peptide_address = cpp.pooling(lst=lst, addresses=lines, n_pools=m)
check_results = cpp.run_experiment(lst=lst, peptide_address=peptide_address,
    ep_length=ep_length, pools=pools, iters=r, n_pools=m, regime='without dropouts').reset_index(drop=True)


for i in range(len(check_results)):
    ad_pools = check_results['Act Pools'].iloc[i][1:-1].split(', ')
    pool_sum = 0
    for adp in ad_pools:
        pool_sum = pool_sum + len(pools[int(adp)])
    check_results.loc[i, 'pool_sum'] = pool_sum


idx = check_results['pool_sum'].idxmax()
cognate = check_results.loc[idx, 'Epitope']
check_results['Cognate'] = False
check_results.loc[check_results['Epitope'] == cognate, 'Cognate'] = True
check_results['n_pools'] = m
check_results['iters'] = r
check_results['balance_var'] = b_stat_var
check_results['balance_range'] = b_stat_range
check_results['balance_iqr'] = b_stat_iqr

check_results.to_csv(output_path,
    sep = "\t", index = None)
