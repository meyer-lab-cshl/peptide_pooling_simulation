import copepodTCR as cpp
import codepub as cdp
import random
import argparse

parser = argparse.ArgumentParser(description='pooling')

parser.add_argument('-n_pools', type=int)
parser.add_argument('-iters', type=int)
parser.add_argument('-len_lst', type=int)
parser.add_argument('-overlap', type=int)
parser.add_argument('-ep_length', type=int)
parser.add_argument('-pep_length', type=int)
parser.add_argument('-n_proteins', type=int)
parser.add_argument('-output', type=str)
args = parser.parse_args()

n_pools = int(args.n_pools)
iters = int(args.iters)
len_lst = int(args.len_lst)
overlap = int(args.overlap)
ep_length = int(args.ep_length)
pep_length = int(args.pep_length)
output_path = str(args.output)

### Peptides
lst_all = []

for n in range(args.n_proteins):
    lst_pr = []
    length = overlap*len_lst + (100-overlap*len_lst%100)//args.n_proteins
    sequence = cpp.random_amino_acid_sequence(length)
    for i in range(0, len(sequence), overlap):
        ps = sequence[i:i+pep_length]
        if len(ps) == pep_length:
            lst_all.append(ps)
    lst_all = lst_all + lst_pr[:len_lst//3 + 10]
lst = lst_all[:len_lst]

### CPP
b, lines = cdp.rcau(n_pools=n_pools, iters=iters, len_lst=len_lst)
pools, peptide_address = cpp.pooling(lst=lst, addresses=lines, n_pools=n_pools)
check_results = cpp.run_experiment(lst=lst, peptide_address=peptide_address,
    ep_length=ep_length, pools=pools, iters=iters, n_pools=n_pools, regime='without dropouts')
cognate = check_results.sample(1)['Epitope'][0]
check_results['Cognate'] = False
check_results.loc[check_results['Epitope'] == cognate, 'Cognate'] = True

check_results.to_csv(output_path,
	sep = "\t", index = None)