import copepodTCR as cpp
import random
import argparse

parser = argparse.ArgumentParser(description='pooling')

parser.add_argument('-n_pools', type=int)
parser.add_argument('-iters', type=int)
parser.add_argument('-len_lst', type=int)
parser.add_argument('-overlap', type=int)
parser.add_argument('-ep_length', type=int)
parser.add_argument('-pep_length', type=int)
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
length = overlap*len_lst + (100-overlap*len_lst%100)
sequence = cpp.random_amino_acid_sequence(length)
lst_all = []
for i in range(0, len(sequence), overlap):
    ps = sequence[i:i+pep_length]
    if len(ps) == pep_length:
        lst_all.append(ps)
lst = lst_all[:len_lst]

### CPP
b, lines = cpp.address_rearrangement_AU(n_pools=n_pools, iters=iters, len_lst=len_lst)
pools, peptide_address = cpp.pooling(lst=lst, addresses=lines, n_pools=n_pools)
check_results = cpp.run_experiment(lst=lst, peptide_address=peptide_address,
    ep_length=ep_length, pools=pools, iters=iters, n_pools=n_pools, regime='without dropouts')
cognate = check_results.sample(1)['Epitope'][0]
check_results['Cognate'] = False
check_results.loc[check_results['Epitope'] == cognate, 'Cognate'] = True

check_results.to_csv(output_path,
	sep = "\t", index = None)