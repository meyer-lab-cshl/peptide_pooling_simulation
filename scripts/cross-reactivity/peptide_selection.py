import copepodTCR as cpp
import codepub as cdp
import math
import itertools
import numpy as np
import pandas as pd
import random
import argparse

parser = argparse.ArgumentParser(description='Generating peptides...')

parser.add_argument('-len_lst', type=int)
parser.add_argument('-overlap', type=int)
parser.add_argument('-ep_length', type=int)
parser.add_argument('-pep_length', type=int)
parser.add_argument('-output', type=str)
args = parser.parse_args()


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



peptides_df = []

for p in lst:
    for i in range(len(p)):
        p_ep = p[i:i + ep_length]
        if len(p_ep) == ep_length:
            peptides_df.append([p, p_ep, 'False'])
peptides_df = pd.DataFrame(peptides_df, columns = ['Peptide', 'Epitope', 'Cognate'])
cognate = list(peptides_df['Epitope'].drop_duplicates().sample(2))
cognate_weak = cognate[0]
cognate_strong = cognate[1]
peptides_df.loc[peptides_df['Epitope'] == cognate_strong, 'Cognate'] = 'strong'
peptides_df.loc[peptides_df['Epitope'] == cognate_weak, 'Cognate'] = 'weak'

peptides_df.to_csv(output_path, sep = "\t", index = None)
