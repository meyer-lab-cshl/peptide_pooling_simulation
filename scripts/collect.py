import pandas as pd


ins = snakemake.input
output = snakemake.output[0]

try:
	results = pd.read_csv(output, sep = "\t")
except FileNotFoundError:
	cols = ['n_pools', 'iters', 'len_lst', 'pep_length', 'shift', 'ep_length', 'n_proteins', 'mu_off',
	'sigma_off', 'mu_n', 'sigma_n', 'r', 'error', 'sigma_p_r',
	'sigma_n_r', 'low_offset', 'notification', 'cognate', 'predicted', 'possible',
	'conclusion_cognate', 'conclusion_possible']

	results = pd.DataFrame(columns = cols)
	results.to_csv('results/summary_results.tsv',
		sep = "\t", index = None)

for item in ins:
	row = pd.read_csv(item, sep = "\t")
	if len(results) == 0:
		results = row
	else:
		results = pd.concat([row, results])

results.to_csv(output, sep = "\t", index = None)