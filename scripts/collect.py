import pandas as pd


ins = snakemake.input
output = snakemake.output[0]

try:
	results = pd.read_csv(output, sep = "\t")
except FileNotFoundError:
	cols = ['n_pools', 'iters', 'len_lst', 'pep_length', 'shift', 'ep_length', 'mu_off',
	'sigma_off', 'mu_n', 'sigma_n', 'r', 'notification', 'cognate', 'predicted', 'possible']

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