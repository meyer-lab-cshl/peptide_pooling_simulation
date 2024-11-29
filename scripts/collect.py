import pandas as pd


ins = snakemake.input
output = snakemake.output[0]

try:
	results = pd.read_csv(output, sep = "\t")
except FileNotFoundError:
	cols = ['n_pools', 'iters', 'len_lst', 'pep_length', 'shift', 'ep_length', 'n_proteins', 'mu_off',
	'sigma_off', 'mu_n', 'sigma_n', 'sigma_p_r',
	'sigma_n_r', 'low_offset',
	'r', 'error', '# act',
	'true_pools', 'model1_pools', 'model2_pools',
	'TruePositive_1', 'TrueNegative_1', 'FalsePositive_1', 'FalseNegative_1',
	'TruePositive_2', 'TrueNegative_2', 'FalsePositive_2', 'FalseNegative_2',
	'negative_model1', 'positive_model1', 'negative_model2', 'positive_model2',
	'neg_control', 'neg_share',
	'positive_sim', 'negative_sim']

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