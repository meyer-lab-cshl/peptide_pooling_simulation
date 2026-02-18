import pandas as pd


ins = snakemake.input
output = snakemake.output[0]

try:
	results = pd.read_csv(output, sep = "\t")
except FileNotFoundError:
	cols = ['method', 'balance_var', 'balance_range', 'balance_iqr',
	'n_pools', 'iters', 'len_lst', 'pep_length', 'shift', 'ep_length', 'mu_off',
	'sigma_off', 'mu_n', 'sigma_n', 'sigma_p_r',
	'sigma_n_r', 'low_offset', 'r', 'error', 'error_pools',
	'# true act', 'true_pools', 'true_peptides',
	'# act', 'model_pools', 'predicted', 'possible',
	'# act n', 'model_pools_n', 'predicted_n', 'possible_n',
	'conclusion_cognate', 'conclusion_possible',
	'conclusion_cognate_n', 'conclusion_possible_n',
	'pools_indices', 'pools_results', 'pools_var',
	'TruePositive', 'TrueNegative', 'FalsePositive', 'FalseNegative',
	'TruePositive_n', 'TrueNegative_n', 'FalsePositive_n', 'FalseNegative_n',
	'negative_model', 'negative_sim', 'negative_sim_norm',
	'positive_model', 'positive_sim', 'positive_sim_norm',
	'neg_control', 'neg_control_n', 'neg_share']

	results = pd.DataFrame(columns = cols)
	results.to_csv('results/method-comparison/summary_results.tsv',
		sep = "\t", index = None)

for item in ins:
	try:
		row = pd.read_csv(item, sep = "\t")

		if len(results) == 0:
			results = row
		else:
			results = pd.concat([row, results])
	except pd.errors.EmptyDataError:
		print(item)

results.to_csv(output, sep = "\t", index = None)