import pandas as pd


ins = snakemake.input
output = snakemake.output[0]

try:
	results = pd.read_csv(output, sep = "\t")
except FileNotFoundError:
	cols = ['method', 'balance_var', 'balance_range', 'balance_iqr',
	'n_pools', 'iters', 'len_lst', 'pep_length', 'shift', 'ep_length',
	'pool_sum_strong', 'pool_sum_weak',
	'mu_off_strong', 'mu_off_weak', 'sigma_off', 'mu_n', 'sigma_n', 'sigma_p_r',
	'sigma_n_r', 'low_offset', 'r',
	'# true act', 'true_pools', 'true_peptides_strong', 'true_peptides_weak',
	'# act', 'model_pools', 'predicted', 'possible',
	'conclusion_cognate_strong', 'conclusion_possible_strong',
	'conclusion_cognate_weak', 'conclusion_possible_weak',
	'pools_indices', 'pools_results',
	'TruePositive', 'TrueNegative', 'FalsePositive', 'FalseNegative',
	'TruePositive_strong', 'TrueNegative_strong', 'FalsePositive_strong', 'FalseNegative_strong',
	'TruePositive_weak', 'TrueNegative_weak', 'FalsePositive_weak', 'FalseNegative_weak',
	'neg_share']

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