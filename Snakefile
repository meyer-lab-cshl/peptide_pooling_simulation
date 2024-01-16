import pandas as pd
import copepodTCR as cpp
from math import comb

n_pools = [12]
#iters = [4]
len_lst = [200]
overlap = [5]
ep_length = [8]
pep_length = [17]
mu_off = [10]
sigma_off = [1.0]
mu_n = [5]
sigma_n = [1.0]
r = [3]

setup1 = pd.DataFrame(columns = ['n_pools', 'len_lst', 'iters'])
for n1 in n_pools:
	for l1 in len_lst:
		it1 = cpp.find_possible_k_values(n1, l1)
		for item in it1:
			if l1 <= comb(n1, item)*0.8:
				row = {'n_pools':n1, 'len_lst':l1, 'iters':item}
				row = pd.DataFrame([row])
				setup1 = pd.concat([setup1, row])
setup1.to_csv('npools_iters_lenlst_correspondence.tsv', sep = '\t', index = None)

setup2 = pd.DataFrame(columns = ['overlap', 'ep_length', 'pep_length'])
for p1 in pep_length:
    for ov1 in overlap:
        if ov1 < p1:
            for ep1 in ep_length:
                if ep1 < p1:
                	row = {'pep_length':p1, 'ep_length':ep1, 'overlap':ov1}
                	row = pd.DataFrame([row])
                	setup2 = pd.concat([setup2, row])
setup2.to_csv('peplength_eplength_overlap_correspondence.tsv', sep = '\t', index = None)

# Rule all
rule all:
	input:
		expand("results/conclusion_N{setup1.n_pools}_I{setup1.iters}_len{setup1.len_lst}_peptide{setup2.pep_length}_overlap{setup2.overlap}_ep_length{setup2.ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}.tsv",
			setup1 = setup1.itertuples(), setup2 = setup2.itertuples(), pep_length=pep_length, overlap=overlap, ep_length=ep_length, mu_off=mu_off, sigma_off=sigma_off, mu_n=mu_n, sigma_n=sigma_n, r=r),
		expand("results/summary_results.tsv")


# Pooling
rule pooling_scheme:
	output:
		path = "results/pooling_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}.tsv"
	params:
		n_pools="{n_pools}",
		iters="{iters}",
		len_lst="{len_lst}",
		overlap="{overlap}",
		ep_length="{ep_length}",
		pep_length="{pep_length}"
	shell:
		"""
		python scripts/pooling.py \
			-output {output.path} \
			-n_pools {wildcards.n_pools} \
			-iters {wildcards.iters} \
			-len_lst {wildcards.len_lst} \
			-overlap {wildcards.overlap} \
			-ep_length {wildcards.ep_length} \
			-pep_length {wildcards.pep_length}
		"""

# Simulation
rule sim_data:
	input:
		"results/pooling_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}.tsv"
	output:
		"results/simulation_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}.tsv"
	params:
		mu_off="{mu_off}",
		sigma_off="{sigma_off}",
		mu_n="{mu_n}",
		sigma_n="{sigma_n}",
		r="{r}",
		n_pools="{n_pools}",
		iters="{iters}",
		ep_length="{ep_length}"
	shell:
		"""
		python scripts/sim_data.py \
			-check_results {input} \
			-output {output} \
			-mu_off {wildcards.mu_off} \
			-sigma_off {wildcards.sigma_off} \
			-mu_n {wildcards.mu_n} \
			-sigma_n {wildcards.sigma_n} \
			-r {wildcards.r} \
			-n_pools {wildcards.n_pools} \
			-iters {wildcards.iters} \
			-ep_length {wildcards.ep_length} \
		"""

# Results interpetation
rule evaluate_data:
	input:
		scheme="results/pooling_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}.tsv",
		data="results/simulation_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}.tsv"
	output:
		"results/conclusion_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}.tsv"
	params:
		n_pools="{n_pools}",
		iters="{iters}",
		len_lst="{len_lst}",
		pep_length="{pep_length}",
		overlap="{overlap}",
		ep_length="{ep_length}",
		mu_off="{mu_off}",
		sigma_off="{sigma_off}",
		mu_n="{mu_n}",
		sigma_n="{sigma_n}",
		r="{r}"
	shell:
		"""
		python scripts/evaluate_data.py \
			-scheme {input.scheme} \
			-data {input.data} \
			-output {output} \
			-n_pools {wildcards.n_pools} \
			-iters {wildcards.iters} \
			-len_lst {wildcards.len_lst} \
			-pep_length {wildcards.pep_length} \
			-overlap {wildcards.overlap} \
			-ep_length {wildcards.ep_length} \
			-mu_off {wildcards.mu_off} \
			-sigma_off {wildcards.sigma_off} \
			-mu_n {wildcards.mu_n} \
			-sigma_n {wildcards.sigma_n} \
			-r {wildcards.r}
			"""

rule collect:
	input:
		expand("results/conclusion_N{setup1.n_pools}_I{setup1.iters}_len{setup1.len_lst}_peptide{setup2.pep_length}_overlap{setup2.overlap}_ep_length{setup2.ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}.tsv",
			setup1 = setup1.itertuples(), setup2 = setup2.itertuples(), pep_length=pep_length, overlap=overlap, ep_length=ep_length, mu_off=mu_off, sigma_off=sigma_off, mu_n=mu_n, sigma_n=sigma_n, r=r)
	output:
		"results/summary_results.tsv"
	script:
		'scripts/collect.py'

