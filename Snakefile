import pandas as pd
import numpy as np
import copepodTCR as cpp
from math import comb

n_pools = list(range(10, 26, 2))
#n_pools = [15]
len_lst = list(range(100, 1200, 200))
#len_lst = [200]
overlap = list(range(4, 10, 2))
#overlap = [5]
ep_length = list(range(8, 15, 2))
#ep_length = [8]
pep_length = list(range(10, 21, 2))
#pep_length = [17]
n_proteins = list(range(1, 50, 20))
#n_proteins = [1]
mu_off = list(range(0, 45, 10))
#mu_off = [20]
sigma_off = list(np.arange(0, 5, 1))
#sigma_off = [5]
sigma_p_r = list(np.arange(0, 5, 2))
#sigma_p_r = [1]
sigma_n_r = list(np.arange(0, 5, 2))
#sigma_n_r = [1]
low_offset = list(np.arange(0.2, 0.9, 0.2))
#low_offset = 0.6
mu_n = list(range(0, 45, 10))
#mu_n = [0]
sigma_n = list(np.arange(0, 5, 1))
#sigma_n = [5]
r = [1, 2, 3]
#r = [3]
error = [0, 1, 2]
#error = [0]

setup1 = pd.DataFrame(columns = ['n_pools', 'len_lst', 'iters', 'n_proteins', 'error'])
for er in error:
	for np1 in n_proteins:
		for n1 in n_pools:
			for l1 in len_lst:
				it1 = cpp.find_possible_k_values(n1, l1)
				for item in it1:
					if l1 <= comb(n1, item)*0.8 and np1 <= l1:
						if er < item and item < 6:
							row = {'n_pools':n1, 'len_lst':l1, 'iters':item,
							'n_proteins':np1, 'error':er}
							row = pd.DataFrame([row])
							setup1 = pd.concat([setup1, row])
setup1.to_csv('npools_iters_lenlst_nproteins_correspondence.tsv', sep = '\t', index = None)
print('setup1 done')

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
print('setup2 done')

#setup1 = pd.read_csv('npools_iters_lenlst_nproteins_correspondence.tsv', sep = '\t')
#setup2 = pd.read_csv('peplength_eplength_overlap_correspondence.tsv', sep = '\t')

# Rule all
rule all:
	input:
		expand("results/conclusion_N{setup1.n_pools}_I{setup1.iters}_len{setup1.len_lst}_peptide{setup2.pep_length}_overlap{setup2.overlap}_ep_length{setup2.ep_length}_nproteins{setup1.n_proteins}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{setup1.error}.tsv",
			setup1 = setup1.itertuples(), setup2 = setup2.itertuples(), mu_off=mu_off, sigma_off=sigma_off, mu_n=mu_n, sigma_n=sigma_n, r=r, sigma_p_r=sigma_p_r, sigma_n_r=sigma_n_r, low_offset=low_offset, error=error),
		expand("results/summary_results.tsv")


# Pooling
rule pooling_scheme:
	output:
		path = "results/pooling_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_nproteins{n_proteins}.tsv"
	params:
		n_pools="{n_pools}",
		iters="{iters}",
		len_lst="{len_lst}",
		overlap="{overlap}",
		ep_length="{ep_length}",
		pep_length="{pep_length}",
		n_proteins = "{n_proteins}"
	shell:
		"""
		python scripts/pooling.py \
			-output {output.path} \
			-n_pools {wildcards.n_pools} \
			-iters {wildcards.iters} \
			-len_lst {wildcards.len_lst} \
			-overlap {wildcards.overlap} \
			-ep_length {wildcards.ep_length} \
			-pep_length {wildcards.pep_length} \
			-n_proteins {wildcards.n_proteins}
		"""

# Simulation
rule sim_data:
	input:
		"results/pooling_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_nproteins{n_proteins}.tsv"
	output:
		"results/simulation_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_nproteins{n_proteins}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{error}.tsv"
	params:
		mu_off="{mu_off}",
		sigma_off="{sigma_off}",
		mu_n="{mu_n}",
		sigma_n="{sigma_n}",
		r="{r}",
		n_pools="{n_pools}",
		sigma_p_r="{sigma_p_r}",
		sigma_n_r="{sigma_n_r}",
		low_offset="{low_offset}",
		error="{error}"
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
			-sigma_p_r {wildcards.sigma_p_r} \
			-sigma_n_r {wildcards.sigma_n_r} \
			-low_offset {wildcards.low_offset} \
			-error {wildcards.error}
		"""

# Results interpetation
rule evaluate_data:
	input:
		scheme="results/pooling_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_nproteins{n_proteins}.tsv",
		data="results/simulation_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_nproteins{n_proteins}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{error}.tsv"
	output:
		"results/conclusion_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_nproteins{n_proteins}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{error}.tsv"
	params:
		n_pools="{n_pools}",
		iters="{iters}",
		len_lst="{len_lst}",
		pep_length="{pep_length}",
		overlap="{overlap}",
		ep_length="{ep_length}",
		n_proteins="{n_proteins}",
		mu_off="{mu_off}",
		sigma_off="{sigma_off}",
		mu_n="{mu_n}",
		sigma_n="{sigma_n}",
		r="{r}",
		sigma_p_r="{sigma_p_r}",
		sigma_n_r="{sigma_n_r}",
		low_offset="{low_offset}",
		error="{error}"
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
			-n_proteins {wildcards.n_proteins} \
			-mu_off {wildcards.mu_off} \
			-sigma_off {wildcards.sigma_off} \
			-mu_n {wildcards.mu_n} \
			-sigma_n {wildcards.sigma_n} \
			-r {wildcards.r} \
			-sigma_p_r {wildcards.sigma_p_r} \
			-sigma_n_r {wildcards.sigma_n_r} \
			-low_offset {wildcards.low_offset} \
			-error {wildcards.error}
			"""

rule collect:
	input:
		expand("results/conclusion_N{setup1.n_pools}_I{setup1.iters}_len{setup1.len_lst}_peptide{setup2.pep_length}_overlap{setup2.overlap}_ep_length{setup2.ep_length}_nproteins{setup1.n_proteins}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{setup1.error}.tsv",
			setup1 = setup1.itertuples(), setup2 = setup2.itertuples(), mu_off=mu_off, sigma_off=sigma_off, mu_n=mu_n, sigma_n=sigma_n, r=r, sigma_p_r=sigma_p_r, sigma_n_r=sigma_n_r, low_offset=low_offset)
	output:
		"results/summary_results.tsv"
	script:
		'scripts/collect.py'

