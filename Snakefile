import pandas as pd
import numpy as np
import copepodTCR as cpp
import codepub as cdp
import math

method = ['copepodTCR', 'basic']
len_lst = [100, 500, 1000, 1500]
wanted_neg_share = [0.6, 0.8]
overlap = [4, 6]
ep_length = [8, 14]
pep_length = [14]
mu_off = [5, 15, 45]
sigma_off = [3]
sigma_p_r = [3]
sigma_n_r = [5, 10, 20]
low_offset = [0.8]
mu_n = [5, 15, 45]
sigma_n = [5, 10, 20]
r = [2]
error = [0, 1]

#method = ['copepodTCR', 'basic']
#len_lst = [500]
#wanted_neg_share = [0.6]
#overlap = [6]
#ep_length = [14]
#pep_length = [14]
#mu_off = [45]
#sigma_off = [3]
#sigma_p_r = [3]
#sigma_n_r = [5]
#low_offset = [0.8]
#mu_n = [5]
#sigma_n = [5]
#r = [1]
#error = [0]



setup1 = pd.DataFrame(columns = ['method', 'len_lst', 'n_pools', 'iters', 'error'])
for er in error:
	for m1 in method:
		for l1 in len_lst:
				for wng in wanted_neg_share:

					for m in range(5, 30):
						possible_r = cdp.find_possible_k_values(m, l1)

						if len(possible_r) != 0:
							negshares = []
							working_r = []

							for rs in possible_r:
								if math.comb(m, rs)*0.8 >= l1 and math.comb(m, rs+1)*0.8 >= l1:
									negshare_result = (m - rs - 1)/m
									negshares.append(negshare_result)
									working_r.append(rs)
							if len(negshares) > 0:
								for j in range(len(negshares)):
									if math.isclose(wng, negshares[j], rel_tol=0.02):
										rs = working_r[j]

										row = {'method':m1, 'len_lst':l1, 'n_pools':m, 'iters':rs, 'error':er}
										row = pd.DataFrame([row] )
										setup1 = pd.concat([setup1, row], ignore_index = True)
										break

setup1.to_csv('method-lenlst-error_correspondence.tsv', sep = '\t', index = None)
#print('setup1 done')

setup2 = pd.DataFrame(columns = ['overlap', 'ep_length', 'pep_length'])
for p1 in pep_length:
    for ov1 in overlap:
        if ov1 < p1:
            for ep1 in ep_length:
                if ep1 <= p1:
                	row = {'pep_length':p1, 'ep_length':ep1, 'overlap':ov1}
                	row = pd.DataFrame([row])
                	setup2 = pd.concat([setup2, row], ignore_index = True)
#setup2.to_csv('peplength-eplength-overlap_correspondence.tsv', sep = '\t', index = None)
#print('setup2 done')

#setup1 = pd.read_csv('method-lenlst-error_correspondence.tsv', sep = '\t')
#setup2 = pd.read_csv('peplength-eplength-overlap_correspondence.tsv', sep = '\t')

# Rule all
rule all:
	input:
		expand("results/method-comparison/conclusion_{setup1.method}_N{setup1.n_pools}_I{setup1.iters}_len{setup1.len_lst}_peptide{setup2.pep_length}_overlap{setup2.overlap}_ep_length{setup2.ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{setup1.error}.tsv",
			setup1 = setup1.itertuples(), setup2 = setup2.itertuples(), mu_off=mu_off, sigma_off=sigma_off, mu_n=mu_n, sigma_n=sigma_n, r=r, sigma_p_r=sigma_p_r, sigma_n_r=sigma_n_r, low_offset=low_offset),
		expand("results/method-comparison/summary_results.tsv")


# Pooling
rule pooling_scheme:
	output:
		path = "results/method-comparison/pooling_{method}_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}.tsv"
	shell:
		"""
		python scripts/method-comparison/pooling.py \
			-method {wildcards.method} \
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
		"results/method-comparison/pooling_{method}_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}.tsv"
	output:
		output_data = "results/method-comparison/simulation_{method}_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{error}.tsv",
		output_params = "results/method-comparison/simparams_{method}_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{error}.tsv"
	params:
		ident_sim = lambda wildcards : "_".join(wildcards)
	resources:
		mem_mb = 4000,
		runtime = 60
	shell:
		"""
		export PYTENSOR_FLAGS="compiledir=${{TMPDIR:-/tmp}}/pytensor_sim_{params.ident_sim}/{jobid}"
		python scripts/method-comparison/sim_data.py \
			-check_results {input} \
			-output {output.output_data} \
			-output_params {output.output_params} \
			-mu_off {wildcards.mu_off} \
			-sigma_off {wildcards.sigma_off} \
			-mu_n {wildcards.mu_n} \
			-sigma_n {wildcards.sigma_n} \
			-r {wildcards.r} \
			-sigma_p_r {wildcards.sigma_p_r} \
			-sigma_n_r {wildcards.sigma_n_r} \
			-low_offset {wildcards.low_offset} \
			-error {wildcards.error}
		rm -rf ${{TMPDIR:-/tmp}}/pytensor_sim_{params.ident_sim}
		"""

# Results interpetation
rule evaluate_data:
	input:
		scheme="results/method-comparison/pooling_{method}_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}.tsv",
		data="results/method-comparison/simulation_{method}_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{error}.tsv",
		sim_params="results/method-comparison/simparams_{method}_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{error}.tsv"
	output:
		"results/method-comparison/conclusion_{method}_N{n_pools}_I{iters}_len{len_lst}_peptide{pep_length}_overlap{overlap}_ep_length{ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{error}.tsv"
	params:
		ident_ev = lambda wildcards : "_".join(wildcards)
	resources:
		mem_mb = 4000,
		runtime = 60
	shell:
		"""
		export PYTENSOR_FLAGS="compiledir=${{TMPDIR:-/tmp}}/pytensor_ev_{params.ident_ev}/{jobid}"
		python scripts/method-comparison/evaluate_data.py \
			-method {wildcards.method} \
			-scheme {input.scheme} \
			-data {input.data} \
			-sim_params {input.sim_params} \
			-output {output} \
			-len_lst {wildcards.len_lst} \
			-pep_length {wildcards.pep_length} \
			-overlap {wildcards.overlap} \
			-ep_length {wildcards.ep_length} \
			-mu_off {wildcards.mu_off} \
			-sigma_off {wildcards.sigma_off} \
			-mu_n {wildcards.mu_n} \
			-sigma_n {wildcards.sigma_n} \
			-r {wildcards.r} \
			-sigma_p_r {wildcards.sigma_p_r} \
			-sigma_n_r {wildcards.sigma_n_r} \
			-low_offset {wildcards.low_offset} \
			-error {wildcards.error}
		rm -rf ${{TMPDIR:-/tmp}}/pytensor_ev_{params.ident_ev}
		"""

rule collect:
	input:
		expand("results/method-comparison/conclusion_{setup1.method}_N{setup1.n_pools}_I{setup1.iters}_len{setup1.len_lst}_peptide{setup2.pep_length}_overlap{setup2.overlap}_ep_length{setup2.ep_length}_muoff{mu_off}_sigmaoff{sigma_off}_mun{mu_n}_sigman{sigma_n}_r{r}_sigmapr{sigma_p_r}_sigmanr{sigma_n_r}_lowoffset{low_offset}_error{setup1.error}.tsv",
			setup1 = setup1.itertuples(), setup2 = setup2.itertuples(), mu_off=mu_off, sigma_off=sigma_off, mu_n=mu_n, sigma_n=sigma_n, r=r, sigma_p_r=sigma_p_r, sigma_n_r=sigma_n_r, low_offset=low_offset)
	output:
		"results/method-comparison/summary_results.tsv"
	script:
		'scripts/method-comparison/collect.py'

