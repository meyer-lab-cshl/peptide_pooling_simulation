import os
import pandas as pd
import numpy as np
import re

def extract_parameters(filename):
    pattern = (r"simulation_N(?P<n_pools>\d+)_I(?P<iters>\d+)_len(?P<len_lst>\d+)_peptide(?P<pep_length>\d+)_"
               r"overlap(?P<shift>\d+)_ep_length(?P<ep_length>\d+)_nproteins(?P<n_proteins>\d+)_"
               r"muoff(?P<mu_off>\d+)_sigmaoff(?P<sigma_off>\d+)_mun(?P<mu_n>\d+)_sigman(?P<sigma_n>\d+)_"
               r"r(?P<r>\d+)_sigmapr(?P<sigma_p_r>\d+)_sigmanr(?P<sigma_n_r>\d+)_lowoffset(?P<low_offset>[\d\.]+)_"
               r"error(?P<error>\d+)\.tsv")
    match = re.match(pattern, filename)
    if match:
        params = match.groupdict()
        for key in params:
            try:
                params[key] = np.int64(params[key])
            except ValueError:
                params[key] = float(params[key])  # Convert float values where applicable
        return params
    return None

def calculate_ratio(filepath):
    try:
        df = pd.read_csv(filepath, sep='\t')
        if df.empty or 'Percentage' not in df.columns:
            return None
        min_val = df['Percentage'].min()
        max_val = df['Percentage'].max()
        return min_val / max_val if max_val != 0 else None
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

def process_simulation_files(folder):
    data = []
    for file in os.listdir(folder):
        if file.startswith("simulation_") and file.endswith(".tsv"):
            try:
                params = extract_parameters(file)
                if params:
                    filepath = os.path.join(folder, file)
                    ratio = calculate_ratio(filepath)
                    params['ratio'] = ratio
                    data.append(params)
            except Exception as e:
                print(f"Skipping file {file} due to error: {e}")
    df = pd.DataFrame(data)
    return df

folder_path = "./results/"
result_df = process_simulation_files(folder_path)
result_df.to_csv('./results/results_min_max.tsv', sep = "\t", index = False)