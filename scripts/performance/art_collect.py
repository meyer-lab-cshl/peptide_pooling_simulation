#!/usr/bin/env python3

import os
import pandas as pd

def assemble_tsv_files(directory, output_file):
    """
    Assembles all TSV files in the specified directory whose names start with 'conclusion' into a single TSV file.

    Parameters:
        directory (str): The path to the directory containing the TSV files.
        output_file (str): The path to save the resulting assembled TSV file.
    """
    # List to store dataframes
    dataframes = []

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Check if the filename starts with 'conclusion' and ends with '.tsv'
        if filename.startswith("conclusion") and filename.endswith(".tsv"):
            filepath = os.path.join(directory, filename)

            # Read the TSV file into a dataframe
            try:
                df = pd.read_csv(filepath, sep="\t")
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")

    # Combine all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Save the combined dataframe to a TSV file
        combined_df.to_csv(output_file, sep="\t", index=False)
        print(f"Combined file saved as {output_file}")
    else:
        print("No files to combine.")

if __name__ == "__main__":
    directory = './results/'
    output_file = "./results/art_summary.tsv"

    assemble_tsv_files(directory, output_file)