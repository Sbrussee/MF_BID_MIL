import os
import pandas as pd
import numpy as np
import argparse

def calculate_entropy(row):
    p0 = row['y_pred0']
    p1 = row['y_pred1']
    # Ensure the probabilities are valid to avoid log(0)
    if p0 > 0 and p1 > 0:
        return - (p0 * np.log2(p0) + p1 * np.log2(p1))
    else:
        return 0.0

def assign_group(certainty):
    if certainty <= 0.25:
        return '0-25'
    elif certainty <= 0.50:
        return '25-50'
    elif certainty <= 0.75:
        return '50-75'
    else:
        return '75-100'

def convert_parquet_to_csv(source_dir):
    # Traverse the directory recursively
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.parquet'):
                # Construct the full file path
                parquet_file_path = os.path.join(root, file)
                # Read the parquet file into a DataFrame
                df = pd.read_parquet(parquet_file_path)
                # Calculate uncertainty for each row
                df['uncertainty'] = df.apply(calculate_entropy, axis=1)
                # Calculate certainty
                df['certainty'] = 1 - df['uncertainty']
                # Assign group based on certainty
                df['group'] = df['certainty'].apply(assign_group)
                # Construct the CSV file path
                csv_file_path = os.path.splitext(parquet_file_path)[0] + '.csv'
                # Save the DataFrame as a CSV file
                df.to_csv(csv_file_path, index=False)
                print(f"Converted {parquet_file_path} to {csv_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Parquet files to CSV files recursively in a directory.")
    parser.add_argument("--source_dir", type=str, help="Path to the source directory containing Parquet files.")

    args = parser.parse_args()
    source_directory = args.source_dir

    convert_parquet_to_csv(source_directory)
