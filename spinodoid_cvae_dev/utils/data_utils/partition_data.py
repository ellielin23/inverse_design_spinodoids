# utils/data_utils/partition_data.py

import pandas as pd
import numpy as np
import os

def get_theta_pattern(row):
    """
    Given a row with theta1, theta2, theta3 values, return a binary string like '101'
    representing which theta values are nonzero (> 0).
    """
    theta_vals = [row['theta1'], row['theta2'], row['theta3']]
    pattern = ''.join(['1' if theta > 0 else '0' for theta in theta_vals])
    return pattern

def partition_dataset_by_theta(input_csv_path, output_dir):
    """
    Partition the dataset into 7 subsets based on which theta components are > 0.
    Saves each partition as a CSV in output_dir, WITHOUT HEADERS.

    Args:
        input_csv_path (str): Path to the full dataset CSV (no headers).
        output_dir (str): Directory to save the partitioned CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # === manually define column names ===
    theta_cols = ['rho', 'theta1', 'theta2', 'theta3']
    C_cols = [f'C{i}' for i in range(1, 22)]
    column_names = ['id'] + theta_cols + C_cols

    # === load the dataset with manual headers ===
    df = pd.read_csv(input_csv_path, header=None, names=column_names)

    # === add theta pattern column (e.g., '101', '011', etc.) ===
    df['theta_pattern'] = df.apply(get_theta_pattern, axis=1)

    # === group by theta pattern ===
    patterns = df['theta_pattern'].unique()
    print(f"✅ Found {len(patterns)} unique theta patterns:", sorted(patterns))

    for pattern in patterns:
        if pattern == '000':
            print(f"⚠️ Skipping pattern '000' — all theta values are zero.")
            continue
        subset = df[df['theta_pattern'] == pattern].copy()
        output_path = os.path.join(output_dir, f"theta_{pattern}.csv")
        subset.to_csv(output_path, index=False, header=False)
        print(f"✅ Saved {len(subset)} samples to {output_path}")

if __name__ == "__main__":
    input_csv = "data/partition_by_theta/large_dataset_augmented.csv"
    output_folder = "data/partition_by_theta"

    partition_dataset_by_theta(input_csv, output_folder)
