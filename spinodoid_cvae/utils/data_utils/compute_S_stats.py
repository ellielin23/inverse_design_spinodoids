# compute_S_stats.py

import torch
import numpy as np
import os
from dataset import SpinodoidDataset

# === theta patterns to process ===
theta_patterns = ["001", "010", "011", "100", "101", "110", "111"]

# === ensure output directory exists ===
os.makedirs("data", exist_ok=True)

# === loop over each theta pattern ===
for pattern in theta_patterns:
    path = f"data/partition_by_theta/theta_{pattern}.csv"
    dataset = SpinodoidDataset(path)

    # stack all S vectors in dataset
    S_all = torch.stack([S for _, S in dataset])  # (_, S) since __getitem__ returns (P, S)
    S_mean = S_all.mean(dim=0)
    S_std = S_all.std(dim=0)

    # save to disk
    np.save(f"data/partition_by_theta/S_mean_theta_{pattern}.npy", S_mean.numpy())
    np.save(f"data/partition_by_theta/S_std_theta_{pattern}.npy", S_std.numpy())

    print(f"✅ Saved S stats for theta pattern {pattern}")
    print(f"   - Mean: {S_mean}")
    print(f"   - Std : {S_std}")




# # === compute_S_stats.py (non-partitioned version) ===

# import torch
# import numpy as np
# import os
# from dataset import SpinodoidDataset

# # === load dataset ===
# DATA_PATH = "data/train/large_dataset.csv"
# dataset = SpinodoidDataset(DATA_PATH)

# # === stack all S vectors ===
# S_all = torch.stack([S for _, S in dataset])
# S_mean = S_all.mean(dim=0)
# S_std = S_all.std(dim=0)

# # === save to disk ===
# os.makedirs("data", exist_ok=True)
# np.save("data/S_mean.npy", S_mean.numpy())
# np.save("data/S_std.npy", S_std.numpy())

# print("✅ Saved S_mean and S_std to data/")
# print("  Mean :", S_mean)
# print("  Std  :", S_std)
