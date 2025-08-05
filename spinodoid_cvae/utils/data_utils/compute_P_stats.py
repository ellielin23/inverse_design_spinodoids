# compute_P_stats.py

import torch
import numpy as np
import os
from utils.data_utils.dataset import SpinodoidDataset
from config import DATA_PATH


# === theta patterns to process ===
theta_patterns = ["001", "010", "011", "100", "101", "110", "111"]

# === ensure output directory exists ===
os.makedirs("data", exist_ok=True)

# === loop over each theta pattern ===
for pattern in theta_patterns:
    path = f"data/partition_by_theta/theta_{pattern}.csv"
    dataset = SpinodoidDataset(path)
    
    # stack all P vectors in dataset
    P_all = torch.stack([P for P, _ in dataset])
    P_mean = P_all.mean(dim=0)
    P_std = P_all.std(dim=0)

    # save to disk
    np.save(f"data/partition_by_theta/P_mean_theta_{pattern}.npy", P_mean.numpy())
    np.save(f"data/partition_by_theta/P_std_theta_{pattern}.npy", P_std.numpy())

    print(f"✅ Saved stats for theta pattern {pattern}")
    print(f"   - Mean: {P_mean}")
    print(f"   - Std : {P_std}")

# import torch
# import numpy as np
# import os
# from utils.data_utils.dataset import SpinodoidDataset
# from config import DATA_PATH

# # === load dataset ===
# dataset = SpinodoidDataset(DATA_PATH)

# # === stack all P vectors ===
# P_all = torch.stack([P for P, _ in dataset])
# P_mean = P_all.mean(dim=0)
# P_std = P_all.std(dim=0)

# # === save to disk ===
# os.makedirs("data", exist_ok=True)
# np.save("data/P_mean.npy", P_mean.numpy())
# np.save("data/P_std.npy", P_std.numpy())

# print("✅ Saved P_mean and P_std to data/")
# print("  Mean :", P_mean)
# print("  Std  :", P_std)
