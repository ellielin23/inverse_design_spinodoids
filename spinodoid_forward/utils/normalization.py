# compute_P_stats.py

import torch
import numpy as np
import os
from utils.dataset import SpinodoidDataset
from config import DATA_PATH

# === load dataset ===
dataset = SpinodoidDataset(DATA_PATH)

# === stack all P vectors ===
P_all = torch.stack([P for _, P in dataset])
P_mean = P_all.mean(dim=0)
P_std = P_all.std(dim=0)

# === save to disk ===
os.makedirs("data", exist_ok=True)
np.save("data/P_mean.npy", P_mean.numpy())
np.save("data/P_std.npy", P_std.numpy())

print("✅ Saved P_mean and P_std to data/")
print("  Mean :", P_mean)
print("  Std  :", P_std)
