# compute_S_stats.py

import torch
import numpy as np
import os
from dataset import SpinodoidDataset

# === load dataset ===
DATA_PATH = "data/dataset_train_x1000.csv"
dataset = SpinodoidDataset(DATA_PATH)

# === stack all S vectors ===
S_all = torch.stack([S for _, S in dataset])
S_mean = S_all.mean(dim=0)
S_std = S_all.std(dim=0)

# === save to disk ===
os.makedirs("data", exist_ok=True)
np.save("data/S_mean.npy", S_mean.numpy())
np.save("data/S_std.npy", S_std.numpy())

print("✅ Saved S_mean and S_std to data/")
print("  Mean :", S_mean)
print("  Std  :", S_std)
