# compute_P_stats.py

import torch
import numpy as np
import os
from load_data import load_dataset

DATA_PATH = "data/dataset_train_x1000.csv"
OUT_DIR = "data"
EPS = 1e-8

# === load dataset directly ===
P, S, _ = load_dataset(DATA_PATH)

# === compute stats ===
P_mean = P.mean(dim=0)
P_std  = torch.clamp(P.std(dim=0), min=EPS)

# === save to disk ===
os.makedirs(OUT_DIR, exist_ok=True)
np.save(f"{OUT_DIR}/P_mean.npy", P_mean.numpy())
np.save(f"{OUT_DIR}/P_std.npy",  P_std.numpy())

print("✅ Saved P_mean and P_std to data/")
print("  Mean:", P_mean)
print("  Std :", P_std)
