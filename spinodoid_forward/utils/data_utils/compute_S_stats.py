# compute_S_stats.py

import torch, numpy as np, os
from load_data import load_dataset

DATA_PATH = "data/dataset_train_x1000.csv"
OUT_DIR = "data"
EPS = 1e-8

# direct load (returns torch tensors)
P, S, _ = load_dataset(DATA_PATH)

S_mean = S.mean(dim=0)
S_std  = torch.clamp(S.std(dim=0), min=EPS)

os.makedirs(OUT_DIR, exist_ok=True)
np.save(f"{OUT_DIR}/S_mean.npy", S_mean.numpy())
np.save(f"{OUT_DIR}/S_std.npy",  S_std.numpy())

print("✅ Saved S_mean and S_std to data/")
print("  Mean:", S_mean)
print("  Std :", S_std)
