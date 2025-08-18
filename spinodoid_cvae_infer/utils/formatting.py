# utils/formatting.py

import os, torch, random
import numpy as np
import pandas as pd
from pathlib import Path

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def fmt(arr, k=4):
    arr = np.asarray(arr, dtype=float).flatten()
    return "[" + ", ".join(f"{v:.{k}f}" for v in arr) + "]"

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def save_outputs(rows, outdir):
    if not rows:
        print("⚠️ No candidates after peak selection/constraints.")
        return
    df = pd.DataFrame(rows).sort_values(by=["_err", "tag"]).reset_index(drop=True)
    df = df[["tag", "prob_est", "S_hat", "error", "status"]]
    df_pass = df[df["status"] == "PASS"].reset_index(drop=True)
    ensure_dir(outdir)
    df.to_csv(os.path.join(outdir, "all_candidates.csv"), index=False)
    df_pass.to_csv(os.path.join(outdir, "passing_candidates.csv"), index=False)
    best = df.iloc[0]
    print(f"⭐ total={len(df)}, pass={len(df_pass)} | BEST: error={best['error']}, tag={best['tag']}, Ŝ={best['S_hat']}")