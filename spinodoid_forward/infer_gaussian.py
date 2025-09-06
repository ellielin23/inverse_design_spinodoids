# infer_gaussian.py
#
# Command-line inference for the Gaussian forward model.
#
# Examples
# --------
# Single S (comma-separated)
#   python infer_gaussian.py --S 30,30,30,0.8 --out preds.csv
#
# Batch from CSV and save results
#   python infer_gaussian.py --csv data/S_batch.csv --out preds.csv

import argparse, ast, os, sys
import numpy as np
import torch
from models.gaussian_forward import GaussianForwardModel


def parse_args():
    p = argparse.ArgumentParser(
        description="Gaussian forward-model inference: returns μ(S), σ(S) for P in raw units."
    )
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to model checkpoint (.pt). Defaults to checkpoints/gaussian/best/gaussian_ckpt_5.pt")
    p.add_argument("--config", type=str, default=None,
                   help="Path to config file. Defaults to checkpoints/gaussian/best/config_5.txt")
    p.add_argument("--S", type=str, default=None,
                   help="Single S as comma-separated list, e.g. '30,30,30,0.8'.")
    p.add_argument("--csv", type=str, default=None,
                   help="CSV file containing one S per row. "
                        "Accepted column names (case-insensitive): S1,S2,S3,S4 "
                        "or the first 4 numeric columns will be used.")
    p.add_argument("--out", type=str, default=None,
                   help="Filename for CSV results (saved under outputs/). "
                        "Also saves outputs/mu.npy and outputs/sigma.npy.")
    p.add_argument("--stats_dir", type=str, default="data",
                   help="Directory containing S_mean.npy, S_std.npy, P_mean.npy, P_std.npy (default: data)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"], help="Device selection.")
    return p.parse_args()


def load_config_dict(path):
    cfg = {}
    with open(path, "r") as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                k, v = k.strip(), v.strip()
                try:
                    cfg[k] = ast.literal_eval(v)
                except Exception:
                    cfg[k] = v
    return cfg


def load_stats(stats_dir, device):
    S_mean = torch.tensor(np.load(os.path.join(stats_dir, "S_mean.npy")),
                          dtype=torch.float32, device=device)
    S_std  = torch.tensor(np.load(os.path.join(stats_dir, "S_std.npy")),
                          dtype=torch.float32, device=device)
    P_mean = torch.tensor(np.load(os.path.join(stats_dir, "P_mean.npy")),
                          dtype=torch.float32, device=device)
    P_std  = torch.tensor(np.load(os.path.join(stats_dir, "P_std.npy")),
                          dtype=torch.float32, device=device)
    return S_mean, S_std, P_mean, P_std


def norm_S(S, S_mean, S_std): return (S - S_mean) / S_std
def denorm_P(Pn, P_mean, P_std): return Pn * P_std + P_mean


def parse_single_S(s_str):
    try:
        parts = [float(x.strip()) for x in s_str.split(",")]
    except Exception:
        raise ValueError(f"Could not parse --S '{s_str}'. Expected comma-separated floats.")
    if len(parts) != 4:
        raise ValueError(f"--S must have 4 values (got {len(parts)}).")
    return np.array(parts, dtype=np.float32).reshape(1, 4)


def load_S_from_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    # try named columns first (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    wanted = [cols[k] for k in ["s1", "s2", "s3", "s4"] if k in cols]
    if len(wanted) == 4:
        S = df[wanted].to_numpy(dtype=np.float32)
    else:
        # fallback: first 4 numeric columns
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) < 4:
            raise ValueError("CSV must contain at least 4 numeric columns or named columns S1..S4.")
        S = df[num_cols[:4]].to_numpy(dtype=np.float32)
        wanted = num_cols[:4]
    return S, df, wanted


def main():
    args = parse_args()

    # defaults (match your current layout)
    if args.ckpt is None:
        args.ckpt = "checkpoints/gaussian/best/gaussian_ckpt_5.pt"
    if args.config is None:
        args.config = "checkpoints/gaussian/best/config_5.txt"

    if not os.path.exists(args.ckpt) or not os.path.exists(args.config):
        print(f"ERROR: Could not find checkpoint/config at {args.ckpt}, {args.config}", file=sys.stderr)
        sys.exit(1)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # config & model
    cfg = load_config_dict(args.config)
    S_DIM = int(cfg.get("S_DIM", 4))
    P_DIM = int(cfg.get("P_DIM", 9))
    HIDDEN_DIMS = cfg.get("HIDDEN_DIMS", [128, 64])
    if isinstance(HIDDEN_DIMS, str):
        try:
            HIDDEN_DIMS = ast.literal_eval(HIDDEN_DIMS)
        except Exception:
            HIDDEN_DIMS = [128, 64]

    model = GaussianForwardModel(S_DIM, P_DIM, hidden_dims=HIDDEN_DIMS).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # stats
    S_mean, S_std, P_mean, P_std = load_stats(args.stats_dir, device)

    # input
    if (args.S is None) == (args.csv is None):
        print("ERROR: Provide exactly one of --S or --csv.", file=sys.stderr)
        sys.exit(2)

    if args.S is not None:
        S_np = parse_single_S(args.S)
        ids = [0]; src_cols = ["S1", "S2", "S3", "S4"]
    else:
        S_np, src_df, src_cols = load_S_from_csv(args.csv)
        ids = list(range(len(S_np)))

    # forward pass
    S_t = torch.tensor(S_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        Sn = norm_S(S_t, S_mean, S_std)
        mu_n, log_sigma_n = model(Sn)              # (N, P_DIM)
        mu_raw    = denorm_P(mu_n, P_mean, P_std)  # (N, P_DIM)
        sigma_raw = torch.exp(log_sigma_n) * P_std # (N, P_DIM)

    mu_np, sig_np = mu_raw.cpu().numpy(), sigma_raw.cpu().numpy()

    # console output (readable)
    np.set_printoptions(precision=6, suppress=True)
    for i, idx in enumerate(ids):
        print(f"\nS[{idx}] = {S_np[i].tolist()}")
        print("  mu    =", mu_np[i].tolist())
        print("  sigma =", sig_np[i].tolist())

    # outputs
    if args.out:
        import pandas as pd
        os.makedirs("outputs", exist_ok=True)

        # CSV
        csv_path = os.path.join("outputs", args.out)
        out = {name: S_np[:, j] for j, name in enumerate(src_cols)}
        for j in range(P_DIM):
            out[f"mu_{j+1}"] = mu_np[:, j]
        for j in range(P_DIM):
            out[f"sigma_{j+1}"] = sig_np[:, j]
        pd.DataFrame(out).to_csv(csv_path, index=False)
        print(f"\nSaved predictions CSV to: {csv_path}")

        # NumPy arrays
        np.save(os.path.join("outputs", "mu.npy"), mu_np)
        np.save(os.path.join("outputs", "sigma.npy"), sig_np)
        print("Saved NumPy arrays to: outputs/mu.npy and outputs/sigma.npy")


if __name__ == "__main__":
    main()
