# infer.py

"""
Usage: python infer.py --csv <data_path>
    - Optional arguments
        --models : 'Comma-separated subset like "001,011,111" or "all" (default = all 7)
        --pass-threshold" : PASS if tensor error < this fraction (e.g., 0.08 for 8%)
        --prob-threshold" : Peak probability threshold
        --samples : Latent samples per model
        --bandwidth : "auto" or float (e.g., 0.3) for KDE
        --seed : default=42
        --device : choices=[None, "cpu", "cuda", "mps"], default=None
        --outdir : default="outputs"

Assumptions:
  - Models live in: release_v1/models/<tag>/ with the following files:
        - config_<tag>.json
        - decoder_<tag>.pt
        - P_mean_<tag>.npy
        - P_std_<tag>.npy
        - S_mean_<tag>.npy
        - S_std_<tag>.npy
  - Input CSV has NO headers, each row = 21 Mandel upper-triangle values
  - Error metric = ||C_pred - C_true||_F / ||C_true||_F
  - Output columns: tag, peak_idx, bw_used, prob_est, S_hat, error, status
"""

import os, sys, json, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from utils.formatting import set_seed, fmt, ensure_dir, save_outputs
from utils.data_processing import read_c_csv_unlabeled
from utils.model_loaders import load_decoder, load_fNN_model, load_all_models_release
from utils.evaluate import eval_one_row

MODELS_DIR = Path("models")
DEFAULT_TAGS = ["001","010","100","011","101","110","111"]

def main():
    ap = argparse.ArgumentParser(description="Parallel CVAE inverse design (release-only; C input; tensor error %).")
    ap.add_argument("--csv", type=str, required=True, help="CSV with unlabeled 21 Mandel components per row.")
    ap.add_argument("--tags", type=str, default="all", help='Comma-separated subset like "001,011,111" or "all" (default = all 7).')
    ap.add_argument("--pass-threshold", type=float, default=0.08, help="PASS if tensor error < this fraction (e.g., 0.08 for 8%%).")
    ap.add_argument("--prob-threshold", type=float, default=0.10, help="Peak probability threshold.")
    ap.add_argument("--samples", type=int, default=1000, help="Latent samples per model.")
    ap.add_argument("--bandwidth", type=str, default="auto", help='"auto" or float (e.g., 0.3) for KDE.')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, choices=[None, "cpu", "cuda", "mps"], default=None)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # ensure default models dir exists
    if not MODELS_DIR.exists():
        print(f"Models dir not found: {MODELS_DIR}", file=sys.stderr); sys.exit(2)

    # parse tags: "all" => DEFAULT_TAGS, else split user list
    if args.tags.strip().lower() == "all":
        tags = DEFAULT_TAGS
    else:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        # quick sanity: must be a subset of the canonical 7
        unknown = sorted(set(tags) - set(DEFAULT_TAGS))
        if unknown:
            print(f"Unknown tag(s): {', '.join(unknown)}. Allowed: {', '.join(DEFAULT_TAGS)}", file=sys.stderr)
            sys.exit(2)

    # validate each requested tag has files present
    for tag in tags:
        tdir = MODELS_DIR / tag
        required = [f"config_{tag}.json", f"decoder_{tag}.pt",
                    f"P_mean_{tag}.npy", f"P_std_{tag}.npy", f"S_mean_{tag}.npy", f"S_std_{tag}.npy"]
        missing = [fn for fn in required if not (tdir / fn).exists()]
        if missing:
            print(f"Missing files for tag {tag} in {tdir}: {', '.join(missing)}", file=sys.stderr)
            sys.exit(2)

    # parse bandwidth
    bw = args.bandwidth
    if isinstance(bw, str) and bw.lower() != "auto":
        try:
            bw = float(bw)
        except Exception:
            print("Invalid --bandwidth. Use 'auto' or a float like 0.3.", file=sys.stderr); sys.exit(2)

    print(f"📦 Loading decoders and fNN from {MODELS_DIR} (tags: {', '.join(tags)}) ...")
    decoders, cfgs, Pm, Ps, Sm, Ss = load_all_models_release(MODELS_DIR, device, tags)
    fNN = load_fNN_model()

    # read C → (C_true, P_true)
    C_true_all, P_true_all = read_c_csv_unlabeled(args.csv)

    # run each row
    for i, (C_true, P_true) in enumerate(zip(C_true_all, P_true_all)):
        row_tag = f"row_{i}"
        print(f"\n🎯 {row_tag}  (P_true={fmt(P_true)})")
        rows = eval_one_row(P_true, C_true, decoders, cfgs, Pm, Ps, Sm, Ss, fNN,
                            pass_thr=args.pass_threshold, prob_thr=args.prob_threshold,
                            samples=args.samples, bw=bw, seed=args.seed, device=device, tags=tags)
        outdir = os.path.join(args.outdir, row_tag)
        meta = {
            "mode": "csv_c_unlabeled_21",
            "row": i,
            "pass_threshold_fraction": args.pass_threshold,
            "prob_threshold": args.prob_threshold,
            "samples": args.samples,
            "bandwidth": args.bandwidth,
            "tags": tags,
            "models_dir": str(MODELS_DIR),
            "device": str(device),
            "seed": args.seed,
            "source_csv": args.csv,
        }
        ensure_dir(outdir)
        Path(os.path.join(outdir, "meta.json")).write_text(json.dumps(meta, indent=2))
        save_outputs(rows, outdir)

if __name__ == "__main__":
    main()