#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infer CVAE candidates across all 7 theta-models and rank by P-reconstruction error (MAE).
Input modes (mutually exclusive):
  --p "p1,...,p9"                     # manual P (9 floats)
  --csv-c path.csv                    # CSV with C-tensor columns; P is derived per row
Outputs per target:
  outputs/<tag>/all_candidates.csv
  outputs/<tag>/passing_candidates.csv
  outputs/<tag>/meta.json
"""

import os, sys, json, argparse, random
import numpy as np, pandas as pd, torch

# === project utils ===
from utils.evaluate_utils.load_paths import load_paths
from utils.evaluate_utils.load_models import load_config, load_decoder, load_fNN_model
from utils.evaluate_utils.sampling import (
    get_S_hats, extract_peaks_with_bandwidth, get_S_hat_peaks, sort_and_select_peaks_by_probability
)
from utils.evaluate_utils.structure_constraints import enforce_theta_domain, filter_S_candidates

# ---------- small helpers ----------
P_COLS = ["C_1111","C_1122","C_1133","C_2222","C_2233","C_3333","C_1212","C_1313","C_2323"]

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def fmt(arr, k=4):
    arr=np.asarray(arr,dtype=float).flatten()
    return "[" + ", ".join(f"{v:.{k}f}" for v in arr) + "]"

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_models_arg(s): return [(t.strip(), int(tr.strip())) for t,tr in (x.split(":") for x in s.split(",") if x.strip())]

def parse_p_str(s):
    vals=[float(x.strip()) for x in s.split(",") if x.strip()!=""]
    if len(vals)!=9: raise ValueError("`--p` expects 9 comma-separated floats.")
    return np.array(vals, dtype=float)

def targets_from_c_csv(csv_path: str):
    """
    Accept either:
      A) CSV with named P columns (C_1111,...,C_2323) -> derive P directly, or
      B) CSV with exactly 21 unlabeled columns (Mandel upper triangle) -> build C tensor then derive P.
    Returns: list[(tag, P_true)]
    """
    # Try reading with headers first
    try:
        df = pd.read_csv(csv_path)
        cols_lower = {c.lower(): c for c in df.columns}
        have_headers = len(df.columns) > 1  # heuristic; if 1 col and commas inside, loadtxt path below will handle
    except Exception:
        have_headers = False
        df = None

    # Case A: labeled P columns available
    if have_headers:
        P_COLS = ["C_1111","C_1122","C_1133","C_2222","C_2233","C_3333","C_1212","C_1313","C_2323"]
        missing = [c for c in P_COLS if c.lower() not in (cols_lower or {})]
        if not missing:
            out = []
            for i, row in df.iterrows():
                P = np.array([float(row[cols_lower[c.lower()]]) for c in P_COLS], dtype=float)
                out.append((f"row_{i}", P))
            return out
        # fall through to try 21-col path if headers exist but not the right ones

    # Case B: unlabeled 21 columns (Mandel 6x6 upper triangle flattened row-wise)
    # Expected order (by rows of upper triangle):
    # (0,0),
    # (0,1),(0,2),(0,3),(0,4),(0,5),
    # (1,1),(1,2),(1,3),(1,4),(1,5),
    # (2,2),(2,3),(2,4),(2,5),
    # (3,3),(3,4),(3,5),
    # (4,4),(4,5),
    # (5,5)
    C_flat_21 = np.loadtxt(csv_path, delimiter=',', dtype=float)
    if C_flat_21.ndim == 1:
        C_flat_21 = C_flat_21.reshape(1, -1)
    if C_flat_21.shape[1] != 21:
        raise ValueError(
            f"{csv_path} has shape {C_flat_21.shape}; expected 21 columns for unlabeled Mandel input."
        )

    # Use your existing pipeline to reconstruct C and derive P
    from utils.data_utils.load_data import full_C_from_C_flat_21, extract_target_properties
    C_tensor = full_C_from_C_flat_21(C_flat_21)                 # (N,3,3,3,3)
    P_mat = extract_target_properties(C_tensor)                  # (N,9)
    return [(f"row_{i}", P_mat[i]) for i in range(P_mat.shape[0])]

# ---------- core ----------
def load_all_models(models, device):
    decoders=[]; cfgs=[]; Pm=[]; Ps=[]; Sm=[]; Ss=[]
    for tag,trial in models:
        paths=load_paths(TRIAL=trial, THETA_MODEL=True, THETA_PATTERN=tag, verbose=False)
        cfg=load_config(paths["config_path"])
        dec=load_decoder(cfg, paths["decoder_path"], flow_type=cfg.get("FLOW_TYPE","planar"), trial=trial, device=device)
        pmean,npstd=np.load(paths["P_mean_path"]),np.load(paths["P_std_path"])
        smean,nsstd=np.load(paths["S_mean_path"]),np.load(paths["S_std_path"])
        npstd=np.where(npstd<1e-8,1.0,npstd); nsstd=np.where(nsstd<1e-8,1.0,nsstd)
        decoders.append(dec); cfgs.append(cfg); Pm.append(pmean); Ps.append(npstd); Sm.append(smean); Ss.append(nsstd)
    return decoders,cfgs,Pm,Ps,Sm,Ss

def p_from_tensor(C):
    # indices: [1111,1122,1133,2222,2233,3333,1212,1313,2323]
    return np.array([C[0,0,0,0], C[0,0,1,1], C[0,0,2,2], C[1,1,1,1], C[1,1,2,2], C[2,2,2,2],
                     C[0,1,0,1], C[0,2,0,2], C[1,2,1,2]], dtype=float)

def eval_one_P(P_true, decoders, cfgs, Pm, Ps, Sm, Ss, fNN, pass_thr, prob_thr, samples, bw, seed, device):
    rows=[]
    for midx,(cfg,p_mean,p_std,s_mean,s_std) in enumerate(zip(cfgs,Pm,Ps,Sm,Ss)):
        decoder=decoders[midx]; latent_dim=int(cfg["LATENT_DIM"])
        Pn=(P_true - p_mean) / (p_std + 1e-8)
        Pn_t=torch.tensor(Pn, dtype=torch.float32, device=device).unsqueeze(0)
        S_norm=get_S_hats(decoder, Pn_t, latent_dim, num_samples=samples, seed=seed, device=device)
        if isinstance(bw,str) and bw.lower()=="auto":
            peaks_norm,bw_used=extract_peaks_with_bandwidth(S_norm, use_auto_bandwidth=True, target_range=(1,10), verbose=False)
        else:
            bw_used=float(bw); peaks_norm=get_S_hat_peaks(S_norm, bandwidth=bw_used)
        peaks_norm,probs,_=sort_and_select_peaks_by_probability(S_norm, peaks_norm, bw_used, prob_threshold=prob_thr, verbose=False)
        peaks=peaks_norm * s_std + s_mean
        peaks=enforce_theta_domain(peaks); peaks=filter_S_candidates(peaks)
        tag=cfg.get("THETA_PATTERN","???")
        for k,S_hat in enumerate(peaks):
            C_pred=fNN(np.expand_dims(S_hat,(0,1))).numpy().reshape(1,3,3,3,3)[0]
            P_hat=p_from_tensor(C_pred)
            p_err=float(np.mean(np.abs(P_hat - P_true))) # MAE over 9 comps
            rows.append({
                "tag": tag, "peak_idx": k, "bw_used": bw_used,
                "prob_est": float(probs[k]) if k<len(probs) else np.nan,
                "S_hat": fmt(S_hat), "theta_deg": fmt(S_hat[:3]), "rho": float(S_hat[3]),
                "P_err_MAE": p_err, "status": "PASS" if p_err < pass_thr else "FAIL",
            })
    return rows

def save_outputs(rows, outdir_base, meta):
    if not rows:
        print("⚠️ No candidates after peak selection/constraints."); return
    df=pd.DataFrame(rows).sort_values(by=["P_err_MAE","tag","peak_idx"]).reset_index(drop=True)
    df_pass=df[df["status"]=="PASS"].reset_index(drop=True)
    ensure_dir(outdir_base)
    df.to_csv(os.path.join(outdir_base,"all_candidates.csv"), index=False)
    df_pass.to_csv(os.path.join(outdir_base,"passing_candidates.csv"), index=False)
    with open(os.path.join(outdir_base,"meta.json"),"w") as f: json.dump(meta,f,indent=2)
    best=df.iloc[0]
    print(f"✅ total={len(df)}, pass={len(df_pass)} | best P-MAE={best['P_err_MAE']:.4f}  tag={best['tag']}  Ŝ={best['S_hat']}")
    print(f"📁 {outdir_base}")

# ---------- CLI ----------
def main():
    ap=argparse.ArgumentParser(description="Parallel CVAE inference (rank by P error).")
    # inputs
    ap.add_argument("--p", type=str, help='Manual P: "p1,...,p9"')
    ap.add_argument("--csv-c", type=str, help="CSV with C_* columns (P derived from 9 comps).")
    # knobs
    ap.add_argument("--pass-threshold", type=float, default=0.08, help="PASS if P-MAE < this.")
    ap.add_argument("--prob-threshold", type=float, default=0.10, help="Peak probability threshold.")
    ap.add_argument("--samples", type=int, default=1000, help="Latent samples per model.")
    ap.add_argument("--bandwidth", type=str, default="auto", help='"auto" or float (e.g., 0.3) for KDE.')
    # env
    ap.add_argument("--models", type=str, default="001:18,010:6,100:9,011:16,101:5,110:3,111:5",
                    help='Comma list "tag:trial,...".')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, choices=[None,"cpu","cuda","mps"], default=None)
    ap.add_argument("--outdir", type=str, default="outputs")
    args=ap.parse_args()

    # mode
    if (args.p is None) == (args.csv_c is None):
        print("Specify exactly one input mode: --p OR --csv-c", file=sys.stderr); sys.exit(2)

    set_seed(args.seed)
    device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    models=parse_models_arg(args.models)
    bw=args.bandwidth
    if isinstance(bw,str) and bw.lower()!="auto":
        try: bw=float(bw)
        except: print("Invalid --bandwidth. Use 'auto' or float like 0.3.", file=sys.stderr); sys.exit(2)

    print("📦 Loading decoders and fNN...")
    decoders,cfgs,Pm,Ps,Sm,Ss=load_all_models(models, device)
    fNN=load_fNN_model()

    # targets
    targets=[("manual", parse_p_str(args.p))] if args.p else targets_from_c_csv(args.csv_c)

    for tag,P_true in targets:
        print(f"\n🎯 Target [{tag}]  P_true={fmt(P_true)}")
        rows=eval_one_P(
            P_true, decoders, cfgs, Pm, Ps, Sm, Ss, fNN,
            pass_thr=args.pass_threshold, prob_thr=args.prob_threshold,
            samples=args.samples, bw=bw, seed=args.seed, device=device
        )
        outdir_base=os.path.join(args.outdir, tag); ensure_dir(outdir_base)
        meta={"tag":tag,"P_true":P_true.tolist(),"pass_threshold":args.pass_threshold,
              "prob_threshold":args.prob_threshold,"samples":args.samples,
              "bandwidth":args.bandwidth,"models":[{"tag":t,"trial":tr} for t,tr in models],
              "device":str(device),"seed":args.seed}
        save_outputs(rows, outdir_base, meta)

if __name__=="__main__":
    main()
