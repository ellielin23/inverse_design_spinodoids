# infer.py

import os, sys, json, argparse, random
import numpy as np, pandas as pd, torch

from utils.evaluate_utils.load_paths import load_paths
from utils.evaluate_utils.load_models import load_config, load_decoder, load_fNN_model
from utils.evaluate_utils.sampling import (
    get_S_hats, extract_peaks_with_bandwidth, get_S_hat_peaks, sort_and_select_peaks_by_probability
)
from utils.evaluate_utils.structure_constraints import enforce_theta_domain, filter_S_candidates
from utils.data_utils.load_data import full_C_from_C_flat_21, extract_target_properties

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def fmt(arr, k=4):
    arr=np.asarray(arr,dtype=float).flatten()
    return "[" + ", ".join(f"{v:.{k}f}" for v in arr) + "]"

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_models_arg(s):
    return [(t.strip(), int(tr.strip())) for t,tr in (x.split(":") for x in s.split(",") if x.strip())]

def euclidean_norm(t: np.ndarray) -> float:
    return float(np.sqrt(np.sum(t**2)))

def compute_tensor_error(C_true: np.ndarray, C_pred: np.ndarray) -> float:
    return euclidean_norm(C_true - C_pred) / euclidean_norm(C_true)  # fraction

def load_all_models(models, device):
    decoders=[]; cfgs=[]; Pm=[]; Ps=[]; Sm=[]; Ss=[]
    for tag,trial in models:
        paths=load_paths(TRIAL=trial, THETA_MODEL=True, THETA_PATTERN=tag, verbose=False)
        cfg=load_config(paths["config_path"])
        dec=load_decoder(cfg, paths["decoder_path"], flow_type=cfg.get("FLOW_TYPE","planar"), trial=trial, device=device)
        pmean,pstd=np.load(paths["P_mean_path"]),np.load(paths["P_std_path"])
        smean,sstd=np.load(paths["S_mean_path"]),np.load(paths["S_std_path"])
        pstd=np.where(pstd<1e-8,1.0,pstd); sstd=np.where(sstd<1e-8,1.0,sstd)
        decoders.append(dec); cfgs.append(cfg); Pm.append(pmean); Ps.append(pstd); Sm.append(smean); Ss.append(sstd)
    return decoders,cfgs,Pm,Ps,Sm,Ss

def read_c_csv_unlabeled(csv_path: str):
    C21 = np.loadtxt(csv_path, delimiter=',', dtype=float)
    if C21.ndim == 1: C21 = C21.reshape(1, -1)
    if C21.shape[1] != 21:
        raise ValueError(f"{csv_path} has shape {C21.shape}; expected 21 columns (Mandel upper triangle).")
    C_true = full_C_from_C_flat_21(C21)          # (N,3,3,3,3)
    P_true = extract_target_properties(C_true)   # (N,9)
    return C_true, P_true

def eval_one_row(P_true, C_true, decoders, cfgs, Pm, Ps, Sm, Ss, fNN,
                 pass_thr, prob_thr, samples, bw, seed, device, models):
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
        peaks = peaks_norm * s_std + s_mean
        peaks = enforce_theta_domain(peaks); peaks = filter_S_candidates(peaks)
        tag = models[midx][0]
        for k,S_hat in enumerate(peaks):
            C_pred = fNN(np.expand_dims(S_hat,(0,1))).numpy().reshape(1,3,3,3,3)[0]
            err_frac = compute_tensor_error(C_true, C_pred)
            err_pct  = f"{round(err_frac*100.0, 2):.2f}%"
            rows.append({
                "tag": tag, "peak_idx": k, "bw_used": bw_used,
                "prob_est": float(probs[k]) if k<len(probs) else np.nan,
                "S_hat": "[" + ", ".join(f"{v:.4f}" for v in S_hat) + "]",
                "error": err_pct,
                "status": "PASS" if err_frac < pass_thr else "FAIL",
                "_err": err_frac,
            })
    return rows

def save_outputs(rows, outdir):
    if not rows:
        print("⚠️ No candidates after peak selection/constraints."); return
    df=pd.DataFrame(rows).sort_values(by=["_err","tag","peak_idx"]).reset_index(drop=True)
    df=df[["tag","peak_idx","bw_used","prob_est","S_hat","error","status"]]
    df_pass=df[df["status"]=="PASS"].reset_index(drop=True)
    ensure_dir(outdir)
    df.to_csv(os.path.join(outdir,"all_candidates.csv"), index=False)
    df_pass.to_csv(os.path.join(outdir,"passing_candidates.csv"), index=False)
    best=df.iloc[0]
    print(f"✅ total={len(df)}, pass={len(df_pass)} | BEST: error={best['error']}, tag={best['tag']}, Ŝ={best['S_hat']}")
    print(f"📁 {outdir}")

def main():
    ap=argparse.ArgumentParser(description="Parallel CVAE inverse design (C-only, tensor error %).")
    ap.add_argument("--csv", type=str, required=True, help="CSV with unlabeled 21 Mandel components per row.")
    ap.add_argument("--pass-threshold", type=float, default=0.08, help="PASS if tensor error < this fraction (e.g., 0.08 for 8%).")
    ap.add_argument("--prob-threshold", type=float, default=0.10, help="Peak probability threshold.")
    ap.add_argument("--samples", type=int, default=1000, help="Latent samples per model.")
    ap.add_argument("--bandwidth", type=str, default="auto", help='"auto" or float (e.g., 0.3) for KDE.')
    ap.add_argument("--models", type=str, default="001:18,010:6,100:9,011:16,101:5,110:3,111:5",
                    help='Comma list "tag:trial,...".')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, choices=[None,"cpu","cuda","mps"], default=None)
    ap.add_argument("--outdir", type=str, default="outputs")
    args=ap.parse_args()

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

    # read C (unlabeled 21 columns) → C_true & P_true
    C_true_all, P_true_all = read_c_csv_unlabeled(args.csv)

    # run each row
    for i,(C_true,P_true) in enumerate(zip(C_true_all, P_true_all)):
        tag=f"row_{i}"
        print(f"\n🎯 {tag}  (P_true={fmt(P_true)})")
        rows=eval_one_row(P_true, C_true, decoders, cfgs, Pm, Ps, Sm, Ss, fNN,
                          pass_thr=args.pass_threshold, prob_thr=args.prob_threshold,
                          samples=args.samples, bw=bw, seed=args.seed, device=device, models=models)
        outdir=os.path.join(args.outdir, tag)
        meta={"mode":"csv_c_unlabeled_21","row":i,"pass_threshold_fraction":args.pass_threshold,
              "prob_threshold":args.prob_threshold,"samples":args.samples,"bandwidth":args.bandwidth,
              "models":[{"tag":t,"trial":tr} for t,tr in models],
              "device":str(device),"seed":args.seed,"source_csv":args.csv}
        ensure_dir(outdir); open(os.path.join(outdir,"meta.json"),"w").write(json.dumps(meta,indent=2))
        save_outputs(rows, outdir)

if __name__=="__main__":
    main()
