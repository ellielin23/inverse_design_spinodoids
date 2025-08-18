# Spinodoid CVAE — Release<br>

CLI for inverse design from elasticity tensors **C** (Mandel 21) to structure parameters **S_hat**, using trained parallel CVAE decoders and Max’s forward **fNN**.<br>

--------------------------------------------------------------------------------
## Quickstart<br>
From the `release/` folder:
- Install dependencies:<br>
    pip install -r requirements.txt<br>
- Run inference (CSV: 21 Mandel values per row, no headers):<br>
    python infer.py --csv data/target_samples.csv<br>

Common options:
- Choose a subset of tags (default = all seven):<br>
    python infer.py --csv data/target_samples.csv --tags 001,011,111<br>
- Tighten pass threshold to 5%:<br>
    python infer.py --csv data/target_samples.csv --pass-threshold 0.05<br>
- Increase latent samples and force CPU:<br>
    python infer.py --csv data/target_samples.csv --samples 1500 --device cpu<br>

--------------------------------------------------------------------------------
## Directory layout (relative to `release/`)<br>
release/<br>
&nbsp;&nbsp;infer.py<br>
&nbsp;&nbsp;requirements.txt<br>
&nbsp;&nbsp;models/<br>
&nbsp;&nbsp;&nbsp;&nbsp;001/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;config_001.json<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;decoder_001.pt<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;P_mean_001.npy<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;P_std_001.npy<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;S_mean_001.npy<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;S_std_001.npy<br>
&nbsp;&nbsp;&nbsp;&nbsp;010/ (same pattern)<br>
&nbsp;&nbsp;&nbsp;&nbsp;100/<br>
&nbsp;&nbsp;&nbsp;&nbsp;011/<br>
&nbsp;&nbsp;&nbsp;&nbsp;101/<br>
&nbsp;&nbsp;&nbsp;&nbsp;110/<br>
&nbsp;&nbsp;&nbsp;&nbsp;111/<br>
&nbsp;&nbsp;fNN/<br>
&nbsp;&nbsp;&nbsp;&nbsp;fNN.h5<br>
&nbsp;&nbsp;utils/<br>
&nbsp;&nbsp;&nbsp;&nbsp;formatting.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;data_processing.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;model_loaders.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;evaluate.py<br><br>

Assumptions:
- Model bundles live in `models/<tag>/`<br>
- fNN model lives at `fNN/fNN.h5`<br>

--------------------------------------------------------------------------------
## Input format<br>
- CSV with **no headers**.<br>
- Each row is **21 Mandel** upper-triangle values of the symmetric **6x6** stiffness matrix **C** (row-major upper triangle). Order of the 21 values:<br><br>
(0,0),<br>
(0,1),(0,2),(0,3),(0,4),(0,5),<br>
(1,1),(1,2),(1,3),(1,4),(1,5),<br>
(2,2),(2,3),(2,4),(2,5),<br>
(3,3),(3,4),(3,5),<br>
(4,4),(4,5),<br>
(5,5)<br><br>
- Minimal example (single row):<br><br>
0.25739938,0.08234422,0.08384212,-0.001122892,0.0018023637,-0.00040648473,0.2489467,0.084682286,7.031013e-05,0.0009808041,0.00050578703,0.26596344,0.0002066667,0.0023057577,0.0029397272,0.16781569,-0.00031259615,0.001966438,0.17230871,8.903016e-05,0.17266414<br><br>
- The script reconstructs the full **3x3x3x3** tensor and extracts **P ∈ R^9**:<br><br>
[C1111, C1122, C1133, C2222, C2233, C3333, C1212, C1313, C2323]<br>

--------------------------------------------------------------------------------
## What the script does (per input row)<br>
1) Convert C(21) to:<br>
&nbsp;&nbsp;- C_true (shape 3x3x3x3)<br>
&nbsp;&nbsp;- P_true (shape 9)<br>
2) For each selected theta-tag (default: 001,010,100,011,101,110,111):<br>
&nbsp;&nbsp;- Normalize P_true with that tag’s (P_mean, P_std)<br>
&nbsp;&nbsp;- Sample latent and find KDE peaks in S_hat<br>
&nbsp;&nbsp;- Enforce structure constraints on S_hat<br>
3) Predict and score:<br>
&nbsp;&nbsp;- C_pred = fNN(S_hat)<br>
&nbsp;&nbsp;- error = ||C_pred - C_true||_F / ||C_true||_F (reported as percent with two decimals)<br>
4) Save all candidates and the subset that PASS the threshold.<br>

--------------------------------------------------------------------------------
## Outputs<br>
For each input row `i` the script writes:<br><br>
outputs/row_i/<br>
&nbsp;&nbsp;all_candidates.csv<br>
&nbsp;&nbsp;passing_candidates.csv<br>
&nbsp;&nbsp;meta.json<br><br>

`all_candidates.csv` / `passing_candidates.csv` columns:<br>
- tag       : theta pattern (one of 001,010,100,011,101,110,111)<br>
- prob_est  : estimated peak probability<br>
- S_hat     : candidate structure parameters as a list string<br>
- error     : relative tensor error as percent (two decimals)<br>
- status    : PASS if error < --pass-threshold, else FAIL<br>
`meta.json` records arguments, tags, device, seed, and source file path for reproducibility.<br>

--------------------------------------------------------------------------------
## CLI usage<br>
- Basic:<br>
python infer.py --csv PATH/TO/file.csv<br><br>

- Options:<br>
--tags            Comma list or "all". Default = all seven (001,010,100,011,101,110,111)<br>
--pass-threshold  PASS if error < fraction. Default = 0.04 (4%)<br>
--prob-threshold  KDE peak probability filter. Default = 0.10<br>
--samples         Latent samples per model. Default = 1000<br>
--bandwidth       "auto" (default) or a float (e.g., 0.3) for KDE<br>
--device          cpu | cuda | mps (Apple). Omit to auto-select<br>
--seed            RNG seed. Default = 42<br>
--outdir          Output folder. Default = outputs<br><br>

- Examples:<br>
python infer.py --csv data/target_samples.csv<br>
python infer.py --csv data/target_samples.csv --tags 001,011,111 --pass-threshold 0.05<br>
python infer.py --csv data/target_samples.csv --samples 1500 --device cpu<br>

--------------------------------------------------------------------------------
## Model bundle requirements (per tag under `models/<tag>/`)<br>
- Required files:<br>
config_<tag>.json<br>
decoder_<tag>.pt<br>
P_mean_<tag>.npy<br>
P_std_<tag>.npy<br>
S_mean_<tag>.npy<br>
S_std_<tag>.npy<br><br>

- Config keys used by the loader:<br>
S_DIM, P_DIM, LATENT_DIM, DECODER_HIDDEN_DIMS, USE_ATTENTION_DECODER, DROPOUT_PROB<br><br>

- Important:<br>
DROPOUT_PROB in the JSON must match training. Mismatched dropout changes layer indices and breaks strict loading.<br>

--------------------------------------------------------------------------------
## Install notes<br>
- Install dependencies:<br> pip install -r requirements.txt<br>
- PyTorch:<br> requirements.txt pins a CPU/MPS build. For CUDA, install the proper wheel from pytorch.org after requirements.<br>
- TensorFlow:<br> On Apple Silicon, requirements install tensorflow-macos + tensorflow-metal. Elsewhere, requirements install standard tensorflow.<br>

--------------------------------------------------------------------------------
## Troubleshooting<br>
- Missing/Unexpected keys when loading a decoder: ensure DECODER_HIDDEN_DIMS and DROPOUT_PROB in the JSON match the checkpoint.<br>
- PyTorch torch.load FutureWarning: loader uses a safe path when supported. Harmless for inference.<br>
- TensorFlow "No training configuration found": harmless. fNN is loaded with compile=False and used only for inference.<br>
- Models folder not found: run from `release/`, ensure `models/<tag>/` exists for selected tags.<br>

--------------------------------------------------------------------------------
## Repro tips<br>
- Keep defaults: `--seed 42`, `--bandwidth auto`.<br>
- Share `meta.json` with results.<br>
- Record versions if needed:<br>
&nbsp;&nbsp;python -V<br>
&nbsp;&nbsp;python -c "import torch, numpy as np, tensorflow as tf; print(torch.__version__, np.__version__, tf.__version__)"<br>

--------------------------------------------------------------------------------
## License / Citation<br>
Add your license and citation information here.<br>