# Spinodoid CVAE — Release

CLI for inverse design from elasticity tensors C (Mandel 21) to structure parameters S_hat, using trained parallel CVAE decoders plus Max's forward fNN.

--------------------------------------------------------------------------------
# QUICKSTART
--------------------------------------------------------------------------------
From the release/ folder:

1) Install dependencies
   pip install -r requirements.txt

2) Run inference (CSV: 21 Mandel values per row, no headers)
   python infer.py --csv data/target_samples.csv

Common options:
   ### choose a subset of tags (default = all seven)
   python infer.py --csv data/target_samples.csv --tags 001,011,111

   ### tighten pass threshold to 5%
   python infer.py --csv data/target_samples.csv --pass-threshold 0.05

   ### increase latent samples and force CPU
   python infer.py --csv data/target_samples.csv --samples 1500 --device cpu

--------------------------------------------------------------------------------
# DIRECTORY LAYOUT (relative to release/)
--------------------------------------------------------------------------------
release/
  infer.py
  requirements.txt
  manifest.json
  models/
    001/
      config_001.json
      decoder_001.pt
      P_mean_001.npy
      P_std_001.npy
      S_mean_001.npy
      S_std_001.npy
    010/  (same pattern)
    100/
    011/
    101/
    110/
    111/
  fNN/
    fNN.h5
    fNN_layers.py
  utils/
    formatting.py
    data_processing.py
    model_loaders.py
    evaluate.py
  data/
    sample_targets.csv

Assumptions:
- Model bundles live in models/<tag>/
- fNN model lives at fNN/fNN.h5

--------------------------------------------------------------------------------
# INPUT FORMAT
--------------------------------------------------------------------------------
- CSV with NO headers.
- Each row is 21 Mandel upper-triangle values of the symmetric 6x6 stiffness matrix C.
- Row-major order of the 21 values:

  (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),
  (1,1),(1,2),(1,3),(1,4),(1,5),
  (2,2),(2,3),(2,4),(2,5),
  (3,3),(3,4),(3,5),
  (4,4),(4,5),
  (5,5)

Example row (single line):
0.25739938,0.08234422,0.08384212,-0.001122892,0.0018023637,-0.00040648473,0.2489467,0.084682286,7.031013e-05,0.0009808041,0.00050578703,0.26596344,0.0002066667,0.0023057577,0.0029397272,0.16781569,-0.00031259615,0.001966438,0.17230871,8.903016e-05,0.17266414

The script reconstructs the full 3x3x3x3 tensor and extracts P in R^9 = [C1111, C1122, C1133, C2222, C2233, C3333, C1212, C1313, C2323].

--------------------------------------------------------------------------------
# WHAT THE SCRIPT DOES (per input row)
--------------------------------------------------------------------------------
1) Convert C(21) to:
   - C_true (shape 3x3x3x3)
   - P_true (shape 9)

2) For each selected theta-tag (default: 001,010,100,011,101,110,111):
   - Normalize P_true with that tag’s (P_mean, P_std)
   - Sample latent, find KDE peaks in S_hat, apply constraints

3) Predict C_pred = fNN(S_hat), then compute relative Frobenius error:
   error = ||C_pred - C_true||_F / ||C_true||_F
   The script reports this as a percent with two decimal places.

4) Save all candidates and the subset that PASS the threshold.

--------------------------------------------------------------------------------
# OUTPUTS
--------------------------------------------------------------------------------
For each input row i the script writes:

outputs/row_i/
  all_candidates.csv
  passing_candidates.csv
  meta.json

CSV columns:
- tag        : theta pattern (one of 001,010,100,011,101,110,111)
- prob_est   : estimated peak probability (aligned with peak_idx)
- S_hat      : candidate structure parameters as a list string
- error      : relative tensor error as percent (two decimals)
- status     : PASS if error < --pass-threshold, else FAIL

meta.json includes arguments, tags, device, seed, and source file path.

--------------------------------------------------------------------------------
# CLI USAGE
--------------------------------------------------------------------------------
python infer.py --csv PATH/TO/file.csv [options]

Options:
--tags            Comma list or "all". Default = all seven (001,010,100,011,101,110,111).
--pass-threshold  PASS if error < fraction. Default = 0.04 (4%).
--prob-threshold  KDE peak probability filter. Default = 0.10.
--samples         Latent samples per model. Default = 1000.
--bandwidth       "auto" (default) or a float (e.g., 0.3) for KDE.
--device          cpu | cuda | mps (Apple). Omit to auto-select.
--seed            RNG seed. Default = 42.
--outdir          Output folder. Default = outputs.

Examples:
python infer.py --csv data/target_samples.csv
python infer.py --csv data/target_samples.csv --tags 001,011,111 --pass-threshold 0.05
python infer.py --csv data/target_samples.csv --samples 1500 --device cpu

--------------------------------------------------------------------------------
# MODEL BUNDLE REQUIREMENTS (per tag under models/<tag>/)
--------------------------------------------------------------------------------
config_<tag>.json
decoder_<tag>.pt
P_mean_<tag>.npy
P_std_<tag>.npy
S_mean_<tag>.npy
S_std_<tag>.npy

Config keys used by the loader:
S_DIM, P_DIM, LATENT_DIM, DECODER_HIDDEN_DIMS, USE_ATTENTION_DECODER, DROPOUT_PROB

Important: DROPOUT_PROB in the JSON must match training. Mismatched dropout changes layer indices and breaks strict loading.

--------------------------------------------------------------------------------
# INSTALL NOTES
--------------------------------------------------------------------------------
Install dependencies:
pip install -r requirements.txt

PyTorch:
- requirements.txt pins a CPU/MPS build. For CUDA, install the proper wheel from pytorch.org after installing requirements.

TensorFlow:
- On Apple Silicon, requirements install tensorflow-macos + tensorflow-metal.
- Elsewhere, requirements install standard tensorflow.

--------------------------------------------------------------------------------
# TROUBLESHOOTING
--------------------------------------------------------------------------------
- Missing/Unexpected keys when loading a decoder:
  Ensure DECODER_HIDDEN_DIMS and DROPOUT_PROB in the JSON match the checkpoint used for that tag.

- PyTorch torch.load FutureWarning:
  The loader uses a safe path when supported. Harmless for inference.

- TensorFlow "No training configuration found" warning:
  Harmless. The fNN is loaded with compile=False and used only for inference.

- Models folder not found:
  Run from release/, ensure models/<tag>/ exists for all selected tags.

--------------------------------------------------------------------------------
# REPRO TIPS
--------------------------------------------------------------------------------
- Keep defaults: --seed 42, --bandwidth auto.
- Share meta.json with results for reproducibility.
- Record versions if needed:
  python -V
  python -c "import torch, numpy as np, tensorflow as tf; print(torch.__version__, np.__version__, tf.__version__)"