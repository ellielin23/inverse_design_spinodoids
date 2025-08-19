# utils/losses.py

import torch, math
from config import BETA_VAR_REG

LOG_2PI = math.log(2.0 * math.pi)

# === gaussian negative log-likelihood ===
def gaussian_nll(mu, log_sigma, target, reduction="mean"):
    """
    mu, log_sigma, target: (B, D) in the SAME (normalized) space.
    Returns mean NLL by default.
    """
    # per-dimension NLL: 0.5*[((y-mu)/sigma)^2 + 2*log_sigma + log(2π)]
    inv_sigma = torch.exp(-log_sigma)
    z = (target - mu) * inv_sigma
    nll_dim = 0.5 * (z**2 + 2.0 * log_sigma + LOG_2PI)

    # optional variance regularizer (usually keep BETA_VAR_REG=0)
    if BETA_VAR_REG and BETA_VAR_REG > 0:
        var_reg = torch.exp(2.0 * log_sigma).mean()
        nll_dim = nll_dim + 0.0 * var_reg  # just to keep graph alive; added below

    # reduce: sum over dims, then mean over batch
    nll = nll_dim.sum(dim=1)
    out = nll.mean() if reduction == "mean" else nll

    if BETA_VAR_REG and BETA_VAR_REG > 0:
        out = out + BETA_VAR_REG * torch.exp(2.0 * log_sigma).mean()

    return out

# === normalizing flow NLL (change-of-variables) ===
def flow_nll(base_mu, base_log_sigma, z_k, log_det_jacobians, reduction="mean"):
    """
    Evaluate log p(P) via z_k = f(P), using base Gaussian N(z | base_mu, base_sigma).
    Inputs:
      - z_k: (B, D) latent after last flow
      - log_det_jacobians: (B,) total log|det ∂f/∂P|
      - base_mu, base_log_sigma: (B, D) or (D,) or None for standard Normal
    Returns mean NLL by default.
    """
    if base_mu is None or base_log_sigma is None:
        # standard Normal base
        log_prob_base = -0.5 * (z_k**2 + LOG_2PI).sum(dim=1)
    else:
        std = torch.exp(base_log_sigma)
        z = (z_k - base_mu) / std
        log_prob_base = -0.5 * (z**2 + 2.0 * base_log_sigma + LOG_2PI).sum(dim=1)

    log_prob = log_prob_base + log_det_jacobians
    nll = -log_prob
    return nll.mean() if reduction == "mean" else nll
