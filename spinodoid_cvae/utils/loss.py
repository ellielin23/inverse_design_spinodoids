# utils/loss.py

import torch
import torch.nn.functional as F
import numpy as np

def reconstruction_loss(S_hat, S):
    """
    Mean squared error loss between predicted and true structure parameters.
    """
    return F.mse_loss(S_hat, S, reduction='mean')

from utils.data_utils.load_data import extract_target_properties

def forward_consistency_loss(S_hat, P_true, fNN, component_weights):
    """
    Computes forward loss using fNN and compares 9 extracted components.
    """
    S_hat_np = S_hat.detach().cpu().numpy().reshape(-1, 1, S_hat.shape[1])
    P_pred_np = fNN.predict_on_batch(S_hat_np)
    P_pred_np = P_pred_np[:, 0]  # shape: (batch, 3, 3, 3, 3)

    # Use your extraction logic on predicted tensor
    P_pred_np_extracted = extract_target_properties(P_pred_np)  # shape: (batch, 9)

    # Convert to torch
    P_pred = torch.tensor(P_pred_np_extracted, dtype=torch.float32, device=S_hat.device)

    # P_true is already shape (batch, 9), from your dataset
    if component_weights is not None:
        weights = torch.tensor(component_weights, dtype=torch.float32, device=S_hat.device)
        loss = ((weights * (P_pred - P_true) ** 2).mean())
    else:
        loss = F.mse_loss(P_pred, P_true, reduction='mean')

    return loss


def kl_divergence(mu, logvar):
    """
    KL divergence between N(mu, sigma^2) and standard normal N(0, I).
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def get_kl_beta(epoch, warmup_epochs=20, max_beta=1.0):
    """
    Linearly increases beta from 0 to max_beta over warmup_epochs.
    After that, keeps it at max_beta.
    """
    return min(max_beta, (epoch + 1) / warmup_epochs * max_beta)


def total_loss(S_hat, S, mu, logvar, log_det=None, beta=1.0):
    """
    Total CVAE loss, optionally flow-aware:
      - With flow: KL = base_KL - log_det_Jacobian
      - Without flow: standard KL term

    Args:
        S_hat (Tensor): Predicted structure vector
        S (Tensor): Ground-truth structure vector
        mu (Tensor): Mean of q(z|x)
        logvar (Tensor): Log-variance of q(z|x)
        log_det (Tensor or None): log-det-Jacobian from flow (optional)
        beta (float): KL divergence weight
    """
    rec = reconstruction_loss(S_hat, S)
    kl = kl_divergence(mu, logvar)
    return rec + beta * kl, rec, kl