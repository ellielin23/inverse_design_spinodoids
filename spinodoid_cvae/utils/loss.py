# utils/loss.py

import torch
import torch.nn.functional as F

def reconstruction_loss(S_hat, S):
    """
    Mean squared error loss between predicted and true structure parameters.
    """
    return F.mse_loss(S_hat, S, reduction='mean')

def kl_divergence(mu, logvar):
    """
    KL divergence between N(mu, sigma^2) and standard normal N(0, I).
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

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
    base_kl = kl_divergence(mu, logvar)

    if log_det is not None:
        kl = base_kl - torch.mean(log_det)
    else:
        kl = base_kl

    return rec + beta * kl, rec, kl