# utils/losses.py

import torch
import math

# === gaussian negative log-likelihood ===
def gaussian_nll(mu, log_sigma, target):
    """
    Computes negative log-likelihood for a diagonal Gaussian.

    Args:
        mu (Tensor): Predicted mean, shape (batch_size, 9)
        log_sigma (Tensor): Predicted log std dev, shape (batch_size, 9)
        target (Tensor): Ground truth properties, shape (batch_size, 9)

    Returns:
        Tensor: Scalar loss (mean NLL over batch)
    """
    var = torch.exp(2 * log_sigma)  # σ² = (e^{logσ})² = e^{2logσ}
    nll = 0.5 * (2 * log_sigma + ((target - mu)**2) / var)
    return nll.mean()

# === normalizing flow negative log-likelihood ===
def flow_nll(base_mu, base_log_sigma, z_k, log_det_jacobians, target):
    """
    Computes the negative log-likelihood under a normalizing flow model.
    
    Args:
        base_mu (Tensor): Mean of base Gaussian, shape (batch_size, 9)
        base_log_sigma (Tensor): Log std dev of base Gaussian, shape (batch_size, 9)
        z_k (Tensor): Flow-transformed sample, shape (batch_size, 9)
        log_det_jacobians (Tensor): Total log-determinant from all flows, shape (batch_size,)
        target (Tensor): Ground truth property vector, shape (batch_size, 9)

    Returns:
        Tensor: Scalar loss (mean negative log-likelihood)
    """
    base_std = torch.exp(base_log_sigma)

    # log-prob of target under base Gaussian
    log_prob_base = -0.5 * torch.sum(
        ((target - base_mu) / base_std) ** 2
        + 2 * base_log_sigma
        + math.log(2 * math.pi),
        dim=1
    )  # shape: (batch_size,)

    # Adjust for flow transformation using change-of-variables formula
    log_prob_flow = log_prob_base + log_det_jacobians  # shape: (batch_size,)

    return -torch.mean(log_prob_flow)