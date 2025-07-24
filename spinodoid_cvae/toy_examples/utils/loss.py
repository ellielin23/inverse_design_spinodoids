# utils/loss.py

import torch
import torch.nn.functional as F

def reconstruction_loss(S_hat, S):
    """
    Mean squared error loss between predicted and true structure parameters.
    Args:
        S_hat (Tensor): Predicted structure vector, shape (batch_size, 4)
        S (Tensor): Ground-truth structure vector, shape (batch_size, 4)
    Returns:
        Scalar MSE loss
    """
    return F.mse_loss(S_hat, S, reduction='mean')

def kl_divergence(mu, logvar):
    """
    Computes the KL divergence between N(mu, sigma^2) and standard normal N(0, I).
    Args:
        mu (Tensor): Mean of latent distribution, shape (batch_size, latent_dim)
        logvar (Tensor): Log variance of latent distribution, shape (batch_size, latent_dim)
    Returns:
        Scalar KL divergence loss
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def total_loss(S_hat, S, mu, logvar, beta=1.0):
    """
    Total CVAE loss: reconstruction + beta * KL
    """
    rec = reconstruction_loss(S_hat, S)
    kl = kl_divergence(mu, logvar)
    return rec + beta * kl, rec, kl
