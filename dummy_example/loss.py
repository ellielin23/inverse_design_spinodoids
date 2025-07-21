# loss.py

import torch
import torch.nn.functional as F
from config import BETA

def reconstruction_loss(S_hat, S):
    """
    Mean squared error between predicted and true structure parameters.
    """
    return F.mse_loss(S_hat, S, reduction='mean')


def kl_divergence(mu, logvar):
    """
    KL divergence between learned latent distribution and standard normal.
    Formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


# def physics_consistency_loss(P_hat, P):
#     """
#     Mean squared error between predicted and target material properties.
#     """
#     return F.mse_loss(P_hat, P, reduction='mean')


def vae_loss(S_hat, S, mu, logvar, P_hat, P):
    """
    Combined loss for the Conditional VAE.
    Includes:
      - Reconstruction loss (structure)
      - KL divergence loss
      - Physics consistency loss (TO BE ADDED LATER)
    """
    recon = reconstruction_loss(S_hat, S)
    kl = kl_divergence(mu, logvar)
    # physics = physics_consistency_loss(P_hat, P)

    total_loss = recon + BETA * kl
    return total_loss