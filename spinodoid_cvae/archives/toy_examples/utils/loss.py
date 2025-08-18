# utils/loss.py

import torch
import torch.nn.functional as F
import numpy as np
from utils.load_data import full_C_from_C_flat_21, extract_target_properties

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


# map full tensor component names to indices in the 9D property vector
COMPONENT_INDEX_MAP = {
    "C111": 0,
    "C122": 1,
    "C133": 2,
    "C222": 3,
    "C233": 4,
    "C333": 5,
    "C212": 6,  # C212 equivalent
    "C313": 7,
    "C323": 8,
}


def forward_consistency_loss(S_hat, P, fNN, component_name="C212"):
    """
    Computes the forward loss: how well Ŝ (predicted structure) maps back to the true property P
    via Max's forward model and extract_target_properties.

    Args:
        S_hat (Tensor): shape (batch_size, 4)
        P (Tensor): shape (batch_size, 1) — true value of the specific component (e.g., C212)
        fNN (Keras Model): Max's trained forward model
        component_name (str): which component of the 9D vector to use (e.g., "C212")

    Returns:
        Scalar forward consistency loss (MSE between predicted and true component)
    """
    COMPONENT_INDEX_MAP = {
        "C111": 0,
        "C122": 1,
        "C133": 2,
        "C222": 3,
        "C233": 4,
        "C333": 5,
        "C212": 6,  # shear
        "C313": 7,
        "C323": 8,
    }
    component_idx = COMPONENT_INDEX_MAP[component_name]

    # Convert to numpy and format for fNN input: (N, 1, 4)
    S_hat_np = S_hat.detach().cpu().numpy()
    S_hat_np = S_hat_np[:, np.newaxis, :]  # shape: (N, 1, 4)

    # Forward pass through fNN → (N, 1, 3, 3, 3, 3)
    C_tensor_pred = fNN.predict(S_hat_np, verbose=0)

    # Remove singleton dimension → (N, 3, 3, 3, 3)
    C_tensor_pred = np.squeeze(C_tensor_pred, axis=1)

    # Extract 9 properties (→ shape: N × 9)
    from utils.load_data import extract_target_properties
    P_pred_9 = extract_target_properties(C_tensor_pred)

    # Get target component
    P_pred_target = P_pred_9[:, component_idx]  # shape: (N,)
    P_pred_target = torch.tensor(P_pred_target, dtype=torch.float32)

    # Ensure shape match
    if len(P.shape) == 2 and P.shape[1] == 1:
        P = P.squeeze(1)

    return F.mse_loss(P_pred_target, P)


def total_loss(S_hat, S, mu, logvar, beta=1.0):
    """
    Total CVAE loss: reconstruction + beta * KL
    """
    rec = reconstruction_loss(S_hat, S)
    kl = kl_divergence(mu, logvar)
    return rec + beta * kl, rec, kl


def total_loss_with_forward(S_hat, S, mu, logvar, P, fNN, component_name, beta=1.0, lamb=1.0):
    """
    CVAE loss with forward consistency:
        total = recon + beta * KL + λ * forward_consistency
    """
    rec = reconstruction_loss(S_hat, S)
    kl = kl_divergence(mu, logvar)
    fwd = forward_consistency_loss(S_hat, P, fNN, component_name)
    total = rec + beta * kl + lamb * fwd
    return total, rec, kl, fwd

