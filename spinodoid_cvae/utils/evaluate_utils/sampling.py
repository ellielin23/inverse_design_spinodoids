# utils/evaluate_utils/sampling.py

import torch
import numpy as np
from sklearn.cluster import MeanShift

def get_S_hats(decoder, P_val, latent_dim, num_samples=1000, device='cpu'):
    """
    Samples structure vectors S_hat from the decoder given a property vector P_val.
    Works with both Decoder and FlowDecoder.

    Args:
        decoder: Trained decoder model (standard or flow-based).
        P_val (torch.Tensor): Target property vector, shape (1, P_dim).
        latent_dim (int): Dimension of latent space.
        num_samples (int): Number of samples.
        device (str): Torch device.
    
    Returns:
        np.ndarray: Array of sampled S_hat vectors, shape (num_samples, S_dim).
    """
    P_tensor = P_val.repeat(num_samples, 1).to(device)
    z_samples = torch.randn((num_samples, latent_dim)).to(device)
    with torch.no_grad():
        S_hats = decoder(z_samples, P_tensor)
    return S_hats.cpu().numpy() if not isinstance(S_hats, tuple) else S_hats[0].cpu().numpy()


def get_S_hat_peaks(S_hats, bandwidth=5.0):
    """
    Applies MeanShift clustering to extract representative peak candidates from sampled S_hat vectors.

    Args:
        S_hats (np.ndarray): Sampled structure vectors, shape (num_samples, S_dim).
        bandwidth (float): Bandwidth for MeanShift.
    
    Returns:
        np.ndarray: Cluster centers (peak representatives), shape (n_peaks, S_dim).
    """
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(S_hats)
    return ms.cluster_centers_
