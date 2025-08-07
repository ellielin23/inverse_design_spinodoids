# utils/evaluate_utils/sampling.py

import torch
import numpy as np
from sklearn.cluster import MeanShift
import pandas as pd

def get_S_hats(decoder, P_val, latent_dim, num_samples=1000, seed=42, device='cpu'):
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
    if seed is not None:
        torch.manual_seed(seed)
    z_samples = torch.randn((num_samples, latent_dim)).to(device)
    with torch.no_grad():
        S_hats = decoder(z_samples, P_tensor)
    return S_hats.cpu().numpy() if not isinstance(S_hats, tuple) else S_hats[0].cpu().numpy()


def get_S_hat_peaks(S_hats, bandwidth):
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


def auto_select_bandwidth(S_hats, target_range=(5, 8), search_space=None):
    """
    Automatically selects a bandwidth that yields a number of peaks in the desired range.
    
    Args:
        S_hats (ndarray): Sampled structure vectors (num_samples, S_dim)
        target_range (tuple): Desired number of peaks (min, max)
        search_space (iterable or None): Bandwidth values to search over
    
    Returns:
        selected_bw (float or None): Chosen bandwidth
        peaks (ndarray): Cluster centers at chosen bandwidth
    """
    if search_space is None:
        search_space = np.linspace(0.1, 20.0, 200)

    for bw in search_space:
        peaks = get_S_hat_peaks(S_hats, bandwidth=bw)
        if target_range[0] <= len(peaks) <= target_range[1]:
            return bw, peaks

    return None, None



def extract_peaks_with_bandwidth(S_hats, use_auto_bandwidth=False, manual_bw=4.0, target_range=(5, 8)):
    """
    Extract peaks from sampled S_hats using mean shift clustering.

    Args:
        S_hats (np.ndarray): Sampled structure vectors, shape (N, S_dim).
        use_auto_bandwidth (bool): Whether to auto-select bandwidth.
        manual_bw (float): Bandwidth to use if not auto-selecting.
        target_range (tuple): Min/max number of peaks to target when auto-selecting bandwidth.

    Returns:
        S_hat_peaks (np.ndarray): Peak structure vectors.
        bw_used (float): The bandwidth used.
    """
    import numpy as np
    from .sampling import get_S_hat_peaks, auto_select_bandwidth

    if use_auto_bandwidth:
        selected_bw, S_hat_peaks = auto_select_bandwidth(S_hats, target_range=target_range)
        bw_used = selected_bw or manual_bw  # fallback
        if S_hat_peaks is not None:
            print(f"\n✅ [Auto] Selected bandwidth: {bw_used:.2f} → Found {len(S_hat_peaks)} peak(s)")
        else:
            print("\n❌ [Auto] Could not find a bandwidth that yields desired number of peaks.")
            S_hat_peaks = []
    else:
        S_hat_peaks = get_S_hat_peaks(S_hats, bandwidth=manual_bw)
        bw_used = manual_bw
        print(f"\n✅ [Manual] Used bandwidth: {bw_used:.2f} → Found {len(S_hat_peaks)} peak(s)")

    return S_hat_peaks, bw_used



def extract_peaks_with_bandwidth_no_print(S_hats, use_auto_bandwidth=False, manual_bw=4.0, target_range=(5, 8)):
    """
    Extract peaks from sampled S_hats using mean shift clustering.

    Args:
        S_hats (np.ndarray): Sampled structure vectors, shape (N, S_dim).
        use_auto_bandwidth (bool): Whether to auto-select bandwidth.
        manual_bw (float): Bandwidth to use if not auto-selecting.
        target_range (tuple): Min/max number of peaks to target when auto-selecting bandwidth.

    Returns:
        S_hat_peaks (np.ndarray): Peak structure vectors.
        bw_used (float): The bandwidth used.
    """
    import numpy as np
    from .sampling import get_S_hat_peaks, auto_select_bandwidth

    if use_auto_bandwidth:
        selected_bw, S_hat_peaks = auto_select_bandwidth(S_hats, target_range=target_range)
        bw_used = selected_bw or manual_bw  # fallback
        if S_hat_peaks is None:
            print("\n❌ [Auto] Could not find a bandwidth that yields desired number of peaks.")
            S_hat_peaks = []
    else:
        S_hat_peaks = get_S_hat_peaks(S_hats, bandwidth=manual_bw)
        bw_used = manual_bw

    return S_hat_peaks, bw_used



def sort_peaks_by_empirical_probability(S_hats, S_hat_peaks, bw_used, verbose=True):
    """
    Sorts S_hat_peaks by empirical probability using MeanShift clustering.
    
    Args:
        S_hats (np.ndarray): All sampled S vectors, shape (N, S_dim)
        S_hat_peaks (np.ndarray): Initial peak estimates, shape (k, S_dim)
        bw_used (float): Bandwidth used for clustering
        verbose (bool): Whether to print info about peak frequencies
    
    Returns:
        sorted_centers (np.ndarray): Peaks sorted by descending empirical frequency
        sorted_probs (np.ndarray): Corresponding probabilities for each peak
        sorted_counts (np.ndarray): Number of points in each cluster
    """
    import numpy as np
    from sklearn.cluster import MeanShift
    
    if S_hat_peaks is None:
        return None, None, None

    ms = MeanShift(bandwidth=bw_used)
    cluster_labels = ms.fit_predict(S_hats)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    total = len(S_hats)
    empirical_probs = counts / total

    sorted_indices = np.argsort(-empirical_probs)
    sorted_probs = empirical_probs[sorted_indices]
    sorted_counts = counts[sorted_indices]
    sorted_centers = ms.cluster_centers_[sorted_indices]

    if verbose:
        print("\n✅ Sorted empirical probabilities for each peak:")
        for i, (count, prob, center) in enumerate(zip(sorted_counts, sorted_probs, sorted_centers)):
            center_str = np.array2string(center, precision=3, separator=', ')
            print(f"  Peak {i}: {count} samples ({prob:.3f})")

    return sorted_centers, sorted_probs, sorted_counts


def format_array(arr, precision=5):
    return "[" + ", ".join(f"{x:.{precision}f}" for x in arr) + "]"


def make_candidate_table(S_hat_peaks, S_true):
    """
    Create a pretty df showing each Ŝ, the true S, and ΔS.
    """
    rows = []
    for S_hat in S_hat_peaks:
        delta = S_hat - S_true
        rows.append({
            "Ŝ": format_array(S_hat),
            "S_true": format_array(S_true),
            "ΔS": format_array(delta)
        })
    df = pd.DataFrame(rows)
    from IPython.display import display
    display(df)
    return df
