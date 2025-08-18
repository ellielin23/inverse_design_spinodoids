# evaluate.py

import torch
import numpy as np
from sklearn.cluster import MeanShift

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


def extract_peaks_with_bandwidth(S_hats, use_auto_bandwidth=False, manual_bw=4.0, target_range=(5, 8), verbose=True):
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
    if use_auto_bandwidth:
        selected_bw, S_hat_peaks = auto_select_bandwidth(S_hats, target_range=target_range)
        bw_used = selected_bw or manual_bw  # fallback
        if S_hat_peaks is not None:
            if verbose:
                print(f"\n✅ [Auto] Selected bandwidth: {bw_used:.2f} → Found {len(S_hat_peaks)} peak(s)")
        else:
            if verbose:
                print("\n❌ [Auto] Could not find a bandwidth that yields desired number of peaks.")
            S_hat_peaks = []
    else:
        S_hat_peaks = get_S_hat_peaks(S_hats, bandwidth=manual_bw)
        bw_used = manual_bw
        print(f"\n✅ [Manual] Used bandwidth: {bw_used:.2f} → Found {len(S_hat_peaks)} peak(s)")

    return S_hat_peaks, bw_used


def sort_and_select_peaks_by_probability(S_hats, S_hat_peaks, bw_used, prob_threshold=0.10, verbose=True):
    """
    Sorts S_hat_peaks by empirical probability using MeanShift clustering
    and filters out peaks below a given probability threshold.
    
    Args:
        S_hats (np.ndarray): All sampled S vectors, shape (N, S_dim)
        S_hat_peaks (np.ndarray): Initial peak estimates, shape (k, S_dim)
        bw_used (float): Bandwidth used for clustering
        prob_threshold (float): Minimum empirical probability to keep a peak
        verbose (bool): Whether to print info about peak frequencies
    
    Returns:
        sorted_centers (np.ndarray): Peaks sorted by descending empirical frequency
                                     and passing the probability threshold
        sorted_probs (np.ndarray): Corresponding probabilities for each retained peak
        sorted_counts (np.ndarray): Number of points in each retained cluster
    """    
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

    # === probability filter ===
    mask = sorted_probs >= prob_threshold
    sorted_centers = sorted_centers[mask]
    sorted_probs = sorted_probs[mask]
    sorted_counts = sorted_counts[mask]

    if verbose:
        print(f"\n✅ Sorted empirical probabilities for each peak (≥ {prob_threshold:.2f}):")
        for i, (count, prob, center) in enumerate(zip(sorted_counts, sorted_probs, sorted_centers)):
            center_str = np.array2string(center, precision=3, separator=', ')
            print(f"  Peak {i}: {count} samples ({prob:.3f})")

    return sorted_centers, sorted_probs, sorted_counts


def euclidean_norm(t: np.ndarray) -> float:
    return float(np.sqrt(np.sum(t**2)))


def compute_tensor_error(C_true: np.ndarray, C_pred: np.ndarray) -> float:
    return euclidean_norm(C_true - C_pred) / euclidean_norm(C_true)


def enforce_theta_domain(S_hat_peaks):
    """
    Enforce Max's suggested non-connected domain on theta components:
    - If theta_i < 0.75, set to 0
    - If 0.75 <= theta_i <= 15, set to 15
    Leaves other values unchanged.

    Args:
        S_hat_peaks (np.ndarray): Array of shape (n_peaks, 4) with candidate S vectors

    Returns:
        np.ndarray: Modified candidate peaks with theta domain enforced
    """
    S_modified = np.copy(S_hat_peaks)
    for i, s in enumerate(S_modified):
        for j in range(3):  # for each theta component (first 3)
            if s[j] < 0.75:
                S_modified[i, j] = 0
            elif 0.75 <= s[j] <= 15:
                S_modified[i, j] = 15
    return S_modified


def filter_S_candidates(S_hat_peaks):
    """
    Filter candidate S peaks to only those that satisfy domain constraints:
    - Each theta component is either 0 or in [15, 90]
    - rho in [0.3, 1]

    Args:
        S_hat_peaks (np.ndarray): Array of shape (n_peaks, 4) with candidate S vectors

    Returns:
        np.ndarray: Filtered candidate peaks
    """
    filtered = []
    for s in S_hat_peaks:
        theta = s[:3]
        rho = s[3]

        theta_condition = np.all((theta == 0) | ((theta >= 15) & (theta <= 90)))
        rho_condition = (0.3 <= rho <= 1)

        if theta_condition and rho_condition:
            filtered.append(s)

    return np.array(filtered)


def eval_one_row(P_true, C_true, decoders, cfgs, Pm, Ps, Sm, Ss, fNN,
                 pass_thr, prob_thr, samples, bw, seed, device, tags):
    rows = []
    with torch.no_grad():
        for midx, (cfg, p_mean, p_std, s_mean, s_std) in enumerate(zip(cfgs, Pm, Ps, Sm, Ss)):
            decoder    = decoders[midx]
            latent_dim = int(cfg["LATENT_DIM"])

            # === normalize P for this tag ===
            Pn   = (P_true - p_mean) / (p_std + 1e-8)
            Pn_t = torch.tensor(Pn, dtype=torch.float32, device=device).unsqueeze(0)

            # === sample Ŝ (normalized) ===
            S_norm = get_S_hats(decoder, Pn_t, latent_dim,
                                num_samples=samples, seed=seed, device=device)

            # === peak extraction ===
            if isinstance(bw, str) and bw.lower() == "auto":
                peaks_norm, bw_used = extract_peaks_with_bandwidth(
                    S_norm, use_auto_bandwidth=True, target_range=(1, 10), verbose=False
                )
            else:
                bw_used = float(bw)
                peaks_norm = get_S_hat_peaks(S_norm, bandwidth=bw_used)

            # === probability filter + sort ===
            peaks_norm, probs, _ = sort_and_select_peaks_by_probability(
                S_norm, peaks_norm, bw_used, prob_threshold=prob_thr, verbose=False
            )

            # === denorm + constraints ===
            peaks = peaks_norm * s_std + s_mean
            peaks = enforce_theta_domain(peaks)
            peaks = filter_S_candidates(peaks)

            tag = tags[midx]
            for k, S_hat in enumerate(peaks):
                # forward to Ĉ and compute tensor error vs C_true
                C_pred   = fNN(np.expand_dims(S_hat, (0, 1))).numpy().reshape(1, 3, 3, 3, 3)[0]
                err_frac = compute_tensor_error(C_true, C_pred)    # fraction
                err_pct  = f"{round(err_frac * 100.0, 2):.2f}%"

                rows.append({
                    "tag": tag,
                    "prob_est": float(probs[k]) if k < len(probs) else np.nan,
                    "S_hat": "[" + ", ".join(f"{v:.4f}" for v in S_hat) + "]",
                    "error": err_pct,
                    "status": "PASS" if err_frac < pass_thr else "FAIL",
                    "_err": err_frac,  # numeric for sorting
                })
    return rows