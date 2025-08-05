# utils/evaluate_utils/structure_constraints.py

import numpy as np

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
