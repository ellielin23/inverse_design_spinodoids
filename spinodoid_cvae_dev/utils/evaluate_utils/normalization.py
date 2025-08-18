# utils/evaluate_utils/normalization.py

import numpy as np

def compute_normalized_mse(P_preds, P_true):
    """
    Computes component-wise normalized MSE:
        nMSE_i = MSE_i / mean(P_true_i^2)

    Args:
        P_preds: (num_peaks, 9) predicted properties
        P_true:  (9,) true property vector (broadcasted)

    Returns:
        normalized_mse: (9,) array
    """
    P_preds = np.array(P_preds)
    P_true = np.array(P_true)

    mse = np.mean((P_preds - P_true) ** 2, axis=0)
    denom = np.mean(P_true ** 2, axis=0)
    normalized_mse = mse / denom
    return normalized_mse