# utils/evaluate_utils/error.py

import numpy as np

def euclidean_norm(tensor):
    """Compute Frobenius norm of a tensor (scalar result)."""
    return np.sqrt(np.sum(tensor ** 2))

def compute_tensor_error(C_true, C_pred):
    """Compute relative error between C_pred and C_true."""
    return euclidean_norm(C_true - C_pred) / euclidean_norm(C_true)