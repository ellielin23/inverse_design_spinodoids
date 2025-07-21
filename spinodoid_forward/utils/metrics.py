import numpy as np
from scipy.stats import entropy

def average_kl(P_pred, P_true, bins=30):
    """
    Computes average KL divergence across 9 dimensions.
    Args:
        P_pred: np.array (N, 9) - predicted samples
        P_true: np.array (M, 9) - empirical samples
    Returns:
        avg_kl: float - average KL across dimensions
    """
    kl_values = []
    for i in range(9):
        pred = P_pred[:, i]
        true = P_true[:, i]

        edges = np.histogram_bin_edges(np.concatenate([pred, true]), bins=bins)
        p_hist, _ = np.histogram(true, bins=edges, density=True)
        q_hist, _ = np.histogram(pred, bins=edges, density=True)

        # avoid log(0)
        p_hist += 1e-10
        q_hist += 1e-10

        kl = entropy(p_hist, q_hist)
        kl_values.append(kl)

    return np.mean(kl_values)
