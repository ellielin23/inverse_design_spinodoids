# utils/evaluate_utils/visualize_output.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_S_hat_space(S_hats, S_true, S_hat_peaks):
    """
    Performs PCA on the S_hat samples and plots the projections of:
    - Sampled \hat{S}
    - True S
    - Detected peaks
    """
    pca = PCA(n_components=2)
    S_pca = pca.fit_transform(S_hats)
    S_true_pca = pca.transform(S_true.reshape(1, -1))
    S_peaks_pca = pca.transform(S_hat_peaks)

    plt.figure(figsize=(6, 5))
    plt.scatter(S_pca[:, 0], S_pca[:, 1], alpha=0.3, label=r"Sampled $\hat{S}$", color='gray')
    plt.scatter(S_true_pca[0, 0], S_true_pca[0, 1], color='red', marker='x', s=100, label=r"$S_{true}$")
    plt.scatter(S_peaks_pca[:, 0], S_peaks_pca[:, 1], color='blue', marker='o', s=80, label="Peaks")
    plt.title(r"PCA Projection of $\hat{S}$ Samples with Detected Peaks")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_P_preds_vs_true(P_preds, P_true):
    """
    Plots all P_pred vs P_true bar plots in a grid layout.

    Args:
        P_preds (list of np.ndarray): List of predicted property vectors (each shape (9,))
        P_true (np.ndarray): Ground-truth property vector (shape (9,))
    """
    labels = [
        "C1111", "C1122", "C1133", "C2222", "C2233", "C3333",
        "C1212", "C1313", "C2323"
    ]
    num_peaks = len(P_preds)
    cols = 3
    rows = (num_peaks + cols - 1) // cols
    width = 0.35
    x = np.arange(len(labels))

    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axs = axs.flatten()

    for i in range(num_peaks):
        ax = axs[i]
        ax.bar(x - width/2, P_true, width, label='True P', color='lightcoral')
        ax.bar(x + width/2, P_preds[i], width, label='Predicted P', color='skyblue')
        ax.set_title(f"Peak {i + 1}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    for j in range(num_peaks, len(axs)):
        axs[j].axis('off')

    axs[0].legend(loc='upper right')
    fig.suptitle("Elastic Components: Predicted vs True for All Peaks", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def evaluate_peaks(S_hat_peaks, P_target, fNN, extract_target_properties):
    """
    Evaluates each peak \hat{S} using Max's fNN and prints per-peak error metrics.

    Returns:
        - P_preds: list of predicted P vectors
        - errors: list of L2 errors
        - mses: list of MSEs
    """
    P_preds = []
    errors = []
    mses = []

    print(f"{'Peak':<6} {'||P_pred - P_true||':<22} {'MSE (per peak)':<15}")
    print("-" * 45)

    for i, S_peak in enumerate(S_hat_peaks):
        S_peak_tf = np.expand_dims(S_peak, axis=(0, 1))  # shape: (1, 1, 4)
        C_pred = fNN(S_peak_tf).numpy().reshape(1, 3, 3, 3, 3)
        P_pred = extract_target_properties(C_pred)[0]
        P_preds.append(P_pred)

        l2_error = np.linalg.norm(P_pred - P_target)
        mse = np.mean((P_pred - P_target) ** 2)
        errors.append(l2_error)
        mses.append(mse)

        print(f"{i:<6} {l2_error:<22.4f} {mse:<15.4f}")

    mean_mse = np.mean(mses)
    print(f"\n✅ Mean MSE across all peaks: {mean_mse:.4f}")
    return P_preds, errors, mses
