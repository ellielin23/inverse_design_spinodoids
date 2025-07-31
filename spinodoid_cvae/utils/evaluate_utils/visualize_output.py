# utils/evaluate_utils/visualize_output.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_S_hat_space(S_hats, S_true, S_hat_peaks):
    """
    Projects sampled structure vectors \hat{S} into 2D using PCA and visualizes:
    - The distribution of sampled \hat{S} values
    - The ground-truth structure S_true
    - The detected high-density peaks from the \hat{S} distribution

    Args:
        S_hats (np.ndarray): Array of sampled structure vectors, shape (N_samples, S_dim)
        S_true (np.ndarray): Ground-truth structure vector, shape (S_dim,)
        S_hat_peaks (np.ndarray): Peak candidates from S_hats, shape (N_peaks, S_dim)
    """
    # === PCA transformation ===
    pca = PCA(n_components=2)
    S_pca = pca.fit_transform(S_hats)
    S_true_pca = pca.transform(S_true.reshape(1, -1))
    S_peaks_pca = pca.transform(S_hat_peaks)

    # === Plot ===
    plt.figure(figsize=(7, 6))

    # Plot sampled \hat{S}
    plt.scatter(S_pca[:, 0], S_pca[:, 1],
                alpha=0.25, color='gray', label=r"Sampled $\hat{\mathcal{S}}$")

    # Plot peaks
    plt.scatter(S_peaks_pca[:, 0], S_peaks_pca[:, 1],
                color='#007acc', s=80, label="Detected Peaks")

    # Plot true S
    plt.scatter(S_true_pca[0, 0], S_true_pca[0, 1],
                color='crimson', marker='x', s=120, linewidths=2, label=r"$\mathcal{S}_{\mathrm{true}}$")

    # === Formatting ===
    plt.title(r"PCA Projection of Sampled $\hat{\mathcal{S}}$ with Detected Peaks", fontsize=14, weight='bold')
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# def plot_all_P_preds_vs_true(P_preds, P_true):
#     """
#     Plots all P_pred vs P_true bar plots in a grid layout with 2 columns per row.

#     Args:
#         P_preds (list of np.ndarray): List of predicted property vectors (each shape (9,))
#         P_true (np.ndarray): Ground-truth property vector (shape (9,))
#     """
#     labels = [
#         "C1111", "C1122", "C1133", "C2222", "C2233", "C3333",
#         "C1212", "C1313", "C2323"
#     ]
#     num_peaks = len(P_preds)
#     cols = 2
#     rows = (num_peaks + cols - 1) // cols  # ceil division
#     width = 0.35
#     x = np.arange(len(labels))

#     fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows))
#     axs = axs.flatten()

#     for i in range(num_peaks):
#         ax = axs[i]
#         ax.bar(x - width / 2, P_true, width, label='True P', color='lightcoral')
#         ax.bar(x + width / 2, P_preds[i], width, label='Predicted P', color='skyblue')
#         ax.set_title(f"Peak {i + 1}")
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, rotation=45)
#         ax.grid(True, axis='y', linestyle='--', alpha=0.6)

#     # Turn off unused subplots
#     for j in range(num_peaks, len(axs)):
#         axs[j].axis('off')

#     # Add legend to the first subplot only
#     axs[0].legend(loc='upper right')
#     fig.suptitle("Elastic Components: Predicted vs True for All Peaks", fontsize=16)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_all_P_preds_vs_true(P_preds, P_true):
    """
    Plots predicted vs true elastic tensors for each candidate,
    in the given order, with professional styling.
    """
    labels = [
        "C₁₁₁₁", "C₁₁₂₂", "C₁₁₃₃", "C₂₂₂₂", "C₂₂₃₃", "C₃₃₃₃",
        "C₁₂₁₂", "C₁₃₁₃", "C₂₃₂₃"
    ]
    num_peaks = len(P_preds)
    cols = 2
    rows = (num_peaks + cols - 1) // cols
    width = 0.35
    x = np.arange(len(labels))

    # === set style ===
    mpl.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
        "figure.dpi": 150,
    })

    fig, axs = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.2 * rows))
    axs = axs.flatten()

    for i in range(num_peaks):
        ax = axs[i]
        ax.bar(x - width / 2, P_true, width, label='True', color='#D9534F')
        ax.bar(x + width / 2, P_preds[i], width, label='Predicted', color='#5BC0DE')
        ax.set_title(f"Candidate {i+1}", pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_ylabel("Elastic Value")
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)

    for j in range(num_peaks, len(axs)):
        axs[j].axis('off')

    # === Add single shared legend in top right ===
    handles, labels_ = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper right', frameon=False, bbox_to_anchor=(0.98, 0.99))

    # === Main title closer to plots ===
    fig.suptitle("Predicted vs True Elastic Components per Ŝ Candidate",
                 fontsize=16, weight='bold', y=0.97)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

import matplotlib.pyplot as plt

def plot_per_component_predictions(P_preds, P_true):
    """
    Plots predicted values for each component (C₁₁₁₁, ..., C₂₃₂₃)
    across all peaks, alongside the ground truth value.
    """
    component_labels = [
        "C₁₁₁₁", "C₁₁₂₂", "C₁₁₃₃",
        "C₂₂₂₂", "C₂₂₃₃", "C₃₃₃₃",
        "C₁₂₁₂", "C₁₃₁₃", "C₂₃₂₃"
    ]

    P_preds = np.array(P_preds)  # shape: (num_peaks, 9)
    P_true = np.array(P_true).flatten()  # shape: (9,)
    num_components = P_preds.shape[1]

    fig, axs = plt.subplots(3, 3, figsize=(14, 10))
    axs = axs.flatten()

    for i in range(num_components):
        ax = axs[i]
        ax.plot(P_preds[:, i], 'o-', label='Predicted', color='#5BC0DE')
        ax.axhline(P_true[i], color='red', linestyle='--', label='True')
        ax.set_title(component_labels[i])
        ax.set_xlabel("Candidate Peak Index")
        ax.set_ylabel("Elastic Value")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend()

    fig.suptitle("Predicted vs True Values for Each Elastic Component", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()

def evaluate_peaks(S_hat_peaks, P_target, fNN, extract_target_properties, P_mean, P_std):
    """
    Evaluates each peak Ŝ using Max's fNN and prints per-peak error metrics in unnormalized space.
    Returns:
        - P_preds: list of predicted P vectors (unnormalized)
        - errors: list of L2 errors (unnormalized)
        - mses: list of MSEs (unnormalized)
    """
    P_preds = []
    errors = []
    mses = []

    # unnormalize ground truth
    P_target_unnorm = P_target * P_std + P_mean

    print(f"{'Peak':<6} {'||P_pred - P_true||':<22} {'MSE (per peak)':<15}")
    print("-" * 45)

    for i, S_peak in enumerate(S_hat_peaks):
        S_peak_tf = np.expand_dims(S_peak, axis=(0, 1))  # shape: (1, 1, 4)
        C_pred = fNN(S_peak_tf).numpy().reshape(1, 3, 3, 3, 3)
        P_pred_norm = extract_target_properties(C_pred)[0]

        # unnormalize prediction
        P_pred = P_pred_norm * P_std + P_mean
        P_preds.append(P_pred)

        l2_error = np.linalg.norm(P_pred - P_target_unnorm)
        mse = np.mean((P_pred - P_target_unnorm) ** 2)
        errors.append(l2_error)
        mses.append(mse)

        print(f"{i:<6} {l2_error:<22.4f} {mse:<15.4f}")

    mean_mse = np.mean(mses)
    print(f"\n✅ Mean MSE across all peaks (unnormalized): {mean_mse:.4f}")
    return P_preds, errors, mses


