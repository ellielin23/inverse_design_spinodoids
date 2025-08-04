# utils/evaluate_utils/visualize_output.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib as mpl
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

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

    # === plot ===
    plt.figure(figsize=(7, 6))

    # plot sampled S-hat
    plt.scatter(S_pca[:, 0], S_pca[:, 1],
                alpha=0.25, color='gray', label=r"Sampled $\hat{\mathcal{S}}$")

    # plot peaks
    plt.scatter(S_peaks_pca[:, 0], S_peaks_pca[:, 1],
                color='#007acc', s=80, label="Detected Peaks")

    # plot true S
    plt.scatter(S_true_pca[0, 0], S_true_pca[0, 1],
                color='crimson', marker='x', s=120, linewidths=2, label=r"$\mathcal{S}_{\mathrm{true}}$")

    # === formatting ===
    plt.title(r"PCA Projection of Sampled $\hat{\mathcal{S}}$ with Detected Peaks", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.show()


def compare_top_k_S_peaks(S_hat_peaks, S_true, k=5):
    """
    Compares the top-k Ŝ peaks to the true structure S_true.

    Args:
        S_hat_peaks (np.ndarray): Array of Ŝ peak vectors, shape (n_peaks, S_dim)
        S_true (np.ndarray): Ground truth S vector, shape (S_dim,)
        k (int): Number of top peaks to compare (default = 5)

    Returns:
        pd.DataFrame: Table with columns ['Peak', 'Ŝ', 'S_true', 'ΔS']
    """
    k = min(k, len(S_hat_peaks))
    rows = []

    S_true_rounded = np.round(S_true, 5)

    for i in range(k):
        S_hat = np.round(S_hat_peaks[i], 5)
        S_diff = np.round(S_hat_peaks[i] - S_true, 5)

        rows.append({
            "Peak": f"{i+1}",
            "Ŝ": S_hat,
            "S_true": S_true_rounded,
            "ΔS": S_diff
        })

    df = pd.DataFrame(rows)
    from IPython.display import HTML, display
    display(HTML(df.to_html(index=False)))
    return None


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
    cols = 3
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


def plot_all_P_preds_vs_true_with_fNN_uncertainty(P_preds, P_target_unnorm, S_true, fNN, P_std, P_mean):
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from utils.data_utils.load_data import extract_target_properties

    # === compute fNN(S_true) to serve as comparison ===
    S_true_tensor = tf.convert_to_tensor(S_true.reshape(1, 1, -1), dtype=tf.float32)
    C_pred = fNN(S_true_tensor).numpy().reshape(1, 3, 3, 3, 3)  # (1, 3, 3, 3, 3)
    P_hat_from_S_true_norm = extract_target_properties(C_pred)[0]  # (9,)
    P_hat_from_S_true = P_hat_from_S_true_norm * P_std + P_mean  # unnormalize

    num_candidates = len(P_preds)
    num_components = len(P_target_unnorm)
    cols = 3
    rows = (num_candidates + 1) // cols

    component_labels = [r"$C_{111}$", r"$C_{112}$", r"$C_{113}$", r"$C_{222}$",
                        r"$C_{223}$", r"$C_{333}$", r"$C_{211}$", r"$C_{313}$", r"$C_{323}$"]

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    fig.suptitle("Predicted vs True Elastic Components per S Candidate\nwith fNN(S_true) Comparison", fontsize=16)

    for i in range(num_candidates):
        row, col = divmod(i, cols)
        ax = axes[row, col] if rows > 1 else axes[col]

        pred = P_preds[i]
        true = P_target_unnorm
        fNN_ref = P_hat_from_S_true

        x = np.arange(num_components)
        width = 0.25

        ax.bar(x - width, true, width, label="True", color="red")
        ax.bar(x, fNN_ref, width, label="Pred (fNN S_true)", color="lightgreen")
        ax.bar(x + width, pred, width, label="Pred (CVAE Ŝ)", color="royalblue")

        ax.set_title(f"Candidate {i + 1}")
        ax.set_xticks(x)
        ax.set_xticklabels(component_labels, rotation=45)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.3)

        if i == 0:
            ax.legend()

    # Hide unused subplots
    for j in range(num_candidates, rows * cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_per_component_bars(P_preds, P_true_norm, P_mean, P_std):
    """
    For each component of P, plot a grouped bar chart:
    - First bar: true value (unnormalized)
    - Following bars: predicted values from each Ŝ peak (already unnormalized)
    """
    component_labels = [
        "C₁₁₁₁", "C₁₁₂₂", "C₁₁₃₃",
        "C₂₂₂₂", "C₂₂₃₃", "C₃₃₃₃",
        "C₁₂₁₂", "C₁₃₁₃", "C₂₃₂₃"
    ]

    P_preds = np.array(P_preds)  # (num_peaks, 9)
    P_true = P_true_norm * P_std + P_mean  # <<< UNNORMALIZE TRUE VECTOR

    num_preds = P_preds.shape[0]
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = axs.flatten()

    colors = cm.get_cmap('tab10', num_preds)  # distinct colors for preds

    for i in range(9):
        ax = axs[i]
        values = [P_true[i]] + list(P_preds[:, i])
        bar_positions = np.arange(len(values))
        bar_width = 0.8

        # true bar
        ax.bar(bar_positions[0], values[0], color='red', label='True')

        # predicted bars
        for j in range(num_preds):
            ax.bar(bar_positions[j + 1], values[j + 1], color=colors(j), label=f'P {j+1}' if i == 0 else None)

        ax.set_title(component_labels[i], fontsize=12, pad=5)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(['True'] + [f'P̂{j+1}' for j in range(num_preds)], rotation=45)
        ax.set_ylabel("Value")
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # legend outside the plot, only once
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 0.96), frameon=False)

    fig.suptitle("Elastic Components: True vs Predicted (Grouped by Component)", fontsize=16, weight='bold', y=1.03)
    plt.tight_layout()
    plt.show()



def evaluate_peaks(S_hat_peaks, P_target, fNN, P_mean, P_std):
    """
    Evaluates each peak Ŝ using Max's fNN and returns a clean pandas DataFrame.
    Returns:
        - P_preds: list of predicted P vectors (unnormalized)
        - errors: list of L2 errors (unnormalized)
        - mses: list of MSEs (unnormalized)
        - df: pd.DataFrame with all of the above
    """
    from utils.data_utils.load_data import extract_target_properties
    P_preds = []
    errors = []
    mses = []

    # unnormalize ground truth
    P_target_unnorm = P_target * P_std + P_mean

    rows = []

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

        rows.append({
            "Peak": i,
            "||P-hat - P||": round(l2_error, 4),
            "MSE": round(mse, 5)
        })

    overall_avg = round(np.mean(mses), 5)
    print(f"\n✅ Mean MSE across all peaks (unnormalized): {overall_avg:.5f}\n")

    df = pd.DataFrame(rows)
    from IPython.display import HTML, display
    display(HTML(df.to_html(index=False)))
    return P_preds, errors, mses, df


def evaluate_avg_mse_per_target_pair(decoder, P_all, S_all, latent_dim, fNN, N=20, seed=42, device=None):
    from utils.evaluate_utils.sampling import get_S_hats, get_S_hat_peaks
    from utils.data_utils.load_data import extract_target_properties

    P_mean = np.load("data/P_mean.npy")
    P_std = np.load("data/P_std.npy")

    rows = []

    for i in range(N):
        P_target = P_all[i].unsqueeze(0).to(device)
        P_target_np = P_target.cpu().numpy().flatten()
        P_target_unnorm = P_target_np * P_std + P_mean

        S_hats = get_S_hats(decoder, P_target, latent_dim, num_samples=1000, seed=seed+i, device=device)
        S_hat_peaks = get_S_hat_peaks(S_hats, bandwidth=4.0)

        mses = []
        for S_peak in S_hat_peaks:
            S_peak_tf = np.expand_dims(S_peak, axis=(0, 1))  # shape: (1, 1, 4)
            C_pred = fNN(S_peak_tf).numpy().reshape(1, 3, 3, 3, 3)
            P_pred_norm = extract_target_properties(C_pred)[0]
            P_pred = P_pred_norm * P_std + P_mean
            mse = np.mean((P_pred - P_target_unnorm) ** 2)
            mses.append(mse)

        avg_mse = np.mean(mses)
        rows.append({"(Sᵢ, Pᵢ) index": f"{i}", "Avg MSE": round(avg_mse, 5)})

    overall_avg = round(np.mean([row["Avg MSE"] for row in rows]), 5)
    print(f"\n✅ Overall average MSE across all (Sᵢ, Pᵢ) pairs: {overall_avg:.5f}")
    
    from IPython.display import HTML, display
    df = pd.DataFrame(rows)
    display(HTML(df.to_html(index=False)))

    return df


def evaluate_peaks_optimized(S_hat_peaks, P_target, fNN, P_mean, P_std):
    """
    Optimized version of evaluate_peaks with vectorized operations and reduced overhead.
    Returns:
        - P_preds: ndarray of predicted P vectors (unnormalized) 
        - errors: ndarray of L2 errors (unnormalized)
        - mses: ndarray of MSEs (unnormalized) 
        - df: pd.DataFrame with all of the above
    """
    from utils.data_utils.load_data import extract_target_properties
    
    # Move imports to top to avoid repeated imports
    # Vectorize preprocessing
    S_peaks_batch = np.expand_dims(S_hat_peaks, axis=1)  # shape: (n_peaks, 1, 4)
    
    # Batch forward pass through neural network
    C_preds = fNN(S_peaks_batch).numpy().reshape(-1, 3, 3, 3, 3)  # (n_peaks, 3, 3, 3, 3)
    
    # Vectorized property extraction
    P_preds_norm = np.array([extract_target_properties(C_pred.reshape(1, 3, 3, 3, 3))[0] 
                            for C_pred in C_preds])  # (n_peaks, 9)
    
    # Vectorized unnormalization
    P_target_unnorm = P_target * P_std + P_mean
    P_preds = P_preds_norm * P_std + P_mean  # (n_peaks, 9)
    
    # Vectorized error calculations
    diffs = P_preds - P_target_unnorm  # (n_peaks, 9)
    errors = np.linalg.norm(diffs, axis=1)  # (n_peaks,)
    mses = np.mean(diffs**2, axis=1)  # (n_peaks,)
    
    # Create DataFrame efficiently
    df_data = {
        "Peak": np.arange(len(S_hat_peaks)),
        "||P-hat - P||": np.round(errors, 4),
        "MSE": np.round(mses, 5)
    }
    df = pd.DataFrame(df_data)
    
    # Display results
    overall_avg = np.round(np.mean(mses), 5)
    print(f"\n✅ Mean MSE across all peaks (unnormalized): {overall_avg:.5f}\n")
    
    from IPython.display import HTML, display
    display(HTML(df.to_html(index=False)))
    
    return P_preds, errors, mses, df
