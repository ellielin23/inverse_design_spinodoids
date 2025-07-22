# evaluate_utils.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from utils.load_data import extract_target_properties

def get_S_hats(decoder, P_val, latent_dim, num_samples=1000):
    P_tensor = P_val.repeat(num_samples, 1).to(P_val.device)
    z_samples = torch.randn((num_samples, latent_dim)).to(P_val.device)
    with torch.no_grad():
        S_hats = decoder(z_samples, P_tensor)
    return S_hats.cpu().numpy()

def get_S_hat_peaks(S_hats, bandwidth=20.0):
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(S_hats)
    return ms.cluster_centers_

def find_optimal_bandwidth(S_hats, max_peaks=10, min_bw=1.0, max_bw=50.0, step=1.0):
    for bw in np.arange(min_bw, max_bw + step, step):
        ms = MeanShift(bandwidth=bw, bin_seeding=True)
        ms.fit(S_hats)
        if len(ms.cluster_centers_) <= max_peaks:
            return bw
    return max_bw

def compute_P_pred_stats(S_hat_peaks, fNN, P_target):
    P_preds = []
    l2_errors = []
    mses = []

    for S_peak in S_hat_peaks:
        S_peak_tf = np.expand_dims(S_peak, axis=(0, 1))  # (1, 1, 4)
        C_pred = fNN(S_peak_tf).numpy().reshape(1, 3, 3, 3, 3)
        P_pred = extract_target_properties(C_pred)[0]

        l2 = np.linalg.norm(P_pred - P_target)
        mse = np.mean((P_pred - P_target) ** 2)

        P_preds.append(P_pred)
        l2_errors.append(l2)
        mses.append(mse)

    return P_preds, l2_errors, mses

def plot_all_P_preds_vs_true(P_preds, P_true, save_path):
    labels = [
        "C1111", "C1122", "C1133", "C2222", "C2233", "C3333",
        "C1212", "C1313", "C2323"
    ]
    num_peaks = len(P_preds)
    rows, cols = 2, 3
    width = 0.35
    x = np.arange(len(labels))

    fig, axs = plt.subplots(rows, cols, figsize=(18, 8))
    axs = axs.flatten()

    for i in range(num_peaks):
        ax = axs[i]
        ax.bar(x - width/2, P_true, width, label='True P', color='lightcoral')
        ax.bar(x + width/2, P_preds[i], width, label='Predicted P', color='skyblue')
        ax.set_title(f"Peak {i + 1}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    axs[0].legend(loc='upper right')
    fig.suptitle("Elastic Components: Predicted vs True for All Peaks", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

def write_stats_to_file(path, l2_errors, mses):
    with open(path, 'w') as f:
        f.write(f"{'Peak':<6} {'||P_pred - P_true||':<22} {'MSE (per peak)':<15}\n")
        f.write("-" * 45 + "\n")
        for i, (l2, mse) in enumerate(zip(l2_errors, mses)):
            f.write(f"{i:<6} {l2:<22.4f} {mse:<15.4f}\n")
        f.write(f"\n✅ Mean MSE across all peaks: {np.mean(mses):.4f}\n")
