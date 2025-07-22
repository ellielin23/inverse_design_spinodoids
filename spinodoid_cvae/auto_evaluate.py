# auto_evaluate.py

import torch
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from models.decoder import Decoder
from utils.load_data import load_dataset, extract_target_properties
from utils.fNN_layers import *
from utils.mathops import dyad
import tensorflow as tf
import importlib
import os

from config_dictionary import config  # this replaces config.py

# === set up ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === load decoder ===
decoder = Decoder(
    config['S_DIM'], config['P_DIM'], config['LATENT_DIM'],
    hidden_dims=config['DECODER_HIDDEN_DIMS']
).to(device)
decoder.load_state_dict(torch.load(config['DECODER_SAVE_PATH'], map_location=device))
decoder.eval()

# === load validation sample ===
P_all, S_all = load_dataset(config['DATA_PATH'])
P_val = P_all[0].unsqueeze(0).to(device)
S_true = S_all[0].cpu().numpy()

# === get S_hats from decoder ===
def get_S_hats(P_val, num_samples):
    P_rep = P_val.repeat(num_samples, 1)
    z_samples = torch.randn((num_samples, config['LATENT_DIM'])).to(device)
    with torch.no_grad():
        S_hats = decoder(z_samples, P_rep)
    return S_hats.cpu().numpy()

# === get peak candidates ===
def get_S_hat_peaks(S_hats, bandwidth):
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(S_hats)
    return ms.cluster_centers_

S_hats = get_S_hats(P_val, num_samples=config['NUM_SAMPLES'])
S_hat_peaks = get_S_hat_peaks(S_hats, bandwidth=config['PEAK_BANDWIDTH'])

# === load Max's forward model ===
custom_objects = {
    'PermutationEquivariantLayer': PermutationEquivariantLayer,
    'DoubleContractionLayer': DoubleContractionLayer,
    'EnforceIsotropyLayer': EnforceIsotropyLayer,
    'NormalizationLayer': NormalizationLayer
}
fNN = tf.keras.models.load_model(config['FNN_PATH'], custom_objects=custom_objects)

P_target = P_val.cpu().numpy().flatten()

P_preds = []
errors = []
mses = []
print(f"{'Peak':<6} {'||P_pred - P_true||':<22} {'MSE (per peak)':<15}")
print("-" * 45)

for i, S_peak in enumerate(S_hat_peaks):
    S_peak_tf = np.expand_dims(S_peak, axis=(0, 1))
    C_pred = fNN(S_peak_tf).numpy().reshape(1, 3, 3, 3, 3)
    P_pred = extract_target_properties(C_pred)[0]
    P_preds.append(P_pred)

    l2 = np.linalg.norm(P_pred - P_target)
    mse = np.mean((P_pred - P_target) ** 2)
    errors.append(l2)
    mses.append(mse)
    print(f"{i:<6} {l2:<22.4f} {mse:<15.4f}")

print(f"\n✅ Mean MSE across all peaks: {np.mean(mses):.4f}")

# === plot results ===
def plot_all_P_preds_vs_true(P_preds, P_true):
    labels = [
        "C1111", "C1122", "C1133", "C2222", "C2233", "C3333",
        "C1212", "C1313", "C2323"
    ]
    num_peaks = len(P_preds)
    x = np.arange(len(labels))
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    axs = axs.flatten()
    width = 0.35

    for i in range(min(num_peaks, len(axs))):
        ax = axs[i]
        ax.bar(x - width / 2, P_true, width, label='True P', color='lightcoral')
        ax.bar(x + width / 2, P_preds[i], width, label='Pred P', color='skyblue')
        ax.set_title(f"Peak {i}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(True)
    axs[0].legend()
    plt.suptitle("P_pred vs P_true for all peaks")
    plt.tight_layout()
    plt.show()

plot_all_P_preds_vs_true(P_preds, P_target)
