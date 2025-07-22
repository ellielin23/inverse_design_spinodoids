# auto_evaluate.py

import os
import torch
import numpy as np
from sklearn.cluster import MeanShift
from utils.load_data import load_dataset, extract_target_properties
from utils.mathops import dyad
from utils.fNN_layers import *
import tensorflow as tf
from functools import partial
from models.decoder import Decoder
from config import DATA_PATH
from utils.evaluate_utils import (
    get_S_hats, get_S_hat_peaks, compute_P_pred_stats,
    plot_all_P_preds_vs_true, write_stats_to_file
)

def main():
    # === load pretrained fNN ===
    custom_objects = {
        'PermutationEquivariantLayer': PermutationEquivariantLayer,
        'DoubleContractionLayer': DoubleContractionLayer,
        'EnforceIsotropyLayer': EnforceIsotropyLayer,
        'NormalizationLayer': NormalizationLayer
    }
    fNN = tf.keras.models.load_model('utils/max_fNN.h5', custom_objects=custom_objects)
    print("✅ Loaded Max's forward model")

    # === load dataset ===
    P_all, S_all = load_dataset(DATA_PATH)
    P_val = P_all[0].unsqueeze(0)  # shape (1, 9)
    S_true = S_all[0].numpy()      # shape (4,)
    P_target = P_val.numpy().flatten()

    # === loop over all trial directories ===
    base_dir = "auto_checkpoints"
    for folder in sorted(os.listdir(base_dir)):
        trial_path = os.path.join(base_dir, folder)
        if not os.path.isdir(trial_path):
            continue

        trial_num = folder.split('_')[-1]
        decoder_path = os.path.join(trial_path, f"decoder_{trial_num}.pt")

        if not os.path.exists(decoder_path):
            print(f"❌ Decoder not found for {folder}")
            continue

        # load config
        config_path = os.path.join(trial_path, f"config_{trial_num}.txt")
        with open(config_path, "r") as f:
            config_lines = f.readlines()
        config_dict = {line.split(":")[0].strip(): eval(line.split(":")[1].strip()) for line in config_lines}

        # rebuild decoder
        latent_dim = config_dict['latent_dim']
        decoder_dims = config_dict['decoder_dims']
        decoder = Decoder(4, 9, latent_dim, decoder_dims)
        decoder.load_state_dict(torch.load(decoder_path))
        decoder.eval()

        # === generate S_hat samples ===
        S_hats = get_S_hats(decoder, P_val, latent_dim, num_samples=1000)
        S_hat_peaks = get_S_hat_peaks(S_hats, bandwidth=20.0)

        # === evaluate each peak ===
        P_preds, errors, mses = compute_P_pred_stats(S_hat_peaks, P_target, fNN)

        # === plot ===
        plot_all_P_preds_vs_true(P_preds, P_target, save_path=os.path.join(trial_path, "P_comparison.png"))

        # === save stats ===
        write_stats_to_file(trial_path, trial_num, P_preds, P_target, errors, mses)

    print("\n✅ All evaluations completed.")

if __name__ == "__main__":
    main()
