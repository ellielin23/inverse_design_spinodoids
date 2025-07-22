# auto_evaluate.py

import os
import torch
import numpy as np
from utils.load_data import load_dataset, extract_target_properties
from utils.mathops import dyad
from utils.fNN_layers import *
import tensorflow as tf
from functools import partial
from models.decoder import Decoder
from config import DATA_PATH
from utils.evaluate_utils import (
    get_S_hats, get_S_hat_peaks, compute_P_pred_stats,
    plot_all_P_preds_vs_true, write_stats_to_file,
    find_optimal_bandwidth
)
import csv

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

    # === summary csv setup ===
    summary_path = "auto_checkpoints/auto_results_summary.csv"
    write_header = not os.path.exists(summary_path)

    with open(summary_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                'Trial', 'LatentDim', 'EncoderDims', 'DecoderDims',
                'BatchSize', 'LR', 'Beta', 'Epochs', 'Bandwidth', 'MeanMSE', 'MeanL2Error'
            ])

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

            # === find optimal bandwidth ===
            bandwidth = find_optimal_bandwidth(S_hats, max_peaks=10)
            S_hat_peaks = get_S_hat_peaks(S_hats, bandwidth=bandwidth)

            # === evaluate each peak ===
            P_preds, errors, mses = compute_P_pred_stats(S_hat_peaks, fNN, P_target)

            # === plot ===
            plot_all_P_preds_vs_true(P_preds, P_target, save_path=os.path.join(trial_path, "P_comparison.png"))

            # === save stats ===
            write_stats_to_file(os.path.join(trial_path, "stats.txt"), errors, mses)

            # === append summary CSV ===
            writer.writerow([
                trial_num,
                latent_dim,
                config_dict['encoder_dims'],
                decoder_dims,
                config_dict['batch_size'],
                config_dict['learning_rate'],
                config_dict['beta'],
                config_dict['epochs'],
                bandwidth,
                np.mean(mses),
                np.mean(errors)
            ])

    print("\n✅ All evaluations completed.")

if __name__ == "__main__":
    main()

