# utils/evaluate_utils/load_paths.py

import os

def load_paths(TRIAL, THETA_MODEL=False, THETA_PATTERN=None, USE_FLOW=False, FLOW_TYPE="realnvp"):
    """
    Dynamically construct paths for config, checkpoints, means/stds, and data
    based on model type (theta-model, flow-based, or vanilla CVAE).

    Returns:
        dict: {
            trial_dir, DATA_PATH, config_path,
            encoder_path, decoder_path,
            P_mean_path, P_std_path,
            S_mean_path, S_std_path
        }
    """
    paths = {}

    if THETA_MODEL:
        assert THETA_PATTERN is not None, "THETA_PATTERN must be provided if THETA_MODEL is True"
        paths["trial_dir"]     = f"parallel_checkpoints/theta_{THETA_PATTERN}/{THETA_PATTERN}_trial_{TRIAL}"
        paths["DATA_PATH"]     = f"data/partition_by_theta/theta_{THETA_PATTERN}.csv"
        paths["config_path"]   = os.path.join(paths["trial_dir"], f"{THETA_PATTERN}_config_{TRIAL}.txt")
        paths["encoder_path"]  = os.path.join(paths["trial_dir"], f"encoder_ckpt_{TRIAL}.pt")
        paths["decoder_path"]  = os.path.join(paths["trial_dir"], f"decoder_ckpt_{TRIAL}.pt")
        paths["P_mean_path"]   = f"data/partition_by_theta/P_mean_theta_{THETA_PATTERN}.npy"
        paths["P_std_path"]    = f"data/partition_by_theta/P_std_theta_{THETA_PATTERN}.npy"
        paths["S_mean_path"]   = f"data/partition_by_theta/S_mean_theta_{THETA_PATTERN}.npy"
        paths["S_std_path"]    = f"data/partition_by_theta/S_std_theta_{THETA_PATTERN}.npy"
        print(f"✅ Loaded files from theta_{THETA_PATTERN} for trial {TRIAL}")

    elif not USE_FLOW:
        paths["trial_dir"]     = f"checkpoints/trial_{TRIAL}"
        paths["DATA_PATH"]     = "data/train/large_dataset.csv"
        paths["config_path"]   = os.path.join(paths["trial_dir"], f"config_{TRIAL}.txt")
        paths["encoder_path"]  = os.path.join(paths["trial_dir"], f"encoder_ckpt_{TRIAL}.pt")
        paths["decoder_path"]  = os.path.join(paths["trial_dir"], f"decoder_ckpt_{TRIAL}.pt")
        paths["P_mean_path"]   = "data/P_mean.npy"
        paths["P_std_path"]    = "data/P_std.npy"
        paths["S_mean_path"]   = "data/S_mean.npy"
        paths["S_std_path"]    = "data/S_std.npy"
        print(f"✅ Loaded files from trial {TRIAL} for no flow")

    elif USE_FLOW:
        paths["trial_dir"]     = f"flow_checkpoints/{FLOW_TYPE}/{FLOW_TYPE}_trial_{TRIAL}"
        paths["DATA_PATH"]     = "data/train/large_dataset.csv"
        paths["config_path"]   = os.path.join(paths["trial_dir"], f"config_{TRIAL}.txt")
        paths["encoder_path"]  = os.path.join(paths["trial_dir"], f"encoder_ckpt_{TRIAL}.pt")
        paths["decoder_path"]  = os.path.join(paths["trial_dir"], f"{FLOW_TYPE}_decoder_ckpt_{TRIAL}.pt")
        paths["P_mean_path"]   = "data/P_mean.npy"
        paths["P_std_path"]    = "data/P_std.npy"
        paths["S_mean_path"]   = "data/S_mean.npy"
        paths["S_std_path"]    = "data/S_std.npy"
        print(f"✅ Loaded files from trial {TRIAL} for {FLOW_TYPE} flow")

    return paths
