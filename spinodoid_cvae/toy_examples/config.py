# toy_examples/predict_C111/config.py

# === data path ===
DATA_PATH = "data/large_dataset.csv"

# === model save paths ===
TRIAL = 1
CHECKPOINT_DIR_PATH = f"toy_examples/predict_C111/checkpoints/C111_trial_{TRIAL}"
ENCODER_SAVE_PATH = f"{CHECKPOINT_DIR_PATH}/encoder_ckpt_{TRIAL}.pt"
DECODER_SAVE_PATH = f"{CHECKPOINT_DIR_PATH}/decoder_ckpt_{TRIAL}.pt"
CONFIG_SAVE_PATH = f"{CHECKPOINT_DIR_PATH}/config_{TRIAL}.txt"

# === dimensions ===
S_DIM = 4        # Structure vector: 3 spatial + 1 volume ratio
P_DIM = 1        # Only predicting the C111 component

# === architecture ===
ENCODER_HIDDEN_DIMS = [64, 32]
DECODER_HIDDEN_DIMS = [32, 64]

# === training ===
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
BETA = 1.0        # Weight for KL divergence
