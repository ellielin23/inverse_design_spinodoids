# toy_examples/predict_C111/config.py

# === data path ===
DATA_PATH = "data/large_dataset.csv"

# === model save paths ===
COMPONENT_NAME = "C111"
TRIAL = 10
CHECKPOINT_DIR_PATH = f"checkpoints/{COMPONENT_NAME}_trial_{TRIAL}"
ENCODER_SAVE_PATH = f"{CHECKPOINT_DIR_PATH}/encoder_ckpt_{TRIAL}.pt"
DECODER_SAVE_PATH = f"{CHECKPOINT_DIR_PATH}/decoder_ckpt_{TRIAL}.pt"
CONFIG_SAVE_PATH = f"{CHECKPOINT_DIR_PATH}/config_{TRIAL}.txt"

# === dimensions ===
S_DIM = 4
P_DIM = 1        # only predicting the C111 component
LATENT_DIM = 4

# === architecture ===
ENCODER_HIDDEN_DIMS = [128, 64, 32]
DECODER_HIDDEN_DIMS = [128, 64, 32]

# === training ===
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
BETA = 1.0
