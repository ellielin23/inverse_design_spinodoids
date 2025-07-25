# config.py

# === flow type ===
USE_FLOW_DECODER = True
FLOW_TYPE = "realnvp"
TRIAL = 4

# === model dimensions ===
S_DIM = 4              # structure parameter dimension
P_DIM = 9              # target property dimension
LATENT_DIM = 4         # latent space dimension (can tune later)

# === model architecture ===
ENCODER_HIDDEN_DIMS = [128, 64, 32] # hidden dimensions for encoder
DECODER_HIDDEN_DIMS = [128, 64, 32] # hidden dimensions for decoder
NUM_FLOWS = 6

# === training hyperparameters ===
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 80
BETA = 0.01             # KL divergence weighting
DROPOUT_PROB = 0.2

# === data ===
DATA_PATH = "data/large_dataset.csv"

# === checkpoint paths ===
if USE_FLOW_DECODER:
    CHECKPOINT_DIR_PATH = f'flow_checkpoints/{FLOW_TYPE}/{FLOW_TYPE}_trial_{TRIAL}'
    ENCODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/encoder_ckpt_{TRIAL}.pt'
    DECODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/{FLOW_TYPE}_decoder_ckpt_{TRIAL}.pt'
    CONFIG_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/config_{TRIAL}.txt'
else:
    CHECKPOINT_DIR_PATH = f'checkpoints/trial_{TRIAL}'
    ENCODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/encoder_ckpt_{TRIAL}.pt'
    DECODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/decoder_ckpt_{TRIAL}.pt'
    CONFIG_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/config_{TRIAL}.txt'
