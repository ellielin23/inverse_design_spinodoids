# config_parallel.py

# === set theta and trial ===
THETA_PATTERN = "011"   # change to "100", "010", "001", etc. as needed
TRIAL = 1

# === set attention and flow ===
USE_ATTENTION_ENCODER = False
USE_ATTENTION_DECODER = False
USE_FLOW_DECODER = False
FLOW_TYPE = "realnvp"

# === model dimensions ===
S_DIM = 4              # structure parameter dimension
P_DIM = 9              # target property dimension
LATENT_DIM = 4        # latent space dimension (can tune later)

# === model architecture ===
ENCODER_HIDDEN_DIMS = [128, 64, 32] # hidden dimensions for encoder
DECODER_HIDDEN_DIMS = [128, 64, 32] # hidden dimensions for decoder
NUM_FLOWS = 6

# === training hyperparameters ===
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
NUM_EPOCHS = 100
BETA = 0.1           # KL divergence weighting
DROPOUT_PROB = 0.5

# === data ===
DATA_PATH = f"data/partition_by_theta/theta_{THETA_PATTERN}.csv"

# === checkpoint paths ===
CHECKPOINT_DIR_PATH = f'parallel_checkpoints/theta_{THETA_PATTERN}/{THETA_PATTERN}_trial_{TRIAL}'
ENCODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/encoder_ckpt_{TRIAL}.pt'
DECODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/decoder_ckpt_{TRIAL}.pt'
CONFIG_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/{THETA_PATTERN}_config_{TRIAL}.txt'

