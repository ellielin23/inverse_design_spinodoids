# config.py

# === model dimensions ===
S_DIM = 4              # structure parameter dimension
P_DIM = 9              # target property dimension
LATENT_DIM = 4         # latent space dimension (can tune later)

# === model architecture ===
ENCODER_HIDDEN_DIMS = [128, 64, 32] # hidden dimensions for encoder
DECODER_HIDDEN_DIMS = [128, 64, 32] # hidden dimensions for decoder

# === training hyperparameters ===
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
BETA = 0.5             # KL divergence weighting
NUM_FLOWS = 20
DROPOUT_PROB = 0.1

# === data ===
DATA_PATH = "data/large_dataset.csv"

# === checkpoint paths ===
TRIAL = 17
CHECKPOINT_DIR_PATH = f'checkpoints/trial_{TRIAL}'
ENCODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/encoder_ckpt_{TRIAL}.pt'
DECODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/decoder_ckpt_{TRIAL}.pt'
CONFIG_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/config_{TRIAL}.txt'
