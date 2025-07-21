# config.py

# === model dimensions ===
S_DIM = 4              # structure parameter dimension
P_DIM = 9              # target property dimension
LATENT_DIM = 3         # latent space dimension (can tune later)

# === model architecture ===
ENCODER_HIDDEN_DIMS = [128, 64, 32] # hidden dimensions for encoder
DECODER_HIDDEN_DIMS = [128, 64, 32] # hidden dimensions for decoder

# === training hyperparameters ===
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
BETA = 1.0             # KL divergence weighting

# === data ===
DATA_PATH = "data/large_dataset.csv"

# === checkpoint paths ===
TRIAL = 3
CHECKPOINT_DIR_PATH = f'models/checkpoints/iteration_{TRIAL}'
ENCODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/encoder_ckpt_{TRIAL}.pt'
DECODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/decoder_ckpt_{TRIAL}.pt'
