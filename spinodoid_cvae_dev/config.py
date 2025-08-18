# config.py

# === set theta and trial ===
TAG = "001"   # change to "100", "010", "001", etc. as needed
TRIAL = 19

# === set attention and flow ===
USE_ATTENTION_ENCODER = True
USE_ATTENTION_DECODER = True
USE_FLOW_DECODER = False
FLOW_TYPE = "realnvp"

# === model dimensions ===
S_DIM = 4
P_DIM = 9
LATENT_DIM = 4

# === model architecture ===
ENCODER_HIDDEN_DIMS = [128, 64, 32]
DECODER_HIDDEN_DIMS = [128, 64, 32]
NUM_FLOWS = 4

# === training hyperparameters ===
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
NUM_EPOCHS = 150
BETA = 0.8
DROPOUT_PROB = 0.2

# === data ===
DATA_PATH = f"data/partition_by_theta/theta_{TAG}.csv"

# === checkpoint paths ===
CHECKPOINT_DIR_PATH = f'checkpoints/theta_{TAG}/{TAG}_trial_{TRIAL}'
ENCODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/encoder_ckpt_{TRIAL}.pt'
DECODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/decoder_ckpt_{TRIAL}.pt'
CONFIG_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/{TAG}_config_{TRIAL}.txt'