# config.py

# === model selection ===
MODEL = "flow"       # options: "flow", "gaussian"
FLOW_TYPE = "planar" # options: "planar", "maf", "realnvp"
TRIAL = 5

# === model dimensions ===
S_DIM = 4  # structure parameter dimension
P_DIM = 9  # target property dimension
HIDDEN_DIMS = [128, 64, 32] # hidden dimensions of the neural network
NUM_FLOWS = 3 # number of flow layers

# === training hyperparameters ===
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
NUM_EPOCHS = 500
BETA_VAR_REG = 1e-2 # regularization parameter for variance

# === data ===
DATA_PATH = "data/dataset_train_x1000.csv"
DISTRIBUTIONAL_DATA_PATH = "data/dataset_distributional.csv"

if MODEL == "flow":
    SAVE_DIR = f'checkpoints/{MODEL}/{FLOW_TYPE}/{FLOW_TYPE}_trial_{TRIAL}'
    SAVE_MODEL_PATH = f'{SAVE_DIR}/{FLOW_TYPE}_ckpt_{TRIAL}.pt'
    SAVE_CONFIG_PATH = f'{SAVE_DIR}/config_{TRIAL}.txt'
elif MODEL == "gaussian":
    SAVE_DIR = f'checkpoints/{MODEL}/{MODEL}_trial_{TRIAL}'
    SAVE_MODEL_PATH = f'{SAVE_DIR}/{MODEL}_ckpt_{TRIAL}.pt'
    SAVE_CONFIG_PATH = f'{SAVE_DIR}/config_{TRIAL}.txt'