# config.py

# === model selection ===
MODEL = "gaussian"       # options: "flow", "gaussian"
FLOW_TYPE = "planar" # options: "planar", "maf", "realnvp"
TRIAL = 1

# === model dimensions ===
S_DIM = 4
P_DIM = 9
HIDDEN_DIMS = [128, 64, 32]
NUM_FLOWS = 3

# === training hyperparameters ===
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
NUM_EPOCHS = 400
BETA_VAR_REG = 0.3
SIGMA_MIN = 1e-3            # floor for std in normalized P space
CLIP_GRAD_NORM = 1.0        # optional grad clipping

# === data ===
DATA_PATH = "data/dataset_train_x1000.csv"
DISTRIBUTIONAL_DATA_PATH = "data/dataset_distributional.csv"

# === save paths ===
if MODEL == "flow":
    SAVE_DIR = f"checkpoints/flow/{FLOW_TYPE}/{FLOW_TYPE}_trial_{TRIAL}"
    SAVE_MODEL_PATH = f"{SAVE_DIR}/{FLOW_TYPE}_ckpt_{TRIAL}.pt"
    SAVE_CONFIG_PATH = f"{SAVE_DIR}/config_{TRIAL}.txt"
elif MODEL == "gaussian":
    SAVE_DIR = f"checkpoints/gaussian/gaussian_trial_{TRIAL}"
    SAVE_MODEL_PATH = f"{SAVE_DIR}/gaussian_ckpt_{TRIAL}.pt"
    SAVE_CONFIG_PATH = f"{SAVE_DIR}/config_{TRIAL}.txt"
else:
    raise ValueError(f"Unknown MODEL: {MODEL}")