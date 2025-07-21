# config.py

# === model dimensions ===
S_DIM = 4                         # structure parameter dimension
P_DIM = 9                         # target property dimension
HIDDEN_DIMS = [128, 64, 32]       # hidden dimensions of the neural network
NUM_FLOWS = 3                     # number of planar flows

# === training hyperparameters ===
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

# === data ===
DATA_PATH = "data/dataset_train_x1000.csv"
DISTRIBUTIONAL_DATA_PATH = "data/dataset_distributional.csv"

# === gaussian training ===
TRIAL_GAUSSIAN = 10
GAUSSIAN_SAVE_PATH = f'checkpoints/gaussian/gaussian_ckpt_{TRIAL_GAUSSIAN}.pt'

# === flow training ===
TRIAL_FLOW = 2
FLOW_SAVE_PATH = f'checkpoints/flow/flow_ckpt_{TRIAL_FLOW}.pt'