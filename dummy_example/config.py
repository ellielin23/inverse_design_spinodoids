# config.py

# model dimensions
S_DIM = 1
P_DIM = 1
LATENT_DIM = 3

# training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 200

BETA = 0.1 # weight for KL divergence loss

# file paths
DATA_PATH = 'data/dummy_sine.npz'

TRIAL = 15
CHECKPOINT_DIR_PATH = f'models/checkpoints/iteration_{TRIAL}'
ENCODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/encoder_{TRIAL}.pt'
DECODER_SAVE_PATH = f'{CHECKPOINT_DIR_PATH}/decoder_{TRIAL}.pt'
