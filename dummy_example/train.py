# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.encoder import Encoder
from models.decoder import Decoder
from utils import reparameterize
from loss import vae_loss
from config import *

import numpy as np

# ===== set device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== load in dataset =====
data = np.load(DATA_PATH)
S = torch.tensor(data['S'], dtype=torch.float32) # shape [N, 1]
P = torch.tensor(data['P'], dtype=torch.float32) # shape [N, 1]

dataset = TensorDataset(S, P) # combine S and P into a single dataset
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) # batches the data and shuffles it in each epoch


# ===== initialize models =====
encoder = Encoder(S_dim=S_DIM, P_dim=P_DIM, latent_dim=LATENT_DIM)
decoder = Decoder(S_dim=S_DIM, P_dim=P_DIM, latent_dim=LATENT_DIM)

# ===== define optimizer =====
params = list(encoder.parameters()) + list(decoder.parameters()) # combine the parameters of both encoder and decoder
optimizer = optim.Adam(params, lr=LEARNING_RATE) # use Adam optimizer to update weights during training

# ===== training loop =====
for epoch in range(EPOCHS):
    encoder.train() # set encoder to training mode
    decoder.train() # set decoder to training mode
    total_loss = 0

    for batch_S, batch_P in dataloader:
        batch_S = batch_S.to(device) # move batch_S to device
        batch_P = batch_P.to(device) # move batch_P to device
        
        optimizer.zero_grad() # reset gradients to zero

        # forward pass through encoder: (S, P) → μ, logvar
        mu, logvar = encoder(batch_S, batch_P)

        # reparameterization trick to sample latent z
        z = reparameterize(mu, logvar)

        # decode: (z, P) → S_hat
        S_hat = decoder(z, batch_P)

        # compute total loss
        loss = vae_loss(S_hat, batch_S, mu, logvar, None, None)
        loss.backward() # backpropagate to compute gradients
        optimizer.step() # update model weights
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# ===== ensure checkpoint directory exists =====
import os
os.makedirs(CHECKPOINT_DIR_PATH, exist_ok=True)

# ===== save model checkpoints =====
torch.save(encoder.state_dict(), ENCODER_SAVE_PATH)
torch.save(decoder.state_dict(), DECODER_SAVE_PATH)