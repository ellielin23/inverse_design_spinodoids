import torch
import torch.nn as nn
from pytorch_minimize.optim import MinimizeWrapper
from torch.utils.data import DataLoader, TensorDataset

from models.encoder import Encoder
from models.decoder import Decoder
from utils import reparameterize
from loss import vae_loss
from config import *

import numpy as np
import os

# ===== set device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== load in dataset =====
data = np.load(DATA_PATH)
S = torch.tensor(data['S'], dtype=torch.float64)  # shape [N, 1], use float64
P = torch.tensor(data['P'], dtype=torch.float64)  # shape [N, 1], use float64

dataset = TensorDataset(S, P)
full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)  # full batch

# === extract full batch for closure ===
full_S, full_P = next(iter(full_loader))
full_S = full_S.to(device).double()
full_P = full_P.to(device).double()

# ===== initialize models (convert to float64) =====
encoder = Encoder(S_dim=S_DIM, P_dim=P_DIM, latent_dim=LATENT_DIM).to(device).double()
decoder = Decoder(S_dim=S_DIM, P_dim=P_DIM, latent_dim=LATENT_DIM).to(device).double()

# ===== define SLSQP optimizer =====
params = list(encoder.parameters()) + list(decoder.parameters())
minimizer_args = dict(method='SLSQP', options={'disp': True, 'maxiter': 100})
optimizer = MinimizeWrapper(params, minimizer_args)

# ===== define closure for optimizer =====
def closure():
    optimizer.zero_grad()
    encoder.train()
    decoder.train()

    # forward pass
    mu, logvar = encoder(full_S, full_P)
    z = reparameterize(mu, logvar)
    S_hat = decoder(z, full_P)

    loss = vae_loss(S_hat, full_S, mu, logvar, None, None)
    loss.backward()
    return loss

# ===== run optimizer =====
final_loss = optimizer.step(closure)
# print final loss from optimizer result object
if optimizer.res and hasattr(optimizer.res, "fun"):
    print(f"\nSLSQP optimization finished | Final loss: {optimizer.res.fun:.6f}")
else:
    print("\nSLSQP optimization finished, but final loss unavailable.")

# ===== ensure checkpoint directory exists =====
os.makedirs(CHECKPOINT_DIR_PATH, exist_ok=True)

# ===== save model checkpoints =====
torch.save(encoder.state_dict(), ENCODER_SAVE_PATH)
torch.save(decoder.state_dict(), DECODER_SAVE_PATH)