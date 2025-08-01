# train_gaussian.py

import torch
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from models.gaussian_forward import GaussianForwardModel
from utils.dataset import SpinodoidDataset
from utils.losses import gaussian_nll
from config import *
from torch.optim.lr_scheduler import StepLR

# === sanity check ===
assert MODEL == "gaussian", f"train_gaussian.py should only be used with MODEL='gaussian', but got '{MODEL}'"

# === setup device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === load normalization constants ===
P_mean = np.load("data/P_mean.npy")
P_std = np.load("data/P_std.npy")
P_mean_tensor = torch.tensor(P_mean, dtype=torch.float32, device=device)
P_std_tensor = torch.tensor(P_std, dtype=torch.float32, device=device)

# === load dataset ===
dataset = SpinodoidDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === initialize model and optimizer ===
model = GaussianForwardModel(S_DIM, P_DIM, hidden_dims=HIDDEN_DIMS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

# === training loop ===
losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss_epoch = 0

    for S_batch, P_batch in dataloader:
        S_batch = S_batch.to(device)
        P_batch = P_batch.to(device)

        # === normalize P ===
        P_batch_norm = (P_batch - P_mean_tensor) / P_std_tensor

        # forward pass
        mu, log_sigma = model(S_batch)

        # compute loss
        loss = gaussian_nll(mu, log_sigma, P_batch_norm)
        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1:03d} | NLL: {total_loss_epoch:.4f} | LR: {current_lr:.2e}")
    losses.append(total_loss_epoch)

# === plot loss curve ===
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Negative Log-Likelihood')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Gaussian Forward Model)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === save model and config ===
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(model.state_dict(), SAVE_MODEL_PATH)
print(f"✅ Model saved to {SAVE_MODEL_PATH}")

with open(SAVE_CONFIG_PATH, "w") as f:
    f.write(f"S_DIM: {S_DIM}\n")
    f.write(f"P_DIM: {P_DIM}\n")
    f.write(f"HIDDEN_DIMS: {HIDDEN_DIMS}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
    f.write(f"BETA_VAR_REG: {BETA_VAR_REG}\n")
print(f"✅ Config saved to {SAVE_CONFIG_PATH}")
