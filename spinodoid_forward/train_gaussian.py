# train_gaussian.py

import torch
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from models.gaussian_forward import GaussianForwardModel
from utils.data_utils.dataset import SpinodoidDataset
from utils.losses import gaussian_nll
from config import *
from torch.optim.lr_scheduler import StepLR

# === sanity check ===
assert MODEL == "gaussian", f"train_gaussian.py should only be used with MODEL='gaussian', but got '{MODEL}'"

# === setup device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === load normalization constants ===
S_mean = torch.tensor(np.load("data/S_mean.npy"), dtype=torch.float32, device=device)
S_std  = torch.tensor(np.load("data/S_std.npy"),  dtype=torch.float32, device=device)
P_mean = torch.tensor(np.load("data/P_mean.npy"), dtype=torch.float32, device=device)
P_std  = torch.tensor(np.load("data/P_std.npy"),  dtype=torch.float32, device=device)

def norm_S(S): return (S - S_mean) / S_std
def norm_P(P): return (P - P_mean) / P_std

# === load dataset ===
dataset = SpinodoidDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === initialize model and optimizer ===
model = GaussianForwardModel(S_DIM, P_DIM, hidden_dims=HIDDEN_DIMS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

# optional knobs from config (with safe defaults)
SIGMA_MIN = globals().get("SIGMA_MIN", 1e-3)
CLIP_GRAD_NORM = globals().get("CLIP_GRAD_NORM", None)

# === training loop ===
losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss_epoch = 0.0

    for S_batch, P_batch in dataloader:
        S_batch = S_batch.to(device)
        P_batch = P_batch.to(device)

        # === normalize inputs/targets ===
        Sn = norm_S(S_batch)
        Pn = norm_P(P_batch)

        # forward pass
        mu, log_sigma = model(Sn)

        # numerical safety: avoid sigma -> 0
        log_sigma = torch.clamp(log_sigma, min=float(np.log(SIGMA_MIN)))

        # compute loss (NLL in normalized space)
        loss = gaussian_nll(mu, log_sigma, Pn)

        # optional variance regularizer to discourage exploding σ
        if BETA_VAR_REG and BETA_VAR_REG > 0:
            loss = loss + BETA_VAR_REG * torch.mean(torch.exp(log_sigma))

        # backward
        optimizer.zero_grad(set_to_none=True)   # ✅ prevent grad accumulation
        loss.backward()
        if CLIP_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        total_loss_epoch += float(loss.item())

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
    f.write(f"SIGMA_MIN: {SIGMA_MIN}\n")
print(f"✅ Config saved to {SAVE_CONFIG_PATH}")
