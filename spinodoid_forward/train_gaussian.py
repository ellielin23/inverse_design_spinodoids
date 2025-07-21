# train_gaussian.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.gaussian_forward import GaussianForwardModel
from utils.dataset import SpinodoidDataset
from utils.losses import gaussian_nll
from config import *

# === load dataset ===
dataset = SpinodoidDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === initialize model ===
model = GaussianForwardModel(S_DIM, P_DIM, hidden_dims=HIDDEN_DIMS)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === add learning rate scheduler === # remove
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

# === loss tracker ===
losses = []

# === training loop ===
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss_epoch = 0

    for S_batch, P_batch in dataloader:
        optimizer.zero_grad()

        # forward pass
        mu, log_sigma = model(S_batch)

        # compute loss
        loss = gaussian_nll(mu, log_sigma, P_batch)
        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()

    # step the scheduler every epoch
    scheduler.step() # remove

    current_lr = scheduler.get_last_lr()[0] # remove
    print(f"Epoch {epoch+1:03d} | NLL: {total_loss_epoch:.4f}")
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

# ===== save model checkpoint =====
torch.save(model.state_dict(), GAUSSIAN_SAVE_PATH)
