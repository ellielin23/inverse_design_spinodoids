# train_flow.py

import torch
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.flow_forward import FlowForwardModel
from utils.dataset import SpinodoidDataset
from utils.losses import flow_nll
from config import *

# === setup device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === load dataset ===
dataset = SpinodoidDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === initialize model ===
model = FlowForwardModel(S_DIM, P_DIM, hidden_dims=HIDDEN_DIMS, num_flows=NUM_FLOWS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === track loss ===
losses = []

# === training loop ===
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for S_batch, P_batch in dataloader:
        S_batch = S_batch.to(device)
        P_batch = P_batch.to(device)

        optimizer.zero_grad()

        # forward pass
        z_k, log_q, mu, log_sigma = model(S_batch)

        # compute loss
        loss = flow_nll(mu, log_sigma, z_k, log_q, P_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1:03d} | Flow NLL: {avg_loss:.4f}")

# === plot loss curve ===
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Flow NLL Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Normalizing Flow Forward Model)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# === save model checkpoint ===
torch.save(model.state_dict(), FLOW_SAVE_PATH)