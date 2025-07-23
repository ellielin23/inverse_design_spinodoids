# train_flow.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from models.flow_decoder import FlowDecoder
from utils.dataset import SpinodoidDataset

# === manual config ===
DATA_PATH = "data/dataset_train_x1000.csv"
BATCH_SIZE = 64
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3

S_DIM = 4
P_DIM = 9
LATENT_DIM = 4
DECODER_HIDDEN_DIMS = [128, 64, 32]
NUM_FLOWS = 4

SAVE_DIR = "flow_checkpoints"
TRIAL = 1
os.makedirs(SAVE_DIR, exist_ok=True)

# === load dataset ===
dataset = SpinodoidDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === initialize model ===
flow_decoder = FlowDecoder(
    S_dim=S_DIM,
    P_dim=P_DIM,
    latent_dim=LATENT_DIM,
    dec_hidden_dims=DECODER_HIDDEN_DIMS,
    num_flows=NUM_FLOWS
)

optimizer = optim.Adam(flow_decoder.parameters(), lr=LEARNING_RATE)

# === training loop ===
losses = []

for epoch in range(NUM_EPOCHS):
    flow_decoder.train()
    total_loss_epoch = 0.0

    for P_batch, S_batch in dataloader:
        z0 = torch.randn(P_batch.size(0), LATENT_DIM)

        S_hat, log_det = flow_decoder(z0, P_batch)
        loss = ((S_hat - S_batch) ** 2).mean()  # MSE loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()

    avg_loss = total_loss_epoch / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f}")

# === plot training loss ===
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Flow Decoder Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Flow Decoder Training Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# === save model ===
torch.save(flow_decoder.state_dict(), f"{SAVE_DIR}/flow_decoder_{TRIAL}.pth")
print("✅ Flow decoder model saved to:", SAVE_DIR)
