# train_flow.py

import torch
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from models.flow_forward import FlowForwardModel
from utils.dataset import SpinodoidDataset
from utils.losses import flow_nll
from config import *
from torch.optim.lr_scheduler import StepLR

# === sanity check ===
assert MODEL.lower() == "flow", f"train_flow.py should only be used with MODEL='flow', but got '{MODEL}'"
assert FLOW_TYPE in ["planar", "maf", "realnvp"], f"Invalid FLOW_TYPE: {FLOW_TYPE}"

# === setup device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === load dataset ===
dataset = SpinodoidDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === initialize model and optimizer ===
model = FlowForwardModel(S_DIM, P_DIM, hidden_dims=HIDDEN_DIMS, num_flows=NUM_FLOWS, flow_type=FLOW_TYPE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

# === training loop ===
losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for S_batch, P_batch in dataloader:
        S_batch = S_batch.to(device)
        P_batch = P_batch.to(device)

        optimizer.zero_grad()
        z_k, log_q, mu, log_sigma = model(S_batch)
        loss = flow_nll(mu, log_sigma, z_k, log_q, P_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1:03d} | Flow NLL: {avg_loss:.4f} | LR: {current_lr:.2e}")

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
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(model.state_dict(), SAVE_MODEL_PATH)
print(f"✅ Model saved to {SAVE_MODEL_PATH}")

# === save config for reproducibility ===
with open(SAVE_CONFIG_PATH, "w") as f:
    f.write(f"S_DIM: {S_DIM}\n")
    f.write(f"P_DIM: {P_DIM}\n")
    f.write(f"HIDDEN_DIMS: {HIDDEN_DIMS}\n")
    f.write(f"NUM_FLOWS: {NUM_FLOWS}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
    f.write(f"BETA_VAR_REG: {BETA_VAR_REG}\n")
print(f"✅ Config saved to {SAVE_CONFIG_PATH}")