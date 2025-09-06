# train_flow.py

import os, math, numpy as np, torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from models.flow_forward import FlowForwardModel
from utils.data_utils.dataset import SpinodoidDataset
from utils.losses import flow_nll   # expects normalized space
from config import *

assert MODEL.lower() == "flow", f"train_flow.py should be used with MODEL='flow', got '{MODEL}'"
assert FLOW_TYPE in ["planar", "maf", "realnvp"], f"Invalid FLOW_TYPE: {FLOW_TYPE}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === load dataset ===
dataset = SpinodoidDataset(DATA_PATH)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# === load normalization stats (from TRAIN split) ===
S_mean = torch.tensor(np.load(f"data/S_mean.npy"), dtype=torch.float32, device=device)
S_std  = torch.tensor(np.load(f"data/S_std.npy"),  dtype=torch.float32, device=device)
P_mean = torch.tensor(np.load(f"data/P_mean.npy"), dtype=torch.float32, device=device)
P_std  = torch.tensor(np.load(f"data/P_std.npy"),  dtype=torch.float32, device=device)

def norm_S(S): return (S - S_mean) / S_std
def norm_P(P): return (P - P_mean) / P_std

# === model & optim ===
model = FlowForwardModel(
    S_dim=S_DIM, P_dim=P_DIM,
    hidden_dims=HIDDEN_DIMS,
    num_flows=NUM_FLOWS,
    flow_type=FLOW_TYPE
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

SIGMA_MIN = globals().get("SIGMA_MIN", 1e-3)      # used by flow_nll if relevant
CLIP_GRAD_NORM = globals().get("CLIP_GRAD_NORM", None)

# === train ===
loss_curve = []
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total = 0.0

    for S_batch, P_batch in loader:
        S_batch = S_batch.to(device, non_blocking=True)
        P_batch = P_batch.to(device, non_blocking=True)

        # normalize for training
        Sn = norm_S(S_batch)
        Pn = norm_P(P_batch)

        optimizer.zero_grad(set_to_none=True)

        # forward:
        z_k, logdet_total, mu, log_sigma = model(Sn, Pn)   # <<< ensure your model signature matches
        
        # NLL in normalized space
        loss = flow_nll(None, None, z_k, logdet_total)

        loss.backward()
        if CLIP_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        total += float(loss.item())

    scheduler.step()
    avg = total / len(loader)
    loss_curve.append(avg)
    print(f"Epoch {epoch:03d} | Flow NLL: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

# === plot ===
plt.figure(figsize=(8,5))
plt.plot(loss_curve, label="Flow NLL")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Flow ({FLOW_TYPE}) — Training NLL")
plt.grid(True, linestyle="--", alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

# === save ===
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(model.state_dict(), SAVE_MODEL_PATH)
print(f"✅ Model saved to {SAVE_MODEL_PATH}")

with open(SAVE_CONFIG_PATH, "w") as f:
    f.write(f"S_DIM: {S_DIM}\nP_DIM: {P_DIM}\nHIDDEN_DIMS: {HIDDEN_DIMS}\n")
    f.write(f"NUM_FLOWS: {NUM_FLOWS}\nFLOW_TYPE: {FLOW_TYPE}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\nLEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\nSIGMA_MIN: {SIGMA_MIN}\n")
print(f"✅ Config saved to {SAVE_CONFIG_PATH}")
