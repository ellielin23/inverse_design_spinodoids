# train_gaussian.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.load_data import load_distributional_dataset
from utils.metrics import average_kl
from models.gaussian_forward import GaussianForwardModel
from utils.dataset import SpinodoidDataset
from utils.losses import gaussian_nll
from config import *

# === load dataset ===
dataset = SpinodoidDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === empirical data for KL === # remove
empirical_data = load_distributional_dataset(DISTRIBUTIONAL_DATA_PATH)

# === initialize model ===
model = GaussianForwardModel(S_DIM, P_DIM, hidden_dims=HIDDEN_DIMS)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === add learning rate scheduler === # remove
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

# === loss tracker ===
losses = []
kl_scores = [] # remove
best_kl = float('inf') # remove
best_model_state = None # remove


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

    # step the scheduler every epoch ✅
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    losses.append(total_loss_epoch)

    # === KL tracking (optional block) ===
    model.eval()
    with torch.no_grad():
        S_val = dataset[0][0].unsqueeze(0)
        S_val_repeat = S_val.repeat(100, 1)
        mu, log_sigma = model(S_val_repeat)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        P_pred = mu + sigma * eps
        P_pred_np = P_pred.cpu().numpy()

        S_key = tuple(np.round(S_val.cpu().numpy().flatten(), 5))
        P_true_np = empirical_data[S_key]

        avg_kl = average_kl(P_pred_np, P_true_np)
        kl_scores.append(avg_kl)

        if avg_kl < best_kl:
            best_kl = avg_kl
            best_model_state = model.state_dict()

    # log everything
    print(f"Epoch {epoch+1:03d} | NLL: {total_loss_epoch:.4f} | KL: {avg_kl:.4f} | LR: {current_lr:.2e}")

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
#torch.save(model.state_dict(), GAUSSIAN_SAVE_PATH)
# === save best model ===
torch.save(best_model_state, GAUSSIAN_SAVE_PATH)
print(f"✅ Best model saved with KL = {best_kl:.4f}")

