# train_parallel.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from config_parallel import *
from utils.data_utils.dataset import SpinodoidDataset
from utils.model_utils import get_encoder, get_decoder
from utils.loss import total_loss, get_kl_beta

# === setup device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === load dataset ===
dataset = SpinodoidDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === get data shapes ===
P_dim = dataset.P.shape[1]
S_dim = dataset.S.shape[1]

# === load normalization stats ===
P_mean_path = f"data/partition_by_theta/P_mean_theta_{THETA_PATTERN}.npy"
P_std_path = f"data/partition_by_theta/P_std_theta_{THETA_PATTERN}.npy"
P_mean = torch.tensor(np.load(P_mean_path), dtype=torch.float32, device=device)
P_std = torch.tensor(np.load(P_std_path), dtype=torch.float32, device=device)

S_mean_path = f"data/partition_by_theta/S_mean_theta_{THETA_PATTERN}.npy"
S_std_path = f"data/partition_by_theta/S_std_theta_{THETA_PATTERN}.npy"
S_mean = torch.tensor(np.load(S_mean_path), dtype=torch.float32, device=device)
S_std = torch.tensor(np.load(S_std_path), dtype=torch.float32, device=device)



# === init models ===
encoder = get_encoder(
    use_attention=USE_ATTENTION_ENCODER,
    S_dim=S_DIM,
    P_dim=P_DIM,
    latent_dim=LATENT_DIM,
    hidden_dims=ENCODER_HIDDEN_DIMS
).to(device)

decoder = get_decoder(
    use_flow=USE_FLOW_DECODER,
    use_attention=USE_ATTENTION_DECODER,
    S_dim=S_DIM,
    P_dim=P_DIM,
    latent_dim=LATENT_DIM,
    hidden_dims=DECODER_HIDDEN_DIMS,
    num_flows=NUM_FLOWS,
    dropout_prob=DROPOUT_PROB,
    flow_type=FLOW_TYPE,
    device=device
).to(device)

# === optimizer and scheduler ===
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

# === reparameterization ===
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# === training loop ===
losses, recon_losses, kl_losses = [], [], []
RECON_WEIGHTS = [1.0, 1.0, 1.0, 1.0]  # boost volume ratio weight if needed

for epoch in range(NUM_EPOCHS):
    encoder.train()
    decoder.train()
    total_loss_epoch = 0
    total_rec_loss = 0
    total_kl_loss = 0

    for P_batch, S_batch in dataloader:
        P_batch, S_batch = P_batch.to(device), S_batch.to(device)
        optimizer.zero_grad()

        # normalize P and S with safeguard against zero division
        P_norm = (P_batch - P_mean) / (P_std + 1e-8)
        S_norm = (S_batch - S_mean) / (S_std + 1e-8)

        # forward pass
        mu, logvar = encoder(S_norm, P_norm)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)  # prevent explosion
        z = reparameterize(mu, logvar)
        beta = get_kl_beta(epoch, warmup_epochs=50, max_beta=BETA)

        S_hat, log_det = (
            decoder(z, P_norm) if USE_FLOW_DECODER else (decoder(z, P_norm), None)
        )

        loss, rec, kl = total_loss(
            S_hat, S_norm, mu, logvar,
            log_det=log_det,
            beta=beta,
            component_weights=RECON_WEIGHTS
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        total_loss_epoch += loss.item()
        total_rec_loss += rec.item()
        total_kl_loss += kl.item()

    scheduler.step()
    print(f"Epoch {epoch+1:03d} | LR: {scheduler.get_last_lr()[0]:.5e} | Loss: {total_loss_epoch:.4f} | Rec: {total_rec_loss:.4f} | KL: {total_kl_loss:.4f} | Beta: {beta:.3f}")
    losses.append(total_loss_epoch)
    recon_losses.append(total_rec_loss)
    kl_losses.append(total_kl_loss)

# === save model + config ===
os.makedirs(CHECKPOINT_DIR_PATH, exist_ok=True)
torch.save(encoder.state_dict(), ENCODER_SAVE_PATH)
torch.save(decoder.state_dict(), DECODER_SAVE_PATH)

# === save config file ===
config_dict = {
    "S_DIM": S_dim,
    "P_DIM": P_dim,
    "LATENT_DIM": LATENT_DIM,
    "ENCODER_HIDDEN_DIMS": ENCODER_HIDDEN_DIMS,
    "DECODER_HIDDEN_DIMS": DECODER_HIDDEN_DIMS,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "BETA": BETA,
    "NUM_FLOWS": NUM_FLOWS,
    "DROPOUT_PROB": DROPOUT_PROB,
    "USE_FLOW_DECODER": USE_FLOW_DECODER,
    "USE_ATTENTION_ENCODER": USE_ATTENTION_ENCODER,
    "USE_ATTENTION_DECODER": USE_ATTENTION_DECODER,
    "THETA_PATTERN": f"{THETA_PATTERN}",      # avoid python invalid integer
    "TRIAL": TRIAL
}


# === plot loss curves ===
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Total Loss')
plt.plot(recon_losses, label='Reconstruction Loss')
plt.plot(kl_losses, label='KL Divergence')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss Curve (Trial {TRIAL}, θ pattern {THETA_PATTERN})")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

with open(CONFIG_SAVE_PATH, "w") as f:
    for k, v in config_dict.items():
        if isinstance(v, str):
            f.write(f'{k}: "{v}"\n')  # wrap strings in quotes!!
        else:
            f.write(f"{k}: {v}\n")

print("✅ Model and config saved to", CHECKPOINT_DIR_PATH)
