# train.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
import numpy as np

from utils.model_utils import get_encoder, get_decoder
from utils.data_utils.dataset import SpinodoidDataset
from utils.loss import total_loss, get_kl_beta
from config import *

# === load dataset and device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SpinodoidDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === load P mean/std for normalization ===
P_mean = torch.tensor(np.load("data/P_mean.npy"), dtype=torch.float32, device=device)
P_std = torch.tensor(np.load("data/P_std.npy"), dtype=torch.float32, device=device)

# === initialize encoder ===
encoder = get_encoder(
    use_attention=USE_ATTENTION,
    S_dim=S_DIM,
    P_dim=P_DIM,
    latent_dim=LATENT_DIM,
    hidden_dims=ENCODER_HIDDEN_DIMS
)

# === initialize decoder (regular or flow) ===
decoder = get_decoder(
    use_flow=USE_FLOW_DECODER,
    use_attention=USE_ATTENTION,
    S_dim=S_DIM,
    P_dim=P_DIM,
    latent_dim=LATENT_DIM,
    hidden_dims=DECODER_HIDDEN_DIMS,
    num_flows=NUM_FLOWS,
    dropout_prob=DROPOUT_PROB,
    flow_type=FLOW_TYPE,
    device=device
)

# === optimizer ===
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)  # decay every 30 epochs by 0.5×

# === reparameterization trick ===
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# === loss trackers ===
losses = []
recon_losses = []
kl_losses = []

# === training loop ===
for epoch in range(NUM_EPOCHS):
    encoder.train()
    decoder.train()

    total_loss_epoch = 0
    total_rec_loss = 0
    total_kl_loss = 0

    for P_batch, S_batch in dataloader:
        optimizer.zero_grad()

        # encode
        P_batch_norm = (P_batch - P_mean) / P_std
        mu, logvar = encoder(S_batch, P_batch_norm)

        z = reparameterize(mu, logvar)

        # === use KL warm-up ===
        beta = get_kl_beta(epoch, warmup_epochs=20, max_beta=10.0)  # can adjust warmup_epochs

        if USE_FLOW_DECODER:
            S_hat, log_det = decoder(z, P_batch_norm)
            loss, rec, kl = total_loss(S_hat, S_batch, mu, logvar, log_det=log_det, beta=beta)
        else:
            S_hat = decoder(z, P_batch_norm)
            loss, rec, kl = total_loss(S_hat, S_batch, mu, logvar, beta=beta)

        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        total_rec_loss += rec.item()
        total_kl_loss += kl.item()

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1:03d} | LR: {current_lr:.5e} | Loss: {total_loss_epoch:.4f} | Rec: {total_rec_loss:.4f} | KL: {total_kl_loss:.4f}")
    losses.append(total_loss_epoch)
    recon_losses.append(total_rec_loss)
    kl_losses.append(total_kl_loss)

# === plot loss curves ===
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Total Loss')
plt.plot(recon_losses, label='Reconstruction Loss')
plt.plot(kl_losses, label='KL Divergence')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === save model checkpoints ===
os.makedirs(CHECKPOINT_DIR_PATH, exist_ok=True)
torch.save(encoder.state_dict(), ENCODER_SAVE_PATH)
torch.save(decoder.state_dict(), DECODER_SAVE_PATH)

# === save config ===
config_dict = {
    "S_DIM": S_DIM,
    "P_DIM": P_DIM,
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
    "USE_ATTENTION_DECODER": USE_ATTENTION_DECODER,
    "USE_ATTENTION_ENCODER": USE_ATTENTION_ENCODER,
}

with open(CONFIG_SAVE_PATH, "w") as f:
    for k, v in config_dict.items():
        f.write(f"{k}: {v}\n")

print("✅ Saved model and config.")
