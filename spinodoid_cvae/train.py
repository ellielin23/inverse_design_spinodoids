# train.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from models.encoder import Encoder
from models.decoder import Decoder
from models.flow_decoder import FlowDecoder
from utils.data_utils.dataset import SpinodoidDataset
from utils.loss import total_loss
from config import *

# === load dataset ===
dataset = SpinodoidDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === initialize encoder ===
encoder = Encoder(S_DIM, P_DIM, LATENT_DIM, ENCODER_HIDDEN_DIMS)

# === initialize decoder (regular or flow) ===
if USE_FLOW_DECODER:
    decoder = FlowDecoder(
        S_dim=S_DIM,
        P_dim=P_DIM,
        latent_dim=LATENT_DIM,
        dec_hidden_dims=DECODER_HIDDEN_DIMS,
        num_flows=NUM_FLOWS,
        dropout_prob=DROPOUT_PROB,
        flow_type=FLOW_TYPE
    )
else:
    decoder = Decoder(S_DIM, P_DIM, LATENT_DIM, DECODER_HIDDEN_DIMS)

# === optimizer ===
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE)

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
        mu, logvar = encoder(S_batch, P_batch)
        z = reparameterize(mu, logvar)

        # decode and compute loss
        if USE_FLOW_DECODER:
            S_hat, log_det = decoder(z, P_batch)
            loss, rec, kl = total_loss(S_hat, S_batch, mu, logvar, log_det=log_det, beta=BETA)
        else:
            S_hat = decoder(z, P_batch)
            loss, rec, kl = total_loss(S_hat, S_batch, mu, logvar, beta=BETA)

        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        total_rec_loss += rec.item()
        total_kl_loss += kl.item()

    print(f"Epoch {epoch+1:03d} | Loss: {total_loss_epoch:.4f} | Rec: {total_rec_loss:.4f} | KL: {total_kl_loss:.4f}")
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
}

with open(CONFIG_SAVE_PATH, "w") as f:
    for k, v in config_dict.items():
        f.write(f"{k}: {v}\n")

print("✅ Saved model and config.")
