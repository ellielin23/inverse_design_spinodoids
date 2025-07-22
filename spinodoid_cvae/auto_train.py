# auto_train.py

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from itertools import product

from models.encoder import Encoder
from models.decoder import Decoder
from utils.dataset import SpinodoidDataset
from utils.loss import total_loss
from config import DATA_PATH
from config_dictionary import *

# === fixed constants ===
S_DIM = 4
P_DIM = 9
BASE_SAVE_DIR = "auto_checkpoints"

# === load dataset once ===
dataset = SpinodoidDataset(DATA_PATH)

# === config sweep ===
sweep_id = 0
for latent_dim, enc_dims, dec_dims, batch_size, lr, beta, epoch_count in product(
    latent_dims, encoder_dims_list, decoder_dims_list,
    batch_sizes, learning_rates, betas, epochs
):
    sweep_id += 1
    print(f"\nStarting config {sweep_id}:")
    print(f"  latent_dim={latent_dim}, encoder_dims={enc_dims}, decoder_dims={dec_dims}")
    print(f"  batch_size={batch_size}, lr={lr}, beta={beta}, epochs={epoch_count}")

    # === dataloader ===
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === models ===
    encoder = Encoder(S_DIM, P_DIM, latent_dim, enc_dims)
    decoder = Decoder(S_DIM, P_DIM, latent_dim, dec_dims)
    encoder.train()
    decoder.train()

    # === optimizer ===
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    # === training ===
    for epoch in range(epoch_count):
        epoch_loss = epoch_rec = epoch_kl = 0
        for P_batch, S_batch in dataloader:
            optimizer.zero_grad()
            mu, logvar = encoder(S_batch, P_batch)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            S_hat = decoder(z, P_batch)
            loss, rec, kl = total_loss(S_hat, S_batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_rec += rec.item()
            epoch_kl += kl.item()
        print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.4f} | Rec: {epoch_rec:.4f} | KL: {epoch_kl:.4f}")

    # === save ===
    save_dir = os.path.join(BASE_SAVE_DIR, f"trial_{sweep_id}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(save_dir, f"encoder_{sweep_id}.pt"))
    torch.save(decoder.state_dict(), os.path.join(save_dir, f"decoder_{sweep_id}.pt"))

    # === save config ===
    config_text = {
        'latent_dim': latent_dim,
        'encoder_dims': enc_dims,
        'decoder_dims': dec_dims,
        'batch_size': batch_size,
        'learning_rate': lr,
        'beta': beta,
        'epochs': epoch_count,
    }
    with open(os.path.join(save_dir, f"config_{sweep_id}.txt"), "w") as f:
        for k, v in config_text.items():
            f.write(f"{k}: {v}\n")

print("\n✅ All training runs completed.")
