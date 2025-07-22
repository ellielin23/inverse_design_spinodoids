# auto_train.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.encoder import Encoder
from models.decoder import Decoder
from utils.dataset import SpinodoidDataset
from utils.loss import total_loss
from config import DATA_PATH
from config_dictionary import CONFIGS
import os

def train_model(config):
    # === unpack config ===
    latent_dim = config['latent_dim']
    enc_dims = config['encoder_dims']
    dec_dims = config['decoder_dims']
    batch_size = config['batch_size']
    lr = config['learning_rate']
    beta = config['beta']
    epochs = config['epochs']
    save_dir = config['save_dir']

    # === constants ===
    S_DIM = 4
    P_DIM = 9

    # === dataset ===
    dataset = SpinodoidDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === models ===
    encoder = Encoder(S_DIM, P_DIM, latent_dim, enc_dims)
    decoder = Decoder(S_DIM, P_DIM, latent_dim, dec_dims)
    encoder.train(); decoder.train()

    # === optimizer ===
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    # === training loop ===
    for epoch in range(epochs):
        epoch_loss, epoch_rec, epoch_kl = 0, 0, 0

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
    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(save_dir, f"encoder_{config['trial']}.pt"))
    torch.save(decoder.state_dict(), os.path.join(save_dir, f"decoder_{config['trial']}.pt"))

    # also save the config
    with open(os.path.join(save_dir, f"config_{config['trial']}.txt"), "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    return encoder, decoder

def main():
    for i, config in enumerate(CONFIGS, start=1):
        config = config.copy()  # to avoid modifying the original
        config['save_dir'] = f"auto_checkpoints/trial_{i}"
        config['trial'] = i
        print(f"\n🚀 Starting Trial {i} with config: {config}")
        train_model(config)

if __name__ == "__main__":
    main()