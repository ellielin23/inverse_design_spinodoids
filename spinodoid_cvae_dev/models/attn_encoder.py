# models/attn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnEncoder(nn.Module):
    """
    Encoder with attention for Conditional Variational Autoencoder.
    Takes structure parameters S and target properties P as input,
    processes them with MLP + attention, and outputs latent distribution parameters.
    """

    def __init__(self, S_dim, P_dim, latent_dim, hidden_dims):
        super().__init__()
        self.input_dim = S_dim + P_dim
        self.latent_dim = latent_dim

        # === MLP feature extractor ===
        layers = []
        prev_dim = self.input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.mlp = nn.Sequential(*layers)

        # === attention layer ===
        self.query = nn.Linear(prev_dim, prev_dim)
        self.key   = nn.Linear(prev_dim, prev_dim)
        self.value = nn.Linear(prev_dim, prev_dim)
        self.scale = prev_dim ** 0.5

        # === output layers ===
        self.fc_mu     = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def attention(self, x):
        """
        Single-head self-attention mechanism.
        Args:
            x: Tensor of shape (batch_size, hidden_dim)
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # compute attention weights using scaled dot-product
        scores = torch.matmul(Q.unsqueeze(1), K.unsqueeze(2)) / self.scale 
        attn_weights = F.softmax(scores, dim=-1)
        attended = attn_weights.squeeze(-1) * V 
        return attended

    def forward(self, S, P):
        x = torch.cat([S, P], dim=1)
        x = self.mlp(x) 
        x = self.attention(x)

        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
