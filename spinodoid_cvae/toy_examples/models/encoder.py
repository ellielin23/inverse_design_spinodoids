# models/encoder.py

import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder network for mapping structure vector S and target value P to latent space parameters.
    """
    def __init__(self, S_dim, P_dim, latent_dim, hidden_dims):
        """
        Args:
            S_dim (int): Dimensionality of structure vector (input).
            P_dim (int): Dimensionality of target property (should be 1 for C111 or C212).
            latent_dim (int): Dimensionality of latent space.
            hidden_dims (list): List of hidden layer dimensions.
        """
        super(Encoder, self).__init__()

        input_dim = S_dim + P_dim  # concatenate S and P as input
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, S, P):
        """
        Args:
            S (Tensor): Structure vector, shape (batch_size, S_dim)
            P (Tensor): Target property, shape (batch_size, P_dim)

        Returns:
            mu (Tensor): Latent mean, shape (batch_size, latent_dim)
            logvar (Tensor): Latent log variance, shape (batch_size, latent_dim)
        """
        x = torch.cat([S, P], dim=1)
        h = self.hidden_layers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
