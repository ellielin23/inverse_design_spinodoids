# models/encoder.py

import torch
import torch.nn as nn
from config import ENCODER_HIDDEN_DIMS

class Encoder(nn.Module):
    """
    Encoder for Conditional Variational Autoencoder.
    Takes structure parameters S and target properties P
    as input, and outputs the mean and log variance of the
    latent space distribution.
    """

    def __init__(self, S_dim, P_dim, latent_dim):
        """
        Initialize the encoder network.

        Args:
            S_dim (int): Dimension of the structure parameters.
            P_dim (int): Dimension of the target properties.
            latent_dim (int): Dimension of the latent space.
        """
        super(Encoder, self).__init__()  # initialize the base nn.Module class
        input_dim = S_dim + P_dim

        # hidden layers (based on config)
        layers = []
        prev_dim = input_dim
        for hidden_dim in ENCODER_HIDDEN_DIMS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)

        # output layers: produce the parameters of the latent distribution
        self.fc_mu = nn.Linear(prev_dim, latent_dim)      # outputs mean vector mu of the latent Gaussian
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)  # outputs log(sigma^2) for variance

    def forward(self, S, P):
        """
        Forward pass through the encoder.

        Args:
            S (torch.Tensor): Structure parameters tensor of shape (batch_size, S_dim)
            P (torch.Tensor): Target properties tensor of shape (batch_size, P_dim)

        Returns:
            tuple: Mean and log variance of the latent distribution
        """
        x = torch.cat([S, P], dim=1)  # concatenate structure and property vectors
        x = self.hidden_layers(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar