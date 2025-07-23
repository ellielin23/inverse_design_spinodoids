# models/decoder.py

import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder for simplified CVAE.
    Takes latent vector z and target property P as input,
    and outputs predicted structure vector S-hat.
    """

    def __init__(self, S_dim, P_dim, latent_dim, hidden_dims):
        """
        Args:
            S_dim (int): Dimension of the structure vector (usually 4)
            P_dim (int): Dimension of the property vector (1 for C111 or C212)
            latent_dim (int): Latent space dimension
            hidden_dims (list[int]): Sizes of hidden layers
        """
        super(Decoder, self).__init__()

        input_dim = latent_dim + P_dim
        layers = []

        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, S_dim)

    def forward(self, z, P):
        """
        Forward pass.

        Args:
            z (Tensor): Latent vector, shape (batch_size, latent_dim)
            P (Tensor): Property vector, shape (batch_size, P_dim)

        Returns:
            Tensor: Predicted structure vector S_hat, shape (batch_size, S_dim)
        """
        x = torch.cat([z, P], dim=1)
        x = self.hidden_layers(x)
        return self.output_layer(x)
