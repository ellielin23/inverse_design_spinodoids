# models/decoder.py

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder for Conditional Variational Autoencoder.
    Takes latent vector z and target properties P as input,
    and outputs a predicted structure vector S-hat.
    """

    def __init__(self, S_dim, P_dim, latent_dim, dec_hidden_dims):
        """
        Initialize the decoder network.

        Args:
            S_dim (int): Dimension of the structure parameters.
            P_dim (int): Dimension of the target properties.
            latent_dim (int): Dimension of the latent space.
            dec_hidden_dims (list): List of hidden layer sizes.
            dropout_prob (float): Dropout probability (default 0.1)
        """
        super(Decoder, self).__init__()
        input_dim = latent_dim + P_dim

        # hidden layers (based on config)
        layers = []
        prev_dim = input_dim
        for hidden_dim in dec_hidden_dims: # change
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(dropout_prob))
            prev_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)

        # output layer: produce predicted structure parameters
        self.output_layer = nn.Linear(prev_dim, S_dim)

    def forward(self, z, P):
        """
        Forward pass through the decoder.

        Args:
            z (torch.Tensor): Latent vector sampled from encoder, shape [batch_size, latent_dim]
            P (torch.Tensor): Target properties, shape [batch_size, P_dim]

        Returns:
            S_hat (torch.Tensor): Predicted structure parameters, shape [batch_size, S_dim]
        """
        x = torch.cat([z, P], dim=1)  # concatenate latent vector and property vector
        x = self.hidden_layers(x)
        S_hat = self.output_layer(x)
        return S_hat