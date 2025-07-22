# models/decoder.py

# if running with auto-tuning, replace the config DECODER_HIDDEN_DIMS 
# with a parameter called dec_hidden_dims in the Decoder class

import torch
import torch.nn as nn
from config import DECODER_HIDDEN_DIMS

class Decoder(nn.Module):
    """
    Decoder for Conditional Variational Autoencoder.
    Takes latent vector z and target properties P as input,
    and outputs a predicted structure vector S-hat.
    """

    def __init__(self, S_dim, P_dim, latent_dim):
        """
        Initialize the decoder network.

        Args:
            S_dim (int): Dimension of the structure parameters.
            P_dim (int): Dimension of the target properties.
            latent_dim (int): Dimension of the latent space.
        """
        super(Decoder, self).__init__()
        input_dim = latent_dim + P_dim

        # hidden layers (based on config)
        layers = []
        prev_dim = input_dim
        for hidden_dim in DECODER_HIDDEN_DIMS: # change
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
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