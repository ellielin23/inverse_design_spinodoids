# models/decoder.py

import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder for Conditional Variational Autoencoder.
    Takes latent vector z and target properties P as input,
    and outputs a predicted structure vector S-hat.
    """

    def __init__(self, s_dim, p_dim, latent_dim):
        """
        Initialize the decoder network.

        Args:
            S_dim (int): Dimension of the structure parameters.
            P_dim (int): Dimension of the target properties.
            latent_dim (int): Dimension of the latent space.
        """
        super(Decoder, self).__init__()
        input_dim = latent_dim + p_dim

        # first hidden layer
        self.fc1 = nn.Linear(input_dim, 128) 
        self.relu1 = nn.ReLU()

        # second hidden layer
        self.fc2 = nn.Linear(128, 64) 
        self.relu2 = nn.ReLU()

        # third hidden layer
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()

        # output layer: produce predicted structure parameters
        self.fc4 = nn.Linear(32, s_dim)

    def forward(self, z, P):
        """
        Forward pass through the decoder.

        Args:
            z (torch.Tensor): Latent vector sampled from encoder, shape [batch_size, latent_dim]
            P (torch.Tensor): Target properties, shape [batch_size, P_dim]

        Returns:
            S_hat (torch.Tensor): Predicted structure parameters, shape [batch_size, S_dim]
        """
        x = torch.cat([z, P], dim=1) # concatenate latent vector and property vector
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        S_hat = self.fc4(x) # output predicted structure parameters
        return S_hat