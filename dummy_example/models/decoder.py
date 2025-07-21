# models/decoder.py

import torch
import torch.nn as nn

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

        # first hidden layer
        self.fc1 = nn.Linear(input_dim, 128) 
        self.relu1 = nn.Tanh()
        # self.dropout = nn.Dropout(0.1)  # you can tune 0.1 → 0.2

        # second hidden layer
        self.fc2 = nn.Linear(128, 64) 
        self.relu2 = nn.Tanh()

        # third hidden layer
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.Tanh()

        # output layer: produce predicted structure parameters
        self.fc4 = nn.Linear(32, S_dim)

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
        x = self.relu1(self.fc1(x)) # apply ReLU activation to the output of the first layer
        # x = self.dropout(x)
        x = self.relu2(self.fc2(x)) # apply ReLU activation to the output of the second layer
        x = self.relu3(self.fc3(x)) # apply ReLU activation to the output of the third layer
        S_hat = self.fc4(x) # output predicted structure parameters
        return S_hat