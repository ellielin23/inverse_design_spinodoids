# models/encoder.py

import torch
import torch.nn as nn

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
        super(Encoder, self).__init__() # initialize the base nn.Module class
        input_dim = S_dim + P_dim

        # first hidden layer: maps input to 64 hidden units
        self.fc1 = nn.Linear(input_dim, 128) # fully connected (linear) layer 1
        self.relu1 = nn.Tanh() # activation function

        # second hidden layer: maps input to 64 hidden units
        self.fc2 = nn.Linear(128, 64) # fully connected (linear) layer 1
        self.relu2 = nn.Tanh() # activation function

        # output layers: produce the parameters of the latent distribution
        self.fc_mu = nn.Linear(64, latent_dim) # outputs mean vector mu of the latent gaussian
        self.fc_logvar = nn.Linear(64, latent_dim) # outputs log(sigma^2) for vairance

    def forward(self, S, P):
        """
        Forward pass through the encoder.

        Args:
            S (torch.Tensor): Structure parameters tensor of shape (batch_size, S_dim)
            P (torch.Tensor): Target properties tensor of shape (batch_size, P_dim)

        Returns:
            tuple: Mean and log variance of the latent distribution
        """
        x = torch.cat([S, P], dim=1) # concatenate structure and property vectors
        x = self.relu1(self.fc1(x)) # apply ReLU activation to the output of the first layer
        x = self.relu2(self.fc2(x)) # apply ReLU activation to the output of the second layer
        mu = self.fc_mu(x) # output mean vector mu of the latent gaussian
        logvar = self.fc_logvar(x) # output log variance of the latent gaussian
        return mu, logvar

