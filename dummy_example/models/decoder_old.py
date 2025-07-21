# models/decoder_old.py

import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, s_dim, p_dim, latent_dim):
        super(Decoder, self).__init__()
        input_dim = latent_dim + p_dim

        # first hidden layer
        self.fc1 = nn.Linear(input_dim, 128) 
        self.relu1 = nn.Tanh()

        # second hidden layer
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.Tanh()

        # third hidden layer
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.Tanh()

        # output layer: produce predicted structure parameters
        self.fc4 = nn.Linear(32, s_dim)

    def forward(self, z, P):
        x = torch.cat([z, P], dim=1) # concatenate latent vector and property vector
        x = self.relu1(self.fc1(x)) # apply ReLU to first layer output
        x = self.relu2(self.fc2(x)) # apply ReLU to second layer output
        x = self.relu3(self.fc3(x)) # apply ReLU to third layer output
        S_hat = self.fc4(x) # output predicted structure parameters
        return S_hat