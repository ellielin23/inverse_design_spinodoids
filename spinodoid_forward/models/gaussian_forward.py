# models/gaussian_forward.py

import torch
import torch.nn as nn

class GaussianForwardModel(nn.Module):
    """
    Gaussian probabilistic model of p(P | S), where:
    - S ∈ ℝ⁴: structure parameters (input)
    - P ∈ ℝ⁹: material properties (output)
    The model outputs a Gaussian: N(μ(S), diag(σ²(S)))
    """

    def __init__(self, S_dim=4, P_dim=9, hidden_dims=[128, 64]):
        super().__init__()
        self.S_dim = S_dim
        self.P_dim = P_dim

        # Hidden layers
        layers = []
        input_dim = S_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(input_dim, hdim))
            layers.append(nn.ReLU())
            input_dim = hdim
        self.backbone = nn.Sequential(*layers)

        # output mean and log variance
        self.mu = nn.Linear(input_dim, P_dim)
        self.log_sigma = nn.Linear(input_dim, P_dim)

    def forward(self, S):
        """
        Args:
            S (Tensor): shape (batch_size, 4)
        Returns:
            mu (Tensor): shape (batch_size, 9)
            log_sigma (Tensor): shape (batch_size, 9)
        """
        features = self.backbone(S)
        mu = self.mu(features)
        log_sigma = self.log_sigma(features)
        return mu, log_sigma
