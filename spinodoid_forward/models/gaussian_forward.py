# models/gaussian_forward.py
import math
import torch
import torch.nn as nn

class GaussianForwardModel(nn.Module):
    """
    p(P | S) = N( μ(S), diag(σ^2(S)) )
    Expects S and P to be normalized outside this module.
    """
    def __init__(self, S_dim=4, P_dim=9, hidden_dims=[128, 64], sigma_min=1e-3):
        super().__init__()
        self.S_dim, self.P_dim = S_dim, P_dim
        self.sigma_min = sigma_min
        self.log_sigma_min = math.log(sigma_min)

        layers = []
        in_dim = S_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        self.mu = nn.Linear(in_dim, P_dim)
        self.log_sigma = nn.Linear(in_dim, P_dim)

        # init: start with modest variances (log_sigma ~ -2 → sigma ~ 0.14)
        nn.init.constant_(self.log_sigma.bias, -2.0)

    def forward(self, S):
        """
        S: (B, S_dim) -> returns (mu, log_sigma) each (B, P_dim)
        """
        h = self.backbone(S)
        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        # numerical safety
        log_sigma = torch.clamp(log_sigma, min=self.log_sigma_min)
        return mu, log_sigma
