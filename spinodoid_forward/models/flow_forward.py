# models/flow_forward.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.flow_layers import get_flow_layers


class FlowForwardModel(nn.Module):
    """
    Forward model using Normalizing Flows (Planar, MAF, or RealNVP via nflows).
    Learns p(P | S) by transforming a base Gaussian using flows conditioned on S.
    """
    def __init__(self, S_dim, P_dim, hidden_dims, num_flows, flow_type="planar"):
        super(FlowForwardModel, self).__init__()
        self.S_dim = S_dim
        self.P_dim = P_dim
        self.num_flows = num_flows
        self.flow_type = flow_type.lower()

        # base MLP to generate mean and log_sigma from structure S
        layers = []
        input_dim = S_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.Tanh())
            input_dim = h_dim
        self.mlp = nn.Sequential(*layers)

        self.mu_layer = nn.Linear(input_dim, P_dim)
        self.log_sigma_layer = nn.Linear(input_dim, P_dim)

        # initialize flows
        self.flows = get_flow_layers(P_dim, num_flows, flow_type=flow_type, hidden_dims=hidden_dims)

    def forward(self, S):
        """
        Args:
            S (Tensor): shape (batch_size, S_dim)

        Returns:
            zk (Tensor): Transformed samples, shape (batch_size, P_dim)
            log_q (Tensor): Total log prob under the flow, shape (batch_size,)
            mu (Tensor): Base Gaussian mean, shape (batch_size, P_dim)
            log_sigma (Tensor): Base Gaussian log std, shape (batch_size, P_dim)
        """
        h = self.mlp(S)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)
        sigma = torch.exp(log_sigma)

        # sample from base Gaussian
        eps = torch.randn_like(mu)
        z0 = mu + sigma * eps

        # base log prob
        log_prob_z0 = -0.5 * (((z0 - mu) / sigma) ** 2 + 2 * log_sigma + torch.log(torch.tensor(2 * torch.pi))).sum(dim=1)

        # apply flow
        if self.flow_type == "planar":
            zk = z0
            log_det_sum = 0.
            for flow in self.flows:
                zk, log_det = flow(zk)
                log_det_sum += log_det
            log_q = log_prob_z0 - log_det_sum
        else:
            zk, log_det = self.flows.forward(z0)
            log_q = log_prob_z0 - log_det  # if log_det is already shape (batch_size,)

        return zk, log_q, mu, log_sigma
