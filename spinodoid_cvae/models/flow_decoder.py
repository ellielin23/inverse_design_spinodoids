import torch
import torch.nn as nn
from models.flow_layers import PlanarFlow

class FlowDecoder(nn.Module):
    def __init__(self, S_dim, P_dim, latent_dim, dec_hidden_dims, num_flows=4):
        super(FlowDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_flows = num_flows

        # Flow layers
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(num_flows)])

        # Fully connected decoder network
        input_dim = latent_dim + P_dim
        layers = []
        prev_dim = input_dim
        for h in dec_hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, S_dim)

    def forward(self, z0, P):
        log_det_sum = 0.0
        z = z0

        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det

        x = torch.cat([z, P], dim=1)
        x = self.hidden_layers(x)
        S_hat = self.output_layer(x)

        return S_hat, log_det_sum
