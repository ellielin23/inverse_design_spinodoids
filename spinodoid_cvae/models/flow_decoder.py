import torch
import torch.nn as nn
from utils.flow_utils.flow_layers import get_flow_layers
from utils.flow_utils.planar_flow import PlanarFlow

class FlowDecoder(nn.Module):
    def __init__(self, S_dim, P_dim, latent_dim, dec_hidden_dims, num_flows=4, dropout_prob=0.1, flow_type="planar"):
        super(FlowDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.flow_type = flow_type.lower()

        # === Flow layers ===
        self.flows = get_flow_layers(latent_dim, num_flows, flow_type=self.flow_type, hidden_dims=dec_hidden_dims)

        # === Fully connected decoder network ===
        input_dim = latent_dim + P_dim
        layers = []
        prev_dim = input_dim
        for h in dec_hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_prob))
            prev_dim = h
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, S_dim)

    def forward(self, z0, P):
        """
        Args:
            z0: Latent vector (batch_size, latent_dim)
            P:  Target property vector (batch_size, P_dim)
        Returns:
            S_hat: Predicted structure (batch_size, S_dim)
            log_det_sum: Sum of log-determinants from the flow
        """
        if isinstance(self.flows, nn.ModuleList):  # PlanarFlow
            log_det_sum = 0.0
            z = z0
            for flow in self.flows:
                z, log_det = flow(z)
                log_det_sum += log_det
        else:
            # nflows CompositeTransform (RealNVP, MAF)
            z, log_det = self.flows(z0)
            log_det_sum = log_det

        # Decoder
        x = torch.cat([z, P], dim=1)
        x = self.hidden_layers(x)
        S_hat = self.output_layer(x)

        return S_hat, log_det_sum
