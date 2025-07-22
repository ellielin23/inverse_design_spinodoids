# models/flow_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanarFlow(nn.Module):
    def __init__(self, latent_dim):
        super(PlanarFlow, self).__init__()
        self.latent_dim = latent_dim
        self.u = nn.Parameter(torch.randn(1, latent_dim))
        self.w = nn.Parameter(torch.randn(1, latent_dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        """
        Apply the planar flow transformation.
        Args:
            z: Latent input tensor of shape (batch_size, latent_dim)
        Returns:
            z_new: Transformed latent tensor
            log_det_jacobian: Log-determinant of the Jacobian
        """
        linear = torch.matmul(z, self.w.t()) + self.b  # shape: (batch_size, 1)
        activation = torch.tanh(linear)                # shape: (batch_size, 1)
        z_new = z + self.u * activation

        # Compute the Jacobian determinant
        psi = (1 - torch.tanh(linear) ** 2) * self.w    # shape: (batch_size, latent_dim)
        det_jacobian = 1 + torch.matmul(psi, self.u.t())  # shape: (batch_size, 1)
        log_det = torch.log(torch.abs(det_jacobian) + 1e-8).squeeze()

        return z_new, log_det
