# utils/flow_utils/planar_flow.py

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
        # === compute linear transformation ===
        linear = torch.matmul(z, self.w.t()) + self.b  # shape: (batch_size, 1)
        activation = torch.tanh(linear)                # shape: (batch_size, 1)

        # === use the "u_hat" trick to ensure invertibility ===
        wu = torch.matmul(self.w, self.u.t())          # shape: (1, 1)
        mwu = -1 + F.softplus(wu)                      # shape: (1, 1)
        u_hat = self.u + (mwu - wu) * self.w / (torch.norm(self.w, p=2) ** 2 + 1e-8)

        # === apply transformation ===
        z_new = z + u_hat * activation                 # broadcasting (batch_size, latent_dim)

        # === compute log-det-Jacobian ===
        psi = (1 - torch.tanh(linear) ** 2) * self.w.expand(z.size(0), -1)  # safe broadcasting
        det_jacobian = 1 + torch.matmul(psi, u_hat.t())  # shape: (batch_size, 1)
        log_det = torch.log(torch.abs(det_jacobian) + 1e-8).squeeze(-1)     # shape: (batch_size,)

        return z_new, log_det