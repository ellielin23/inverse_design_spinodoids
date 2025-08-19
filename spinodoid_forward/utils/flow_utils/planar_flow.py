# utils/planar_flow.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanarFlow(nn.Module):
    """
    Implements a single Planar Flow transformation:
        z' = z + u * tanh(w^T z + b)
    """
    def __init__(self, dim):
        """
        Args:
            dim (int): Dimensionality of the latent space z
        """
        super(PlanarFlow, self).__init__()
        self.dim = dim
        self.u = nn.Parameter(torch.randn(dim))
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        """
        Apply the planar flow transformation to input z

        Args:
            z (Tensor): shape (batch_size, dim)

        Returns:
            z_new (Tensor): Transformed sample
            log_det_jacobian (Tensor): Log-determinant of the Jacobian
        """
        # inner linear part: w^T z + b
        linear = F.linear(z, self.w.unsqueeze(0), self.b)  # shape: (batch_size, 1)
        h = torch.tanh(linear)                            # shape: (batch_size, 1)
        z_new = z + self.u * h                            # broadcasting u over batch

        # compute the derivative h' for Jacobian
        psi = (1 - torch.tanh(linear) ** 2) * self.w      # shape: (batch_size, dim)

        # compute log-determinant: log |1 + u^T psi|
        uT_psi = torch.matmul(psi, self.u.unsqueeze(-1)).squeeze(-1)  # shape: (batch_size,)
        log_abs_det = torch.log(torch.abs(1 + uT_psi) + 1e-8)         # small epsilon for stability

        return z_new, log_abs_det
