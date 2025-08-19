# models/flow_forward.py

import math
import torch
import torch.nn as nn
from utils.flow_utils.flow_layers import get_flow_layers  # should accept context_dim if conditional

LOG_2PI = math.log(2.0 * math.pi)

class FlowForwardModel(nn.Module):
    """
    Conditional normalizing flow for p(P | S).
    We standardize P with μ(S), σ(S): z0 = (P - μ)/σ, then pass z0 through flows.
    log p(P|S) = log p0(z_k) + log|detJ_flows| - sum(log σ).
    """
    def __init__(self, S_dim, P_dim, hidden_dims, num_flows, flow_type="planar", conditional=True):
        super().__init__()
        self.S_dim = S_dim
        self.P_dim = P_dim
        self.flow_type = flow_type.lower()
        self.conditional = conditional

        # conditioner: S -> features -> (μ, logσ) for affine standardization of P
        layers, in_dim = [], S_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(in_dim, P_dim)
        self.log_sigma_layer = nn.Linear(in_dim, P_dim)

        # flow stack operating on standardized P; pass context=S if conditional
        context_dim = S_dim if conditional else 0
        self.flows = get_flow_layers(P_dim, num_flows,
                                     flow_type=self.flow_type,
                                     hidden_dims=hidden_dims,
                                     context_dim=context_dim)

    def forward(self, S, P):
        """
        Args:
            S: (B, S_dim)  conditioning
            P: (B, P_dim)  data (already normalized-to-dataset scale if you do global norm outside)
        Returns:
            z_k:   (B, P_dim) latent after last flow
            logdet_total: (B,) total log |∂z/∂P|
            mu:    (B, P_dim) affine mean
            log_sigma: (B, P_dim) affine log-std
        """
        h = self.mlp(S)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)
        sigma = torch.exp(log_sigma)

        # affine standardization (adds Jacobian term -sum log σ)
        z0 = (P - mu) / sigma                        # (B, D)
        logdet_affine = -log_sigma.sum(dim=1)        # (B,)

        # flow transform
        if isinstance(self.flows, nn.ModuleList):
            # planar stack (no context)
            z = z0
            logdet_flows = torch.zeros(z.size(0), device=z.device)
            for f in self.flows:
                z, ld = f(z)                # each returns (out, logdet) with shape (B, D) and (B,)
                logdet_flows = logdet_flows + ld
            z_k = z
        else:
            # maf / realnvp (supports context)
            if self.conditional:
                z_k, logdet_flows = self.flows(z0, context=S)
            else:
                z_k, logdet_flows = self.flows(z0)

        logdet_total = logdet_affine + logdet_flows  # (B,)

        return z_k, logdet_total, mu, log_sigma

    @staticmethod
    def standard_normal_logprob(z):
        # (B,D) -> (B,)
        return -0.5 * (z.pow(2) + LOG_2PI).sum(dim=1)

    def nll(self, S, P):
        """Convenience: negative log-likelihood for a batch."""
        z_k, logdet_total, _, _ = self(S, P)
        log_p0 = self.standard_normal_logprob(z_k)
        log_p = log_p0 + logdet_total
        return (-log_p).mean()

    @torch.no_grad()
    def sample(self, S, num_samples=1):
        """
        Sampling:
          1) draw z_k ~ N(0,I)
          2) invert flows to z0
          3) de-standardize: P = μ + σ * z0
        """
        B = S.size(0)
        h = self.mlp(S)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)
        sigma = torch.exp(log_sigma)

        z_k = torch.randn(B * num_samples, self.P_dim, device=S.device)
        S_rep = S.repeat_interleave(num_samples, dim=0)

        if isinstance(self.flows, nn.ModuleList):
            # invert planar stack: go backwards
            z = z_k
            for f in reversed(self.flows):
                z, _ = f.inverse(z)         # assume PlanarFlow.inverse returns (x, logdet)
            z0 = z
        else:
            if self.conditional:
                z0, _ = self.flows.inverse(z_k, context=S_rep)
            else:
                z0, _ = self.flows.inverse(z_k)

        mu_rep = mu.repeat_interleave(num_samples, dim=0)
        sigma_rep = sigma.repeat_interleave(num_samples, dim=0)
        P_samples = mu_rep + sigma_rep * z0
        return P_samples.view(B, num_samples, self.P_dim)
