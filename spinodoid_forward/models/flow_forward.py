# models/flow_forward.py

import math
import torch
import torch.nn as nn
from utils.flow_utils.flow_layers import get_flow_layers  # must accept context_dim if conditional

LOG_2PI = math.log(2.0 * math.pi)

class FlowForwardModel(nn.Module):
    """
    Conditional normalizing flow for p(P | S).

    We standardize P with μ(S), σ(S): z0 = (P - μ)/σ, then pass z0 through flows.
    log p(P|S) = log p0(z_k) + log|detJ_flows| - sum(log σ).
    """

    def __init__(self,
                 S_dim: int,
                 P_dim: int,
                 hidden_dims,
                 num_flows: int,
                 flow_type: str = "planar",
                 conditional: bool = True,
                 rho_min: float = 0.3,
                 rho_max: float = 1.0):
        super().__init__()
        self.S_dim = S_dim
        self.P_dim = P_dim
        self.flow_type = flow_type.lower()
        self.conditional = conditional
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)

        # conditioner: S -> features -> (μ, logσ)
        layers, in_dim = [], S_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(in_dim, P_dim)
        self.log_sigma_layer = nn.Linear(in_dim, P_dim)

        # flow stack on standardized P; pass context=S if conditional
        context_dim = S_dim if conditional else 0
        self.flows = get_flow_layers(
            P_dim,
            num_flows,
            flow_type=self.flow_type,
            hidden_dims=hidden_dims,
            context_dim=context_dim
        )

    def forward(self, S: torch.Tensor, P: torch.Tensor):
        """
        Args:
            S: (B, S_dim) conditioning (same scale you used during training)
            P: (B, P_dim) data (same scale you used during training)
        Returns:
            z_k:           (B, P_dim) latent after last flow
            logdet_total:  (B,) total log|∂z/∂P|
            mu:            (B, P_dim) affine mean
            log_sigma:     (B, P_dim) affine log-std
        """
        h = self.mlp(S)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)
        sigma = torch.exp(log_sigma)

        # affine standardization (adds Jacobian term -sum log σ)
        z0 = (P - mu) / sigma
        logdet_affine = -log_sigma.sum(dim=1)

        # flow transform
        if isinstance(self.flows, nn.ModuleList):
            # planar stack (no context)
            z = z0
            logdet_flows = torch.zeros(z.size(0), device=z.device)
            for f in self.flows:
                z, ld = f(z)  # each returns (out, logdet) with (B,D) and (B,)
                logdet_flows = logdet_flows + ld
            z_k = z
        else:
            # maf / realnvp (supports context)
            if self.conditional:
                z_k, logdet_flows = self.flows(z0, context=S)
            else:
                z_k, logdet_flows = self.flows(z0)

        logdet_total = logdet_affine + logdet_flows
        return z_k, logdet_total, mu, log_sigma

    @staticmethod
    def standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (z.pow(2) + LOG_2PI).sum(dim=1)

    def nll(self, S: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood for a batch."""
        z_k, logdet_total, _, _ = self(S, P)
        log_p0 = self.standard_normal_logprob(z_k)
        log_p = log_p0 + logdet_total
        return (-log_p).mean()

    @torch.no_grad()
    def sample(self,
               S: torch.Tensor,
               num_samples: int = 1,
               rho_raw: torch.Tensor | None = None,
               rho_min: float | None = None,
               rho_max: float | None = None) -> torch.Tensor:
        """
        Sampling:
          1) draw tempered base noise z_k ~ N(0, scale(ρ)^2 I)
          2) invert flows to z0
          3) de-standardize: P = μ + σ * z0

        Args:
            S:        (B, S_dim)
            num_samples: number of samples per S
            rho_raw:  (B,) or (B,1) raw ρ values (i.e., S[...,3]) in original units.
                      If None, no tempering (scale=1).
            rho_min/rho_max: optional overrides. Defaults come from the model.
        Returns:
            P_samples: (B, num_samples, P_dim) in the same scale as P used for training.
        """
        B = S.size(0)
        h = self.mlp(S)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)
        sigma = torch.exp(log_sigma)

        # === rho-based tempering of base noise (per batch) ===
        rmin = self.rho_min if rho_min is None else float(rho_min)
        rmax = self.rho_max if rho_max is None else float(rho_max)

        if rho_raw is None:
            scale = torch.ones(B, device=S.device)
        else:
            rho = rho_raw.view(B).to(S.device)
            denom = max(1e-12, (rmax - rmin))
            # scale = 1 at rho=rmin  ->  0 at rho=rmax
            scale = torch.clamp((rmax - rho) / denom, 0.0, 1.0)

        # base noise
        z_k = torch.randn(B * num_samples, self.P_dim, device=S.device)
        scale_rep = scale.repeat_interleave(num_samples, dim=0).unsqueeze(-1)
        z_k = scale_rep * z_k  # tempered base noise

        S_rep = S.repeat_interleave(num_samples, dim=0)

        # invert flows
        if isinstance(self.flows, nn.ModuleList):
            z = z_k
            for f in reversed(self.flows):
                z, _ = f.inverse(z)
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
