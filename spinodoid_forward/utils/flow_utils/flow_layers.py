# utils/flow_layers.py

import torch
import torch.nn as nn
from functools import partial

from nflows.transforms import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
from nflows.nn.nets import ResidualNet

from .planar_flow import PlanarFlow

def _alt_mask(D, start_with_one=False):
    # 0101... or 1010...
    base = torch.tensor([0, 1], dtype=torch.float32)
    m = base.repeat((D + 1) // 2)[:D]
    if start_with_one:
        m = 1.0 - m
    return m

def get_flow_layers(P_dim, num_flows, flow_type="planar", hidden_dims=None, context_dim=0):
    """
    Returns:
      - 'planar'  -> nn.ModuleList of PlanarFlow (no context)
      - 'maf'/'realnvp' -> CompositeTransform that supports (x, context=...)
    """
    flow_type = flow_type.lower()
    hidden_dims = hidden_dims or [128]
    hidden = hidden_dims[0]
    ctx = context_dim if context_dim and context_dim > 0 else None

    if flow_type == "planar":
        # planar flow usually doesn't use context; FlowForwardModel should loop over ModuleList
        return nn.ModuleList([PlanarFlow(P_dim) for _ in range(num_flows)])

    elif flow_type == "maf":
        # Add permutation between autoregressive transforms (reverse or random)
        transforms = []
        for k in range(num_flows):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=P_dim,
                    hidden_features=hidden,
                    context_features=ctx,
                )
            )
            transforms.append(RandomPermutation(features=P_dim))
        return CompositeTransform(transforms)

    elif flow_type == "realnvp":
        # alternating binary masks + permutations between couplings
        transforms = []

        # residualNet factory expected signature: (in_features, out_features, **kwargs)
        def make_resnet(in_features, out_features):
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden,
                context_features=ctx,
                num_blocks=2,
                activation=torch.nn.ReLU(),
                dropout_probability=0.0,
                use_batch_norm=False,
            )

        for k in range(num_flows):
            mask = _alt_mask(P_dim, start_with_one=bool(k % 2))
            coupling = AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=make_resnet,
            )
            transforms.append(coupling)
            transforms.append(ReversePermutation(features=P_dim))
        return CompositeTransform(transforms)

    else:
        raise ValueError(f"Unknown flow type: {flow_type}")