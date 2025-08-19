# utils/flow_layers.py

import torch
import torch.nn as nn
from nflows.transforms import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.nn.nets import MLP, ResidualNet
from utils.planar_flow import PlanarFlow
from functools import partial


def get_flow_layers(P_dim, num_flows, flow_type="planar", hidden_dims=None):
    if flow_type == "planar":
        return nn.ModuleList([PlanarFlow(P_dim) for _ in range(num_flows)])

    elif flow_type == "maf":
        hidden_dims = hidden_dims or [128]
        transforms = [
            MaskedAffineAutoregressiveTransform(
                features=P_dim,
                hidden_features=hidden_dims[0],
                context_features=None
            ) for _ in range(num_flows)
        ]
        return CompositeTransform(transforms)

    elif flow_type == "realnvp":
        hidden_features = hidden_dims[0]
        transforms = []
        transform_net_create_fn = partial(
            ResidualNet,
            hidden_features=hidden_features,
            context_features=None
        )

        for i in range(num_flows):
            mask = torch.tensor([i % 2] * P_dim, dtype=torch.float32)

            coupling = AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=transform_net_create_fn
            )
            transforms.append(coupling)

        return CompositeTransform(transforms)

    else:
        raise ValueError(f"Unknown flow type: {flow_type}")
