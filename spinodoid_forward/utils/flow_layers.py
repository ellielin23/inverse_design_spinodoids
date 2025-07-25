# utils/flow_layers.py

import torch
import torch.nn as nn
from nflows.transforms import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.nn.nets import MLP
from utils.planar_flow import PlanarFlow


def get_flow_layers(P_dim, num_flows, flow_type="planar"):
    """
    Returns a flow stack based on the requested flow_type.

    Args:
        P_dim (int): Dimensionality of the target property vector.
        num_flows (int): Number of flow layers to stack.
        flow_type (str): One of ['planar', 'maf', 'realnvp']

    Returns:
        nn.ModuleList or CompositeTransform
    """
    flow_type = flow_type.lower()

    if flow_type == "planar":
        return nn.ModuleList([PlanarFlow(P_dim) for _ in range(num_flows)])

    elif flow_type == "maf":
        hidden_features = 128
        transforms = []
        for _ in range(num_flows):
            maf = MaskedAffineAutoregressiveTransform(
                features=P_dim,
                hidden_features=hidden_features,
                context_features=None
            )
            transforms.append(maf)
        return CompositeTransform(transforms)

    elif flow_type == "realnvp":
        hidden_features = 128
        transforms = []
        for i in range(num_flows):
            # alternate binary mask (0,1,0,1,... or 1,0,1,0,...)
            mask = torch.tensor([(i + j) % 2 for j in range(P_dim)], dtype=torch.float32)

            coupling = AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=lambda in_features, out_features: MLP(
                    [hidden_features, hidden_features],
                    in_features=in_features,
                    out_features=out_features,
                    activation=nn.ReLU()
                )
            )
            transforms.append(coupling)
        return CompositeTransform(transforms)

    else:
        raise ValueError(f"Unknown flow type: {flow_type}")
