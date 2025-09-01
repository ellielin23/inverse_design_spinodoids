# utils/model_utils.py

import torch
from models.decoder import Decoder
from models.flow_decoder import FlowDecoder
from models.flow_attn_decoder import FlowAttnDecoder
from models.attn_decoder import AttnDecoder
from models.encoder import Encoder
from models.attn_encoder import AttnEncoder

def get_decoder(use_flow, use_attention, S_dim, P_dim, latent_dim,
                hidden_dims, num_flows=4, dropout_prob=0.1, flow_type="planar", device="cpu"):
    if use_flow:
        if use_attention:
            decoder = FlowAttnDecoder(
                S_dim=S_dim, P_dim=P_dim, latent_dim=latent_dim,
                dec_hidden_dims=hidden_dims, num_flows=num_flows,
                dropout_prob=dropout_prob, flow_type=flow_type
            )
        else:
            decoder = FlowDecoder(
                S_dim=S_dim, P_dim=P_dim, latent_dim=latent_dim,
                dec_hidden_dims=hidden_dims, num_flows=num_flows,
                dropout_prob=dropout_prob, flow_type=flow_type
            )
    else:
        if use_attention:
            decoder = AttnDecoder(
                S_dim=S_dim, P_dim=P_dim, latent_dim=latent_dim,
                dec_hidden_dims=hidden_dims, dropout_prob=dropout_prob
            )
        else:
            decoder = Decoder(S_dim, P_dim, latent_dim, hidden_dims)

    return decoder.to(device)


def get_encoder(use_attention, S_dim, P_dim, latent_dim, hidden_dims):
    """
    Returns the appropriate encoder instance based on configuration.

    Args:
        use_attention (bool): If True, use attention-based encoder.
        S_dim (int): Structure parameter dimension.
        P_dim (int): Property dimension.
        latent_dim (int): Latent space dimension.
        hidden_dims (list): Hidden layer dimensions.
    """
    if use_attention:
        return AttnEncoder(S_dim, P_dim, latent_dim, hidden_dims)
    else:
        return Encoder(S_dim, P_dim, latent_dim, hidden_dims)
