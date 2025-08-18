# utils/loaders.py

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Sequence
import numpy as np
import torch
import torch.nn as nn


# =============== DECODERS ===============

def _mlp(in_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout and dropout > 0:
            layers += [nn.Dropout(dropout)]
        prev = h
    return nn.Sequential(*layers)

class Decoder(nn.Module):
    def __init__(self, S_dim, P_dim, latent_dim, dec_hidden_dims):
        super(Decoder, self).__init__()
        input_dim = latent_dim + P_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in dec_hidden_dims: # change
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, S_dim)

    def forward(self, z, P):
        x = torch.cat([z, P], dim=1)
        x = self.hidden_layers(x)
        S_hat = self.output_layer(x)
        return S_hat

class AttnDecoder(nn.Module):
    def __init__(self, S_dim, P_dim, latent_dim, dec_hidden_dims, dropout_prob=0.1):
        super(AttnDecoder, self).__init__()
        input_dim = latent_dim + P_dim

        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, batch_first=True)

        layers = []
        prev_dim = input_dim
        for h in dec_hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            prev_dim = h

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, S_dim)

    def forward(self, z, P):
        x = torch.cat([z, P], dim=1)  # [B, latent_dim + P_dim]
        x = x.unsqueeze(1)            # [B, 1, D]
        x, _ = self.attn(x, x, x)     # self-attention on single token
        x = x.squeeze(1)              # [B, D]
        x = self.hidden_layers(x)
        S_hat = self.output_layer(x)
        return S_hat


def get_decoder(
    use_flow: bool,                  # ignored (kept for config compatibility)
    use_attention: bool,
    S_dim: int,
    P_dim: int,
    latent_dim: int,
    hidden_dims,
    num_flows=None, dropout_prob: float = 0.1, flow_type=None,
    device="cpu", **kwargs
):
    cls = AttnDecoder if use_attention else Decoder
    dec = cls(S_dim=S_dim, P_dim=P_dim, latent_dim=latent_dim,
              dec_hidden_dims=hidden_dims, dropout_prob=dropout_prob).to(device)
    dec.eval()
    return dec


def load_decoder(config, decoder_path, device):
    dec = get_decoder(
        use_flow=False,
        use_attention=config.get("USE_ATTENTION_DECODER", False),
        S_dim=config["S_DIM"], P_dim=config["P_DIM"],
        latent_dim=config["LATENT_DIM"],
        hidden_dims=config["DECODER_HIDDEN_DIMS"],
        dropout_prob=config.get("DROPOUT_PROB", 0.1),
        device=device,
    )

    try:
        state = torch.load(str(decoder_path), map_location=device, weights_only=True)
    except TypeError:
        # older pytorch without weights_only
        state = torch.load(str(decoder_path), map_location=device)

    dec.load_state_dict(state, strict=True)
    dec.eval()
    return dec


# =============== FORWARD SURROGATE (fNN) ===============

def load_fNN_model(path: str | Path = "fNN/fNN.h5"):
    """
    Load Max's forward model with custom layers.
    TensorFlow is imported lazily to keep this module light unless needed.
    """
    try:
        import tensorflow as tf  # lazy import
    except Exception as e:
        raise ImportError("TensorFlow is required to load the fNN model. Install tensorflow>=2.x") from e

    # import custom layers only when loading the model
    try:
        from fNN.fNN_layers import (
            PermutationEquivariantLayer,
            DoubleContractionLayer,
            EnforceIsotropyLayer,
            NormalizationLayer,
        )
    except Exception as e:
        raise ImportError("Could not import custom fNN layers from fNN/fNN_layers.py") from e

    custom_objects = {
        "PermutationEquivariantLayer": PermutationEquivariantLayer,
        "DoubleContractionLayer": DoubleContractionLayer,
        "EnforceIsotropyLayer": EnforceIsotropyLayer,
        "NormalizationLayer": NormalizationLayer,
    }

    return tf.keras.models.load_model(str(path), custom_objects=custom_objects, compile=False)


# =============== release model bundle loader ===============

def load_all_models_release(models_dir: str | Path, device: str | torch.device, tags: List[str]):
    """
    Load per-tag decoder + stats from release_v1/models/<tag>/.
    Returns decoders, cfgs, P_mean, P_std, S_mean, S_std lists in tag order.
    """
    models_dir = Path(models_dir)
    decoders, cfgs, Pm, Ps, Sm, Ss = [], [], [], [], [], []
    for tag in tags:
        tdir = models_dir / tag
        cfg_path = tdir / f"config_{tag}.json"
        dec_path = tdir / f"decoder_{tag}.pt"
        Pm_path  = tdir / f"P_mean_{tag}.npy"
        Ps_path  = tdir / f"P_std_{tag}.npy"
        Sm_path  = tdir / f"S_mean_{tag}.npy"
        Ss_path  = tdir / f"S_std_{tag}.npy"

        missing = [p.name for p in [cfg_path, dec_path, Pm_path, Ps_path, Sm_path, Ss_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"[{tag}] Missing files in {tdir}: {', '.join(missing)}")

        cfg = json.loads(cfg_path.read_text())
        dec = load_decoder(cfg, dec_path, device=device)
        pmean, pstd = np.load(Pm_path), np.load(Ps_path)
        smean, sstd = np.load(Sm_path), np.load(Ss_path)
        pstd = np.where(pstd < 1e-8, 1.0, pstd)
        sstd = np.where(sstd < 1e-8, 1.0, sstd)

        decoders.append(dec); cfgs.append(cfg)
        Pm.append(pmean); Ps.append(pstd); Sm.append(smean); Ss.append(sstd)

    return decoders, cfgs, Pm, Ps, Sm, Ss
