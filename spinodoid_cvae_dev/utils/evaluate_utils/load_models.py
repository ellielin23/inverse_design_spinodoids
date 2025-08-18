# utils/evalute_utils/load_models.py

import torch
import tensorflow as tf
from utils.model_utils import get_encoder, get_decoder
from utils.fNN_utils.fNN_layers import (
    PermutationEquivariantLayer,
    DoubleContractionLayer,
    EnforceIsotropyLayer,
    NormalizationLayer
)

def load_fNN_model(path='utils/fNN_utils/max_fNN.h5'):
    custom_objects = {
        'PermutationEquivariantLayer': PermutationEquivariantLayer,
        'DoubleContractionLayer': DoubleContractionLayer,
        'EnforceIsotropyLayer': EnforceIsotropyLayer,
        'NormalizationLayer': NormalizationLayer
    }
    fNN = tf.keras.models.load_model(path, custom_objects=custom_objects)
    print("✅ Loaded Max's forward model")
    return fNN

def load_config(config_path):
    with open(config_path, "r") as f:
        lines = f.readlines()
    config = {line.split(":")[0].strip(): eval(line.split(":")[1].strip()) for line in lines}
    return config

def load_decoder(config, decoder_path, flow_type, trial, device):
    decoder = get_decoder(
        use_flow=config.get("USE_FLOW_DECODER", False),
        use_attention=config.get("USE_ATTENTION_DECODER", False),
        S_dim=config["S_DIM"],
        P_dim=config["P_DIM"],
        latent_dim=config["LATENT_DIM"],
        hidden_dims=config["DECODER_HIDDEN_DIMS"],
        num_flows=config.get("NUM_FLOWS", 4),
        dropout_prob=config.get("DROPOUT_PROB", 0.1),
        flow_type=flow_type,
        device=device
    )
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()
    print(f"✅ Loaded decoder from trial {trial}")
    return decoder

def load_encoder(config, encoder_path, trial, device):
    encoder = get_encoder(
        use_attention=config.get("USE_ATTENTION_ENCODER", False),
        S_dim=config["S_DIM"],
        P_dim=config["P_DIM"],
        latent_dim=config["LATENT_DIM"],
        hidden_dims=config["ENCODER_HIDDEN_DIMS"]
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    print(f"✅ Loaded encoder from trial {trial}")
    return encoder
