# config_dictionary.py

import itertools

# === define config options ===
latent_dims = [3, 4, 5]
encoder_dims_list = [
    [128, 64, 32],
    [256, 128, 64],
]
decoder_dims_list = [
    [128, 64, 32],
    [256, 128, 64],
]
batch_sizes = [64]
learning_rates = [1e-3, 5e-4]
betas = [1.0, 0.5]
epochs = [300, 500]

# === generate config dicts ===
CONFIGS = []
seen_configs = set()

for latent_dim, enc_dims, dec_dims, batch_size, lr, beta, ep in itertools.product(
    latent_dims, encoder_dims_list, decoder_dims_list, batch_sizes, learning_rates, betas, epochs
):
    config = {
        'latent_dim': latent_dim,
        'encoder_dims': enc_dims,
        'decoder_dims': dec_dims,
        'batch_size': batch_size,
        'learning_rate': lr,
        'beta': beta,
        'epochs': ep
    }

    # convert to a hashable representation
    config_hash = frozenset((k, tuple(v) if isinstance(v, list) else v) for k, v in config.items() if k != 'save_dir')

    if config_hash not in seen_configs:
        CONFIGS.append(config)
        seen_configs.add(config_hash)