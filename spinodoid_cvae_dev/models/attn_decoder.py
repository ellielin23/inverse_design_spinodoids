import torch
import torch.nn as nn

class AttnDecoder(nn.Module):
    """
    Attention-only conditional decoder (no flow).
    Takes latent z and target properties P, attends over the fused token,
    then maps through MLP → S_hat.
    """

    def __init__(self, S_dim, P_dim, latent_dim, dec_hidden_dims, dropout_prob=0.1):
        super(AttnDecoder, self).__init__()
        input_dim = latent_dim + P_dim

        # === attention layer ===
        # num_heads=1 so embed_dim doesn't need to be divisible by >1
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, batch_first=True)

        # === fully connected decoder network (with attention) ===
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
        """
        Args:
            z (torch.Tensor): [batch_size, latent_dim]
            P (torch.Tensor): [batch_size, P_dim]
        Returns:
            S_hat (torch.Tensor): [batch_size, S_dim]
        """
        x = torch.cat([z, P], dim=1)
        x = x.unsqueeze(1)            
        x, _ = self.attn(x, x, x)     
        x = x.squeeze(1)              
        x = self.hidden_layers(x)
        S_hat = self.output_layer(x)
        return S_hat
