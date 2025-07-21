import torch

def reparameterize(mu, logvar):
    """
    Apply the reparameterization trick:
    z = mu + sigma * epsilon, where epsilon ~ N(0, I)

    Args:
        mu (Tensor): Mean of the latent distribution, shape [batch_size, latent_dim]
        logvar (Tensor): Log variance, shape [batch_size, latent_dim]

    Returns:
        z (Tensor): Sampled latent vector, shape [batch_size, latent_dim]
    """
    std = torch.exp(0.5 * logvar)         # convert log-variance to std dev
    eps = torch.randn_like(std)           # random noise ~ N(0,1), same shape as std
    z = mu + std * eps                    # sampled z using reparameterization trick
    return z

def make_closure(model, input_tensor, target_tensor, loss_fn, optimizer):
    """
    Returns a closure function compatible with optimizers like SLSQP from pytorch-minimize.
    """
    def closure():
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        loss.backward()
        return loss
    return closure

