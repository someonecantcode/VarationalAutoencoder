import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.layers(x)


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.hidden_dim % (2**config.hidden_layers) == 0, "not divisible by layers"
        
        self.encode = nn.Sequential(
            FFN(config.input_dim, config.hidden_dim),
            *[FFN(config.hidden_dim // (2 ** i), config.hidden_dim // (2 ** (i+1))) for i in range(config.hidden_layers)]
        )

        z_dim = config.hidden_dim // (2 ** config.hidden_layers)

        self.mulayer = nn.Linear(z_dim, config.latent_channels)
        self.logvar = nn.Linear(z_dim, config.latent_channels)

        
        self.decode = nn.Sequential(
            FFN(config.latent_channels, z_dim),
            *[FFN(z_dim * (2 ** i), z_dim*(2 ** (i+1))) for i in range(config.hidden_layers)],
            nn.Linear(config.hidden_dim, config.input_dim), # proj back up
        )

    def forward(self, x, targets=None, beta=1.0):
        x = self.encode(x)
        mu = self.mulayer(x)
        logvar = self.logvar(x)
        
        reparam = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        output = self.decode(reparam)
        
        if targets is None:
            loss = None
        else:
            bce_loss = F.binary_cross_entropy_with_logits(output, targets, reduction="sum") / self.config.batch_size
            # mse_loss = F.mse_loss(input=output, target=targets, reduction="sum") / batch_size
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / self.config.batch_size
            
            loss = bce_loss + (beta) * kl_loss
        return output, loss, bce_loss, kl_loss
