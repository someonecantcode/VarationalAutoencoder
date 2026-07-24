"""
Variational Autoencoder (VAE) for MNIST.

Architecture:
  Encoder: 784 -> 512 -> 256 -> latent_dim (mu, logvar)
  Decoder: latent_dim -> 256 -> 512 -> 784

Usage:
  model = VariationalAutoencoder(input_dim=784, latent_dim=20)
  recon, mu, logvar = model(x)
  loss = vae_loss(recon, x, mu, logvar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# hyperparams
@dataclass
class autoencConfig:
    input_dim: int = 784
    hidden_layers: int = 2
    latent_dim: int = 32
    lr = 3e-4


class VariationalAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        # --- Encoder ---
        encoder_layers = []
        prev_dim = config.input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(
            *[nn.Linear(prev_dim, h_dim),
                            nn.ReLU(), for _ in range(n_flattens-1)]
        )

        # VAE-specific: predict mu and log_var from the last hidden layer
        self.fc_mu = nn.Linear(prev_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, config.latent_dim)

        # --- Decoder ---
        decoder_layers = []
        prev_dim = config.latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input into latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to input space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, reparameterize, decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    VAE loss = reconstruction loss + beta * KL divergence.

    Args:
        recon: Decoder output (logits, before sigmoid).
        x: Original input, values in [0, 1].
        mu: Predicted mean of latent distribution.
        logvar: Predicted log-variance of latent distribution.
        beta: Beta-VAE weighting factor (default 1.0).

    Returns:
        Scalar loss tensor.
    """
    recon_loss = F.binary_cross_entropy_with_logits(recon, x, reduction="sum")
    # KL divergence: KL(N(mu, std) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def read_idx3_images(path: str) -> torch.Tensor:
    """Read IDX3-format MNIST images and return normalized float32 tensor."""
    import numpy as np

    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint8, offset=16)
    images = data.reshape(-1, 784).astype(np.float32) / 255.0
    return torch.tensor(images, dtype=torch.float32)


def get_device() -> torch.device:
    """Return the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    # --- Quick smoke test ---
    device = get_device()
    print(f"Device: {device}")

    model = VariationalAutoencoder(input_dim=784, latent_dim=20).to(device)
    x = torch.randn(4, 784, device=device)

    recon, mu, logvar = model(x)
    loss = vae_loss(recon, x, mu, logvar)

    print(f"Input shape:  {x.shape}")
    print(f"Recon shape:  {recon.shape}")
    print(f"Mu shape:     {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print(f"Loss:         {loss.item():.4f}")
    print("VAE forward pass OK.")