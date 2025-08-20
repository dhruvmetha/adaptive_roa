import torch
import torch.nn as nn

class LatentEncoder(nn.Module):
    """VAE-style encoder for latent variables"""
    
    def __init__(self, in_dim=4, latent_dim=8, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x0, y):
        """
        Encode (x0, y) pair to latent parameters
        x0: [B, 2] = [θ0, ω0] in raw coordinates
        y:  [B, 2] = [θ1, ω1] in raw coordinates
        Returns: mu [B, latent_dim], logvar [B, latent_dim]
        """
        h = self.net(torch.cat([x0, y], dim=-1))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

def reparameterize(mu, logvar, std_min=1e-5):
    """Reparameterization trick for VAE sampling"""
    std = (0.5 * logvar).exp().clamp_min(std_min)
    eps = torch.randn_like(std)
    return mu + eps * std