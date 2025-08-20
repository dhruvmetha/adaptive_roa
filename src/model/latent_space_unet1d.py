import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal (Fourier) time embedding.
    t: [B] in [0,1]. Returns [B, dim].
    """
    half = dim // 2
    device = t.device
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = t[:, None] * emb[None, :]  # [B, half]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.SiLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class LatentCircularUNet1D(nn.Module):
    """
    Latent model for circular FM:
      - Encoder E: R^3 (embedded state) -> R^L
      - Decoder D: R^L -> R^3 (embedded state)
      - Velocity head F: predicts v_z given (z_t, t_emb, cond=z0)
    """
    def __init__(self, latent_dim: int = 8, hidden_dims=(128, 128), time_emb_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim

        # enc/dec (tiny MLPs — adjust if needed)
        self.encoder = MLP(3, (64, 64), latent_dim)
        self.decoder = MLP(latent_dim, (64, 64), 3)

        # velocity predictor in latent space
        in_dim = latent_dim + latent_dim + time_emb_dim  # z_t + cond + time_emb
        self.vel_head = MLP(in_dim, hidden_dims, latent_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        x:      expected to be z_t (latent) [B, L]
        t:      [B] in [0,1]
        cond:   expected to be z0 (latent) [B, L]
        returns predicted v_z [B, L]
        """
        if x.dim() != 2:
            x = x.view(x.shape[0], -1)
        if condition.dim() != 2:
            condition = condition.view(condition.shape[0], -1)

        t_emb = timestep_embedding(t, self.time_emb_dim)
        h = torch.cat([x, condition, t_emb], dim=1)
        v = self.vel_head(h)
        return v

    # helper utils (exposed for inference convenience)
    def encode(self, x_emb: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_emb)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)