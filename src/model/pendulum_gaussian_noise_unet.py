"""
1D UNet for Pendulum Gaussian Noise Flow Matching

SIMPLIFIED model WITHOUT:
- Latent variables
- Conditioning on start state

Model signature: f(x_embedded, t) → velocity
"""
import torch
import torch.nn as nn


class PendulumGaussianNoiseUNet1D(nn.Module):
    """
    Simplified 1D UNet for Pendulum Gaussian Noise Flow Matching

    Input: embedded state (3D) + time (64D)
    Output: velocity in tangent space (2D)

    NO latent variables, NO conditioning - simpler architecture!
    """

    def __init__(self,
                 embedded_dim: int = 3,      # (sin θ, cos θ, θ̇_norm)
                 time_emb_dim: int = 64,
                 hidden_dims: list = [256, 512, 256],
                 output_dim: int = 2):       # (dθ, dθ̇)

        super().__init__()

        self.embedded_dim = embedded_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Time embedding (sinusoidal)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim // 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU()
        )

        # Total input dimension (NO latent, NO condition!)
        total_input_dim = embedded_dim + time_emb_dim

        # UNet architecture
        layers = []
        prev_dim = total_input_dim

        # Encoder
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - SIMPLIFIED signature!

        Args:
            x: Embedded state [B, 3] (sin θ, cos θ, θ̇_norm)
            t: Time [B] or [B, 1]

        Returns:
            Velocity [B, 2] (dθ, dθ̇)
        """
        # Ensure t is [B, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Time embedding
        t_emb = self.time_mlp(t)

        # Concatenate inputs (NO latent, NO condition!)
        combined = torch.cat([x, t_emb], dim=-1)

        # Forward through network
        return self.network(combined)

    def get_model_info(self):
        """Return model information"""
        return {
            'model_type': self.__class__.__name__,
            'embedded_dim': self.embedded_dim,
            'time_emb_dim': self.time_emb_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
