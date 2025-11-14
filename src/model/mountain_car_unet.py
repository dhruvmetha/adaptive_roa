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

class MountainCarUNet(nn.Module):
    """
    Mountain Car UNet for Latent Conditional Flow Matching:
    - Takes Mountain Car state x_t (position_norm, velocity_norm), time t, latent z, condition
    - Predicts velocity in ℝ² tangent space: (dposition/dt, dvelocity/dt)

    Pure Euclidean manifold (ℝ²):
    - Input dimension: 2D (position_norm, velocity_norm)
    - Output dimension: 2D (dposition/dt, dvelocity/dt)
    - Condition dimension: 2D (same as embedded state)
    - No embedding transformation needed (unlike CartPole with SO2 angle)
    """
    def __init__(self,
                 embedded_dim: int = 2,        # Mountain Car state dimension (pure Euclidean)
                 latent_dim: int = 2,          # Latent variable dimension
                 condition_dim: int = 2,       # Condition dimension (embedded start state)
                 time_emb_dim: int = 64,       # Time embedding dimension
                 hidden_dims = [256, 512, 256], # Hidden layer dimensions
                 output_dim: int = 2,          # Output dimension (2D velocity)
                 use_input_embeddings: bool = False,
                 input_emb_dim: int = 64):
        super().__init__()
        self.embedded_dim = embedded_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_input_embeddings = use_input_embeddings
        self.input_emb_dim = input_emb_dim

        # Input embeddings for richer representations
        if use_input_embeddings:
            self.state_embedding = nn.Linear(embedded_dim, input_emb_dim)
            self.latent_embedding = nn.Linear(latent_dim, input_emb_dim)
            self.condition_embedding = nn.Linear(condition_dim, input_emb_dim)

            # Total input: 3 * input_emb_dim + time_emb_dim
            total_input_dim = 3 * input_emb_dim + time_emb_dim
        else:
            # Total input: embedded_state + time_emb + latent + condition
            total_input_dim = embedded_dim + time_emb_dim + latent_dim + condition_dim

        # Velocity predictor in ℝ² tangent space
        self.vel_head = MLP(total_input_dim, hidden_dims, output_dim)

    def forward(self,
                x_t: torch.Tensor,        # [B, 2] embedded state (position_norm, velocity_norm)
                t: torch.Tensor,          # [B] time
                z: torch.Tensor,          # [B, latent_dim] latent vector
                condition: torch.Tensor   # [B, 2] embedded start state
                ) -> torch.Tensor:
        """
        Forward pass for Mountain Car Latent Conditional Flow Matching

        Args:
            x_t: Current embedded state [B, 2]
            t: Time [B]
            z: Latent variable [B, latent_dim]
            condition: Start state condition [B, 2]

        Returns:
            Predicted velocity [B, 2] in tangent space (dposition/dt, dvelocity/dt)
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]

        if self.use_input_embeddings:
            # Rich embedding approach
            state_emb = self.state_embedding(x_t)           # [B, input_emb_dim]
            latent_emb = self.latent_embedding(z)           # [B, input_emb_dim]
            condition_emb = self.condition_embedding(condition)  # [B, input_emb_dim]

            # Concatenate all embeddings
            x_input = torch.cat([state_emb, t_emb, latent_emb, condition_emb], dim=1)
        else:
            # Simple concatenation approach
            x_input = torch.cat([x_t, t_emb, z, condition], dim=1)  # [B, total_input_dim]

        # Predict velocity in tangent space
        velocity = self.vel_head(x_input)  # [B, 2]

        return velocity

    def get_model_info(self) -> dict:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "embedded_dim": self.embedded_dim,
            "latent_dim": self.latent_dim,
            "condition_dim": self.condition_dim,
            "time_emb_dim": self.time_emb_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "use_input_embeddings": self.use_input_embeddings,
            "total_parameters": total_params,
            "model_type": "Mountain Car Latent Conditional Flow Matching"
        }
