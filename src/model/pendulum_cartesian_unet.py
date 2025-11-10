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

class PendulumCartesianUNet(nn.Module):
    """
    Pendulum Cartesian UNet for Conditional Flow Matching:
    - Takes embedded state x_t (x, y, ẋ, ẏ), time t, and condition
    - Predicts velocity in ℝ⁴ tangent space: (dx/dt, dy/dt, dẋ/dt, dẏ/dt)

    Key differences from angular pendulum:
    - All Euclidean manifold (no SO2 components)
    - Input dimension: 4D (x, y, ẋ, ẏ) - identity embedding
    - Output dimension: 4D (dx/dt, dy/dt, dẋ/dt, dẏ/dt)
    - Condition dimension: 4D (same as embedded state)
    """
    def __init__(self,
                 embedded_dim: int = 4,        # Pendulum Cartesian embedded state dimension
                 condition_dim: int = 4,       # Condition dimension (embedded start state)
                 time_emb_dim: int = 64,       # Time embedding dimension
                 hidden_dims = [256, 512, 256], # Hidden layer dimensions
                 output_dim: int = 4,          # Output dimension (4D velocity)
                 use_input_embeddings: bool = False,
                 input_emb_dim: int = 64):
        super().__init__()
        self.embedded_dim = embedded_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_input_embeddings = use_input_embeddings
        self.input_emb_dim = input_emb_dim

        # Input embeddings for richer representations
        if use_input_embeddings:
            self.state_embedding = nn.Linear(embedded_dim, input_emb_dim)
            self.condition_embedding = nn.Linear(condition_dim, input_emb_dim)

            # Total input: state_emb + time_emb + condition_emb
            total_input_dim = 2 * input_emb_dim + time_emb_dim
        else:
            # Total input: embedded_state + time_emb + condition
            total_input_dim = embedded_dim + time_emb_dim + condition_dim

        # Velocity predictor in ℝ⁴ tangent space
        self.vel_head = MLP(total_input_dim, hidden_dims, output_dim)

    def forward(self,
                x_t: torch.Tensor,        # [B, 4] embedded state (x, y, ẋ, ẏ)
                t: torch.Tensor,          # [B] time
                condition: torch.Tensor   # [B, 4] embedded start state
                ) -> torch.Tensor:
        """
        Forward pass for Pendulum Cartesian Conditional Flow Matching

        Args:
            x_t: Current embedded state [B, 4]
            t: Time [B]
            condition: Start state condition [B, 4]

        Returns:
            Predicted velocity [B, 4] in tangent space (dx/dt, dy/dt, dẋ/dt, dẏ/dt)
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]

        if self.use_input_embeddings:
            # Rich embedding approach
            state_emb = self.state_embedding(x_t)           # [B, input_emb_dim]
            condition_emb = self.condition_embedding(condition)  # [B, input_emb_dim]
            x_input = torch.cat([state_emb, t_emb, condition_emb], dim=1)
        else:
            # Simple concatenation approach
            x_input = torch.cat([x_t, t_emb, condition], dim=1)

        # Predict velocity in tangent space
        velocity = self.vel_head(x_input)  # [B, 4]

        return velocity

    def get_model_info(self) -> dict:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "embedded_dim": self.embedded_dim,
            "condition_dim": self.condition_dim,
            "time_emb_dim": self.time_emb_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "use_input_embeddings": self.use_input_embeddings,
            "total_parameters": total_params,
            "model_type": "Pendulum Cartesian Conditional Flow Matching"
        }
