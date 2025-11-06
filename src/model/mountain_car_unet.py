"""Mountain Car UNet Model

System-specific UNet for Mountain Car latent conditional flow matching.

Following the pattern from CartPoleUNet and HumanoidUNet.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal (Fourier) time embedding.

    Args:
        t: Time tensor [B]
        dim: Embedding dimension

    Returns:
        Time embeddings [B, dim]
    """
    half = dim // 2
    device = t.device
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class MLP(nn.Module):
    """Multi-layer perceptron with SiLU activation."""

    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int):
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
    """UNet model for Mountain Car conditional flow matching.

    State: [position, velocity] (2D pure Euclidean)
    Embedded state: [position, velocity] (2D, no sin/cos needed)

    Model input: [embedded_state, time_embedding, condition]
    Model output: velocity field [d_position, d_velocity]

    Note: NO latent variables (following Humanoid pattern)
    """

    def __init__(
        self,
        embedded_dim: int = 2,
        condition_dim: int = 2,
        time_emb_dim: int = 128,
        hidden_dims: List[int] = [256, 512, 256],
        output_dim: int = 2,
        use_input_embeddings: bool = False,
        input_emb_dim: int = 128
    ):
        """Initialize MountainCar UNet.

        Args:
            embedded_dim: Dimension of embedded state (2 for MountainCar)
            condition_dim: Dimension of conditioning (2 for MountainCar)
            time_emb_dim: Dimension of time embedding
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (2 for MountainCar)
            use_input_embeddings: Whether to use rich input embeddings
            input_emb_dim: Dimension of input embeddings (if used)
        """
        super().__init__()

        self.embedded_dim = embedded_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_input_embeddings = use_input_embeddings
        self.input_emb_dim = input_emb_dim

        # Calculate total input dimension
        if use_input_embeddings:
            # Rich embedding path: embed each component separately
            self.state_embedding = nn.Linear(embedded_dim, input_emb_dim)
            self.condition_embedding = nn.Linear(condition_dim, input_emb_dim)
            total_input_dim = time_emb_dim + 2 * input_emb_dim
        else:
            # Simple concatenation path
            total_input_dim = embedded_dim + time_emb_dim + condition_dim

        # Velocity prediction head
        self.vel_head = MLP(total_input_dim, hidden_dims, output_dim)

        print(f"✅ Initialized MountainCarUNet:")
        print(f"   Embedded dim: {embedded_dim}")
        print(f"   Condition dim: {condition_dim}")
        print(f"   Time embedding dim: {time_emb_dim}")
        print(f"   Hidden dims: {hidden_dims}")
        print(f"   Output dim: {output_dim}")
        print(f"   Total input dim: {total_input_dim}")
        print(f"   Use input embeddings: {use_input_embeddings}")

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total parameters: {total_params:,}")

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_t: Current state [B, embedded_dim]
            t: Time [B]
            condition: Conditioning (start state) [B, condition_dim]

        Returns:
            Velocity field [B, output_dim]
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)

        if self.use_input_embeddings:
            # Rich embedding path
            x_emb = self.state_embedding(x_t)
            c_emb = self.condition_embedding(condition)
            x_input = torch.cat([t_emb, x_emb, c_emb], dim=1)
        else:
            # Simple concatenation path
            x_input = torch.cat([x_t, t_emb, condition], dim=1)

        # Predict velocity
        velocity = self.vel_head(x_input)
        return velocity

    def get_model_info(self) -> dict:
        """Get model architecture information.

        Returns:
            Dictionary with model specifications
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'MountainCarUNet',
            'embedded_dim': self.embedded_dim,
            'condition_dim': self.condition_dim,
            'time_emb_dim': self.time_emb_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'use_input_embeddings': self.use_input_embeddings,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
