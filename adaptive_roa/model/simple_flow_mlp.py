"""
Universal Simple Flow MLP for Latent Conditional Flow Matching.

Consolidates all system-specific UNet files into one universal implementation.
Same concat-and-MLP architecture, just parameterized by dimensions.

    Input: concat([x_t, timestep_emb(t), z, condition]) -> [B, total_input_dim]
    MLP:   Linear -> SiLU -> Linear -> SiLU -> ... -> Linear -> velocity
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal (Fourier) time embedding. t: [B] in [0,1]. Returns [B, dim]."""
    half = dim // 2
    device = t.device
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = t[:, None] * emb[None, :]  # [B, half]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SimpleFlowMLP(nn.Module):
    """
    Universal simple MLP velocity model for latent conditional flow matching.

    Works for any system (pendulum, cartpole, quadrotor 2D/3D, etc.)
    by parameterizing embedded_dim, condition_dim, output_dim, and hidden_dims.

    Forward signature: forward(x_t, t, z, condition) -> velocity
    """

    def __init__(
        self,
        embedded_dim: int,
        latent_dim: int,
        condition_dim: int,
        time_emb_dim: int = 64,
        hidden_dims: list = [256, 512, 256],
        output_dim: int = 2,
        use_input_embeddings: bool = False,
        input_emb_dim: int = 64,
    ):
        super().__init__()
        self.embedded_dim = embedded_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_input_embeddings = use_input_embeddings
        self.input_emb_dim = input_emb_dim

        if use_input_embeddings:
            self.state_embedding = nn.Linear(embedded_dim, input_emb_dim)
            self.latent_embedding = nn.Linear(latent_dim, input_emb_dim)
            self.condition_embedding = nn.Linear(condition_dim, input_emb_dim)
            total_input_dim = 3 * input_emb_dim + time_emb_dim
        else:
            total_input_dim = embedded_dim + time_emb_dim + latent_dim + condition_dim

        # Build MLP: Linear -> SiLU -> ... -> Linear
        layers = []
        last = total_input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.SiLU()]
            last = h
        layers += [nn.Linear(last, output_dim)]
        self.vel_head = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity in tangent space.

        Args:
            x_t: Embedded interpolated state [B, embedded_dim]
            t: Time parameter [B] in [0,1]
            z: Latent vector [B, latent_dim]
            condition: Embedded start state [B, condition_dim]

        Returns:
            Predicted velocity [B, output_dim] in tangent space
        """
        if x_t.dim() != 2 or x_t.shape[1] != self.embedded_dim:
            x_t = x_t.view(x_t.shape[0], -1)
        if condition.dim() != 2 or condition.shape[1] != self.condition_dim:
            condition = condition.view(condition.shape[0], -1)
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            z = z.view(z.shape[0], -1)

        t_emb = timestep_embedding(t, self.time_emb_dim)

        if self.use_input_embeddings:
            state_emb = F.silu(self.state_embedding(x_t))
            latent_emb = F.silu(self.latent_embedding(z))
            condition_emb = F.silu(self.condition_embedding(condition))
            h = torch.cat([state_emb, latent_emb, condition_emb, t_emb], dim=1)
        else:
            h = torch.cat([x_t, t_emb, z, condition], dim=1)

        return self.vel_head(h)

    def get_model_info(self) -> dict:
        """Return architecture metadata used by training logs."""
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
            "model_type": "SimpleFlowMLP (Universal)",
        }
