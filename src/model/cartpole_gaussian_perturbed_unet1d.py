"""
CartPole Gaussian-Perturbed Flow Matching UNet

Simplified model WITHOUT latent variables or conditioning.
Input: embedded state + time only
Output: velocity field
"""
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
        emb = F.pad(emb, (0, 1))
    return emb


class MLP(nn.Module):
    """Multi-layer perceptron with SiLU activations"""
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


class CartPoleGaussianPerturbedUNet1D(nn.Module):
    """
    Simplified CartPole Flow Matching UNet (Gaussian-Perturbed Variant)

    Key simplifications from latent conditional version:
    - NO latent variable input (removed z)
    - NO conditioning input (removed start state condition)
    - Simpler model signature: f(x_t, t) → velocity

    Architecture:
    - Input: embedded CartPole state (5D) + time embedding (64D)
    - Hidden layers: [256, 512, 1024, 512, 256] with SiLU activations
    - Output: 4D velocity in tangent space (dx/dt, dθ/dt, dẋ/dt, dθ̇/dt)

    Manifold: ℝ²×S¹×ℝ
    - Input state: [x_norm, sin(θ), cos(θ), ẋ_norm, θ̇_norm]
    - Output velocity: [dx/dt, dθ/dt, dẋ/dt, dθ̇/dt]
    """

    def __init__(self,
                 embedded_dim: int = 5,        # CartPole embedded state dimension
                 time_emb_dim: int = 64,       # Time embedding dimension
                 hidden_dims = [256, 512, 1024, 512, 256],  # Hidden layer dimensions
                 output_dim: int = 4,          # Output dimension (4D velocity)
                 use_input_embeddings: bool = False,
                 input_emb_dim: int = 64):
        super().__init__()
        self.embedded_dim = embedded_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_input_embeddings = use_input_embeddings
        self.input_emb_dim = input_emb_dim

        # Input embeddings for richer representations (optional)
        if use_input_embeddings:
            self.state_embedding = nn.Linear(embedded_dim, input_emb_dim)
            # Total input: input_emb_dim + time_emb_dim
            total_input_dim = input_emb_dim + time_emb_dim
        else:
            # Simple concatenation: embedded_state + time_emb
            # MUCH SIMPLER than latent conditional (no z, no condition)
            total_input_dim = embedded_dim + time_emb_dim

        # Velocity predictor in ℝ²×S¹×ℝ tangent space
        self.vel_head = MLP(total_input_dim, hidden_dims, output_dim)

    def forward(self,
                x_t: torch.Tensor,        # [B, 5] embedded state
                t: torch.Tensor           # [B] time
                ) -> torch.Tensor:
        """
        Forward pass for Gaussian-Perturbed Flow Matching

        SIMPLIFIED SIGNATURE (no latent z, no condition):
        - Old signature: forward(x_t, t, z, condition)
        - New signature: forward(x_t, t)

        Args:
            x_t: Current embedded state [B, 5] as (x_norm, sin(θ), cos(θ), ẋ_norm, θ̇_norm)
            t: Time [B]

        Returns:
            Predicted velocity [B, 4] in tangent space (dx/dt, dθ/dt, dẋ/dt, dθ̇/dt)
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]

        if self.use_input_embeddings:
            # Rich embedding approach
            state_emb = self.state_embedding(x_t)  # [B, input_emb_dim]
            x_input = torch.cat([state_emb, t_emb], dim=1)
        else:
            # Simple concatenation (default)
            x_input = torch.cat([x_t, t_emb], dim=1)  # [B, embedded_dim + time_emb_dim]

        # Predict velocity in tangent space
        velocity = self.vel_head(x_input)  # [B, 4]

        return velocity

    def get_model_info(self) -> dict:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "embedded_dim": self.embedded_dim,
            "time_emb_dim": self.time_emb_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "use_input_embeddings": self.use_input_embeddings,
            "total_parameters": total_params,
            "model_type": "CartPole Gaussian-Perturbed Flow Matching",
            "differences_from_latent_conditional": [
                "No latent variable input (removed z)",
                "No conditioning input (removed start state)",
                "Simpler model signature: f(x_t, t) only",
                f"Fewer parameters: ~{total_params:,} vs ~2M in latent conditional"
            ]
        }
