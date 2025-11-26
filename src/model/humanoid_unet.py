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

class HumanoidUNet(nn.Module):
    """
    Humanoid UNet for Latent Conditional Flow Matching:
    - Takes embedded state x_t (67D), time t, latent z (8D), condition (67D)
    - Predicts velocity in ℝ³⁴ × S² × ℝ³⁰ tangent space: (67D velocity)

    Architecture Design:
    - Deeper than CartPole to handle 67D complexity
    - Optional input embeddings for richer representations
    - Designed for Product(Euclidean(34), Sphere(3), Euclidean(30)) manifold

    Key features:
    - Embedded state: 67D (identity for Euclidean + Sphere components)
    - Latent: 8D (larger than 2D for pendulum/cartpole)
    - Condition: 67D (start state)
    - Hidden dims: [256, 512, 1024, 1024, 512, 256] (deeper for complexity)
    - Total capacity: ~1-2M parameters (vs 500K for UniversalUNet)
    """
    def __init__(self,
                 embedded_dim: int = 67,              # Humanoid embedded state dimension
                 latent_dim: int = 8,                 # Latent variable dimension (larger for 67D)
                 condition_dim: int = 67,             # Condition dimension (embedded start state)
                 time_emb_dim: int = 128,             # Time embedding dimension (increased)
                 hidden_dims = [256, 512, 1024, 1024, 512, 256],  # Deeper for 67D
                 output_dim: int = 67,                # Output dimension (67D velocity)
                 use_input_embeddings: bool = True,   # Enable by default for 67D
                 input_emb_dim: int = 128,            # Larger embedding dimension
                 dropout: float = 0.1):               # Regularization for large model
        super().__init__()
        self.embedded_dim = embedded_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_input_embeddings = use_input_embeddings
        self.input_emb_dim = input_emb_dim
        self.dropout = dropout

        # Input embeddings for richer representations
        # For 67D state, embeddings help the network learn better representations
        if use_input_embeddings:
            self.state_embedding = nn.Sequential(
                nn.Linear(embedded_dim, input_emb_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            self.latent_embedding = nn.Sequential(
                nn.Linear(latent_dim, input_emb_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            self.condition_embedding = nn.Sequential(
                nn.Linear(condition_dim, input_emb_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            )

            # Total input: 3 * input_emb_dim + time_emb_dim
            total_input_dim = 3 * input_emb_dim + time_emb_dim
        else:
            # Total input: embedded_state + time_emb + latent + condition
            total_input_dim = embedded_dim + time_emb_dim + latent_dim + condition_dim

        # Velocity predictor in ℝ³⁴ × S² × ℝ³⁰ tangent space
        # Deeper architecture for 67D complexity
        self.vel_head = MLP(total_input_dim, hidden_dims, output_dim)

    def forward(self,
                x_t: torch.Tensor,        # [B, 67] embedded state
                t: torch.Tensor,          # [B] time
                z: torch.Tensor,          # [B, 8] latent vector
                condition: torch.Tensor   # [B, 67] embedded start state
                ) -> torch.Tensor:
        """
        Forward pass for Humanoid Latent Conditional Flow Matching

        Args:
            x_t: Current embedded state [B, 67]
                 - Dims 0-33: Euclidean components (identity)
                 - Dims 34-36: Sphere components (3D unit vector)
                 - Dims 37-66: Euclidean components (identity)
            t: Time [B] in [0, 1]
            z: Latent variable [B, 8]
            condition: Start state condition [B, 67] (same structure as x_t)

        Returns:
            Predicted velocity [B, 67] in tangent space
            - Tangent to ℝ³⁴ × S² × ℝ³⁰ manifold
            - Facebook FM's Sphere() manifold handles geodesic projection
        """
        # Ensure correct shapes
        if x_t.dim() != 2 or x_t.shape[1] != self.embedded_dim:
            x_t = x_t.view(x_t.shape[0], -1)
        if condition.dim() != 2 or condition.shape[1] != self.condition_dim:
            condition = condition.view(condition.shape[0], -1)
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            z = z.view(z.shape[0], -1)

        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]

        if self.use_input_embeddings:
            # Rich input representations
            x_emb = self.state_embedding(x_t)           # [B, input_emb_dim]
            z_emb = self.latent_embedding(z)            # [B, input_emb_dim]
            cond_emb = self.condition_embedding(condition)  # [B, input_emb_dim]

            # Concatenate all embeddings
            inp = torch.cat([x_emb, t_emb, z_emb, cond_emb], dim=1)
        else:
            # Simple concatenation
            inp = torch.cat([x_t, t_emb, z, condition], dim=1)

        # Predict velocity in tangent space
        velocity = self.vel_head(inp)  # [B, 67]

        return velocity

    def get_architecture_info(self) -> dict:
        """Get information about the model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "HumanoidUNet",
            "embedded_dim": self.embedded_dim,
            "latent_dim": self.latent_dim,
            "condition_dim": self.condition_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "time_emb_dim": self.time_emb_dim,
            "use_input_embeddings": self.use_input_embeddings,
            "input_emb_dim": self.input_emb_dim if self.use_input_embeddings else None,
            "dropout": self.dropout,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        return (f"HumanoidUNet("
                f"dims=[{self.embedded_dim}+{self.latent_dim}+{self.condition_dim}→{self.hidden_dims}→{self.output_dim}], "
                f"latent={self.latent_dim}, "
                f"use_emb={self.use_input_embeddings}, "
                f"params={total_params:,})")
