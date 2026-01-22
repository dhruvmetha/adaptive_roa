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


class Quadrotor3DUNet(nn.Module):
    """
    Quadrotor 3D UNet for Latent Conditional Flow Matching:
    - Takes 13D embedded state x_t, time t, latent z, and condition (start state)
    - Predicts velocity directly in tangent space (12D)

    State format (representation): (x, y, z, qw, qx, qy, qz, ẋ, ẏ, ż, p, q, r) - 13D
    Tangent format (output): (v_x, v_y, v_z, ω_x, ω_y, ω_z, a_x, a_y, a_z, α_p, α_q, α_r) - 12D

    Key dimensions:
    - Input/embedded dimension: 13 (state in representation space)
    - Output dimension: 12 (velocity in tangent space)

    Note on SO(3) geometry:
    - SO(3) has 4D quaternion representation but 3D tangent space (rotation vectors)
    - FB FM's Product manifold with (SO3(), 4, 3) uses logmap/expmap for this conversion
    - The model outputs 12D tangent velocity directly; expmap converts back to 13D state

    The model predicts tangent-space velocities:
    - Position velocity: (v_x, v_y, v_z) ∈ ℝ³
    - Rotation velocity: (ω_x, ω_y, ω_z) ∈ so(3) - axis-angle representation
    - Linear velocity derivative: (a_x, a_y, a_z) ∈ ℝ³
    - Angular velocity derivative: (α_p, α_q, α_r) ∈ ℝ³
    """

    def __init__(self,
                 embedded_dim: int = 13,       # Quadrotor state dimension (representation)
                 latent_dim: int = 4,          # Latent variable dimension
                 condition_dim: int = 13,      # Condition dimension (embedded start state)
                 time_emb_dim: int = 64,       # Time embedding dimension
                 hidden_dims=[512, 1024, 512], # Hidden layer dimensions (larger for 13D)
                 output_dim: int = 12,         # Output dimension (12D tangent space)
                 use_input_embeddings: bool = False,
                 input_emb_dim: int = 128):
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

        # Velocity predictor
        self.vel_head = MLP(total_input_dim, hidden_dims, output_dim)

    def forward(self,
                x_t: torch.Tensor,        # [B, 13] state (embedded representation)
                t: torch.Tensor,          # [B] time
                z: torch.Tensor,          # [B, latent_dim] latent vector
                condition: torch.Tensor   # [B, 13] start state condition
                ) -> torch.Tensor:
        """
        Forward pass for Quadrotor 3D Latent Conditional Flow Matching

        Args:
            x_t: Current state [B, 13] in representation space
            t: Time [B]
            z: Latent variable [B, latent_dim]
            condition: Start state condition [B, 13]

        Returns:
            Predicted velocity [B, 12] in tangent space
            (FB FM expmap converts to 13D state for integration)
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]

        if self.use_input_embeddings:
            # Rich embedding approach
            state_emb = self.state_embedding(x_t)              # [B, input_emb_dim]
            latent_emb = self.latent_embedding(z)              # [B, input_emb_dim]
            condition_emb = self.condition_embedding(condition) # [B, input_emb_dim]

            # Concatenate all embeddings
            x_input = torch.cat([state_emb, t_emb, latent_emb, condition_emb], dim=1)
        else:
            # Simple concatenation approach
            x_input = torch.cat([x_t, t_emb, z, condition], dim=1)  # [B, total_input_dim]

        # Predict velocity directly in tangent space (12D)
        # FB FM's expmap converts tangent velocity back to state space for integration
        velocity = self.vel_head(x_input)  # [B, 12]

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
            "model_type": "Quadrotor3D Latent Conditional Flow Matching"
        }
