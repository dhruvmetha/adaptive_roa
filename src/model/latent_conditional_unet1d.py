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

class LatentConditionalUNet1D(nn.Module):
    """
    Latent Conditional Flow Matching UNet:
    - Takes embedded state x_t (sin θ, cos θ, θ̇), time t, latent z, condition
    - Predicts velocity in S¹ × ℝ tangent space: (dθ/dt, dθ̇/dt)
    """
    def __init__(self, 
                 embedded_dim: int,
                 latent_dim: int,
                 condition_dim: int,
                 time_emb_dim: int,
                 hidden_dims,
                 output_dim: int,
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
        
        # Velocity predictor in S¹ × ℝ tangent space
        self.vel_head = MLP(total_input_dim, hidden_dims, output_dim)

    def forward(self, 
                x_t: torch.Tensor,        # [B, 3] embedded state (sin θ, cos θ, θ̇)
                t: torch.Tensor,          # [B] time
                z: torch.Tensor,          # [B, 2] latent vector
                condition: torch.Tensor   # [B, 3] embedded start state
                ) -> torch.Tensor:
        """
        Predict velocity in S¹ × ℝ tangent space
        
        Args:
            x_t: Embedded interpolated state [B, 3]
            t: Time parameter [B] in [0,1]
            z: Latent vector [B, 2] 
            condition: Embedded start state [B, 3]
            
        Returns:
            Predicted velocity [B, 2] in tangent space (dθ/dt, dθ̇/dt)
        """
        # Ensure correct shapes
        if x_t.dim() != 2 or x_t.shape[1] != self.embedded_dim:
            x_t = x_t.view(x_t.shape[0], -1)
        if condition.dim() != 2 or condition.shape[1] != self.condition_dim:
            condition = condition.view(condition.shape[0], -1)
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            z = z.view(z.shape[0], -1)
        
        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)
        
        if self.use_input_embeddings:
            # Apply learned embeddings for richer representations
            state_emb = F.silu(self.state_embedding(x_t))
            latent_emb = F.silu(self.latent_embedding(z))
            condition_emb = F.silu(self.condition_embedding(condition))
            
            # Concatenate embedded inputs
            h = torch.cat([state_emb, latent_emb, condition_emb, t_emb], dim=1)
        else:
            # Standard concatenation
            h = torch.cat([x_t, t_emb, z, condition], dim=1)
        
        # Predict velocity in tangent space
        vel = self.vel_head(h)
        
        return vel