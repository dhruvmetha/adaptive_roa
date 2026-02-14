"""
AdaLN Residual MLP for Latent Conditional Flow Matching.

DiT-style Adaptive Layer Norm residual blocks with hybrid input conditioning.
Condition is both concatenated at input (direct access) and used for per-layer
AdaLN modulation. Zero-initialization on all modulation layers and final output.

Architecture:
    Conditioning encoder:
        c = MLP([timestep_emb(t); z; condition]) -> [B, cond_dim]

    Input projection:
        h = Linear([x_t; timestep_emb(t); z; condition]) -> [B, hidden_dim]
        (hybrid: raw conditioning concatenated at input too)

    AdaLN Residual Blocks (x num_blocks):
        gamma, beta, alpha = Linear(c) -> 3 x hidden_dim  [zero-initialized]
        h' = gamma * LayerNorm(h) + beta                  [AdaLN modulation]
        h' = Linear(SiLU(Linear(h')))                     [FFN with expansion]
        h = h + alpha * h'                                 [gated residual]

    Output:
        shift, scale = Linear(c) -> 2 x hidden_dim        [zero-initialized]
        velocity = Linear(modulate(LayerNorm(h)))          [final layer, zero-initialized]
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive modulation: scale * x + shift."""
    return x * (1.0 + scale) + shift


class AdaLNResidualBlock(nn.Module):
    """Single residual block with AdaLN modulation and gated residual connection."""

    def __init__(self, hidden_dim: int, cond_dim: int, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # FFN with expansion
        expanded_dim = hidden_dim * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # AdaLN modulation: produces gamma, beta, alpha (3 * hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_dim),
        )

        # Zero-initialize the modulation output
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden state [B, hidden_dim]
            c: Conditioning vector [B, cond_dim]
        Returns:
            Updated hidden state [B, hidden_dim]
        """
        gamma, beta, alpha = self.adaLN_modulation(c).chunk(3, dim=-1)
        h_mod = modulate(self.norm(h), beta, gamma)
        h_ffn = self.ffn(h_mod)
        return h + alpha * h_ffn


class AdaLNResidualMLP(nn.Module):
    """
    AdaLN Residual MLP velocity model for latent conditional flow matching.

    Hybrid conditioning: condition concatenated at input (direct access) +
    AdaLN modulation per block (per-layer modulation from time+latent+condition).

    Forward signature: forward(x_t, t, z, condition) -> velocity
    """

    def __init__(
        self,
        embedded_dim: int,
        latent_dim: int,
        condition_dim: int,
        time_emb_dim: int = 64,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        output_dim: int = 2,
        cond_dim: int = 256,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedded_dim = embedded_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.cond_dim = cond_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # --- Conditioning encoder ---
        # Produces the AdaLN conditioning vector c from time, latent, and condition
        cond_input_dim = time_emb_dim + latent_dim + condition_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_input_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # --- Input projection ---
        # Hybrid: raw concatenation of all inputs projected to hidden_dim
        input_dim = embedded_dim + time_emb_dim + latent_dim + condition_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # --- AdaLN Residual Blocks ---
        self.blocks = nn.ModuleList([
            AdaLNResidualBlock(hidden_dim, cond_dim, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

        # --- Final layer with AdaLN ---
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )
        self.final_linear = nn.Linear(hidden_dim, output_dim)

        # Zero-initialize final modulation and output
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

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

        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)

        # Conditioning vector for AdaLN
        c = self.cond_encoder(torch.cat([t_emb, z, condition], dim=1))

        # Input projection (hybrid: raw inputs concatenated)
        h = self.input_proj(torch.cat([x_t, t_emb, z, condition], dim=1))

        # Residual blocks with AdaLN
        for block in self.blocks:
            h = block(h, c)

        # Final output with AdaLN
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        velocity = self.final_linear(h)

        return velocity

    def get_model_info(self) -> dict:
        """Return architecture metadata used by training logs."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "embedded_dim": self.embedded_dim,
            "latent_dim": self.latent_dim,
            "condition_dim": self.condition_dim,
            "time_emb_dim": self.time_emb_dim,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "output_dim": self.output_dim,
            "cond_dim": self.cond_dim,
            "mlp_ratio": self.mlp_ratio,
            "total_parameters": total_params,
            "model_type": "AdaLN Residual MLP",
        }
