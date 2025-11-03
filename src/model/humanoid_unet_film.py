import math
from typing import List, Optional

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


class ConditionEncoder(nn.Module):
    """
    Encodes conditioning inputs (time and condition vector) into a compact context
    used to modulate FiLM residual blocks.
    """

    def __init__(self, in_dim: int, cond_dim: int, hidden_dims: Optional[List[int]] = None, dropout_p: float = 0.0):
        super().__init__()
        hidden_dims = hidden_dims or []
        dims: List[int] = [in_dim] + hidden_dims + [cond_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation after last layer
                layers.append(nn.SiLU())
                if dropout_p > 0:
                    layers.append(nn.Dropout(dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualFiLMBlock(nn.Module):
    """
    Pre-norm residual MLP block with FiLM modulation from a conditioning context.
    y = skip(x) + scale * Dropout( W2( FiLM( SiLU( W1( LN(x) ) ) ) ) )
    FiLM(h) = (1 + gamma) * h + beta, where [gamma, beta] = Linear(cond_context).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        dropout_p: float = 0.0,
        residual_scale: float = 1.0,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual_scale = residual_scale

        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.film = nn.Linear(cond_dim, 2 * out_dim)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

        if zero_init_residual:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, cond_context: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = F.silu(h)

        gamma_beta = self.film(cond_context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        h = h * (1 + gamma) + beta

        h = self.fc2(h)
        h = self.dropout(h)
        return self.skip(x) + self.residual_scale * h


class HumanoidUNetFiLM(nn.Module):
    """
    Humanoid UNet variant with Residual FiLM conditioning.

    Inputs:
      - embedded humanoid state x_t (67D), time t, condition (67D embedded start state)
    Output:
      - Predicts velocity in ℝ³⁴ × S² × ℝ³⁰ (67D)

    Conditioning:
      - Concatenate inputs as in the baseline for the main path
      - Build a separate conditioning context from [t_emb, condition{_emb}] to FiLM-modulate every block
    """

    def __init__(
        self,
        embedded_dim: int = 67,
        condition_dim: int = 67,
        time_emb_dim: int = 128,
        hidden_dims: List[int] = [256, 512, 512, 256],
        output_dim: int = 67,
        use_input_embeddings: bool = False,
        input_emb_dim: int = 128,
        film_cond_dim: int = 256,
        film_hidden_dims: Optional[List[int]] = None,
        dropout_p: float = 0.0,
        residual_scale: Optional[float] = None,
        zero_init_blocks: bool = True,
        zero_init_out: bool = False,
    ) -> None:
        super().__init__()
        self.embedded_dim = embedded_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_input_embeddings = use_input_embeddings
        self.input_emb_dim = input_emb_dim
        self.film_cond_dim = film_cond_dim
        self.dropout_p = dropout_p
        

        # Optional input embeddings for richer representations
        if use_input_embeddings:
            self.state_embedding = nn.Linear(embedded_dim, input_emb_dim)
            self.condition_embedding = nn.Linear(condition_dim, input_emb_dim)
            total_input_dim = 2 * input_emb_dim + time_emb_dim
            cond_encoder_in = input_emb_dim + time_emb_dim  # [condition_emb, t_emb]
        else:
            total_input_dim = embedded_dim + time_emb_dim + condition_dim
            cond_encoder_in = condition_dim + time_emb_dim  # [condition, t_emb]

        # Conditioning encoder (shared across blocks)
        self.cond_encoder = ConditionEncoder(
            in_dim=cond_encoder_in,
            cond_dim=film_cond_dim,
            hidden_dims=film_hidden_dims or [],
            dropout_p=dropout_p,
        )

        # Residual FiLM MLP stack
        dims = [total_input_dim] + hidden_dims
        num_blocks = len(hidden_dims)
        block_residual_scale = residual_scale if residual_scale is not None else (1.0 / math.sqrt(max(num_blocks, 1)))

        blocks: List[ResidualFiLMBlock] = []
        for i in range(num_blocks):
            blocks.append(
                ResidualFiLMBlock(
                    in_dim=dims[i],
                    out_dim=dims[i + 1],
                    cond_dim=film_cond_dim,
                    dropout_p=dropout_p,
                    residual_scale=block_residual_scale,
                    zero_init_residual=zero_init_blocks,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.norm_out = nn.LayerNorm(hidden_dims[-1]) if hidden_dims else nn.Identity()
        self.out_proj = nn.Linear(hidden_dims[-1], output_dim)
        if zero_init_out:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FiLM residual conditioning.

        Args:
            x_t: Current embedded state [B, 67]
            t: Time [B]
            condition: Start state condition [B, 67]

        Returns:
            Predicted velocity [B, 67]
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]

        if self.use_input_embeddings:
            state_emb = self.state_embedding(x_t)           # [B, input_emb_dim]
            condition_emb = self.condition_embedding(condition)  # [B, input_emb_dim]
            x_input = torch.cat([state_emb, t_emb, condition_emb], dim=1)
            cond_in = torch.cat([condition_emb, t_emb], dim=1)
        else:
            x_input = torch.cat([x_t, t_emb, condition], dim=1)
            cond_in = torch.cat([condition, t_emb], dim=1)

        cond_context = self.cond_encoder(cond_in)

        h = x_input
        for block in self.blocks:
            h = block(h, cond_context)

        h = self.norm_out(h)
        h = F.silu(h)
        velocity = self.out_proj(h)

        return velocity

    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "embedded_dim": self.embedded_dim,
            "condition_dim": self.condition_dim,
            "time_emb_dim": self.time_emb_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "use_input_embeddings": self.use_input_embeddings,
            "film_cond_dim": self.film_cond_dim,
            "total_parameters": total_params,
            "model_type": "Humanoid Residual FiLM",
        }


