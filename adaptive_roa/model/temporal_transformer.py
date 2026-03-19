"""
Temporal Diffusion Transformer for trajectory-level flow matching.

Ported from local_dynamics' TemporalDiffusionTransformer with adaptations:
- Removed genMoPlan dependencies (inlined SinusoidalPosEmb, removed TemporalModel base)
- Adapted forward signature: forward(x_t_embedded, t, z, condition) -> [B, T, output_dim]
  to match olympics-classifier convention where models take (x_t, t, z, condition)
- z and condition are combined into a single global query vector

Architecture:
    Input: x_t_embedded [B, T, embed_dim] (already embedded per-timestep)
    Time: t [B] flow time -> sinusoidal embedding -> MLP
    Global conditioning: concat(z, condition) -> GlobalQueryProcessor -> AdaLN modulation
    Transformer blocks with AdaLN-Zero, LayerScale, optional windowed attention
    Output: [B, T, output_dim] velocity in tangent space
"""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for scalar inputs (e.g., diffusion timestep)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GlobalQueryProcessor(nn.Module):
    """Process global query vectors into conditioning embeddings for AdaLN."""

    def __init__(self, global_query_dim, global_query_embed_dim):
        super().__init__()
        self.global_query_dim = global_query_dim
        self.global_query_embed_dim = global_query_embed_dim

        self.temporal_encoder = nn.Linear(global_query_dim, global_query_embed_dim)

        self.query_mlp = nn.Sequential(
            nn.Linear(global_query_embed_dim, global_query_embed_dim * 4),
            nn.Mish(),
            nn.Linear(global_query_embed_dim * 4, global_query_embed_dim),
        )

    def forward(self, global_query: torch.Tensor) -> torch.Tensor:
        encoded = self.temporal_encoder(global_query)
        return self.query_mlp(encoded)


class DiffusionTransformerBlock(nn.Module):
    """Transformer block with AdaLN-Zero style norm modulation and LayerScale."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        time_embed_dim: int,
        global_query_embed_dim: int = 0,
        dropout: float = 0.1,
        use_windowed_attention: bool = False,
        attention_window_size: int = 0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_global_query = global_query_embed_dim > 0
        self.use_windowed_attention = use_windowed_attention
        self.attention_window_size = attention_window_size

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # AdaLN-Zero modulations
        self.time_gamma_attn = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
        self.time_beta_attn = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
        self.time_gamma_ff = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
        self.time_beta_ff = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))

        if self.use_global_query:
            self.global_gamma_attn = nn.Sequential(
                nn.Mish(), nn.Linear(global_query_embed_dim, hidden_dim)
            )
            self.global_beta_attn = nn.Sequential(
                nn.Mish(), nn.Linear(global_query_embed_dim, hidden_dim)
            )
            self.global_gamma_ff = nn.Sequential(
                nn.Mish(), nn.Linear(global_query_embed_dim, hidden_dim)
            )
            self.global_beta_ff = nn.Sequential(
                nn.Mish(), nn.Linear(global_query_embed_dim, hidden_dim)
            )

        # LayerScale parameters for residuals
        self.alpha_attn = nn.Parameter(torch.tensor(1e-2))
        self.alpha_ff = nn.Parameter(torch.tensor(1e-2))

        self.dropout_layer = nn.Dropout(dropout)

    def _zero_init_modulation(self):
        """Zero-initialize modulation projection layers (AdaLN-Zero)."""
        zero_linears = [
            self.time_gamma_attn[-1], self.time_beta_attn[-1],
            self.time_gamma_ff[-1], self.time_beta_ff[-1],
        ]
        if self.use_global_query:
            zero_linears.extend([
                self.global_gamma_attn[-1], self.global_beta_attn[-1],
                self.global_gamma_ff[-1], self.global_beta_ff[-1],
            ])
        for lin in zero_linears:
            nn.init.zeros_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        q_global: torch.Tensor = None,
    ) -> torch.Tensor:
        # Sub-block 1: Self-attention with AdaLN-Zero and LayerScale
        y1 = self.norm1(x)
        b, s, _ = y1.shape

        gamma_attn = self.time_gamma_attn(t).unsqueeze(1).expand(b, s, -1)
        beta_attn = self.time_beta_attn(t).unsqueeze(1).expand(b, s, -1)
        if self.use_global_query and q_global is not None:
            gamma_attn = gamma_attn + self.global_gamma_attn(q_global).unsqueeze(1).expand(b, s, -1)
            beta_attn = beta_attn + self.global_beta_attn(q_global).unsqueeze(1).expand(b, s, -1)

        y1 = y1 * (1 + gamma_attn) + beta_attn

        attn_mask = None
        if self.use_windowed_attention and self.attention_window_size > 0:
            seq_len = s
            if self.attention_window_size < seq_len - 1:
                device = y1.device
                idx = torch.arange(seq_len, device=device)
                dist = (idx[None, :] - idx[:, None]).abs()
                mask_bool = dist > self.attention_window_size
                attn_mask = torch.zeros((seq_len, seq_len), device=device, dtype=y1.dtype)
                attn_mask.masked_fill_(mask_bool, float("-inf"))

        attn_out = self.self_attn(y1, y1, y1, attn_mask=attn_mask)[0]
        x = x + self.alpha_attn * self.dropout_layer(attn_out)

        # Sub-block 2: Feedforward with AdaLN-Zero and LayerScale
        y2 = self.norm2(x)
        gamma_ff = self.time_gamma_ff(t).unsqueeze(1).expand(b, s, -1)
        beta_ff = self.time_beta_ff(t).unsqueeze(1).expand(b, s, -1)
        if self.use_global_query and q_global is not None:
            gamma_ff = gamma_ff + self.global_gamma_ff(q_global).unsqueeze(1).expand(b, s, -1)
            beta_ff = beta_ff + self.global_beta_ff(q_global).unsqueeze(1).expand(b, s, -1)

        y2 = y2 * (1 + gamma_ff) + beta_ff
        ff_out = self.ff(y2)
        x = x + self.alpha_ff * ff_out
        return x


class TemporalTransformer(nn.Module):
    """
    Transformer for trajectory-level flow matching.

    Forward signature matches olympics-classifier convention:
        forward(x_t_embedded, t, z, condition) -> [B, T, output_dim]

    where z and condition are combined into a global query for AdaLN modulation.
    """

    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 2,
        condition_dim: int = 3,
        # Transformer architecture
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = None,
        dropout: float = 0.1,
        time_embed_dim: int = None,
        global_query_embed_dim: int = None,
        use_positional_encoding: bool = True,
        use_windowed_attention: bool = False,
        attention_window_size: int = 0,
    ):
        super().__init__()

        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4
        if time_embed_dim is None:
            time_embed_dim = hidden_dim
        if global_query_embed_dim is None:
            global_query_embed_dim = hidden_dim

        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, sequence_length, hidden_dim) * 0.02
            )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Global query: combine z and condition
        global_query_dim = latent_dim + condition_dim
        self.global_query_processor = GlobalQueryProcessor(
            global_query_dim, global_query_embed_dim
        )

        self.layers = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    time_embed_dim=time_embed_dim,
                    global_query_embed_dim=global_query_embed_dim,
                    dropout=dropout,
                    use_windowed_attention=use_windowed_attention,
                    attention_window_size=attention_window_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Apply AdaLN-Zero zero-initialization after global init
        for layer in self.layers:
            if hasattr(layer, "_zero_init_modulation"):
                layer._zero_init_modulation()

    def forward(
        self,
        x_t_embedded: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the Temporal Transformer.

        Args:
            x_t_embedded: [B, T, input_dim] already embedded per-timestep
            t: [B] flow time
            z: [B, latent_dim] stochastic latent
            condition: [B, condition_dim] embedded start state

        Returns:
            [B, T, output_dim] predicted velocity per timestep
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # Global query from z and condition
        global_query = torch.cat([z, condition], dim=-1)
        q_global = self.global_query_processor(global_query)

        # Project input to hidden dim
        x = self.input_projection(x_t_embedded)
        if self.use_positional_encoding:
            x = x + self.positional_encoding

        # Transformer layers
        for layer in self.layers:
            x = layer(x, t_emb, q_global)

        # Output projection
        x = self.output_projection(x)
        return x

    def get_model_info(self) -> dict:
        """Return architecture metadata for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "model_type": "TemporalTransformer",
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": len(self.layers),
            "total_parameters": total_params,
        }
