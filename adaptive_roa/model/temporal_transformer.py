"""
Temporal Diffusion Transformer for trajectory-level flow matching.

Ported from local_dynamics' TemporalDiffusionTransformer with adaptations:
- Removed genMoPlan dependencies (inlined SinusoidalPosEmb, removed TemporalModel base)
- Adapted forward signature: forward(x_t_embedded, t, z, condition) -> [B, T, output_dim]
  to match olympics-classifier convention where models take (x_t, t, z, condition)
- z and condition are combined into a single global query token

Architecture:
    Input: x_t_embedded [B, T, embed_dim] (already embedded per-timestep)
    Time: t [B] flow time -> sinusoidal embedding -> MLP -> AdaLN modulation
    Global conditioning: concat(z, condition) -> QueryEncoder -> cross-attention KV
    Transformer blocks: self-attn (AdaLN) -> cross-attn (global query) -> FF (AdaLN)
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


class QueryEncoder(nn.Module):
    """Encode global query tokens via MLP projection.

    Projects the concatenated (z, condition) vector into hidden_dim space
    to serve as K/V tokens for cross-attention in each transformer block.
    Output shape: [B, 1, hidden_dim] (single query token).
    """

    def __init__(self, global_query_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(global_query_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, global_query: torch.Tensor) -> torch.Tensor:
        # global_query: [B, global_query_dim]
        q = self.projection(global_query)  # [B, hidden_dim]
        return q.unsqueeze(1)  # [B, 1, hidden_dim] — single query token


class DiffusionTransformerBlock(nn.Module):
    """Transformer block with AdaLN-Zero, cross-attention, and LayerScale.

    Architecture per block:
        1. Self-attention with AdaLN-Zero (time modulation) + LayerScale
        2. Cross-attention with global query (Q=horizon, KV=query tokens) + LayerScale
        3. Feedforward with AdaLN-Zero (time modulation) + LayerScale
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        time_embed_dim: int,
        use_cross_attention: bool = True,
        dropout: float = 0.1,
        use_windowed_attention: bool = False,
        attention_window_size: int = 0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        self.use_windowed_attention = use_windowed_attention
        self.attention_window_size = attention_window_size

        # Sub-block 1: Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # AdaLN-Zero for self-attention (time modulation)
        self.time_gamma_attn = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
        self.time_beta_attn = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
        self.alpha_attn = nn.Parameter(torch.tensor(1e-2))

        # Sub-block 2: Cross-attention with global query
        if self.use_cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads,
                dropout=dropout, batch_first=True
            )
            self.time_gamma_cross = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
            self.time_beta_cross = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
            self.alpha_cross = nn.Parameter(torch.tensor(1e-2))

        # Sub-block 3: Feedforward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # AdaLN-Zero for feedforward (time modulation)
        self.time_gamma_ff = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
        self.time_beta_ff = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
        self.alpha_ff = nn.Parameter(torch.tensor(1e-2))

        self.dropout_layer = nn.Dropout(dropout)

    def _zero_init_modulation(self):
        """Zero-initialize modulation projection layers (AdaLN-Zero)."""
        zero_linears = [
            self.time_gamma_attn[-1], self.time_beta_attn[-1],
            self.time_gamma_ff[-1], self.time_beta_ff[-1],
        ]
        if self.use_cross_attention:
            zero_linears.extend([
                self.time_gamma_cross[-1], self.time_beta_cross[-1],
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
        b, s, _ = x.shape

        # Sub-block 1: Self-attention with AdaLN-Zero and LayerScale
        y1 = self.norm1(x)
        gamma_attn = self.time_gamma_attn(t).unsqueeze(1).expand(b, s, -1)
        beta_attn = self.time_beta_attn(t).unsqueeze(1).expand(b, s, -1)
        y1 = y1 * (1 + gamma_attn) + beta_attn

        attn_mask = None
        if self.use_windowed_attention and self.attention_window_size > 0:
            if self.attention_window_size < s - 1:
                device = y1.device
                idx = torch.arange(s, device=device)
                dist = (idx[None, :] - idx[:, None]).abs()
                mask_bool = dist > self.attention_window_size
                attn_mask = torch.zeros((s, s), device=device, dtype=y1.dtype)
                attn_mask.masked_fill_(mask_bool, float("-inf"))

        attn_out = self.self_attn(y1, y1, y1, attn_mask=attn_mask)[0]
        x = x + self.alpha_attn * self.dropout_layer(attn_out)

        # Sub-block 2: Cross-attention with global query (Q=x, KV=q_global)
        if self.use_cross_attention and q_global is not None:
            y_cross = self.norm_cross(x)
            gamma_cross = self.time_gamma_cross(t).unsqueeze(1).expand(b, s, -1)
            beta_cross = self.time_beta_cross(t).unsqueeze(1).expand(b, s, -1)
            y_cross = y_cross * (1 + gamma_cross) + beta_cross
            cross_out = self.cross_attn(y_cross, q_global, q_global)[0]
            x = x + self.alpha_cross * self.dropout_layer(cross_out)

        # Sub-block 3: Feedforward with AdaLN-Zero and LayerScale
        y2 = self.norm2(x)
        gamma_ff = self.time_gamma_ff(t).unsqueeze(1).expand(b, s, -1)
        beta_ff = self.time_beta_ff(t).unsqueeze(1).expand(b, s, -1)
        y2 = y2 * (1 + gamma_ff) + beta_ff
        ff_out = self.ff(y2)
        x = x + self.alpha_ff * ff_out

        return x


class TemporalTransformer(nn.Module):
    """
    Transformer for trajectory-level flow matching.

    Forward signature matches olympics-classifier convention:
        forward(x_t_embedded, t, z, condition) -> [B, T, output_dim]

    z and condition are concatenated into a global query, encoded via
    QueryEncoder, and used as K/V tokens in cross-attention at each block.
    Time modulates self-attention and feedforward via AdaLN-Zero.
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
        use_positional_encoding: bool = True,
        use_windowed_attention: bool = False,
        attention_window_size: int = 0,
    ):
        super().__init__()

        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4
        if time_embed_dim is None:
            time_embed_dim = hidden_dim

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

        # Global query: combine z and condition -> encode for cross-attention K/V
        global_query_dim = latent_dim + condition_dim
        self.query_encoder = QueryEncoder(global_query_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    time_embed_dim=time_embed_dim,
                    use_cross_attention=True,
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

        # Global query: encode (z, condition) as cross-attention K/V tokens
        global_query = torch.cat([z, condition], dim=-1)  # [B, latent_dim + condition_dim]
        q_global = self.query_encoder(global_query)  # [B, 1, hidden_dim]

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
