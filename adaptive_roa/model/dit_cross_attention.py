"""
DiT with Cross-Attention for Latent Conditional Flow Matching.

Transformer-style architecture that tokenizes state components and uses
cross-attention between state tokens and condition tokens. Designed for
higher-dimensional systems with natural component structure (e.g., position,
orientation, velocity groups).

Architecture:
    Tokenization:
        x_t -> split by state_component_dims -> project each to token_dim -> + pos_emb
        condition -> split by cond_component_dims -> project each to token_dim -> + pos_emb

    Conditioning signal (for AdaLN):
        c = MLP([timestep_emb(t); z]) -> [B, cond_proj_dim]

    N x DiTCrossAttentionBlock:
        Self-Attention: state tokens attend to each other (with AdaLN modulation)
        Cross-Attention: state tokens query condition tokens (with AdaLN modulation)
        FFN: feedforward with AdaLN modulation and gated residual

    Detokenize:
        per-token Linear (ZERO-INIT) -> concat -> velocity [B, output_dim]
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaln_mlp import modulate, timestep_embedding


class ComponentTokenizer(nn.Module):
    """Splits a flat vector by component dims and projects each group to token_dim."""

    def __init__(self, component_dims: List[int], token_dim: int):
        super().__init__()
        self.component_dims = component_dims
        self.num_tokens = len(component_dims)
        self.projections = nn.ModuleList([
            nn.Linear(d, token_dim) for d in component_dims
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, sum(component_dims)]
        Returns:
            tokens: [B, num_tokens, token_dim]
        """
        parts = x.split(self.component_dims, dim=-1)
        tokens = [proj(part) for proj, part in zip(self.projections, parts)]
        return torch.stack(tokens, dim=1)


class ComponentDetokenizer(nn.Module):
    """Projects each token back to its output dim and concatenates."""

    def __init__(self, output_component_dims: List[int], token_dim: int):
        super().__init__()
        self.output_component_dims = output_component_dims
        self.projections = nn.ModuleList([
            nn.Linear(token_dim, d) for d in output_component_dims
        ])
        # Zero-initialize only biases (not weights - needed for gradient flow)
        # Unlike models with residual skip connections, DiT has no skip path from input to output
        for proj in self.projections:
            nn.init.zeros_(proj.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, num_tokens, token_dim]
        Returns:
            output: [B, sum(output_component_dims)]
        """
        parts = [proj(tokens[:, i]) for i, proj in enumerate(self.projections)]
        return torch.cat(parts, dim=-1)


class DiTCrossAttentionBlock(nn.Module):
    """Single DiT block with self-attention, cross-attention, and FFN, all with AdaLN."""

    def __init__(self, token_dim: int, n_heads: int, cond_proj_dim: int,
                 mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.token_dim = token_dim
        self.n_heads = n_heads

        # --- Self-Attention sublayer ---
        self.norm_sa = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            token_dim, n_heads, dropout=dropout, batch_first=True
        )

        # --- Cross-Attention sublayer ---
        self.norm_ca_q = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.norm_ca_kv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            token_dim, n_heads, dropout=dropout, batch_first=True
        )

        # --- FFN sublayer ---
        self.norm_ff = nn.LayerNorm(token_dim, elementwise_affine=False)
        expanded_dim = token_dim * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, token_dim),
            nn.Dropout(dropout),
        )

        # --- AdaLN modulation ---
        # 9 modulation params: (shift, scale, gate) x 3 sublayers
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_proj_dim, 9 * token_dim),
        )
        # Zero-initialize
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State tokens [B, N_s, token_dim]
            cond_tokens: Condition tokens [B, N_c, token_dim]
            c: AdaLN conditioning vector [B, cond_proj_dim]
        Returns:
            Updated state tokens [B, N_s, token_dim]
        """
        # Unpack 9 modulation params
        mods = self.adaLN_modulation(c)  # [B, 9 * token_dim]
        (shift_sa, scale_sa, gate_sa,
         shift_ca, scale_ca, gate_ca,
         shift_ff, scale_ff, gate_ff) = mods.chunk(9, dim=-1)

        # Expand for broadcasting: [B, 1, token_dim]
        shift_sa = shift_sa.unsqueeze(1)
        scale_sa = scale_sa.unsqueeze(1)
        gate_sa = gate_sa.unsqueeze(1)
        shift_ca = shift_ca.unsqueeze(1)
        scale_ca = scale_ca.unsqueeze(1)
        gate_ca = gate_ca.unsqueeze(1)
        shift_ff = shift_ff.unsqueeze(1)
        scale_ff = scale_ff.unsqueeze(1)
        gate_ff = gate_ff.unsqueeze(1)

        # --- Self-Attention ---
        x_mod = modulate(self.norm_sa(x), shift_sa, scale_sa)
        x_sa, _ = self.self_attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate_sa * x_sa

        # --- Cross-Attention ---
        q = modulate(self.norm_ca_q(x), shift_ca, scale_ca)
        kv = self.norm_ca_kv(cond_tokens)
        x_ca, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + gate_ca * x_ca

        # --- FFN ---
        x_mod = modulate(self.norm_ff(x), shift_ff, scale_ff)
        x_ff = self.ffn(x_mod)
        x = x + gate_ff * x_ff

        return x


class DiTCrossAttentionModel(nn.Module):
    """
    DiT with Cross-Attention velocity model for latent conditional flow matching.

    Tokenizes state and condition by component groups, uses self-attention among
    state tokens and cross-attention from state to condition tokens.

    Forward signature: forward(x_t, t, z, condition) -> velocity
    """

    def __init__(
        self,
        embedded_dim: int,
        latent_dim: int,
        condition_dim: int,
        time_emb_dim: int = 64,
        output_dim: int = 2,
        token_dim: int = 128,
        n_heads: int = 4,
        num_blocks: int = 4,
        mlp_ratio: int = 4,
        cond_proj_dim: int = 256,
        dropout: float = 0.0,
        state_component_dims: Optional[List[int]] = None,
        condition_component_dims: Optional[List[int]] = None,
        output_component_dims: Optional[List[int]] = None,
        # Legacy params (accepted for Hydra config compat, ignored)
        hidden_dims: Optional[List[int]] = None,
        use_input_embeddings: bool = False,
        input_emb_dim: int = 64,
    ):
        super().__init__()
        self.embedded_dim = embedded_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.output_dim = output_dim
        self.token_dim = token_dim
        self.n_heads = n_heads
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        self.cond_proj_dim = cond_proj_dim

        # Default component dims: treat entire vector as one token
        if state_component_dims is None:
            state_component_dims = [embedded_dim]
        if condition_component_dims is None:
            condition_component_dims = list(state_component_dims)
        if output_component_dims is None:
            output_component_dims = [output_dim]

        self.state_component_dims = state_component_dims
        self.condition_component_dims = condition_component_dims
        self.output_component_dims = output_component_dims

        assert sum(state_component_dims) == embedded_dim, \
            f"state_component_dims {state_component_dims} must sum to embedded_dim {embedded_dim}"
        assert sum(condition_component_dims) == condition_dim, \
            f"condition_component_dims {condition_component_dims} must sum to condition_dim {condition_dim}"
        assert sum(output_component_dims) == output_dim, \
            f"output_component_dims {output_component_dims} must sum to output_dim {output_dim}"
        assert len(state_component_dims) == len(output_component_dims), \
            f"state and output must have same number of tokens"

        num_state_tokens = len(state_component_dims)
        num_cond_tokens = len(condition_component_dims)

        # --- Tokenizers ---
        self.state_tokenizer = ComponentTokenizer(state_component_dims, token_dim)
        self.cond_tokenizer = ComponentTokenizer(condition_component_dims, token_dim)
        self.detokenizer = ComponentDetokenizer(output_component_dims, token_dim)

        # --- Learned positional embeddings ---
        self.state_pos_emb = nn.Parameter(torch.zeros(1, num_state_tokens, token_dim))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, num_cond_tokens, token_dim))
        nn.init.normal_(self.state_pos_emb, std=0.02)
        nn.init.normal_(self.cond_pos_emb, std=0.02)

        # --- Conditioning encoder (time + latent -> AdaLN vector) ---
        cond_input_dim = time_emb_dim + latent_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_input_dim, cond_proj_dim),
            nn.SiLU(),
            nn.Linear(cond_proj_dim, cond_proj_dim),
        )

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            DiTCrossAttentionBlock(token_dim, n_heads, cond_proj_dim, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

        # --- Final AdaLN layer ---
        self.final_norm = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_proj_dim, 2 * token_dim),
        )
        # Zero-initialize final modulation
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

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

        # AdaLN conditioning from time + latent
        c = self.cond_encoder(torch.cat([t_emb, z], dim=1))

        # Tokenize state and condition
        state_tokens = self.state_tokenizer(x_t) + self.state_pos_emb
        cond_tokens = self.cond_tokenizer(condition) + self.cond_pos_emb

        # Transformer blocks
        x = state_tokens
        for block in self.blocks:
            x = block(x, cond_tokens, c)

        # Final AdaLN + detokenize
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        # Expand for token sequence: [B, 1, token_dim]
        x = modulate(self.final_norm(x), shift.unsqueeze(1), scale.unsqueeze(1))
        velocity = self.detokenizer(x)

        return velocity

    def get_model_info(self) -> dict:
        """Return architecture metadata used by training logs."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "embedded_dim": self.embedded_dim,
            "latent_dim": self.latent_dim,
            "condition_dim": self.condition_dim,
            "time_emb_dim": self.time_emb_dim,
            "output_dim": self.output_dim,
            "token_dim": self.token_dim,
            "n_heads": self.n_heads,
            "num_blocks": self.num_blocks,
            "mlp_ratio": self.mlp_ratio,
            "cond_proj_dim": self.cond_proj_dim,
            "state_component_dims": self.state_component_dims,
            "condition_component_dims": self.condition_component_dims,
            "output_component_dims": self.output_component_dims,
            "total_parameters": total_params,
            "model_type": "DiT Cross-Attention",
        }
