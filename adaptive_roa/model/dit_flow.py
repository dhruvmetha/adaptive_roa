"""
DiT Flow Model with Cross-Attention for Latent Conditional Flow Matching.

Transformer blocks with per-dimension tokenization, self-attention over state
tokens, cross-attention to condition tokens, and AdaLN from time+latent.

Architecture:
    Tokenization:
        x_t [B, E] -> per-dim projection -> [B, E, D]   (E tokens of dim D)
        condition [B, C] -> per-dim projection -> [B, C, D]

    AdaLN conditioning (time + latent only):
        c = MLP([timestep_emb(t); z]) -> [B, cond_dim]

    DiT Blocks (x num_blocks):
        1. AdaLN Self-Attention over state tokens
        2. Cross-Attention: state tokens query condition tokens
        3. AdaLN FFN

    Output:
        AdaLN final -> per-token readout -> flatten -> Linear -> velocity [B, output_dim]
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
    """Apply adaptive modulation: (1 + scale) * x + shift."""
    return x * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    """
    DiT Transformer block: AdaLN self-attention + cross-attention + AdaLN FFN.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # --- Self-Attention with AdaLN ---
        self.norm_sa = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # AdaLN modulation for self-attention: gamma, beta, alpha
        self.adaLN_sa = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_dim),
        )

        # --- Cross-Attention with AdaLN ---
        self.norm_ca_q = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_ca_kv = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # AdaLN modulation for cross-attention: gamma, beta, alpha
        self.adaLN_ca = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_dim),
        )

        # --- FFN with AdaLN ---
        self.norm_ff = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        expanded_dim = hidden_dim * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        # AdaLN modulation for FFN: gamma, beta, alpha
        self.adaLN_ff = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_dim),
        )

        # Zero-initialize all modulation outputs
        for mod in [self.adaLN_sa, self.adaLN_ca, self.adaLN_ff]:
            nn.init.zeros_(mod[-1].weight)
            nn.init.zeros_(mod[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        cond_tokens: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: State tokens [B, S, D]
            cond_tokens: Condition tokens [B, C, D]
            c: AdaLN conditioning vector [B, cond_dim]
        Returns:
            Updated state tokens [B, S, D]
        """
        # c needs to be [B, 1, ...] for broadcasting over sequence dim
        c_expanded = c.unsqueeze(1)  # [B, 1, cond_dim]

        # 1. Self-Attention with AdaLN
        gamma_sa, beta_sa, alpha_sa = self.adaLN_sa(c_expanded).chunk(3, dim=-1)
        h_mod = modulate(self.norm_sa(h), beta_sa, gamma_sa)
        h_sa, _ = self.self_attn(h_mod, h_mod, h_mod)
        h = h + alpha_sa * h_sa

        # 2. Cross-Attention with AdaLN
        gamma_ca, beta_ca, alpha_ca = self.adaLN_ca(c_expanded).chunk(3, dim=-1)
        q = modulate(self.norm_ca_q(h), beta_ca, gamma_ca)
        kv = self.norm_ca_kv(cond_tokens)
        h_ca, _ = self.cross_attn(q, kv, kv)
        h = h + alpha_ca * h_ca

        # 3. FFN with AdaLN
        gamma_ff, beta_ff, alpha_ff = self.adaLN_ff(c_expanded).chunk(3, dim=-1)
        h_mod = modulate(self.norm_ff(h), beta_ff, gamma_ff)
        h_ff = self.ffn(h_mod)
        h = h + alpha_ff * h_ff

        return h


class DiTFlowModel(nn.Module):
    """
    DiT Flow Model with per-dimension tokenization and cross-attention.

    Each scalar dimension of the state becomes a token. Sequence length =
    embedded_dim (3-13), so attention is trivially cheap. Condition tokens
    are attended to via cross-attention; time+latent modulate via AdaLN.

    Forward signature: forward(x_t, t, z, condition) -> velocity
    """

    def __init__(
        self,
        embedded_dim: int,
        latent_dim: int,
        condition_dim: int,
        time_emb_dim: int = 64,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        num_heads: int = 4,
        output_dim: int = 2,
        cond_dim: int = 256,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedded_dim = embedded_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.cond_dim = cond_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # --- Per-dimension tokenization ---
        # Each scalar -> hidden_dim token via learned projection + positional embedding
        self.state_token_proj = nn.Linear(1, hidden_dim)
        self.cond_token_proj = nn.Linear(1, hidden_dim)

        # Learned positional embeddings for state and condition tokens
        self.state_pos_emb = nn.Parameter(torch.zeros(1, embedded_dim, hidden_dim))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, condition_dim, hidden_dim))

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.state_pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cond_pos_emb, std=0.02)

        # --- AdaLN conditioning encoder (time + latent only) ---
        adaln_input_dim = time_emb_dim + latent_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(adaln_input_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # --- DiT Blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, cond_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

        # --- Final output layer with AdaLN ---
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )
        # Per-token readout -> flatten -> output projection
        self.token_readout = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Linear(embedded_dim, output_dim)

        # Zero-initialize final layers (but NOT token_readout.weight or output_proj.weight - needed for gradient flow)
        # Unlike AdaLN which has a residual path, DiT has no skip connection from input to output
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.token_readout.bias)
        nn.init.zeros_(self.output_proj.bias)

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
        B = x_t.shape[0]

        if x_t.dim() != 2 or x_t.shape[1] != self.embedded_dim:
            x_t = x_t.view(B, -1)
        if condition.dim() != 2 or condition.shape[1] != self.condition_dim:
            condition = condition.view(B, -1)
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            z = z.view(B, -1)

        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)

        # --- Tokenize ---
        # [B, E] -> [B, E, 1] -> [B, E, D]
        state_tokens = self.state_token_proj(x_t.unsqueeze(-1)) + self.state_pos_emb
        cond_tokens = self.cond_token_proj(condition.unsqueeze(-1)) + self.cond_pos_emb

        # --- AdaLN conditioning from time + latent ---
        c = self.cond_encoder(torch.cat([t_emb, z], dim=1))

        # --- DiT Blocks ---
        h = state_tokens
        for block in self.blocks:
            h = block(h, cond_tokens, c)

        # --- Output ---
        shift, scale = self.final_adaLN(c).unsqueeze(1).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)  # [B, E, D]
        h = self.token_readout(h).squeeze(-1)  # [B, E]
        velocity = self.output_proj(h)  # [B, output_dim]

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
            "num_heads": self.num_heads,
            "output_dim": self.output_dim,
            "cond_dim": self.cond_dim,
            "mlp_ratio": self.mlp_ratio,
            "total_parameters": total_params,
            "model_type": "DiT Flow Model (Cross-Attention)",
        }
