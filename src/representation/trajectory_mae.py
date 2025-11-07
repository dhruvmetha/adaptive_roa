"""
Trajectory MAE with BERT-style architecture.

Key differences from original:
1. Encoder processes FULL sequence (with mask tokens)
2. Decoder is simple MLP (per-position, no attention)
3. Reconstructs ALL positions, loss on masked only
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time steps."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            x + positional encoding (batch_size, seq_len, d_model)
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class CartPoleStateEmbedding(nn.Module):
    """Manifold-aware embedding for CartPole states."""

    def __init__(self, embed_dim: int, use_learned_embedding: bool = True):
        """
        Args:
            embed_dim: Dimension to embed states to
            use_learned_embedding: If True, use learned linear layer after manifold embedding
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_learned_embedding = use_learned_embedding

        # CartPole state: (x, theta, x_dot, theta_dot) -> (x, sin(theta), cos(theta), x_dot, theta_dot)
        # Input: 4D -> After manifold embedding: 5D
        manifold_dim = 5

        if use_learned_embedding:
            self.projection = nn.Linear(manifold_dim, embed_dim)
        else:
            # Just use manifold embedding directly (requires embed_dim == 5)
            assert embed_dim == manifold_dim, "Without learned embedding, embed_dim must be 5"
            self.projection = nn.Identity()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Embed CartPole states with circular angle handling.

        Args:
            states: (batch_size, seq_len, 4) - normalized (x, theta, x_dot, theta_dot)

        Returns:
            embedded: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = states.shape

        x = states[..., 0:1]          # (batch_size, seq_len, 1)
        theta = states[..., 1:2]      # (batch_size, seq_len, 1) - normalized
        x_dot = states[..., 2:3]      # (batch_size, seq_len, 1)
        theta_dot = states[..., 3:4]  # (batch_size, seq_len, 1)

        # Convert normalized theta to radians for circular encoding
        theta_rad = theta * 2 * math.pi  # Map [0,1] to [0, 2π]

        sin_theta = torch.sin(theta_rad)  # (batch_size, seq_len, 1)
        cos_theta = torch.cos(theta_rad)  # (batch_size, seq_len, 1)

        # Concatenate manifold features: (x, sin(theta), cos(theta), x_dot, theta_dot)
        manifold_features = torch.cat([x, sin_theta, cos_theta, x_dot, theta_dot], dim=-1)
        # (batch_size, seq_len, 5)

        # Project to embedding dimension
        embedded = self.projection(manifold_features)  # (batch_size, seq_len, embed_dim)

        return embedded


class TrajectoryMAE(nn.Module):
    """
    Trajectory MAE with BERT-style architecture.

    Architecture:
    1. Full sequence with mask tokens → Encoder → Embeddings for all positions
    2. Per-position MLP decoder → Reconstruct all positions
    3. Loss computed only on masked positions
    """

    def __init__(
        self,
        state_dim: int = 4,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 1001,
        use_learned_embedding: bool = True,
        decoder_hidden_dim: Optional[int] = None,
    ):
        """
        Initialize Trajectory MAE.

        Args:
            state_dim: Dimension of state (4 for CartPole)
            embed_dim: Embedding dimension for transformer
            encoder_depth: Number of transformer encoder layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed_dim
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            use_learned_embedding: Whether to use learned linear projection after manifold embedding
            decoder_hidden_dim: Hidden dimension for decoder MLP (if None, uses embed_dim * mlp_ratio)
        """
        super().__init__()

        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # State embedding (manifold-aware for CartPole)
        self.state_embedding = CartPoleStateEmbedding(
            embed_dim=embed_dim,
            use_learned_embedding=use_learned_embedding
        )

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len=max_seq_len)

        # Mask token (learned embedding for masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Transformer Encoder (processes FULL sequence with mask tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        # Simple MLP Decoder (per-position, processes each embedding independently)
        if decoder_hidden_dim is None:
            decoder_hidden_dim = int(embed_dim * mlp_ratio)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim // 2, 5)  # Output: (x, sin(theta), cos(theta), x_dot, theta_dot)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            states: Input states (batch_size, seq_len, state_dim)
            mask: Boolean mask (batch_size, seq_len) - True for MASKED positions
            padding_mask: Padding mask (batch_size, seq_len) - True for REAL tokens

        Returns:
            reconstructed: Predicted manifold features for ALL positions (batch_size, seq_len, 5)
        """
        batch_size, seq_len, _ = states.shape

        # 1. Embed all states
        embedded = self.state_embedding(states)  # (batch_size, seq_len, embed_dim)

        # 2. Replace masked positions with mask token
        mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)  # (batch_size, seq_len, embed_dim)

        # Use mask to select: mask_token where masked, embedded where visible
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        encoder_input = torch.where(mask_expanded, mask_tokens, embedded)  # (batch_size, seq_len, embed_dim)

        # 3. Add positional encoding
        encoder_input = self.pos_encoding(encoder_input)  # (batch_size, seq_len, embed_dim)

        # 4. Encode (processes FULL sequence)
        if padding_mask is not None:
            # Invert padding mask: True for positions to ignore
            src_key_padding_mask = ~padding_mask
            encoded = self.encoder(encoder_input, src_key_padding_mask=src_key_padding_mask)
        else:
            encoded = self.encoder(encoder_input)
        # encoded: (batch_size, seq_len, embed_dim)

        # 5. Decode ALL positions independently with simple MLP
        # Reshape to (batch_size * seq_len, embed_dim)
        encoded_flat = encoded.reshape(batch_size * seq_len, self.embed_dim)

        # Decode
        reconstructed_flat = self.decoder(encoded_flat)  # (batch_size * seq_len, 5)

        # Reshape back
        reconstructed = reconstructed_flat.reshape(batch_size, seq_len, 5)

        return reconstructed

    def get_encoder_representation(
        self,
        states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        aggregate: str = 'mean',
    ) -> torch.Tensor:
        """
        Get trajectory representation from encoder (for downstream tasks).

        Note: No masking applied during inference - processes full sequence.

        Args:
            states: Input states (batch_size, seq_len, state_dim)
            padding_mask: Padding mask (batch_size, seq_len)
            aggregate: How to aggregate sequence ('mean', 'max', 'last')

        Returns:
            representation: Trajectory embedding (batch_size, embed_dim)
        """
        batch_size, seq_len, _ = states.shape

        # Embed states (no masking for inference)
        embedded = self.state_embedding(states)  # (batch_size, seq_len, embed_dim)

        # Add positional encoding
        embedded = self.pos_encoding(embedded)  # (batch_size, seq_len, embed_dim)

        # Encode
        if padding_mask is not None:
            src_key_padding_mask = ~padding_mask
            encoded = self.encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        else:
            encoded = self.encoder(embedded)

        # Aggregate
        if aggregate == 'mean':
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(-1).float()
                sum_encoded = (encoded * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1)
                representation = sum_encoded / count.clamp(min=1)
            else:
                representation = encoded.mean(dim=1)
        elif aggregate == 'max':
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(-1)
                encoded_masked = encoded.masked_fill(~mask_expanded, float('-inf'))
                representation = encoded_masked.max(dim=1)[0]
            else:
                representation = encoded.max(dim=1)[0]
        elif aggregate == 'last':
            if padding_mask is not None:
                last_indices = padding_mask.sum(dim=1) - 1
                representation = encoded[torch.arange(batch_size), last_indices]
            else:
                representation = encoded[:, -1]
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        return representation
