"""
Contrastive Encoder for Self-Supervised Representation Learning

Maps normalized states to L2-normalized embeddings for metric learning.
Follows existing model patterns (MLP architecture with ReLU activations).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ContrastiveEncoder(nn.Module):
    """
    MLP encoder for contrastive representation learning

    Architecture:
    - Input: Normalized state [batch_size, input_dim]
    - Hidden layers: Configurable MLPs with ReLU + Dropout
    - Output: L2-normalized embedding [batch_size, embedding_dim]

    The L2 normalization ensures embeddings lie on a hypersphere,
    making cosine similarity a natural distance metric.
    """

    def __init__(self,
                 input_dim: int,
                 embedding_dim: int = 128,
                 hidden_channels: List[int] = [512, 256],
                 dropout: float = 0.1,
                 normalize_output: bool = True):
        """
        Initialize contrastive encoder

        Args:
            input_dim: Dimension of input state (e.g., 67 for humanoid, 4 for cartpole)
            embedding_dim: Dimension of output embedding (default: 128)
            hidden_channels: List of hidden layer sizes (default: [512, 256])
            dropout: Dropout probability (default: 0.1)
            normalize_output: If True, L2-normalize output embeddings (default: True)
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.normalize_output = normalize_output

        # Build MLP encoder (following existing MLP pattern)
        layers = []
        current_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_channels:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Final projection to embedding space
        layers.append(nn.Linear(current_dim, embedding_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode state to embedding

        Args:
            x: Input state [batch_size, input_dim] (normalized)

        Returns:
            Embedding [batch_size, embedding_dim] (L2-normalized if normalize_output=True)
        """
        # Pass through MLP encoder
        embedding = self.encoder(x)

        # L2 normalization for stable metric learning
        if self.normalize_output:
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def __repr__(self):
        return (f"ContrastiveEncoder(input_dim={self.input_dim}, "
                f"embedding_dim={self.embedding_dim}, "
                f"normalize_output={self.normalize_output})")


class TemporalContrastiveEncoder(nn.Module):
    """
    Temporal encoder with 1D convolutions for capturing local dynamics

    Architecture:
    - Input: Sequence of states [batch_size, seq_len, state_dim]
    - 1D CNN layers for temporal feature extraction
    - Global pooling + MLP for embedding
    - Output: L2-normalized embedding [batch_size, embedding_dim]

    Optional advanced architecture for capturing trajectory dynamics.
    """

    def __init__(self,
                 input_dim: int,
                 embedding_dim: int = 128,
                 hidden_channels: List[int] = [256, 128],
                 conv_channels: List[int] = [64, 128],
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 normalize_output: bool = True):
        """
        Initialize temporal contrastive encoder

        Args:
            input_dim: Dimension of input state
            embedding_dim: Dimension of output embedding
            hidden_channels: MLP hidden layer sizes after convolution
            conv_channels: 1D conv channel sizes
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            normalize_output: If True, L2-normalize output embeddings
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.normalize_output = normalize_output

        # 1D Convolutional layers for temporal feature extraction
        conv_layers = []
        in_channels = input_dim

        for out_channels in conv_channels:
            conv_layers.append(nn.Conv1d(in_channels, out_channels,
                                         kernel_size=kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.ReLU())
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv_encoder = nn.Sequential(*conv_layers)

        # MLP after global pooling
        mlp_layers = []
        current_dim = conv_channels[-1]

        for hidden_dim in hidden_channels:
            mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        mlp_layers.append(nn.Linear(current_dim, embedding_dim))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal sequence to embedding

        Args:
            x: Input sequence [batch_size, seq_len, state_dim] or
               single state [batch_size, state_dim]

        Returns:
            Embedding [batch_size, embedding_dim]
        """
        # Handle single state input (add sequence dimension)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, state_dim] -> [B, 1, state_dim]

        # Transpose for Conv1d: [B, seq_len, state_dim] -> [B, state_dim, seq_len]
        x = x.transpose(1, 2)

        # 1D convolution
        features = self.conv_encoder(x)  # [B, conv_channels[-1], seq_len]

        # Global average pooling over time
        pooled = features.mean(dim=-1)  # [B, conv_channels[-1]]

        # MLP projection
        embedding = self.mlp(pooled)

        # L2 normalization
        if self.normalize_output:
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def __repr__(self):
        return (f"TemporalContrastiveEncoder(input_dim={self.input_dim}, "
                f"embedding_dim={self.embedding_dim}, "
                f"normalize_output={self.normalize_output})")
