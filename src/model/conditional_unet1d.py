import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for conditioning"""
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(condition_dim, feature_dim)
        self.beta_proj = nn.Linear(condition_dim, feature_dim)
        
    def forward(self, features, condition):
        """
        Apply FiLM conditioning: gamma * features + beta
        
        Args:
            features: [batch_size, feature_dim]
            condition: [batch_size, condition_dim]
        """
        gamma = self.gamma_proj(condition)
        beta = self.beta_proj(condition)
        return gamma * features + beta

class ConditionalResidualBlock(nn.Module):
    def __init__(self, dim, time_emb_dim, condition_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, dim)
        self.film_layer = FiLMLayer(condition_dim, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x, time_emb, condition):
        # Time conditioning
        time_cond = self.time_mlp(time_emb)
        
        # Apply time conditioning
        h = x + time_cond
        
        # Apply FiLM conditioning
        h = self.film_layer(h, condition)
        
        # Residual connection
        return x + self.net(h)

class ConditionalUNet1D(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,  # Current state (embedded 3D: sin θ, cos θ, θ̇)
        condition_dim: int = 3,  # Start state (embedded 3D: sin θ, cos θ, θ̇)
        output_dim: int = 3,  # Velocity prediction (embedded 3D)
        hidden_dims: list = [64, 128, 256],
        time_emb_dim: int = 128
    ):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        
        # Condition projection to enrich conditioning information
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 2),
            nn.SiLU(),
            nn.Linear(condition_dim * 2, condition_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Encoder blocks with FiLM conditioning
        self.encoder_blocks = nn.ModuleList()
        dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(dims) - 1):
            self.encoder_blocks.append(nn.Sequential(
                ConditionalResidualBlock(dims[i], time_emb_dim, condition_dim),
                nn.Linear(dims[i], dims[i+1])
            ))
        
        # Bottleneck with FiLM conditioning
        self.bottleneck = ConditionalResidualBlock(dims[-1], time_emb_dim, condition_dim)
        
        # Decoder blocks with FiLM conditioning
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.decoder_blocks.append(nn.Sequential(
                nn.Linear(dims[i] + dims[i-1], dims[i-1]),  # +skip connection
                ConditionalResidualBlock(dims[i-1], time_emb_dim, condition_dim)
            ))
        
        # Output projection
        self.output_proj = nn.Linear(dims[0], output_dim)
        
    def forward(self, x, t, condition):
        """
        Args:
            x: current state tensor [batch_size, 3] - (sin θ, cos θ, θ̇)
            t: time tensor [batch_size] - flow matching time
            condition: start state [batch_size, 3] - (sin θ, cos θ, θ̇)
        """
        # Time embedding
        time_emb = self.time_embedding(t)
        
        # Process condition
        condition_processed = self.condition_proj(condition)
        
        # Input projection
        h = self.input_proj(x)
        
        # Encoder with skip connections and FiLM conditioning
        skip_connections = [h]
        for encoder_block in self.encoder_blocks:
            res_block, linear = encoder_block
            h = res_block(h, time_emb, condition_processed)
            h = linear(h)
            skip_connections.append(h)
        
        # Bottleneck with FiLM conditioning
        h = self.bottleneck(h, time_emb, condition_processed)
        
        # Decoder with FiLM conditioning
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skip_connections[:-1])):
            linear, res_block = decoder_block
            h = torch.cat([h, skip], dim=-1)  # Skip connection
            h = linear(h)
            h = res_block(h, time_emb, condition_processed)
        
        # Output projection
        output = self.output_proj(h)
        
        return output