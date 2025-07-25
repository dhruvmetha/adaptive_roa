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

class ResidualBlock(nn.Module):
    def __init__(self, dim, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x, time_emb):
        time_cond = self.time_mlp(time_emb)
        return x + self.net(x + time_cond)

class UNet1D(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,  # Current state (2D) + condition (2D)
        output_dim: int = 2,  # Velocity prediction (2D)
        hidden_dims: list = [64, 128, 256],
        time_emb_dim: int = 128
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(dims) - 1):
            self.encoder_blocks.append(nn.Sequential(
                ResidualBlock(dims[i], time_emb_dim),
                nn.Linear(dims[i], dims[i+1])
            ))
        
        # Bottleneck
        self.bottleneck = ResidualBlock(dims[-1], time_emb_dim)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.decoder_blocks.append(nn.Sequential(
                nn.Linear(dims[i] + dims[i-1], dims[i-1]),  # +skip connection
                ResidualBlock(dims[i-1], time_emb_dim)
            ))
        
        # Output projection
        self.output_proj = nn.Linear(dims[0], output_dim)
        
    def forward(self, x, t, condition=None):
        """
        x: input tensor [batch_size, 2] - current state in flow
        t: time tensor [batch_size] - flow matching time
        condition: conditional input [batch_size, 2] - start state
        """
        # Time embedding
        time_emb = self.time_embedding(t)
        
        # Concatenate input with condition (start state)
        if condition is not None:
            x = torch.cat([x, condition], dim=-1)
        
        # Input projection
        h = self.input_proj(x)
        
        # Encoder with skip connections
        skip_connections = [h]
        for encoder_block in self.encoder_blocks:
            res_block, linear = encoder_block
            h = res_block(h, time_emb)
            h = linear(h)
            skip_connections.append(h)
        
        # Bottleneck
        h = self.bottleneck(h, time_emb)
        
        # Decoder
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skip_connections[:-1])):
            linear, res_block = decoder_block
            h = torch.cat([h, skip], dim=-1)  # Skip connection
            h = linear(h)
            h = res_block(h, time_emb)
        
        # Output projection
        output = self.output_proj(h)
        
        return output