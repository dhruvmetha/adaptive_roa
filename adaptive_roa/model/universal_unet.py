"""
Universal U-Net architecture that adapts to any dynamical system
"""
import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time conditioning"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time conditioning"""
    
    def __init__(self, in_dim, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, x, time_emb):
        time_emb = self.time_mlp(time_emb)
        return self.net(x + time_emb) + x


class UniversalUNet(nn.Module):
    """
    Universal U-Net that adapts to any dynamical system
    
    Automatically sizes itself based on:
    - Input dimension: embedded_state + condition  
    - Output dimension: tangent space dimension
    - System-agnostic architecture
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dims: list = [64, 128, 256], 
                 time_emb_dim: int = 128):
        """
        Initialize universal U-Net
        
        Args:
            input_dim: Input dimension (embedded_state + condition)
            output_dim: Output dimension (tangent space)
            hidden_dims: Hidden layer dimensions for encoder/decoder
            time_emb_dim: Time embedding dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([])
        in_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims:
            self.encoder_blocks.append(nn.Sequential(
                ResidualBlock(in_dim, time_emb_dim),
                nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
            ))
            in_dim = hidden_dim
        
        # Bottleneck
        self.bottleneck = ResidualBlock(hidden_dims[-1], time_emb_dim)
        
        # Decoder blocks with skip connections
        self.decoder_blocks = nn.ModuleList([])
        
        # Create decoder blocks to match encoder in reverse
        decoder_dims = list(reversed(hidden_dims[:-1]))  # Reverse except last
        
        for i, out_dim in enumerate(decoder_dims):
            if i == 0:
                # First decoder block: bottleneck + skip from encoder
                in_dim = hidden_dims[-1] + hidden_dims[-(i+2)]
            else:
                # Subsequent decoder blocks: prev output + skip from encoder
                in_dim = decoder_dims[i-1] + hidden_dims[-(i+2)]
            
            self.decoder_blocks.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                ResidualBlock(out_dim, time_emb_dim)
            ))
        
        # Output projection to tangent space
        final_dim = hidden_dims[0] if decoder_dims else hidden_dims[-1]
        self.output_proj = nn.Linear(final_dim, output_dim)

    def forward(self, x, t, condition):
        """
        Forward pass
        
        Args:
            x: current embedded state [batch_size, embedding_dim]
            t: time [batch_size]
            condition: conditioning embedded state [batch_size, embedding_dim]
        
        Returns:
            tangent_velocity: velocity in tangent space [batch_size, tangent_dim]
        """
        # Get time embeddings
        t_emb = self.time_embedding(t)
        
        # Concatenate input with condition
        h = torch.cat([x, condition], dim=-1)  # [batch_size, input_dim]
        h = self.input_proj(h)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder - store skip connections after processing each block
        for i, block in enumerate(self.encoder_blocks):
            if isinstance(block, nn.Sequential):
                h = block[0](h, t_emb)  # ResidualBlock
                if len(block) > 1 and not isinstance(block[1], nn.Identity):
                    h = block[1](h)  # Linear projection
                # Store skip connection after full block processing
                if i < len(self.encoder_blocks) - 1:  # Don't store last layer
                    skip_connections.append(h)
            else:
                h = block(h, t_emb)
                if i < len(self.encoder_blocks) - 1:
                    skip_connections.append(h)
        
        # Bottleneck
        h = self.bottleneck(h, t_emb)
        
        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i+1)]  # Skip from encoder (reversed order)
            h = torch.cat([h, skip], dim=-1)
            h = block[0](h)  # Linear
            h = block[1](h, t_emb)  # ResidualBlock
        
        # Output projection to tangent space
        tangent_velocity = self.output_proj(h)
        
        return tangent_velocity
    
    def get_architecture_info(self) -> dict:
        """Get information about the model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim, 
            "hidden_dims": self.hidden_dims,
            "time_emb_dim": self.time_emb_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "encoder_blocks": len(self.encoder_blocks),
            "decoder_blocks": len(self.decoder_blocks)
        }
    
    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        return (f"UniversalUNet("
                f"dims=[{self.input_dim}→{self.hidden_dims}→{self.output_dim}], "
                f"params={total_params:,})")