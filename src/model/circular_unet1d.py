import torch
import torch.nn as nn
import numpy as np

class SinusoidalPositionEmbeddings(nn.Module):
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

class CircularUNet1D(nn.Module):
    def __init__(self, input_dim=6, output_dim=3, hidden_dims=[64, 128, 256], time_emb_dim=128):
        """
        Circular-aware UNet for S¹ × ℝ manifold
        
        Args:
            input_dim: 6 = current_state(3) + condition(3) 
            output_dim: 3 = velocity on S¹ × ℝ
            hidden_dims: encoder/decoder dimensions
            time_emb_dim: time embedding dimension
        """
        super().__init__()
        
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        
        # Input projection: (sin,cos,θ̇,sin_cond,cos_cond,θ̇_cond) → first hidden dim
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
        # hidden_dims = [64, 128, 256] → decoder: 256 → 128 → 64
        decoder_dims = list(reversed(hidden_dims[:-1]))  # [128, 64]
        
        for i, out_dim in enumerate(decoder_dims):
            if i == 0:
                # First decoder block: bottleneck (256) + skip (128) → 128
                in_dim = hidden_dims[-1] + hidden_dims[-2]  # 256 + 128 = 384
            else:
                # Second decoder block: prev output (128) + skip (64) → 64  
                in_dim = decoder_dims[i-1] + hidden_dims[-(i+2)]  # 128 + 64 = 192
            
            self.decoder_blocks.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                ResidualBlock(out_dim, time_emb_dim)
            ))
        
        # Output projection to tangent space of S¹ × ℝ
        self.output_proj = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, x, t, condition):
        """
        Forward pass
        
        Args:
            x: current state (sin(θ), cos(θ), θ̇) [batch_size, 3]
            t: time [batch_size]
            condition: conditioning state (sin(θ₀), cos(θ₀), θ̇₀) [batch_size, 3]
        
        Returns:
            velocity in tangent space [batch_size, 3]
        """
        # Get time embeddings
        t_emb = self.time_embedding(t)
        
        # Concatenate input with condition
        h = torch.cat([x, condition], dim=-1)  # [batch_size, 6]
        h = self.input_proj(h)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder - store skip connections after processing each block
        for i, block in enumerate(self.encoder_blocks):
            if isinstance(block, nn.Sequential):
                h = block[0](h, t_emb)  # ResidualBlock
                if len(block) > 1 and not isinstance(block[1], nn.Identity):  # Linear projection
                    h = block[1](h)
                # Store skip connection after full block processing
                if i < len(self.encoder_blocks) - 1:  # Don't store last layer (becomes bottleneck input)
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
        
        # Output projection
        velocity = self.output_proj(h)
        
        # Project to tangent space of S¹ × ℝ
        velocity_projected = self.project_to_tangent_space(velocity, x)
        
        return velocity_projected
    
    def project_to_tangent_space(self, velocity, x):
        """
        Project velocity to tangent space of S¹ × ℝ
        
        For S¹ component: velocity must be orthogonal to position vector
        For ℝ component: no constraint
        """
        sin_theta, cos_theta = x[..., 0], x[..., 1]
        v_sin, v_cos, v_theta_dot = velocity[..., 0], velocity[..., 1], velocity[..., 2]
        
        # Project S¹ component to tangent space: v ⟂ position
        # Remove radial component: v - (v·pos)pos
        dot_product = v_sin * sin_theta + v_cos * cos_theta
        v_sin_projected = v_sin - dot_product * sin_theta
        v_cos_projected = v_cos - dot_product * cos_theta
        
        # ℝ component (angular velocity) has no constraint
        return torch.stack([v_sin_projected, v_cos_projected, v_theta_dot], dim=-1)