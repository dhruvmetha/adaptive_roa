"""
Simple Direct Latent Circular Flow Matching Inference
"""
import torch
from torch import Tensor
from typing import Tuple
import numpy as np

from src.utils.angles import wrap_to_pi


class SimpleLatentCircularInference:
    """
    Simple inference for direct noise conditioning approach
    
    Much simpler than VAE approach:
    - No encoder used at all
    - Just sample z ~ N(0,I) and integrate
    - Different noise → different endpoints
    """
    
    def __init__(self, model: torch.nn.Module, latent_dim: int = 8, steps: int = 32):
        """
        Initialize simple inference module
        
        Args:
            model: Trained flow matching model
            latent_dim: Dimension of latent noise variable z
            steps: Number of integration steps (RK4)
        """
        self.model = model.eval()
        self.latent_dim = latent_dim
        self.steps = steps

    def features(self, theta: Tensor, omega: Tensor) -> Tensor:
        """Convert state to circular features [sin θ, cos θ, ω]"""
        return torch.cat([theta.sin(), theta.cos(), omega], dim=-1)

    def cond_vec(self, theta0: Tensor, omega0: Tensor, z: Tensor) -> Tensor:
        """Create conditioning vector [sin θ₀, cos θ₀, ω₀, z]"""
        return torch.cat([theta0.sin(), theta0.cos(), omega0, z], dim=-1)

    @torch.no_grad()
    def _rhs(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        """
        Right-hand side of ODE: dx/dt = v_θ(x, t; x₀, z)
        
        Args:
            x: Current state [B, 2] = [θ, ω]
            t: Time [B, 1] 
            cond: Conditioning vector [B, 3+L] = [sin θ₀, cos θ₀, ω₀, z]
            
        Returns:
            Velocity field [B, 2] = [dθ/dt, dω/dt]
        """
        theta, omega = x[:, :1], x[:, 1:2]
        phi = self.features(theta, omega)                    # [B, 3]
        feats = torch.cat([phi, t, cond], dim=-1)            # [B, 7+L]
        
        # Model expects single tensor input when using concatenated features
        u = self.model(feats)                                # [B, 2] transport velocity
        return u

    @torch.no_grad()
    def integrate(self, x0: Tensor, cond: Tensor) -> Tensor:
        """
        Integrate ODE from t=0 to t=1 using RK4
        
        Args:
            x0: Initial state [B, 2] = [θ₀, ω₀]
            cond: Conditioning vector [B, 3+L] = [sin θ₀, cos θ₀, ω₀, z]
            
        Returns:
            Final endpoint [B, 2] = [θ₁, ω₁]
        """
        B = x0.size(0)
        h = 1.0 / self.steps
        x = x0.clone()
        t = torch.zeros(B, 1, device=x0.device)
        
        for _ in range(self.steps):
            k1 = self._rhs(x, t, cond)
            k2 = self._rhs(self._advance(x, 0.5*h, k1), t + 0.5*h, cond)
            k3 = self._rhs(self._advance(x, 0.5*h, k2), t + 0.5*h, cond)
            k4 = self._rhs(self._advance(x, h, k3), t + h, cond)
            
            dx = (k1 + 2*k2 + 2*k3 + k4) / 6.0
            x = self._advance(x, h, dx)
            
            # Keep θ wrapped to (-π, π]
            x[:, :1] = wrap_to_pi(x[:, :1])
            t = t + h
            
        return x

    @staticmethod
    def _advance(x: Tensor, h: float, k: Tensor) -> Tensor:
        """Advance state by h*k"""
        return torch.cat([x[:, :1] + h * k[:, :1], x[:, 1:2] + h * k[:, 1:2]], dim=-1)

    @torch.no_grad()
    def sample_endpoints(self, x0: Tensor, num_samples: int = 32) -> Tensor:
        """
        Sample multiple endpoints using random noise
        
        Args:
            x0: Initial states [B, 2] = [θ₀, ω₀]
            num_samples: Number of samples per initial state
            
        Returns:
            Sampled endpoints [B, num_samples, 2]
        """
        B = x0.size(0)
        
        # Sample random noise z ~ N(0, I) - this is the key difference!
        z = torch.randn(B * num_samples, self.latent_dim, device=x0.device)
        
        # Repeat initial states for each sample
        x0_rep = x0.repeat_interleave(num_samples, dim=0)  # [B*num_samples, 2]
        
        # Create conditioning vectors
        theta0, omega0 = x0_rep[:, :1], x0_rep[:, 1:2]
        cond = self.cond_vec(theta0, omega0, z)            # [B*num_samples, 3+L]
        
        # Integrate to get endpoints
        y_hat = self.integrate(x0_rep, cond)               # [B*num_samples, 2]
        
        # Reshape to [B, num_samples, 2]
        return y_hat.view(B, num_samples, 2)

    @torch.no_grad()
    def predict_attractor_distribution(self, x0: Tensor, num_samples: int = 64, 
                                     attractor_centers: Tensor = None) -> Tensor:
        """
        Predict probability distribution over attractors
        
        Args:
            x0: Initial states [B, 2]
            num_samples: Number of samples for Monte Carlo estimation
            attractor_centers: Known attractor locations [num_attractors, 2]
                              If None, uses default pendulum attractors
            
        Returns:
            Attractor probabilities [B, num_attractors]
        """
        if attractor_centers is None:
            # Default pendulum attractors at (0, 0) and (π, 0)
            attractor_centers = torch.tensor([[0.0, 0.0], [np.pi, 0.0]], 
                                           device=x0.device, dtype=x0.dtype)
        
        # Sample endpoints
        samples = self.sample_endpoints(x0, num_samples)  # [B, num_samples, 2]
        B, N, _ = samples.shape
        num_attractors = attractor_centers.shape[0]
        
        # Compute distances to each attractor
        # Handle circular distance for angle coordinate
        theta_samples = samples[:, :, 0]  # [B, N]
        omega_samples = samples[:, :, 1]  # [B, N]
        
        # Initialize attractor counts
        attractor_counts = torch.zeros(B, num_attractors, device=x0.device)
        
        for i, (theta_att, omega_att) in enumerate(attractor_centers):
            # Circular distance for angle
            theta_dist = torch.abs(wrap_to_pi(theta_samples - theta_att))
            omega_dist = torch.abs(omega_samples - omega_att)
            
            # Combined distance (could use different weighting)
            dist = torch.sqrt(theta_dist**2 + omega_dist**2)  # [B, N]
            
            # Count samples close to this attractor (within threshold)
            threshold = 0.2  # Configurable
            close_to_attractor = (dist < threshold).float()
            attractor_counts[:, i] = close_to_attractor.sum(dim=1)
        
        # Convert counts to probabilities
        total_counts = attractor_counts.sum(dim=1, keepdim=True)
        probabilities = attractor_counts / (total_counts + 1e-8)  # Avoid division by zero
        
        return probabilities

    @torch.no_grad()
    def compute_uncertainty(self, x0: Tensor, num_samples: int = 64) -> Tensor:
        """
        Compute prediction uncertainty using entropy of attractor distribution
        
        Args:
            x0: Initial states [B, 2]
            num_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            Entropy values [B] - higher values indicate more uncertainty
        """
        probs = self.predict_attractor_distribution(x0, num_samples)  # [B, num_attractors]
        
        # Compute entropy: H = -sum(p * log(p))
        epsilon = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
        
        return entropy