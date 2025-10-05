"""
Base class for dynamical systems with Lie group structure
"""
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

try:
    import theseus as th
    THESEUS_AVAILABLE = True
except ImportError:
    THESEUS_AVAILABLE = False
    print("Warning: Theseus not available. Some functionality will be limited.")


class ManifoldComponent:
    """Represents a single manifold component in the state space"""
    
    def __init__(self, manifold_type: str, dim: int, name: str):
        """
        Args:
            manifold_type: Type of manifold ("SO2", "SO3", "SE2", "SE3", "Real")  
            dim: Dimension of the component
            name: Human-readable name (e.g., "angle", "position")
        """
        self.manifold_type = manifold_type
        self.dim = dim
        self.name = name
        
    @property
    def tangent_dim(self) -> int:
        """Dimension of tangent space"""
        if self.manifold_type == "SO2":
            return 1  # Angular velocity
        elif self.manifold_type == "SO3":  
            return 3  # Angular velocity vector
        elif self.manifold_type == "SE2":
            return 3  # [vx, vy, ω]
        elif self.manifold_type == "SE3":
            return 6  # [vx, vy, vz, ωx, ωy, ωz]
        elif self.manifold_type == "Real":
            return self.dim
        else:
            raise ValueError(f"Unknown manifold type: {self.manifold_type}")
    
    @property
    def embedding_dim(self) -> int:
        """Dimension in embedding space"""
        if self.manifold_type == "SO2":
            return 2  # (sin θ, cos θ)
        elif self.manifold_type == "SO3":
            return 4  # Quaternion (w, x, y, z)
        elif self.manifold_type == "SE2":
            return 3  # (x, y, θ) - but θ needs embedding
        elif self.manifold_type == "SE3":  
            return 7  # (x, y, z, qw, qx, qy, qz)
        elif self.manifold_type == "Real":
            return self.dim
        else:
            raise ValueError(f"Unknown manifold type: {self.manifold_type}")


class DynamicalSystem(ABC):
    """
    Abstract base class for dynamical systems with Lie group structure
    
    Each system defines:
    1. Manifold structure of its state space
    2. Embedding/extraction methods for neural networks
    3. State normalization bounds
    """
    
    def __init__(self):
        self._manifold_components = self.define_manifold_structure()
        self._state_bounds = self.define_state_bounds()
        
    @abstractmethod
    def define_manifold_structure(self) -> List[ManifoldComponent]:
        """
        Define the manifold structure of the system's state space
        
        Returns:
            List of ManifoldComponent objects describing each state component
        """
        pass
    
    @abstractmethod 
    def define_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define normalization bounds for each state component
        
        Returns:
            Dictionary mapping component names to (min, max) bounds
        """
        pass
    
    @property
    def manifold_components(self) -> List[ManifoldComponent]:
        """Get manifold components"""
        return self._manifold_components
    
    @property
    def state_dim(self) -> int:
        """Total dimension of raw state"""
        return sum(comp.dim for comp in self._manifold_components)
    
    @property
    def embedding_dim(self) -> int:
        """Total dimension of embedded state for neural network"""
        return sum(comp.embedding_dim for comp in self._manifold_components)
    
    @property
    def tangent_dim(self) -> int:
        """Total dimension of tangent space (velocity prediction)"""
        return sum(comp.tangent_dim for comp in self._manifold_components)
    
    @property
    def state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get state bounds"""
        return self._state_bounds
    
    def embed_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Embed raw state into neural network input space
        
        Args:
            state: Raw state tensor [..., state_dim]
            
        Returns:
            embedded: Embedded state tensor [..., embedding_dim]
        """
        embedded_components = []
        start_idx = 0
        
        for comp in self._manifold_components:
            end_idx = start_idx + comp.dim
            component_state = state[..., start_idx:end_idx]
            
            if comp.manifold_type == "SO2":
                # θ → (sin θ, cos θ)
                theta = component_state[..., 0]
                embedded = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
                
            elif comp.manifold_type == "Real":
                # Pass through unchanged
                embedded = component_state
                
            else:
                raise NotImplementedError(f"Embedding for {comp.manifold_type} not implemented yet")
            
            embedded_components.append(embedded)
            start_idx = end_idx
            
        return torch.cat(embedded_components, dim=-1)
    
    def extract_state(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Extract raw state from embedded representation
        
        Args:
            embedded: Embedded state tensor [..., embedding_dim]
            
        Returns:
            state: Raw state tensor [..., state_dim]
        """
        state_components = []
        start_idx = 0
        
        for comp in self._manifold_components:
            end_idx = start_idx + comp.embedding_dim
            embedded_component = embedded[..., start_idx:end_idx]
            
            if comp.manifold_type == "SO2":
                # (sin θ, cos θ) → θ
                sin_theta = embedded_component[..., 0]
                cos_theta = embedded_component[..., 1]
                theta = torch.atan2(sin_theta, cos_theta)
                state = theta.unsqueeze(-1)
                
            elif comp.manifold_type == "Real":
                # Pass through unchanged
                state = embedded_component
                
            else:
                raise NotImplementedError(f"Extraction for {comp.manifold_type} not implemented yet")
            
            state_components.append(state)
            start_idx = end_idx
            
        return torch.cat(state_components, dim=-1)
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state to [0, 1] range"""
        normalized_components = []
        start_idx = 0
        
        for comp in self._manifold_components:
            end_idx = start_idx + comp.dim
            component_state = state[..., start_idx:end_idx]
            
            if comp.name in self.state_bounds:
                min_bound, max_bound = self.state_bounds[comp.name]
                normalized = (component_state - min_bound) / (max_bound - min_bound)
            else:
                # No normalization bounds specified
                normalized = component_state
                
            normalized_components.append(normalized)
            start_idx = end_idx
            
        return torch.cat(normalized_components, dim=-1)
    
    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Denormalize state from [0, 1] back to original range"""
        denormalized_components = []
        start_idx = 0
        
        for comp in self._manifold_components:
            end_idx = start_idx + comp.dim
            component_state = state[..., start_idx:end_idx]
            
            if comp.name in self.state_bounds:
                min_bound, max_bound = self.state_bounds[comp.name]
                denormalized = component_state * (max_bound - min_bound) + min_bound
            else:
                # No normalization bounds specified
                denormalized = component_state
                
            denormalized_components.append(denormalized)
            start_idx = end_idx
            
        return torch.cat(denormalized_components, dim=-1)
    
    def get_component_slice(self, component_name: str, tensor_type: str = "state") -> slice:
        """
        Get slice indices for a specific component in different tensor representations
        
        Args:
            component_name: Name of the component
            tensor_type: "state", "embedded", or "tangent"
            
        Returns:
            slice object for indexing the component
        """
        start_idx = 0
        
        for comp in self._manifold_components:
            if tensor_type == "state":
                dim = comp.dim
            elif tensor_type == "embedded":
                dim = comp.embedding_dim
            elif tensor_type == "tangent":
                dim = comp.tangent_dim
            else:
                raise ValueError(f"Unknown tensor type: {tensor_type}")
            
            if comp.name == component_name:
                return slice(start_idx, start_idx + dim)
            
            start_idx += dim
            
        raise ValueError(f"Component {component_name} not found")
    
    def __repr__(self) -> str:
        components_str = ", ".join([
            f"{comp.name}({comp.manifold_type})" for comp in self._manifold_components
        ])
        return f"{self.__class__.__name__}(manifolds=[{components_str}])"


# Legacy compatibility
BaseSystem = DynamicalSystem