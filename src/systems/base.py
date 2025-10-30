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
            manifold_type: Type of manifold ("SO2", "Real")
            dim: Dimension of the component
            name: Human-readable name (e.g., "angle", "position")
        """
        self.manifold_type = manifold_type
        self.dim = dim
        self.name = name


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
    def state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get state bounds"""
        return self._state_bounds
    
    def get_circular_indices(self) -> List[int]:
        """
        Get indices of SO2 (circular) components in state vector

        This is useful for operations that need to treat circular coordinates
        specially (e.g., angle wrapping during perturbation).

        Returns:
            List of integer indices where SO2 components appear in the state vector

        Example:
            CartPole state: [x, θ, ẋ, θ̇] → returns [1] (θ at index 1)
            Pendulum state: [θ, θ̇] → returns [0] (θ at index 0)
        """
        indices = []
        idx = 0

        for comp in self._manifold_components:
            if comp.manifold_type == "SO2":
                indices.append(idx)
            idx += comp.dim

        return indices

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

    def __repr__(self) -> str:
        components_str = ", ".join([
            f"{comp.name}({comp.manifold_type})" for comp in self._manifold_components
        ])
        return f"{self.__class__.__name__}(manifolds=[{components_str}])"


# Legacy compatibility
BaseSystem = DynamicalSystem