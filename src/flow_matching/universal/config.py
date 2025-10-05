"""
Configuration for universal flow matching
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from ...systems.base import DynamicalSystem


@dataclass
class UniversalFlowMatchingConfig:
    """Configuration for universal flow matching models"""
    
    # Integration parameters
    num_integration_steps: int = 100
    
    # Model architecture  
    hidden_dims: Tuple[int, ...] = (64, 128, 256)
    time_emb_dim: int = 128
    
    # Training parameters
    sigma: float = 0.0  # Noise level for flow matching
    
    # System-specific parameters (set automatically)
    system: Optional[DynamicalSystem] = None
    state_dim: Optional[int] = None
    embedding_dim: Optional[int] = None  
    tangent_dim: Optional[int] = None
    
    def __post_init__(self):
        """Initialize system-dependent parameters"""
        if self.system is not None:
            self.state_dim = self.system.state_dim
            self.embedding_dim = self.system.embedding_dim
            self.tangent_dim = self.system.tangent_dim
    
    @classmethod
    def for_system(cls, system: DynamicalSystem, **kwargs) -> 'UniversalFlowMatchingConfig':
        """
        Create configuration for a specific dynamical system
        
        Args:
            system: DynamicalSystem instance
            **kwargs: Additional configuration parameters
            
        Returns:
            UniversalFlowMatchingConfig instance
        """
        config = cls(system=system, **kwargs)
        return config
    
    @property
    def model_input_dim(self) -> int:
        """Input dimension for the model (current + condition)"""
        if self.embedding_dim is None:
            raise ValueError("System not set. Use UniversalFlowMatchingConfig.for_system()")
        return self.embedding_dim * 2  # current state + condition
    
    @property  
    def model_output_dim(self) -> int:
        """Output dimension for the model (tangent space)"""
        if self.tangent_dim is None:
            raise ValueError("System not set. Use UniversalFlowMatchingConfig.for_system()")
        return self.tangent_dim
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the configured system"""
        if self.system is None:
            return {"system": "None configured"}
            
        return {
            "system_type": type(self.system).__name__,
            "manifold_structure": [
                f"{comp.name}({comp.manifold_type})" 
                for comp in self.system.manifold_components
            ],
            "dimensions": {
                "state": self.state_dim,
                "embedding": self.embedding_dim,
                "tangent": self.tangent_dim,
                "model_input": self.model_input_dim,
                "model_output": self.model_output_dim
            },
            "state_bounds": self.system.state_bounds
        }
    
    def __repr__(self) -> str:
        if self.system is None:
            return f"UniversalFlowMatchingConfig(no_system)"
        
        return (f"UniversalFlowMatchingConfig("
                f"system={type(self.system).__name__}, "
                f"dims=[{self.state_dim}→{self.embedding_dim}→{self.tangent_dim}], "
                f"steps={self.num_integration_steps})")