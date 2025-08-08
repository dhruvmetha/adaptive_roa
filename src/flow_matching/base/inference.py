"""
Abstract base class for flow matching inference
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, List
from pathlib import Path

from .config import FlowMatchingConfig


class BaseFlowMatchingInference(ABC):
    """
    Abstract base class for flow matching inference
    
    Provides common functionality for loading models, state normalization,
    and trajectory integration while allowing subclasses to implement
    variant-specific state handling.
    """
    
    def __init__(self, checkpoint_path: str, config: Optional[FlowMatchingConfig] = None):
        """
        Initialize inference with trained checkpoint
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config: Configuration object (uses default if None)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or FlowMatchingConfig()
        self.checkpoint_path = checkpoint_path
        
        # Load model (implemented by subclasses)
        self.model = self._load_model()
        self.model.eval()
        self.model = self.model.to(self.device)
    
    @abstractmethod
    def _load_model(self):
        """
        Load the trained model from checkpoint
        
        This method must be implemented by subclasses to handle the specific
        model architecture and checkpoint loading for each variant.
        
        Returns:
            Loaded model
        """
        pass
    
    @abstractmethod
    def _prepare_state_for_integration(self, state: torch.Tensor) -> torch.Tensor:
        """
        Prepare state for integration (e.g., embedding for circular)
        
        Args:
            state: Raw state tensor
            
        Returns:
            Prepared state tensor
        """
        pass
    
    @abstractmethod  
    def _extract_state_from_integration(self, integrated_state: torch.Tensor) -> torch.Tensor:
        """
        Extract final state from integration (e.g., extract angles from embedding)
        
        Args:
            integrated_state: State after integration
            
        Returns:
            Extracted state tensor
        """
        pass
    
    @abstractmethod
    def _get_model_input_dim(self) -> int:
        """Get the input dimension for the model"""
        pass
    
    def _project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project state back to manifold (optional, no-op by default)
        
        Subclasses can override this to enforce manifold constraints
        during integration (e.g., unit circle for circular flow).
        
        Args:
            x: Current state tensor
            
        Returns:
            Projected state tensor
        """
        return x
    
    def normalize_state(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Normalize state to [0, 1] range"""
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        
        min_bounds, max_bounds = self.config.state_bounds
        min_bounds = torch.tensor(min_bounds, device=state.device, dtype=state.dtype)
        max_bounds = torch.tensor(max_bounds, device=state.device, dtype=state.dtype)
        
        return (state - min_bounds) / (max_bounds - min_bounds)
    
    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Denormalize state from [0, 1] back to original range"""
        min_bounds, max_bounds = self.config.state_bounds
        min_bounds = torch.tensor(min_bounds, device=state.device, dtype=state.dtype)
        max_bounds = torch.tensor(max_bounds, device=state.device, dtype=state.dtype)
        
        return state * (max_bounds - min_bounds) + min_bounds
    
    @torch.no_grad()
    def predict_endpoint(self, 
                        start_states: torch.Tensor, 
                        num_steps: Optional[int] = None,
                        return_path: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict endpoint(s) for given start state(s)
        
        Args:
            start_states: Initial states [batch_size, 2] or [2] for single prediction
            num_steps: Number of integration steps (uses config default if None)
            return_path: If True, return the full integration path
            
        Returns:
            predicted_endpoints: Final states [batch_size, 2] or [2]
            paths (optional): Full integration paths [batch_size, num_steps+1, 2]
        """
        # Handle single state input
        single_input = start_states.dim() == 1
        if single_input:
            start_states = start_states.unsqueeze(0)
        
        start_states = start_states.to(self.device)
        batch_size = start_states.shape[0]
        
        # Use config default if num_steps not specified
        if num_steps is None:
            num_steps = self.config.num_integration_steps
        
        # Normalize input states
        start_states_norm = self.normalize_state(start_states)
        
        # Prepare states for integration (variant-specific)
        x = self._prepare_state_for_integration(start_states_norm)
        
        # Integration setup
        dt = 1.0 / num_steps
        
        # Store path if requested
        if return_path:
            path = [self._extract_state_from_integration(x.clone())]
        
        # Integrate using Euler method
        for i in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * i * dt
            
            # Predict velocity (using normalized start states as condition)
            velocity = self.model.model(x, t, condition=self._prepare_state_for_integration(start_states_norm))
            
            # Euler step
            x = x + velocity * dt
            
            # Project back to manifold (e.g., unit circle for circular flow)
            x = self._project_to_manifold(x)
            
            if return_path:
                path.append(self._extract_state_from_integration(x.clone()))
        
        # Extract final states and denormalize
        final_states = self._extract_state_from_integration(x)
        predicted_endpoints = self.denormalize_state(final_states)
        
        # Handle single output
        if single_input:
            predicted_endpoints = predicted_endpoints.squeeze(0)
        
        if return_path:
            # Process path
            path_tensor = torch.stack(path, dim=1)  # [batch_size, num_steps+1, state_dim]
            
            # Denormalize path
            path_denorm = torch.stack([
                self.denormalize_state(step) for step in path_tensor.transpose(0, 1)
            ], dim=1)
            
            if single_input:
                path_denorm = path_denorm.squeeze(0)
            
            return predicted_endpoints, path_denorm
        
        return predicted_endpoints
    
    def predict_single(self, 
                      angle: float, 
                      angular_velocity: float,
                      return_path: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Convenient method for single prediction
        
        Args:
            angle: Initial angle (float)
            angular_velocity: Initial angular velocity (float)
            return_path: If True, return integration path
            
        Returns:
            endpoint: Predicted endpoint as numpy array
            path (optional): Integration path as numpy array
        """
        start_state = torch.tensor([angle, angular_velocity], dtype=torch.float32)
        
        if return_path:
            endpoint, path = self.predict_endpoint(start_state, return_path=True)
            return endpoint.cpu().numpy(), path.cpu().numpy()
        else:
            endpoint = self.predict_endpoint(start_state)
            return endpoint.cpu().numpy()
    
    def batch_predict(self, start_states_list: List[List[float]]) -> np.ndarray:
        """
        Predict endpoints for a batch of start states
        
        Args:
            start_states_list: List of [angle, angular_velocity] pairs
            
        Returns:
            predictions: Predicted endpoints as numpy array [n_samples, 2]
        """
        start_tensor = torch.tensor(start_states_list, dtype=torch.float32)
        predictions = self.predict_endpoint(start_tensor)
        return predictions.cpu().numpy()