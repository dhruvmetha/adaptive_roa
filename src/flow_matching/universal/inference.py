"""
Universal flow matching inference supporting multiple dynamical systems
"""
import torch
import numpy as np
from typing import Union, Tuple, Optional, List
from pathlib import Path

from ..base.inference import BaseFlowMatchingInference
from .config import UniversalFlowMatchingConfig
from .flow_matcher import UniversalFlowMatcher
from ...systems.base import DynamicalSystem
from ...manifold_integration import TheseusIntegrator


class UniversalFlowMatchingInference(BaseFlowMatchingInference):
    """
    Universal inference for flow matching models
    
    Automatically adapts to any dynamical system by using:
    1. System-defined state embedding/extraction
    2. System-aware manifold integration via TheseusIntegrator
    3. Proper normalization using system bounds
    """
    
    def __init__(self, 
                 checkpoint_path: str, 
                 system: DynamicalSystem,
                 config: Optional[UniversalFlowMatchingConfig] = None):
        """
        Initialize universal flow matching inference
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            system: DynamicalSystem defining manifold structure
            config: Configuration object
        """
        self.system = system
        self.integrator = TheseusIntegrator(system)
        
        # Create system-specific config if not provided
        if config is None:
            config = UniversalFlowMatchingConfig.for_system(system)
        elif config.system != system:
            config.system = system
            config.__post_init__()
            
        super().__init__(checkpoint_path, config)
    
    def _load_model(self):
        """Load universal flow matching model from checkpoint"""
        from ...model.universal_unet import UniversalUNet
        import torch.optim as optim
        
        # Create model architecture matching the system
        model_net = UniversalUNet(
            input_dim=self.config.model_input_dim,
            output_dim=self.config.model_output_dim, 
            hidden_dims=list(self.config.hidden_dims),
            time_emb_dim=self.config.time_emb_dim
        )
        
        # Create dummy optimizer and scheduler for loading (not used in inference)
        dummy_optimizer = lambda params: optim.Adam(params, lr=1e-3)
        dummy_scheduler = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        # Load model from checkpoint
        model = UniversalFlowMatcher.load_from_checkpoint(
            self.checkpoint_path,
            system=self.system,
            model=model_net,
            config=self.config,
            optimizer=dummy_optimizer,
            scheduler=dummy_scheduler,
            strict=False  # In case of minor mismatches
        )
        
        return model
    
    def _prepare_state_for_integration(self, state: torch.Tensor) -> torch.Tensor:
        """
        Prepare state for integration using system embedding
        
        Args:
            state: Normalized state tensor [batch_size, state_dim]
            
        Returns:
            embedded: Embedded state tensor [batch_size, embedding_dim]
        """
        return self.system.embed_state(state)
    
    def _extract_state_from_integration(self, embedded_state: torch.Tensor) -> torch.Tensor:
        """
        Extract state from integration using system extraction
        
        Args:
            embedded_state: Embedded state after integration [batch_size, embedding_dim]
            
        Returns:
            state: Extracted state tensor [batch_size, state_dim]
        """
        return self.system.extract_state(embedded_state)
    
    def _get_model_input_dim(self) -> int:
        """Get the input dimension for the model"""
        return self.config.model_input_dim
    
    @torch.no_grad()
    def predict_endpoint(self, 
                        start_states: torch.Tensor, 
                        num_steps: Optional[int] = None,
                        return_path: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict endpoint(s) using universal manifold integration
        
        Args:
            start_states: Initial states [batch_size, state_dim] or [state_dim] for single prediction
            num_steps: Number of integration steps (uses config default if None)
            return_path: If True, return the full integration path
            
        Returns:
            predicted_endpoints: Final states [batch_size, state_dim] or [state_dim]
            paths (optional): Full integration paths [batch_size, num_steps+1, state_dim]
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
        
        # Normalize input states using system bounds
        start_states_norm = self.system.normalize_state(start_states)
        
        # Prepare states for integration (embed for neural network)
        x_embedded = self._prepare_state_for_integration(start_states_norm)
        
        # Integration setup
        dt = 1.0 / num_steps
        
        # Store path if requested (in raw state coordinates)
        if return_path:
            path = [start_states_norm.clone()]
        
        # Current state in raw coordinates for manifold integration
        current_state = start_states_norm.clone()
        
        # Integrate using universal manifold integration
        for i in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * i * dt
            
            # Predict tangent velocity using neural network
            velocity_tangent = self.model.model(
                x_embedded, t, 
                condition=self._prepare_state_for_integration(start_states_norm)
            )
            
            # Integrate one step on the manifold using Theseus
            next_state = self.integrator.integrate_step(current_state, velocity_tangent, dt)
            
            # Update current state and embedding for next iteration
            current_state = next_state
            x_embedded = self._prepare_state_for_integration(current_state)
            
            if return_path:
                path.append(current_state.clone())
        
        # Denormalize final states using system bounds
        predicted_endpoints = self.system.denormalize_state(current_state)
        
        # Handle single output
        if single_input:
            predicted_endpoints = predicted_endpoints.squeeze(0)
        
        if return_path:
            # Process path - denormalize all steps
            path_tensor = torch.stack(path, dim=1)  # [batch_size, num_steps+1, state_dim]
            
            # Denormalize path
            path_denorm = torch.stack([
                self.system.denormalize_state(step) for step in path_tensor.transpose(0, 1)
            ], dim=1)
            
            if single_input:
                path_denorm = path_denorm.squeeze(0)
            
            return predicted_endpoints, path_denorm
        
        return predicted_endpoints
    
    def normalize_state(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Normalize state using system-defined bounds"""
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        return self.system.normalize_state(state.to(self.device))
    
    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Denormalize state using system-defined bounds"""
        return self.system.denormalize_state(state)
    
    def predict_single_from_components(self, **state_components) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Convenient method for single prediction using component names
        
        Example:
            # For pendulum
            result = inferencer.predict_single_from_components(angle=1.5, angular_velocity=0.0)
            
            # For cartpole  
            result = inferencer.predict_single_from_components(
                cart_position=0.0, cart_velocity=0.0, 
                pole_angle=0.1, pole_angular_velocity=0.0
            )
        
        Args:
            **state_components: Named state components
            
        Returns:
            endpoint: Predicted endpoint as numpy array
        """
        # Build state tensor from components
        state_values = []
        for comp in self.system.manifold_components:
            if comp.name not in state_components:
                raise ValueError(f"Missing state component: {comp.name}")
            
            comp_value = state_components[comp.name]
            if comp.dim == 1:
                state_values.append(comp_value)
            else:
                state_values.extend(comp_value)
        
        start_state = torch.tensor(state_values, dtype=torch.float32)
        endpoint = self.predict_endpoint(start_state)
        return endpoint.cpu().numpy()
    
    def get_system_info(self) -> dict:
        """Get information about the configured system"""
        return self.config.get_system_info()
    
    def __repr__(self) -> str:
        return (f"UniversalFlowMatchingInference("
                f"system={type(self.system).__name__}, "
                f"checkpoint={Path(self.checkpoint_path).name}, "
                f"manifolds={len(self.system.manifold_components)})")