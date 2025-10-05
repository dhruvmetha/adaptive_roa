"""
Universal flow matching implementation supporting multiple dynamical systems
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import lightning.pytorch as pl

from ..base.flow_matcher import BaseFlowMatcher
from .config import UniversalFlowMatchingConfig
from ...systems.base import DynamicalSystem
from ...manifold_integration import TheseusIntegrator


class UniversalFlowMatcher(BaseFlowMatcher):
    """
    Universal flow matching with automatic manifold handling
    
    Supports any dynamical system by:
    1. Using system-defined embedding/extraction
    2. Predicting tangent velocities in natural coordinates
    3. Computing targets using system-aware geodesic interpolation
    4. Integrating using proper Lie group operations
    """
    
    def __init__(self, 
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 config: Optional[UniversalFlowMatchingConfig] = None):
        """
        Initialize universal flow matcher
        
        Args:
            system: DynamicalSystem defining manifold structure
            model: Neural network model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            config: Configuration object
        """
        self.system = system
        self.integrator = TheseusIntegrator(system)
        
        # Create system-specific config if not provided
        if config is None:
            config = UniversalFlowMatchingConfig.for_system(system)
        elif config.system is None:
            config.system = system
            config.__post_init__()
            
        super().__init__(model, optimizer, scheduler, config)
    
    def prepare_start_state(self, start_states: torch.Tensor) -> torch.Tensor:
        """
        Prepare start states using system-defined embedding for neural network conditioning
        
        Args:
            start_states: Initial states [batch_size, state_dim] 
            
        Returns:
            Embedded start states [batch_size, embedding_dim]
        """
        # Only embed start states - they're needed for neural network conditioning
        start_embedded = self.system.embed_state(start_states)
        return start_embedded
    
    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute universal flow matching loss
        
        Args:
            batch: Dictionary containing 'start_state' and 'end_state' tensors
            
        Returns:
            Flow matching loss
        """
        # Extract RAW states from batch (not pre-embedded)
        start_states = batch["start_state_original"]  # [batch_size, state_dim] 
        end_states = batch["end_state_original"]      # [batch_size, state_dim]
        
        # Only embed start states (needed for neural network conditioning)
        x0_embedded = self.prepare_start_state(start_states)
        
        # Sample random times
        batch_size = start_states.shape[0]
        t = torch.rand(batch_size, device=self.device)
        
        # Interpolate and compute target velocity (using raw states)
        x_t, target_velocity = self.interpolate_and_compute_target(
            x0_embedded, t, start_states, end_states
        )
        
        # Predict velocity using the model
        predicted_velocity = self.forward(x_t, t, condition=x0_embedded)
        
        # Compute MSE loss between predicted and target velocities
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        
        return loss
    
    def interpolate_and_compute_target(self, 
                                     x0_embedded: torch.Tensor,
                                     t: torch.Tensor,
                                     start_states: torch.Tensor,
                                     end_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform interpolation and compute target velocities using system manifold structure
        
        Args:
            x0_embedded: Start embedded states [batch_size, embedding_dim]
            t: Time parameter [batch_size]
            start_states: Start raw states [batch_size, state_dim] 
            end_states: End raw states [batch_size, state_dim]
            
        Returns:
            Tuple of (interpolated_embedded, target_velocity_tangent)
        """
        # Interpolate in raw state space using manifold structure
        interpolated_states = self.interpolate_states(start_states, end_states, t)
        
        # Embed interpolated states
        x_t = self.system.embed_state(interpolated_states)
        
        # Compute target tangent velocities
        target_velocity = self.compute_target_velocity(start_states, end_states, t)
        
        return x_t, target_velocity
    
    def interpolate_states(self, 
                          start_states: torch.Tensor, 
                          end_states: torch.Tensor, 
                          t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate states respecting manifold structure
        
        Args:
            start_states: Start states [batch_size, state_dim]
            end_states: End states [batch_size, state_dim] 
            t: Time parameter [batch_size]
            
        Returns:
            interpolated_states: [batch_size, state_dim]
        """
        # Handle scalar t
        if t.dim() == 0:
            t = t.expand(start_states.shape[0])
        
        # Ensure t has correct shape for broadcasting
        t = t.unsqueeze(-1)  # [batch_size, 1]
        
        # Decompose states by manifold components
        start_components = self._decompose_state(start_states)
        end_components = self._decompose_state(end_states)
        
        interpolated_components = []
        
        for comp, start_comp, end_comp in zip(
            self.system.manifold_components, start_components, end_components
        ):
            if comp.manifold_type == "SO2":
                # Geodesic interpolation on circle
                theta0 = start_comp[..., 0]  # [batch_size]
                theta1 = end_comp[..., 0]    # [batch_size]
                
                # Compute shortest angular path
                angular_diff = theta1 - theta0
                angular_diff = torch.atan2(torch.sin(angular_diff), torch.cos(angular_diff))
                
                # Interpolate along geodesic
                theta_t = theta0 + t.squeeze(-1) * angular_diff
                interpolated = theta_t.unsqueeze(-1)
                
            elif comp.manifold_type == "Real":
                # Linear interpolation on real line
                interpolated = (1 - t) * start_comp + t * end_comp
                
            else:
                # For other manifold types, fall back to linear interpolation
                # TODO: Implement proper geodesic interpolation for SO3, SE2, SE3
                interpolated = (1 - t) * start_comp + t * end_comp
                print(f"Warning: Using linear interpolation for {comp.manifold_type}. "
                      f"Consider implementing proper geodesic interpolation.")
            
            interpolated_components.append(interpolated)
        
        return torch.cat(interpolated_components, dim=-1)
    
    def compute_target_velocity(self,
                               start_states: torch.Tensor,
                               end_states: torch.Tensor, 
                               t: torch.Tensor) -> torch.Tensor:
        """
        Compute target velocities in tangent space using Theseus for consistency
        
        Args:
            start_states: Start states [batch_size, state_dim]
            end_states: End states [batch_size, state_dim]
            t: Time parameter [batch_size]
            
        Returns:
            target_velocity: [batch_size, tangent_dim]
        """
        # Import Theseus if available
        try:
            import theseus as th
            THESEUS_AVAILABLE = True
        except ImportError:
            THESEUS_AVAILABLE = False
        
        # Decompose states by manifold components
        start_components = self._decompose_state(start_states)
        end_components = self._decompose_state(end_states)
        
        velocity_components = []
        
        for comp, start_comp, end_comp in zip(
            self.system.manifold_components, start_components, end_components
        ):
            if comp.manifold_type == "SO2":
                target_vel = self._compute_so2_target_velocity(start_comp, end_comp, THESEUS_AVAILABLE)
                
            elif comp.manifold_type == "SO3":
                target_vel = self._compute_so3_target_velocity(start_comp, end_comp, THESEUS_AVAILABLE)
                
            elif comp.manifold_type == "SE2":
                target_vel = self._compute_se2_target_velocity(start_comp, end_comp, THESEUS_AVAILABLE)
                
            elif comp.manifold_type == "SE3":
                target_vel = self._compute_se3_target_velocity(start_comp, end_comp, THESEUS_AVAILABLE)
                
            elif comp.manifold_type == "Real":
                # Velocity on real line (no Theseus needed)
                target_vel = end_comp - start_comp  # [batch_size, dim]
                
            else:
                # For unknown manifold types, fall back to difference
                target_vel = end_comp - start_comp
                print(f"Warning: Using linear velocity for unknown manifold type {comp.manifold_type}.")
            
            velocity_components.append(target_vel)
        
        return torch.cat(velocity_components, dim=-1)
    
    def _compute_so2_target_velocity(self, 
                                   start_comp: torch.Tensor, 
                                   end_comp: torch.Tensor,
                                   theseus_available: bool) -> torch.Tensor:
        """
        Compute SO2 target velocity using Theseus log map for consistency
        
        Args:
            start_comp: Start angles [batch_size, 1]
            end_comp: End angles [batch_size, 1]
            theseus_available: Whether Theseus is available
            
        Returns:
            target_velocity: [batch_size, 1] - angular velocity
        """
        if theseus_available:
            try:
                import theseus as th
                
                theta0 = start_comp[..., 0]  # [batch_size]
                theta1 = end_comp[..., 0]    # [batch_size]
                batch_size = theta0.shape[0]
                
                # Create SE2 objects with zero translation (for SO2 behavior)
                zero_translation = torch.zeros(batch_size, 2, device=theta0.device, dtype=theta0.dtype)
                
                start_poses = torch.cat([zero_translation, theta0.unsqueeze(-1)], dim=-1)  # [batch, 3]
                end_poses = torch.cat([zero_translation, theta1.unsqueeze(-1)], dim=-1)    # [batch, 3]
                
                start_se2 = th.SE2(x_y_theta=start_poses)
                end_se2 = th.SE2(x_y_theta=end_poses)
                
                # Compute relative transformation: start^{-1} * end
                relative_se2 = start_se2.inverse().compose(end_se2)
                
                # Get tangent vector using log map
                log_map_result = relative_se2.log_map()  # [batch_size, 3]
                
                # Extract rotation component (angular velocity)
                target_angular_vel = log_map_result[..., 2:3]  # [batch_size, 1]
                
                return target_angular_vel
                
            except Exception as e:
                print(f"Warning: Theseus SO2 target computation failed ({e}), using fallback")
        
        # Fallback: Manual shortest path computation
        theta0 = start_comp[..., 0]
        theta1 = end_comp[..., 0] 
        
        angular_diff = theta1 - theta0
        angular_diff = torch.atan2(torch.sin(angular_diff), torch.cos(angular_diff))
        
        return angular_diff.unsqueeze(-1)  # [batch_size, 1]
    
    def _compute_so3_target_velocity(self,
                                   start_comp: torch.Tensor,
                                   end_comp: torch.Tensor, 
                                   theseus_available: bool) -> torch.Tensor:
        """
        Compute SO3 target velocity using Theseus log map
        
        Args:
            start_comp: Start quaternions [batch_size, 4]
            end_comp: End quaternions [batch_size, 4] 
            theseus_available: Whether Theseus is available
            
        Returns:
            target_velocity: [batch_size, 3] - angular velocity vector
        """
        if theseus_available:
            try:
                import theseus as th
                
                start_so3 = th.SO3(quaternion=start_comp)
                end_so3 = th.SO3(quaternion=end_comp)
                
                # Compute relative rotation: start^{-1} * end
                relative_so3 = start_so3.inverse().compose(end_so3)
                
                # Get tangent vector using log map
                target_angular_vel = relative_so3.log_map()  # [batch_size, 3]
                
                return target_angular_vel
                
            except Exception as e:
                print(f"Warning: Theseus SO3 target computation failed ({e}), using fallback")
        
        # Fallback: Not implemented - SO3 log map is complex
        print("Warning: SO3 target velocity fallback not implemented. Using zero velocity.")
        return torch.zeros_like(start_comp[..., :3])  # [batch_size, 3]
    
    def _compute_se2_target_velocity(self,
                                   start_comp: torch.Tensor,
                                   end_comp: torch.Tensor,
                                   theseus_available: bool) -> torch.Tensor:
        """
        Compute SE2 target velocity using Theseus log map
        
        Args:
            start_comp: Start poses [batch_size, 3] as (x, y, θ)
            end_comp: End poses [batch_size, 3] as (x, y, θ)
            theseus_available: Whether Theseus is available
            
        Returns:
            target_velocity: [batch_size, 3] - velocity (vx, vy, ω)
        """
        if theseus_available:
            try:
                import theseus as th
                
                start_se2 = th.SE2(x_y_theta=start_comp)
                end_se2 = th.SE2(x_y_theta=end_comp)
                
                # Compute relative transformation: start^{-1} * end
                relative_se2 = start_se2.inverse().compose(end_se2)
                
                # Get tangent vector using log map
                target_velocity = relative_se2.log_map()  # [batch_size, 3]
                
                return target_velocity
                
            except Exception as e:
                print(f"Warning: Theseus SE2 target computation failed ({e}), using fallback")
        
        # Fallback: Component-wise difference with angle wrapping
        diff = end_comp - start_comp
        
        # Wrap angle component (assuming it's the last dimension)
        if start_comp.shape[-1] >= 3:  # Has angle component
            angle_diff = diff[..., 2]
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            diff[..., 2] = angle_diff
        
        return diff
    
    def _compute_se3_target_velocity(self,
                                   start_comp: torch.Tensor,
                                   end_comp: torch.Tensor,
                                   theseus_available: bool) -> torch.Tensor:
        """
        Compute SE3 target velocity using Theseus log map
        
        Args:
            start_comp: Start poses [batch_size, 7] as (x, y, z, qw, qx, qy, qz)
            end_comp: End poses [batch_size, 7] as (x, y, z, qw, qx, qy, qz)
            theseus_available: Whether Theseus is available
            
        Returns:
            target_velocity: [batch_size, 6] - velocity (vx, vy, vz, ωx, ωy, ωz)
        """
        if theseus_available:
            try:
                import theseus as th
                
                start_se3 = th.SE3(tensor=start_comp)
                end_se3 = th.SE3(tensor=end_comp)
                
                # Compute relative transformation: start^{-1} * end
                relative_se3 = start_se3.inverse().compose(end_se3)
                
                # Get tangent vector using log map
                target_velocity = relative_se3.log_map()  # [batch_size, 6]
                
                return target_velocity
                
            except Exception as e:
                print(f"Warning: Theseus SE3 target computation failed ({e}), using fallback")
        
        # Fallback: Not implemented - SE3 log map is very complex
        print("Warning: SE3 target velocity fallback not implemented. Using zero velocity.")
        return torch.zeros(start_comp.shape[0], 6, device=start_comp.device, dtype=start_comp.dtype)
    
    def _decompose_state(self, state: torch.Tensor) -> list:
        """Decompose state tensor by manifold components"""
        components = []
        start_idx = 0
        
        for comp in self.system.manifold_components:
            end_idx = start_idx + comp.dim
            components.append(state[..., start_idx:end_idx])
            start_idx = end_idx
            
        return components
    
    def __repr__(self) -> str:
        return (f"UniversalFlowMatcher("
                f"system={type(self.system).__name__}, "
                f"manifolds={len(self.system.manifold_components)}, "
                f"dims={self.system.state_dim}→{self.system.embedding_dim}→{self.system.tangent_dim})")