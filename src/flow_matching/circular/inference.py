"""
Circular flow matching inference implementation
"""
import torch
from typing import Optional, Union, Tuple

try:
    import theseus as th
    THESEUS_AVAILABLE = True
except ImportError:
    THESEUS_AVAILABLE = False
    print("Warning: Theseus not available. Using manual angle wrapping for manifold integration.")

from ..base.inference import BaseFlowMatchingInference
from ..base.config import FlowMatchingConfig
from ..utils.state_transformations import embed_circular_state, extract_circular_state
from .flow_matcher import CircularFlowMatcher
from ...model.circular_unet1d import CircularUNet1D


class CircularFlowMatchingInference(BaseFlowMatchingInference):
    """
    Inference for circular flow matching models
    
    Handles loading circular flow matching checkpoints and provides
    inference functionality with proper circular topology handling.
    """
    
    def __init__(self, checkpoint_path: str, config: Optional[FlowMatchingConfig] = None):
        """
        Initialize circular flow matching inference
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config: Configuration object
        """
        super().__init__(checkpoint_path, config)
    
    def _load_model(self):
        """Load circular flow matching model from checkpoint"""
        # Create model architecture matching training setup
        model_net = CircularUNet1D(
            input_dim=6,   # Current embedded state (3D) + condition (3D)
            output_dim=2,  # 2D tangent velocity (dθ/dt, dθ̇/dt)
            hidden_dims=list(self.config.hidden_dims),
            time_emb_dim=self.config.time_emb_dim
        )
        
        # Load model from checkpoint with the architecture
        model = CircularFlowMatcher.load_from_checkpoint(
            self.checkpoint_path,
            model=model_net,
            config=self.config,
            strict=False  # In case of minor mismatches
        )
        
        return model
    
    def _prepare_state_for_integration(self, state: torch.Tensor) -> torch.Tensor:
        """
        Prepare state for integration by embedding in S¹ × ℝ
        
        Args:
            state: Normalized state tensor [batch_size, 2] as (θ, θ̇)
            
        Returns:
            Embedded state tensor [batch_size, 3] as (sin(θ), cos(θ), θ̇)
        """
        return embed_circular_state(state)
    
    def _extract_state_from_integration(self, integrated_state: torch.Tensor) -> torch.Tensor:
        """
        Extract state from integration by converting from S¹ × ℝ embedding
        
        Args:
            integrated_state: Embedded state after integration [batch_size, 3]
            
        Returns:
            Extracted state tensor [batch_size, 2] as (θ, θ̇)
        """
        return extract_circular_state(integrated_state)
    
    def _get_model_input_dim(self) -> int:
        """Get the input dimension for the circular model"""
        return 6  # Current embedded state (3D) + condition (3D)
    
    def _integrate_on_manifold(self, current_2d: torch.Tensor, velocity_2d: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Integrate on S¹ × ℝ manifold using proper geometric integration
        
        Args:
            current_2d: Current state [batch_size, 2] as (θ, θ̇)
            velocity_2d: Tangent velocity [batch_size, 2] as (dθ/dt, dθ̇/dt)
            dt: Integration time step
            
        Returns:
            next_2d: Next state [batch_size, 2] on manifold
        """
        theta, theta_dot = current_2d[..., 0], current_2d[..., 1] 
        dtheta_dt, dtheta_dot_dt = velocity_2d[..., 0], velocity_2d[..., 1]
        
        if THESEUS_AVAILABLE:
            # Use Theseus for proper manifold integration
            try:
                batch_size = theta.shape[0] if theta.dim() > 0 else 1
                
                # Ensure theta has batch dimension
                if theta.dim() == 0:
                    theta = theta.unsqueeze(0)
                    dtheta_dt = dtheta_dt.unsqueeze(0)
                
                # Create SE2 with zero translation to get SO2 behavior
                zero_xy = torch.zeros(batch_size, 2, device=theta.device, dtype=theta.dtype)
                current_poses = torch.cat([zero_xy, theta.unsqueeze(-1)], dim=-1)  # [batch, 3]
                current_se2 = th.SE2(x_y_theta=current_poses)
                
                # Create tangent vector for SE2 (only rotation component)
                zero_vel = torch.zeros_like(dtheta_dt)
                tangent_vec = torch.stack([zero_vel, zero_vel, dtheta_dt * dt], dim=-1)  # [batch, 3]
                
                # Apply exponential map
                exp_tangent = th.SE2.exp_map(tangent_vec)
                next_se2 = current_se2.compose(exp_tangent)
                
                # Extract angle (automatically wrapped)
                theta_new = next_se2.theta
                
                # Handle single element case
                if batch_size == 1 and theta.dim() == 1:
                    theta_new = theta_new.squeeze(0)
                    
            except Exception as e:
                print(f"Warning: Theseus integration failed ({e}), falling back to manual wrapping")
                # Fallback to manual method
                theta_new = theta + dtheta_dt * dt
                theta_new = torch.atan2(torch.sin(theta_new), torch.cos(theta_new))
            
        else:
            # Fallback: Manual angle wrapping
            theta_new = theta + dtheta_dt * dt
            theta_new = torch.atan2(torch.sin(theta_new), torch.cos(theta_new))
        
        # Standard integration for ℝ component (angular velocity)
        theta_dot_new = theta_dot + dtheta_dot_dt * dt
        
        return torch.stack([theta_new, theta_dot_new], dim=-1)
    
    
    @torch.no_grad()
    def predict_endpoint(self, 
                        start_states: torch.Tensor, 
                        num_steps: Optional[int] = None,
                        return_path: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Override to handle 2D velocity conversion during integration
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
        
        # Integrate directly in 2D manifold coordinates
        for i in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * i * dt
            
            # Predict 2D tangent velocity (dθ/dt, dθ̇/dt)
            velocity_2d = self.model.model(x, t, condition=self._prepare_state_for_integration(start_states_norm))
            
            # Extract current (θ, θ̇) from embedded state
            current_2d = self._extract_state_from_integration(x)
            
            # Integrate on S¹ × ℝ manifold using proper geometry
            next_2d = self._integrate_on_manifold(current_2d, velocity_2d, dt)
            
            # Re-embed back to 3D for next iteration
            x = self._prepare_state_for_integration(next_2d)
            
            if return_path:
                path.append(next_2d.clone())
        
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
    
    def visualize_multiple_flow_paths(self,
                                    start_states_list,
                                    save_path=None,
                                    figsize=(16, 8),
                                    max_paths=50):
        """
        Visualize multiple flow paths for circular flow matching
        
        Args:
            start_states_list: List of [angle, velocity] pairs
            save_path: Path to save the plot
            figsize: Figure size
            max_paths: Maximum number of paths to visualize
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Limit number of paths
        n_paths = min(len(start_states_list), max_paths)
        start_states = start_states_list[:n_paths]
        
        # Generate colors
        colors = plt.cm.tab20(np.linspace(0, 1, n_paths))
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Phase space with flow paths
        ax1 = axes[0]
        
        # Plot attractors
        attractors = [[0, 0], [2.1, 0], [-2.1, 0]]
        for attr in attractors:
            circle = plt.Circle(attr, 0.1, color='gray', alpha=0.3)
            ax1.add_patch(circle)
        
        # Plot flow paths
        for i, start_state in enumerate(start_states):
            # Get flow path
            endpoint, path = self.predict_single(
                start_state[0], start_state[1], return_path=True
            )
            
            # Plot path
            ax1.plot(path[:, 0], path[:, 1], color=colors[i], alpha=0.7, linewidth=1)
            ax1.scatter(start_state[0], start_state[1], color='green', s=20, alpha=0.8)
            ax1.scatter(endpoint[0], endpoint[1], color='red', s=20, alpha=0.8)
        
        ax1.set_xlabel('Angle (θ)')
        ax1.set_ylabel('Angular Velocity (θ̇)')
        ax1.set_title(f'Circular Flow Paths ({n_paths} trajectories)')
        ax1.set_xlim(-np.pi, np.pi)
        ax1.set_ylim(-2*np.pi, 2*np.pi)
        ax1.grid(True, alpha=0.3)
        
        # Add π labels
        ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax1.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        ax1.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
        ax1.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
        
        # Plot 2: Endpoint distribution
        ax2 = axes[1]
        
        endpoints = []
        for start_state in start_states:
            endpoint = self.predict_single(start_state[0], start_state[1])
            endpoints.append(endpoint)
        
        endpoints = np.array(endpoints)
        
        # Plot attractors
        for attr in attractors:
            circle = plt.Circle(attr, 0.1, color='gray', alpha=0.3)
            ax2.add_patch(circle)
        
        # Plot start points and endpoints
        start_array = np.array(start_states)
        ax2.scatter(start_array[:, 0], start_array[:, 1], 
                   color='green', s=30, alpha=0.6, label='Start points')
        ax2.scatter(endpoints[:, 0], endpoints[:, 1], 
                   color='red', s=30, alpha=0.6, label='Endpoints')
        
        ax2.set_xlabel('Angle (θ)')
        ax2.set_ylabel('Angular Velocity (θ̇)')
        ax2.set_title('Start Points vs Predicted Endpoints')
        ax2.set_xlim(-np.pi, np.pi)
        ax2.set_ylim(-2*np.pi, 2*np.pi)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add π labels
        ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax2.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        ax2.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
        ax2.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Multiple flow paths visualization saved to {save_path}")
        
        return fig