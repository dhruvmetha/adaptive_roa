"""
Circular flow matching inference implementation
"""
import torch
from typing import Optional

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
            output_dim=3,  # Velocity on S¹ × ℝ (3D)
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
    
    def _project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project back to S¹ × ℝ manifold by normalizing circular components
        
        Args:
            x: Current embedded state [batch_size, 3] as (sin θ, cos θ, θ̇)
            
        Returns:
            Projected state with normalized (sin θ, cos θ) on unit circle
        """
        sin_theta, cos_theta, theta_dot = x[..., 0], x[..., 1], x[..., 2]
        
        # Normalize (sin, cos) to unit circle
        norm = torch.sqrt(sin_theta**2 + cos_theta**2)
        sin_theta_proj = sin_theta / (norm + 1e-8)
        cos_theta_proj = cos_theta / (norm + 1e-8)
        
        # θ̇ component stays unchanged (no constraint on ℝ)
        return torch.stack([sin_theta_proj, cos_theta_proj, theta_dot], dim=-1)
    
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