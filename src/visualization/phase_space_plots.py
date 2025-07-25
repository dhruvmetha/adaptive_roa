"""
Unified phase space plotting functions
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional, Tuple, List, Union
from pathlib import Path

from ..systems.pendulum_config import PendulumConfig


class PhaseSpacePlotter:
    """Unified phase space plotting functionality"""
    
    def __init__(self, config: PendulumConfig = None):
        self.config = config or PendulumConfig()
    
    def setup_phase_space_axes(self, ax: plt.Axes, title: str = "Phase Space") -> plt.Axes:
        """Setup standard phase space axes with labels and ticks"""
        ax.set_xlabel('Angle (q) [radians]', fontsize=12)
        ax.set_ylabel('Angular Velocity (q̇) [rad/s]', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set bounds
        ax.set_xlim(self.config.ANGLE_MIN, self.config.ANGLE_MAX)
        ax.set_ylim(self.config.VELOCITY_MIN, self.config.VELOCITY_MAX)
        
        # Add π-based tick labels for angle
        pi_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        pi_labels = ['-π', '-π/2', '0', 'π/2', 'π']
        ax.set_xticks(pi_ticks)
        ax.set_xticklabels(pi_labels)
        
        # Add π-based tick labels for velocity  
        pi2_ticks = [-2*np.pi, -np.pi, 0, np.pi, 2*np.pi]
        pi2_labels = ['-2π', '-π', '0', 'π', '2π']
        ax.set_yticks(pi2_ticks)
        ax.set_yticklabels(pi2_labels)
        
        # Add grid and reference lines
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        
        return ax
    
    def add_attractors(self, ax: plt.Axes, show_labels: bool = True) -> plt.Axes:
        """Add attractor regions to phase space plot"""
        for i, (attractor, name, color) in enumerate(zip(
            self.config.ATTRACTORS, 
            self.config.ATTRACTOR_NAMES, 
            self.config.ATTRACTOR_COLORS
        )):
            circle = Circle(
                attractor, 
                self.config.ATTRACTOR_RADIUS, 
                color=color, 
                alpha=0.4, 
                label=name if show_labels and i < 3 else None
            )
            ax.add_patch(circle)
        
        return ax
    
    def plot_scatter_comparison(self, 
                              ground_truth: np.ndarray,
                              predictions: np.ndarray,
                              save_path: Optional[Union[str, Path]] = None,
                              figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """Create prediction vs ground truth scatter plots"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Angle comparison
        axes[0].scatter(ground_truth[:, 0], predictions[:, 0], alpha=0.6, s=20)
        axes[0].plot([self.config.ANGLE_MIN, self.config.ANGLE_MAX], 
                    [self.config.ANGLE_MIN, self.config.ANGLE_MAX], 
                    'r--', label='Perfect prediction')
        axes[0].set_xlabel('True Angle')
        axes[0].set_ylabel('Predicted Angle')
        axes[0].set_title('Angle Prediction')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Velocity comparison
        axes[1].scatter(ground_truth[:, 1], predictions[:, 1], alpha=0.6, s=20)
        axes[1].plot([self.config.VELOCITY_MIN, self.config.VELOCITY_MAX], 
                    [self.config.VELOCITY_MIN, self.config.VELOCITY_MAX], 
                    'r--', label='Perfect prediction')
        axes[1].set_xlabel('True Angular Velocity')
        axes[1].set_ylabel('Predicted Angular Velocity')
        axes[1].set_title('Angular Velocity Prediction')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_error_distribution(self,
                              ground_truth: np.ndarray,
                              predictions: np.ndarray,
                              save_path: Optional[Union[str, Path]] = None,
                              figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """Create error distribution plots"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        angle_errors = predictions[:, 0] - ground_truth[:, 0]
        velocity_errors = predictions[:, 1] - ground_truth[:, 1]
        
        axes[0].hist(angle_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Angle Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Angle Error Distribution\\n(MAE: {np.mean(np.abs(angle_errors)):.4f})')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(velocity_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Angular Velocity Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Angular Velocity Error Distribution\\n(MAE: {np.mean(np.abs(velocity_errors)):.4f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_phase_space_comparison(self,
                                  ground_truth: np.ndarray,
                                  predictions: np.ndarray,
                                  start_states: Optional[np.ndarray] = None,
                                  save_path: Optional[Union[str, Path]] = None,
                                  figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Create phase space comparison plot"""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Setup axes
        ax = self.setup_phase_space_axes(ax, "Phase Space Comparison")
        ax = self.add_attractors(ax)
        
        # Plot data
        ax.scatter(ground_truth[:, 0], ground_truth[:, 1], 
                  alpha=0.5, s=20, c='blue', label='True endpoints')
        ax.scatter(predictions[:, 0], predictions[:, 1], 
                  alpha=0.5, s=20, c='orange', label='Predicted endpoints')
        
        if start_states is not None:
            ax.scatter(start_states[:, 0], start_states[:, 1],
                      alpha=0.3, s=15, c='green', label='Start states')
        
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig