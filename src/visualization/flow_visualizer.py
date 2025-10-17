"""
Flow path visualization functionality
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Tuple, List, Union
from pathlib import Path

from src.systems.pendulum_config import PendulumConfig
from .phase_space_plots import PhaseSpacePlotter


class FlowVisualizer:
    """Visualize flow paths and trajectories in phase space"""
    
    def __init__(self, config: PendulumConfig = None):
        self.config = config or PendulumConfig()
        self.plotter = PhaseSpacePlotter(config)
    
    def plot_single_flow_path(self,
                            start_state: np.ndarray,
                            path: np.ndarray,
                            endpoint: Optional[np.ndarray] = None,
                            true_endpoint: Optional[np.ndarray] = None,
                            save_path: Optional[Union[str, Path]] = None,
                            figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """Plot a single flow path with time evolution"""
        
        if endpoint is None:
            endpoint = path[-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Phase space trajectory
        ax1 = self.plotter.setup_phase_space_axes(ax1, "Phase Space Flow Path")
        ax1 = self.plotter.add_attractors(ax1)
        
        # Flow path
        ax1.plot(path[:, 0], path[:, 1], 'b-', alpha=0.8, linewidth=2, label='Flow path')
        
        # Points
        ax1.scatter(start_state[0], start_state[1], color='green', s=150, marker='o',
                   label='Start', zorder=5, edgecolor='black', linewidth=2)
        ax1.scatter(endpoint[0], endpoint[1], color='orange', s=150, marker='s',
                   label='Predicted end', zorder=5, edgecolor='black', linewidth=2)
        
        if true_endpoint is not None:
            ax1.scatter(true_endpoint[0], true_endpoint[1], color='red', s=150, marker='^',
                       label='True end', zorder=5, edgecolor='black', linewidth=2)
        
        ax1.legend()
        
        # Plot 2: Time evolution
        time_steps = np.linspace(0, 1, len(path))
        ax2.plot(time_steps, path[:, 0], 'b-', label='Angle (q)', linewidth=2)
        ax2.plot(time_steps, path[:, 1], 'r-', label='Angular velocity (q̇)', linewidth=2)
        
        # Start and end markers
        ax2.scatter(0, start_state[0], color='green', s=60, marker='o', zorder=5)
        ax2.scatter(0, start_state[1], color='green', s=60, marker='o', zorder=5)
        ax2.scatter(1, endpoint[0], color='orange', s=60, marker='s', zorder=5)
        ax2.scatter(1, endpoint[1], color='orange', s=60, marker='s', zorder=5)
        
        if true_endpoint is not None:
            ax2.scatter(1, true_endpoint[0], color='red', s=60, marker='^', zorder=5)
            ax2.scatter(1, true_endpoint[1], color='red', s=60, marker='^', zorder=5)
        
        ax2.set_xlabel('Flow time (t)')
        ax2.set_ylabel('State value')
        ax2.set_title('State Evolution Over Flow Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_multiple_flow_paths(self,
                               start_states: List[np.ndarray],
                               paths: List[np.ndarray],
                               endpoints: Optional[List[np.ndarray]] = None,
                               save_path: Optional[Union[str, Path]] = None,
                               figsize: Tuple[int, int] = (14, 10),
                               max_paths: int = 250) -> plt.Figure:
        """Plot multiple flow paths in a single phase space plot"""
        
        # Limit number of paths for visibility
        n_paths = min(len(start_states), max_paths)
        start_states = start_states[:n_paths]
        paths = paths[:n_paths]
        if endpoints:
            endpoints = endpoints[:n_paths]
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Setup axes
        ax = self.plotter.setup_phase_space_axes(ax, f'Flow Paths: {n_paths} Trajectories')
        ax = self.plotter.add_attractors(ax, show_labels=False)
        
        # Generate colors
        cmap = plt.cm.get_cmap('tab20')
        colors = [cmap(i / len(start_states)) for i in range(len(start_states))]
        
        # Plot all trajectories
        for i, (start_state, path, color) in enumerate(zip(start_states, paths, colors)):
            # Flow path
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=0.8, alpha=0.7)
            
            # Start point
            ax.scatter(start_state[0], start_state[1], color='darkgreen', s=15, 
                      marker='o', zorder=5, alpha=0.8)
            
            # End point
            end_point = endpoints[i] if endpoints else path[-1]
            ax.scatter(end_point[0], end_point[1], color='darkred', s=15, 
                      marker='s', zorder=5, alpha=0.8)
        
        # Create legend
        custom_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen',
                   markersize=10, markeredgecolor='black', markeredgewidth=2, 
                   label='Start Points'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred',
                   markersize=10, markeredgecolor='black', markeredgewidth=2, 
                   label='End Points')
        ]
        
        # Add attractor legend items
        attractor_legend = []
        for name, color in zip(self.config.ATTRACTOR_NAMES, self.config.ATTRACTOR_COLORS):
            attractor_legend.append(plt.Circle((0, 0), 0.1, color=color, alpha=0.4))
        
        ax.legend(custom_elements + attractor_legend,
                 ['Start Points', 'End Points'] + self.config.ATTRACTOR_NAMES,
                 loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_sample_flow_paths(self,
                             start_states: np.ndarray,
                             paths: List[np.ndarray],
                             endpoints: np.ndarray,
                             true_endpoints: Optional[np.ndarray] = None,
                             n_samples: int = 5,
                             save_dir: Optional[Union[str, Path]] = None) -> List[plt.Figure]:
        """Plot individual sample flow paths"""
        
        # Select random samples
        indices = np.random.choice(len(start_states), n_samples, replace=False)
        figures = []
        
        for i, idx in enumerate(indices):
            start = start_states[idx]
            path = paths[idx]
            pred_end = endpoints[idx]
            true_end = true_endpoints[idx] if true_endpoints is not None else None
            
            save_path = None
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(exist_ok=True)
                save_path = save_dir / f"sample_flow_path_{i+1}.png"
            
            fig = self.plot_single_flow_path(
                start, path, pred_end, true_end, save_path,
                figsize=(10, 8)
            )
            
            # Add title with state information
            fig.suptitle(
                f'Sample Flow Path {i+1}\\n'
                f'Start: ({start[0]:.2f}, {start[1]:.2f}) → '
                f'Pred: ({pred_end[0]:.2f}, {pred_end[1]:.2f})' +
                (f' | True: ({true_end[0]:.2f}, {true_end[1]:.2f})' if true_end is not None else ''),
                fontsize=12
            )
            
            figures.append(fig)
            
            if save_path:
                plt.close(fig)  # Close to save memory when saving
        
        return figures