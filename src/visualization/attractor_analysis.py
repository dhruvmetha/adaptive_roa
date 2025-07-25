"""
Attractor basin analysis and separatrix detection
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path
from tqdm import tqdm

from ..systems.pendulum_config import PendulumConfig
from .phase_space_plots import PhaseSpacePlotter

# Import both flow matching variants for type checking
try:
    from ..flow_matching.standard.inference import StandardFlowMatchingInference
    from ..flow_matching.circular.inference import CircularFlowMatchingInference
except ImportError:
    # Fallback for old imports
    StandardFlowMatchingInference = None
    CircularFlowMatchingInference = None


class AttractorBasinAnalyzer:
    """Analyze attractor basins and detect separatrix regions"""
    
    def __init__(self, config: PendulumConfig = None):
        self.config = config or PendulumConfig()
        self.plotter = PhaseSpacePlotter(config)
        
        # Colors for visualization
        self.basin_colors = [
            '#FF6B6B',  # Red for center attractor basin
            '#4ECDC4',  # Teal for right attractor basin  
            '#45B7D1',  # Blue for left attractor basin
            '#FFA07A',  # Light salmon for separatrix points
            '#D3D3D3'   # Light gray for no attractor points
        ]
        
        # Analysis results storage
        self.grid_points = None
        self.grid_shape = None
        self.basin_labels = None
        self.separatrix_mask = None
        self.endpoints = None
    
    def detect_inferencer_type(self, inferencer) -> str:
        """Detect the type of inferencer for appropriate handling"""
        if StandardFlowMatchingInference and isinstance(inferencer, StandardFlowMatchingInference):
            return 'standard'
        elif CircularFlowMatchingInference and isinstance(inferencer, CircularFlowMatchingInference):
            return 'circular'
        else:
            # Fallback detection based on class name
            class_name = inferencer.__class__.__name__.lower()
            if 'circular' in class_name:
                return 'circular'
            else:
                return 'standard'
    
    def analyze_attractor_basins(self,
                               inferencer,
                               resolution: float = 0.1,
                               batch_size: int = 1000) -> Dict[str, Any]:
        """
        Analyze attractor basins by discretizing state space
        
        Args:
            inferencer: Model inference object with predict_endpoint method
            resolution: Grid resolution for discretization
            batch_size: Batch size for efficient processing
            
        Returns:
            Dictionary with analysis results
        """
        # Detect inferencer type
        inferencer_type = self.detect_inferencer_type(inferencer)
        print(f"Analyzing attractor basins with resolution {resolution}...")
        print(f"Detected inferencer type: {inferencer_type}")
        
        # Create discretized grid
        self.grid_points, self.grid_shape = self.config.create_discretized_grid(resolution)
        n_points = len(self.grid_points)
        
        print(f"Created grid with {n_points} points ({self.grid_shape[0]}x{self.grid_shape[1]})")
        
        # Predict endpoints for all grid points in batches
        self.endpoints = np.zeros_like(self.grid_points)
        
        print("Predicting endpoints for all grid points...")
        for i in tqdm(range(0, n_points, batch_size)):
            end_idx = min(i + batch_size, n_points)
            batch_points = self.grid_points[i:end_idx]
            
            # Convert to tensor and predict
            batch_tensor = torch.tensor(batch_points, dtype=torch.float32)
            batch_endpoints = inferencer.predict_endpoint(batch_tensor)
            
            self.endpoints[i:end_idx] = batch_endpoints.cpu().numpy()
        
        # Classify each grid point by its endpoint
        self.basin_labels = self._classify_basin_membership()
        
        # Detect separatrix points
        self.separatrix_mask = self._detect_separatrix_points()
        
        # Compute statistics
        stats = self._compute_basin_statistics()
        
        print(f"Basin analysis complete!")
        print(f"  Resolution: {resolution}")
        print(f"  Grid points: {n_points}")
        print(f"  Basin distribution: {stats['basin_counts']}")
        print(f"  Separatrix points: {stats['separatrix_count']} ({stats['separatrix_percentage']:.1f}%)")
        
        return {
            'resolution': resolution,
            'grid_points': self.grid_points,
            'grid_shape': self.grid_shape,
            'endpoints': self.endpoints,
            'basin_labels': self.basin_labels,
            'separatrix_mask': self.separatrix_mask,
            'statistics': stats
        }
    
    def _classify_basin_membership(self) -> np.ndarray:
        """
        Classify each grid point by which attractor basin it belongs to
        
        Returns:
            Array of shape [N] with basin labels:
            0, 1, 2 = attractor basins, 3 = separatrix, 4 = no attractor
        """
        n_points = len(self.endpoints)
        print(n_points)
        basin_labels = np.full(n_points, 4, dtype=int)  # Default: no attractor
        print(basin_labels.shape)
        # Check membership in each attractor
        attractor_membership = self.config.is_in_attractor(self.endpoints)  # Shape: [N, 3]
        
        print(attractor_membership.shape)
        
        for i in range(len(self.config.ATTRACTORS)):
            # Points that belong to this attractor
            in_this_attractor = attractor_membership[:, i]
            basin_labels[in_this_attractor] = i
        
        return basin_labels
    
    def _detect_separatrix_points(self) -> np.ndarray:
        """
        Detect separatrix points (belong to no attractor or multiple attractors)
        
        Returns:
            Boolean array of shape [N] indicating separatrix points
        """
        # Points that don't belong to any attractor are separatrix points
        no_attractor_mask = (self.basin_labels == 4)
        
        # Also check for points that belong to multiple attractors
        # (shouldn't happen with current logic, but good to check)
        attractor_membership = self.config.is_in_attractor(self.endpoints)
        multiple_attractors_mask = np.sum(attractor_membership, axis=1) > 1
        
        separatrix_mask = no_attractor_mask | multiple_attractors_mask
        
        # Update basin labels for separatrix points
        self.basin_labels[separatrix_mask] = 3
        
        return separatrix_mask
    
    def _compute_basin_statistics(self) -> Dict[str, Any]:
        """Compute statistics about the basin analysis"""
        n_total = len(self.basin_labels)
        
        # Count points in each category
        basin_counts = {}
        for i in range(len(self.config.ATTRACTORS)):
            count = np.sum(self.basin_labels == i)
            basin_counts[f'attractor_{i}'] = count
            basin_counts[f'attractor_{i}_percent'] = 100 * count / n_total
        
        separatrix_count = np.sum(self.basin_labels == 3)
        no_attractor_count = np.sum(self.basin_labels == 4)
        
        basin_counts['separatrix'] = separatrix_count
        basin_counts['separatrix_percent'] = 100 * separatrix_count / n_total
        basin_counts['no_attractor'] = no_attractor_count
        basin_counts['no_attractor_percent'] = 100 * no_attractor_count / n_total
        
        return {
            'total_points': n_total,
            'basin_counts': basin_counts,
            'separatrix_count': separatrix_count,
            'separatrix_percentage': 100 * separatrix_count / n_total
        }
    
    def visualize_attractor_basins(self,
                                 save_path: Optional[Union[str, Path]] = None,
                                 figsize: Tuple[int, int] = (14, 10),
                                 show_grid_points: bool = False,
                                 point_size: float = 1.0) -> plt.Figure:
        """
        Visualize the attractor basins and separatrix regions
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
            show_grid_points: Whether to show individual grid points
            point_size: Size of grid points if shown
            
        Returns:
            Matplotlib figure
        """
        if self.basin_labels is None:
            raise ValueError("Must run analyze_attractor_basins first")
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Setup phase space axes
        ax = self.plotter.setup_phase_space_axes(ax, "Attractor Basins and Separatrix")
        
        # Create custom colormap
        basin_colors_rgb = [
            [1.0, 0.42, 0.42],  # Red for center attractor
            [0.31, 0.80, 0.77], # Teal for right attractor
            [0.27, 0.72, 0.82], # Blue for left attractor  
            [1.0, 0.63, 0.48],  # Light salmon for separatrix
            [0.83, 0.83, 0.83]  # Light gray for no attractor
        ]
        cmap = ListedColormap(basin_colors_rgb)
        
        if show_grid_points:
            # Show individual grid points
            scatter = ax.scatter(
                self.grid_points[:, 0], 
                self.grid_points[:, 1],
                c=self.basin_labels, 
                cmap=cmap, 
                s=point_size,
                alpha=0.8,
                vmin=0, 
                vmax=4
            )
        else:
            # Show as filled contour/heatmap
            basin_grid = self.basin_labels.reshape(self.grid_shape)
            
            # Create coordinate arrays for imshow
            angle_coords = np.linspace(
                self.config.ANGLE_MIN, 
                self.config.ANGLE_MAX, 
                self.grid_shape[1]
            )
            velocity_coords = np.linspace(
                self.config.VELOCITY_MIN, 
                self.config.VELOCITY_MAX, 
                self.grid_shape[0]  
            )
            
            im = ax.imshow(
                basin_grid,
                extent=[
                    self.config.ANGLE_MIN, self.config.ANGLE_MAX,
                    self.config.VELOCITY_MIN, self.config.VELOCITY_MAX
                ],
                origin='lower',
                cmap=cmap,
                alpha=0.8,
                vmin=0,
                vmax=4,
                aspect='auto'
            )
        
        # Add attractor circles on top
        ax = self.plotter.add_attractors(ax, show_labels=False)
        
        # Create custom legend
        legend_elements = []
        legend_labels = []
        
        for i, (name, color) in enumerate(zip(self.config.ATTRACTOR_NAMES, basin_colors_rgb[:3])):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.8))
            legend_labels.append(f'{name} Basin')
        
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=basin_colors_rgb[3], alpha=0.8))
        legend_labels.append('Separatrix')
        
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=basin_colors_rgb[4], alpha=0.8))  
        legend_labels.append('No Attractor')
        
        # Add attractor markers to legend
        for i, (name, color) in enumerate(zip(self.config.ATTRACTOR_NAMES, self.config.ATTRACTOR_COLORS)):
            legend_elements.append(plt.Circle((0, 0), 0.1, color=color, alpha=0.6))
            legend_labels.append(f'{name} Center')
        
        ax.legend(legend_elements, legend_labels, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Attractor basin visualization saved to {save_path}")
        
        return fig
    
    def visualize_basin_statistics(self,
                                 save_path: Optional[Union[str, Path]] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualize statistics about the basin analysis
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.basin_labels is None:
            raise ValueError("Must run analyze_attractor_basins first")
        
        stats = self._compute_basin_statistics()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Basin size distribution (pie chart)
        basin_names = []
        basin_sizes = []
        basin_colors = []
        
        for i, name in enumerate(self.config.ATTRACTOR_NAMES):
            count = stats['basin_counts'][f'attractor_{i}']
            if count > 0:
                basin_names.append(f'{name} Basin')
                basin_sizes.append(count)
                basin_colors.append(self.basin_colors[i])
        
        if stats['basin_counts']['separatrix'] > 0:
            basin_names.append('Separatrix')
            basin_sizes.append(stats['basin_counts']['separatrix'])
            basin_colors.append(self.basin_colors[3])
        
        if stats['basin_counts']['no_attractor'] > 0:
            basin_names.append('No Attractor')
            basin_sizes.append(stats['basin_counts']['no_attractor'])
            basin_colors.append(self.basin_colors[4])
        
        ax1.pie(basin_sizes, labels=basin_names, colors=basin_colors, autopct='%1.1f%%')
        ax1.set_title('Basin Size Distribution')
        
        # Distance to attractors histogram
        distances_to_closest = np.zeros(len(self.endpoints))
        for i in range(len(self.endpoints)):
            _, distances = self.config.get_closest_attractor(self.endpoints[i:i+1])
            distances_to_closest[i] = distances[0]
        
        ax2.hist(distances_to_closest, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(x=self.config.ATTRACTOR_RADIUS, color='red', linestyle='--', 
                   label=f'Attractor radius ({self.config.ATTRACTOR_RADIUS})')
        ax2.set_xlabel('Distance to Closest Attractor')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Distances to Closest Attractor')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Basin statistics visualization saved to {save_path}")
        
        return fig
    
    def save_analysis_results(self, 
                            output_dir: Union[str, Path],
                            analysis_results: Dict[str, Any]) -> None:
        """
        Save complete analysis results including data and visualizations
        
        Args:
            output_dir: Directory to save results
            analysis_results: Results from analyze_attractor_basins
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save basin visualization
        self.visualize_attractor_basins(
            save_path=output_dir / "attractor_basins.png",
            figsize=(16, 12)
        )
        
        # Save statistics visualization
        self.visualize_basin_statistics(
            save_path=output_dir / "basin_statistics.png"
        )
        
        # Save grid points visualization (zoomed in)
        self.visualize_attractor_basins(
            save_path=output_dir / "attractor_basins_grid_points.png",
            figsize=(16, 12),
            show_grid_points=True,
            point_size=2.0
        )
        
        # Save raw data
        np.savez(
            output_dir / "basin_analysis_data.npz",
            grid_points=analysis_results['grid_points'],
            endpoints=analysis_results['endpoints'],
            basin_labels=analysis_results['basin_labels'],
            separatrix_mask=analysis_results['separatrix_mask'],
            resolution=analysis_results['resolution']
        )
        
        # Save statistics report
        stats = analysis_results['statistics']
        with open(output_dir / "basin_analysis_report.txt", 'w') as f:
            f.write("ATTRACTOR BASIN ANALYSIS REPORT\\n")
            f.write("=" * 50 + "\\n")
            f.write(f"Resolution: {analysis_results['resolution']}\\n")
            f.write(f"Total grid points: {stats['total_points']}\\n")
            f.write(f"Grid shape: {analysis_results['grid_shape']}\\n\\n")
            
            f.write("Basin Distribution:\\n")
            for i, name in enumerate(self.config.ATTRACTOR_NAMES):
                count = stats['basin_counts'][f'attractor_{i}']
                percent = stats['basin_counts'][f'attractor_{i}_percent']
                f.write(f"  {name} Basin: {count} points ({percent:.1f}%)\\n")
            
            f.write(f"  Separatrix: {stats['basin_counts']['separatrix']} points ({stats['basin_counts']['separatrix_percent']:.1f}%)\\n")
            f.write(f"  No Attractor: {stats['basin_counts']['no_attractor']} points ({stats['basin_counts']['no_attractor_percent']:.1f}%)\\n")
        
        print(f"Complete basin analysis saved to {output_dir}")
    
    def _plot_basins_on_axes(self, ax, show_grid_points=False, point_size=1.0):
        """Helper method to plot basins on existing axes"""
        if self.basin_labels is None:
            raise ValueError("Must run analyze_attractor_basins first")
        
        basin_colors_rgb = [
            [1.0, 0.42, 0.42],  # Red for center attractor
            [0.31, 0.80, 0.77], # Teal for right attractor
            [0.27, 0.72, 0.82], # Blue for left attractor  
            [1.0, 0.63, 0.48],  # Light salmon for separatrix
            [0.83, 0.83, 0.83]  # Light gray for no attractor
        ]
        
        if show_grid_points:
            colors = [basin_colors_rgb[label] for label in self.basin_labels]
            ax.scatter(
                self.grid_points[:, 0], 
                self.grid_points[:, 1],
                c=colors,
                s=point_size,
                alpha=0.8
            )
        else:
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(basin_colors_rgb)
            basin_grid = self.basin_labels.reshape(self.grid_shape)
            
            ax.imshow(
                basin_grid,
                extent=[
                    self.config.ANGLE_MIN, self.config.ANGLE_MAX,
                    self.config.VELOCITY_MIN, self.config.VELOCITY_MAX
                ],
                origin='lower',
                cmap=cmap,
                alpha=0.8,
                vmin=0,
                vmax=4,
                aspect='auto'
            )
        
        # Add attractors
        self.plotter.add_attractors(ax, show_labels=False)