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
        # Optional probabilistic outputs
        self.entropy_map = None
        self.pmax_map = None
        self.margin_map = None
        
        # Standard deviation maps for endpoint predictions
        self.endpoint_std_map = None
        self.endpoint_std_x_map = None
        self.endpoint_std_y_map = None
    
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
                               batch_size: int = 1000,
                               use_probabilistic: bool = False,
                               num_samples: int = 64,
                               thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Analyze attractor basins by discretizing state space
        
        Args:
            inferencer: Model inference object with predict_endpoint method
            resolution: Grid resolution for discretization
            batch_size: Batch size for efficient processing
            use_probabilistic: If True and the inferencer supports it, use
                probabilistic attractor distributions to label basins and
                compute uncertainty. If False, use deterministic endpoints.
            num_samples: Number of samples for probabilistic estimation
            thresholds: Optional dict with keys 'entropy', 'pmax', 'margin'
                to control separatrix detection in probabilistic mode
            
        Returns:
            Dictionary with analysis results
        """
        # Detect inferencer type
        inferencer_type = self.detect_inferencer_type(inferencer)
        print(f"Analyzing attractor basins with resolution {resolution}...")
        print(f"Detected inferencer type: {inferencer_type}")
        
        if use_probabilistic and hasattr(inferencer, 'predict_attractor_distribution'):
            return self._analyze_attractor_basins_probabilistic(
                inferencer,
                resolution=resolution,
                batch_size=batch_size,
                num_samples=num_samples,
                thresholds=thresholds or {'entropy': 0.9, 'pmax': 0.55, 'margin': 0.15}
            )
        
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

    def _analyze_attractor_basins_probabilistic(self,
                                                inferencer,
                                                resolution: float,
                                                batch_size: int,
                                                num_samples: int,
                                                thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Probabilistic basin analysis using LCFM attractor distributions."""
        # Create grid
        self.grid_points, self.grid_shape = self.config.create_discretized_grid(resolution)
        n_points = len(self.grid_points)
        print(f"Created grid with {n_points} points ({self.grid_shape[0]}x{self.grid_shape[1]})")
        
        # Outputs
        K = len(self.config.ATTRACTORS)
        basin_labels = np.full(n_points, 4, dtype=int)  # default no-attractor
        separatrix_mask = np.zeros(n_points, dtype=bool)
        entropy_map = np.zeros(n_points, dtype=float)
        pmax_map = np.zeros(n_points, dtype=float)
        margin_map = np.zeros(n_points, dtype=float)
        
        # Also compute mean endpoints (for downstream stats/plots)
        self.endpoints = np.zeros_like(self.grid_points)
        
        # Arrays to store standard deviations across samples
        endpoint_std_map = np.zeros(n_points, dtype=float)  # Magnitude of endpoint std
        endpoint_std_x_map = np.zeros(n_points, dtype=float)  # X-coordinate std
        endpoint_std_y_map = np.zeros(n_points, dtype=float)  # Y-coordinate std
        
        # Attractor centers tensor
        # Use model device if available
        try:
            model_device = next(inferencer.model.parameters()).device  # type: ignore[attr-defined]
        except Exception:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        centers = torch.tensor(self.config.ATTRACTORS, dtype=torch.float32, device=model_device)
        
        print("Estimating attractor probabilities and uncertainty for all grid points...")
        for i in tqdm(range(0, n_points, batch_size)):
            end_idx = min(i + batch_size, n_points)
            batch_points = self.grid_points[i:end_idx]
            batch_tensor = torch.tensor(batch_points, dtype=torch.float32, device=model_device)
            
            # Probabilities over attractors
            probs = inferencer.predict_attractor_distribution(
                batch_tensor, num_samples=num_samples, attractor_centers=centers
            )  # [B, K]
            probs_np = probs.detach().cpu().numpy()
            sums = probs_np.sum(axis=1)
            has_att = sums > 0
            
            # Hard labels via argmax for points with any mass
            if np.any(has_att):
                basin_labels[i:end_idx][has_att] = probs_np[has_att].argmax(axis=1)
            # no-attractor remains 4
            
            # Uncertainty metrics
            eps = 1e-8
            ent = -(probs_np * np.log(probs_np + eps)).sum(axis=1)
            # pmax and margin (difference between top-1 and top-2)
            # Handle K=1 edge case defensively
            if K >= 2:
                top2 = np.partition(probs_np, -2, axis=1)[:, -2:]
                pmax = top2[:, 1]
                margin = pmax - top2[:, 0]
            else:
                pmax = probs_np.max(axis=1)
                margin = np.zeros_like(pmax)
            
            entropy_map[i:end_idx] = ent
            pmax_map[i:end_idx] = pmax
            margin_map[i:end_idx] = margin
            
            # Separatrix criteria
            sep = (ent > thresholds.get('entropy', 0.9)) | \
                  (pmax < thresholds.get('pmax', 0.55)) | \
                  (margin < thresholds.get('margin', 0.15))
            separatrix_mask[i:end_idx] = sep
            # Overwrite basin labels as separatrix where applicable (but keep 4 for true no-attractor)
            override = sep & has_att
            if np.any(override):
                sub = basin_labels[i:end_idx]
                sub[override] = 3
                basin_labels[i:end_idx] = sub
            
            # Mean endpoints and standard deviations from samples
            try:
                samples = inferencer.sample_endpoints(batch_tensor, num_samples=num_samples)  # [B, N, 2]
                samples_np = samples.detach().cpu().numpy()  # [B, N, 2]
                
                # Compute means
                means = samples_np.mean(axis=1)  # [B, 2]
                self.endpoints[i:end_idx] = means
                
                # Compute standard deviations
                stds = samples_np.std(axis=1)  # [B, 2] - std along sample dimension
                endpoint_std_x_map[i:end_idx] = stds[:, 0]  # X-coordinate std
                endpoint_std_y_map[i:end_idx] = stds[:, 1]  # Y-coordinate std
                
                # Magnitude of standard deviation (Euclidean norm of std vector)
                endpoint_std_map[i:end_idx] = np.linalg.norm(stds, axis=1)
                
            except Exception as e:
                # If sampling is unavailable, leave endpoints and stds as zeros for these points
                print(f"Warning: Could not sample endpoints for batch {i//batch_size}: {e}")
                pass
        
        # Store
        self.basin_labels = basin_labels
        self.separatrix_mask = separatrix_mask
        self.entropy_map = entropy_map
        self.pmax_map = pmax_map
        self.margin_map = margin_map
        self.endpoint_std_map = endpoint_std_map
        self.endpoint_std_x_map = endpoint_std_x_map
        self.endpoint_std_y_map = endpoint_std_y_map
        
        # Compute statistics
        stats = self._compute_basin_statistics()
        
        print(f"Probabilistic basin analysis complete!")
        print(f"  Resolution: {resolution}")
        print(f"  Grid points: {n_points}")
        print(f"  Basin distribution: {stats['basin_counts']}")
        print(f"  Separatrix points: {stats['separatrix_count']} ({stats['separatrix_percentage']:.1f}%)")
        if self.endpoint_std_map is not None:
            valid_std_points = np.sum(self.endpoint_std_map > 0)
            mean_std = self.endpoint_std_map[self.endpoint_std_map > 0].mean()
            print(f"  Standard deviation calculated for {valid_std_points} points (mean: {mean_std:.4f})")
        
        return {
            'resolution': resolution,
            'grid_points': self.grid_points,
            'grid_shape': self.grid_shape,
            'endpoints': self.endpoints,
            'basin_labels': self.basin_labels,
            'separatrix_mask': self.separatrix_mask,
            'statistics': stats,
            'entropy': self.entropy_map,
            'pmax': self.pmax_map,
            'margin': self.margin_map,
            'endpoint_std': self.endpoint_std_map,
            'endpoint_std_x': self.endpoint_std_x_map,
            'endpoint_std_y': self.endpoint_std_y_map,
            'thresholds': thresholds  # Store the thresholds used
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
    
    def visualize_uncertainty_maps(self,
                                   save_dir: Optional[Union[str, Path]] = None,
                                   figsize: Tuple[int, int] = (16, 6)) -> Tuple[plt.Figure, plt.Figure]:
        """Visualize entropy and pmax maps if available."""
        if self.entropy_map is None or self.pmax_map is None:
            raise ValueError("Uncertainty maps not available. Run probabilistic analysis first.")
        
        # Helper to render a heatmap given a flat array
        def _render_heatmap(values: np.ndarray, title: str, cmap: str = 'viridis') -> plt.Figure:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax = self.plotter.setup_phase_space_axes(ax, title)
            grid = values.reshape(self.grid_shape)
            im = ax.imshow(
                grid,
                extent=[self.config.ANGLE_MIN, self.config.ANGLE_MAX,
                        self.config.VELOCITY_MIN, self.config.VELOCITY_MAX],
                origin='lower', cmap=cmap, aspect='auto'
            )
            plt.colorbar(im, ax=ax)
            return fig
        
        fig_entropy = _render_heatmap(self.entropy_map, "Entropy (uncertainty)")
        fig_pmax = _render_heatmap(self.pmax_map, "Max attractor probability", cmap='magma')
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            fig_entropy.savefig(save_dir / "uncertainty_entropy.png", dpi=150, bbox_inches='tight')
            fig_pmax.savefig(save_dir / "uncertainty_pmax.png", dpi=150, bbox_inches='tight')
            print(f"Uncertainty maps saved to {save_dir}")
        
        return fig_entropy, fig_pmax
    
    def visualize_standard_deviation_maps(self,
                                        save_dir: Optional[Union[str, Path]] = None,
                                        figsize: Tuple[int, int] = (20, 6)) -> Tuple[plt.Figure, ...]:
        """Visualize endpoint standard deviation maps if available."""
        if self.endpoint_std_map is None:
            raise ValueError("Standard deviation maps not available. Run probabilistic analysis first.")
        
        # Helper to render a heatmap given a flat array
        def _render_heatmap(values: np.ndarray, title: str, cmap: str = 'plasma') -> plt.Figure:
            fig, ax = plt.subplots(1, 1, figsize=(figsize[0]//3, figsize[1]))
            ax = self.plotter.setup_phase_space_axes(ax, title)
            grid = values.reshape(self.grid_shape)
            im = ax.imshow(
                grid,
                extent=[self.config.ANGLE_MIN, self.config.ANGLE_MAX,
                        self.config.VELOCITY_MIN, self.config.VELOCITY_MAX],
                origin='lower', cmap=cmap, aspect='auto'
            )
            plt.colorbar(im, ax=ax, label='Standard Deviation')
            return fig
        
        # Create individual plots
        fig_std_magnitude = _render_heatmap(self.endpoint_std_map, "Endpoint Std (Magnitude)", cmap='plasma')
        fig_std_x = _render_heatmap(self.endpoint_std_x_map, "Endpoint Std (θ component)", cmap='viridis')
        fig_std_y = _render_heatmap(self.endpoint_std_y_map, "Endpoint Std (ω component)", cmap='inferno')
        
        # Also create a combined figure with all three
        fig_combined, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Magnitude plot
        ax = axes[0]
        ax = self.plotter.setup_phase_space_axes(ax, "Endpoint Std (Magnitude)")
        grid = self.endpoint_std_map.reshape(self.grid_shape)
        im1 = ax.imshow(
            grid,
            extent=[self.config.ANGLE_MIN, self.config.ANGLE_MAX,
                    self.config.VELOCITY_MIN, self.config.VELOCITY_MAX],
            origin='lower', cmap='plasma', aspect='auto'
        )
        plt.colorbar(im1, ax=ax, label='Std Magnitude')
        
        # X (theta) component plot
        ax = axes[1]
        ax = self.plotter.setup_phase_space_axes(ax, "Endpoint Std (θ component)")
        grid = self.endpoint_std_x_map.reshape(self.grid_shape)
        im2 = ax.imshow(
            grid,
            extent=[self.config.ANGLE_MIN, self.config.ANGLE_MAX,
                    self.config.VELOCITY_MIN, self.config.VELOCITY_MAX],
            origin='lower', cmap='viridis', aspect='auto'
        )
        plt.colorbar(im2, ax=ax, label='Std (θ)')
        
        # Y (omega) component plot
        ax = axes[2]
        ax = self.plotter.setup_phase_space_axes(ax, "Endpoint Std (ω component)")
        grid = self.endpoint_std_y_map.reshape(self.grid_shape)
        im3 = ax.imshow(
            grid,
            extent=[self.config.ANGLE_MIN, self.config.ANGLE_MAX,
                    self.config.VELOCITY_MIN, self.config.VELOCITY_MAX],
            origin='lower', cmap='inferno', aspect='auto'
        )
        plt.colorbar(im3, ax=ax, label='Std (ω)')
        
        plt.tight_layout()
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            fig_std_magnitude.savefig(save_dir / "endpoint_std_magnitude.png", dpi=150, bbox_inches='tight')
            fig_std_x.savefig(save_dir / "endpoint_std_theta.png", dpi=150, bbox_inches='tight')
            fig_std_y.savefig(save_dir / "endpoint_std_omega.png", dpi=150, bbox_inches='tight')
            fig_combined.savefig(save_dir / "endpoint_std_combined.png", dpi=150, bbox_inches='tight')
            print(f"Standard deviation maps saved to {save_dir}")
        
        # Close individual figures to save memory
        plt.close(fig_std_magnitude)
        plt.close(fig_std_x)
        plt.close(fig_std_y)
        
        return fig_combined, fig_std_magnitude, fig_std_x, fig_std_y
    
    def visualize_probability_heatmap(self,
                                    save_dir: Optional[Union[str, Path]] = None,
                                    analysis_results: Optional[Dict[str, Any]] = None,
                                    figsize: Tuple[int, int] = (14, 10)) -> Optional[plt.Figure]:
        """
        Visualize probability heatmap based on pmax thresholding.
        If an attractor is selected (pmax > threshold), show the probability.
        If no attractor is selected, mark as 0.5 probability.
        
        Args:
            save_dir: Directory to save the plot
            analysis_results: Results dict containing thresholds info
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if no pmax data available
        """
        if self.pmax_map is None:
            print("Warning: No pmax data available. Skipping probability heatmap.")
            return None
        
        # Get pmax threshold from analysis results or use default
        pmax_threshold = 0.55  # default
        if analysis_results and 'thresholds' in analysis_results:
            pmax_threshold = analysis_results['thresholds'].get('pmax', 0.55)
        
        # Create probability map based on pmax thresholding
        probability_map = np.full_like(self.pmax_map, 0.5)  # Default to 0.5
        
        # Where pmax > threshold, use the actual pmax value
        attractor_selected = self.pmax_map > pmax_threshold
        probability_map[attractor_selected] = self.pmax_map[attractor_selected]
        
        # Reshape to grid
        probability_grid = probability_map.reshape(self.grid_shape)
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = self.plotter.setup_phase_space_axes(ax, f"Probability Heatmap (pmax threshold = {pmax_threshold})")
        
        # Create the heatmap
        im = ax.imshow(
            probability_grid,
            extent=[self.config.ANGLE_MIN, self.config.ANGLE_MAX,
                    self.config.VELOCITY_MIN, self.config.VELOCITY_MAX],
            origin='lower',
            cmap='RdYlBu_r',  # Red-Yellow-Blue colormap (reversed)
            alpha=0.8,
            aspect='auto',
            vmin=0.0,
            vmax=1.0
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Probability', rotation=270, labelpad=20, fontsize=12)
        
        # Add attractor markers
        ax = self.plotter.add_attractors(ax, show_labels=True)
        
        # Add statistics text
        n_selected = np.sum(attractor_selected)
        n_total = len(self.pmax_map)
        selection_percent = 100 * n_selected / n_total
        
        stats_text = f"""Statistics:
Points with attractor selected: {n_selected:,} ({selection_percent:.1f}%)
Points with no attractor: {n_total - n_selected:,} ({100 - selection_percent:.1f}%)
pmax threshold: {pmax_threshold}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            output_path = save_dir / "probability_heatmap_pmax.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Probability heatmap saved to {output_path}")
        
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
        
        # Save uncertainty visualizations if available
        if self.entropy_map is not None and self.pmax_map is not None:
            self.visualize_uncertainty_maps(output_dir)
            # Also save probability heatmap
            self.visualize_probability_heatmap(output_dir, analysis_results)
        
        # Save standard deviation visualizations if available
        if self.endpoint_std_map is not None:
            self.visualize_standard_deviation_maps(output_dir)
        
        # Save grid points visualization (zoomed in)
        self.visualize_attractor_basins(
            save_path=output_dir / "attractor_basins_grid_points.png",
            figsize=(16, 12),
            show_grid_points=True,
            point_size=2.0
        )
        
        # Save raw data
        save_data = {
            'grid_points': analysis_results['grid_points'],
            'endpoints': analysis_results['endpoints'],
            'basin_labels': analysis_results['basin_labels'],
            'separatrix_mask': analysis_results['separatrix_mask'],
            'resolution': analysis_results['resolution']
        }
        
        # Add probabilistic data if available
        if 'entropy' in analysis_results:
            save_data['entropy'] = analysis_results['entropy']
        if 'pmax' in analysis_results:
            save_data['pmax'] = analysis_results['pmax']
        if 'margin' in analysis_results:
            save_data['margin'] = analysis_results['margin']
        
        # Add standard deviation data if available
        if 'endpoint_std' in analysis_results:
            save_data['endpoint_std'] = analysis_results['endpoint_std']
        if 'endpoint_std_x' in analysis_results:
            save_data['endpoint_std_x'] = analysis_results['endpoint_std_x']
        if 'endpoint_std_y' in analysis_results:
            save_data['endpoint_std_y'] = analysis_results['endpoint_std_y']
            
        np.savez(output_dir / "basin_analysis_data.npz", **save_data)
        
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
            
            # Add standard deviation statistics if available
            if 'endpoint_std' in analysis_results:
                f.write("\\nEndpoint Prediction Standard Deviation Statistics:\\n")
                std_data = analysis_results['endpoint_std']
                std_x_data = analysis_results['endpoint_std_x']
                std_y_data = analysis_results['endpoint_std_y']
                
                # Only compute stats for non-zero values (where samples were available)
                nonzero_mask = std_data > 0
                if np.any(nonzero_mask):
                    std_valid = std_data[nonzero_mask]
                    std_x_valid = std_x_data[nonzero_mask] 
                    std_y_valid = std_y_data[nonzero_mask]
                    
                    f.write(f"  Magnitude - Mean: {std_valid.mean():.4f}, Std: {std_valid.std():.4f}, Max: {std_valid.max():.4f}\\n")
                    f.write(f"  θ component - Mean: {std_x_valid.mean():.4f}, Std: {std_x_valid.std():.4f}, Max: {std_x_valid.max():.4f}\\n")
                    f.write(f"  ω component - Mean: {std_y_valid.mean():.4f}, Std: {std_y_valid.std():.4f}, Max: {std_y_valid.max():.4f}\\n")
                    f.write(f"  Points with valid std data: {np.sum(nonzero_mask)} / {len(std_data)} ({100*np.sum(nonzero_mask)/len(std_data):.1f}%)\\n")
        
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