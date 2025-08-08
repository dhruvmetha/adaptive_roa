"""
Demo script for the new attractor basin analysis functionality
"""
import os
import torch
import numpy as np
import hydra
from pathlib import Path
from omegaconf import DictConfig

from src.flow_matching.circular import CircularFlowMatchingInference
from src.visualization.attractor_analysis import AttractorBasinAnalyzer
from src.systems.pendulum_config import PendulumConfig


@hydra.main(config_path="../configs", config_name="demo_attractor_analysis.yaml")
def main(cfg: DictConfig):
    print("=" * 60)
    print("ATTRACTOR BASIN ANALYSIS DEMO")
    print("=" * 60)
    
    # Set GPU device from config
    if cfg.device.get("device_id") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device.device_id)
        print(f"Set CUDA_VISIBLE_DEVICES={cfg.device.device_id}")
    
    # Initialize configuration
    config = PendulumConfig()
    
    # Load trained model
    checkpoint_path = cfg.checkpoint_path
    
    print(f"Loading flow matching model from: {checkpoint_path}")
    try:
        inferencer = CircularFlowMatchingInference(checkpoint_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print(f"Please update the checkpoint_path in configs/demo_attractor_analysis.yaml")
        return
    
    # Initialize attractor basin analyzer
    analyzer = AttractorBasinAnalyzer(config)
    
    # Create output directory
    output_dir = Path(cfg.analysis.output_dir)
    
    # Get analysis settings from config
    resolutions = cfg.analysis.resolutions
    batch_size = cfg.analysis.batch_size
    
    # Demo 1: Basic basin analysis with medium resolution
    print("\\n" + "=" * 50)
    print(f"DEMO 1: Basin Analysis with Resolution {resolutions[1]}")
    print("=" * 50)
    
    results = analyzer.analyze_attractor_basins(
        inferencer, 
        resolution=resolutions[1],
        batch_size=batch_size
    )
    
    # Save complete analysis
    analyzer.save_analysis_results(output_dir, results)
    
    # Demo 2: Higher resolution analysis
    print("\\n" + "=" * 50)
    print("DEMO 2: Higher Resolution Analysis (0.05)")
    print("=" * 50)
    
    high_res_results = analyzer.analyze_attractor_basins(
        inferencer,
        resolution=0.05,
        batch_size=500
    )
    
    # Save high resolution analysis
    high_res_dir = output_dir / "high_resolution"
    analyzer.save_analysis_results(high_res_dir, high_res_results)
    
    # Demo 3: Custom resolution analysis
    print("\\n" + "=" * 50)
    print("DEMO 3: Custom Resolution Analysis (0.2)")
    print("=" * 50)
    
    coarse_results = analyzer.analyze_attractor_basins(
        inferencer,
        resolution=0.2,
        batch_size=1000
    )
    
    # Save coarse analysis  
    coarse_dir = output_dir / "coarse_resolution"
    analyzer.save_analysis_results(coarse_dir, coarse_results)
    
    # Demo 4: Analysis comparison
    print("\\n" + "=" * 50)
    print("DEMO 4: Resolution Comparison")
    print("=" * 50)
    
    resolutions = [0.2, 0.1, 0.05]
    all_results = [coarse_results, results, high_res_results]
    
    print("Resolution comparison:")
    print("Resolution | Grid Points | Separatrix Points | Separatrix %")
    print("-" * 60)
    
    for res, res_results in zip(resolutions, all_results):
        stats = res_results['statistics']
        grid_points = stats['total_points']
        separatrix_count = stats['separatrix_count']
        separatrix_percent = stats['separatrix_percentage']
        
        print(f"{res:10.2f} | {grid_points:11d} | {separatrix_count:15d} | {separatrix_percent:10.1f}%")
    
    # Demo 5: Visualize basin boundaries in detail
    print("\\n" + "=" * 50)
    print("DEMO 5: Detailed Basin Boundary Visualization")
    print("=" * 50)
    
    # Create detailed visualization showing both heatmap and scatter
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Coarse resolution heatmap
    analyzer.basin_labels = coarse_results['basin_labels']
    analyzer.grid_points = coarse_results['grid_points']
    analyzer.grid_shape = coarse_results['grid_shape']
    
    ax = axes[0, 0]
    analyzer.plotter.setup_phase_space_axes(ax, f"Coarse Resolution (0.2)")
    analyzer._plot_basins_on_axes(ax, show_grid_points=False)
    
    # Plot 2: Medium resolution heatmap
    analyzer.basin_labels = results['basin_labels']
    analyzer.grid_points = results['grid_points'] 
    analyzer.grid_shape = results['grid_shape']
    
    ax = axes[0, 1]
    analyzer.plotter.setup_phase_space_axes(ax, f"Medium Resolution (0.1)")
    analyzer._plot_basins_on_axes(ax, show_grid_points=False)
    
    # Plot 3: High resolution heatmap
    analyzer.basin_labels = high_res_results['basin_labels']
    analyzer.grid_points = high_res_results['grid_points']
    analyzer.grid_shape = high_res_results['grid_shape']
    
    ax = axes[1, 0]
    analyzer.plotter.setup_phase_space_axes(ax, f"High Resolution (0.05)")
    analyzer._plot_basins_on_axes(ax, show_grid_points=False)
    
    # Plot 4: High resolution with grid points
    ax = axes[1, 1]
    analyzer.plotter.setup_phase_space_axes(ax, f"High Resolution (0.05) - Grid Points")
    analyzer._plot_basins_on_axes(ax, show_grid_points=True, point_size=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "resolution_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Resolution comparison visualization saved")
    
    print("\\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print(f"All results saved to: {output_dir}")
    print("\\nGenerated files:")
    print("- attractor_basins.png: Main basin visualization")
    print("- basin_statistics.png: Statistical analysis")  
    print("- attractor_basins_grid_points.png: Grid point detail")
    print("- basin_analysis_data.npz: Raw analysis data")
    print("- basin_analysis_report.txt: Text report")
    print("- resolution_comparison.png: Multi-resolution comparison")
    print("\\nThe new attractor basin analysis provides:")
    print("• State space discretization with configurable resolution")
    print("• Automatic attractor basin classification")
    print("• Separatrix point detection (points that reach no attractor)")
    print("• Comprehensive visualizations and statistics")
    print("• Support for different analysis resolutions")




if __name__ == "__main__":
    main()