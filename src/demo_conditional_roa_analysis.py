"""
Dense grid ROA analysis for conditional flow matching with variance/std computation.

OPTIMIZED VERSION with mega-batch processing for maximum performance!

Performs:
- Dense grid sampling of pendulum phase space
- 50 samples per grid point using conditional flow matching
- Computes standard deviation and variance of endpoint predictions
- Generates heatmap visualizations of prediction uncertainty

Optimizations implemented:
- MEGA-BATCH processing: Eliminates inner loops, processes all samples simultaneously
- Increased default batch size (500) for better GPU utilization
- Vectorized operations throughout for maximum performance
- Expected speedup: 10-100x faster than naive implementation
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.flow_matching.conditional import ConditionalFlowMatchingInference
from src.systems.pendulum_config import PendulumConfig
from src.visualization.phase_space_plots import PhaseSpacePlotter


def create_dense_grid(theta_range=(-np.pi, np.pi), theta_dot_range=(-2*np.pi, 2*np.pi), resolution=0.1):
    """Create dense grid of pendulum states"""
    theta_vals = np.arange(theta_range[0], theta_range[1] + resolution, resolution)
    theta_dot_vals = np.arange(theta_dot_range[0], theta_dot_range[1] + resolution, resolution)
    
    # Create meshgrid
    theta_grid, theta_dot_grid = np.meshgrid(theta_vals, theta_dot_vals, indexing='ij')
    
    # Flatten to get list of (theta, theta_dot) points
    grid_points = np.column_stack([theta_grid.flatten(), theta_dot_grid.flatten()])
    grid_shape = theta_grid.shape
    
    print(f"Created dense grid: {len(grid_points)} points ({grid_shape[0]}x{grid_shape[1]})")
    print(f"Theta range: [{theta_vals[0]:.3f}, {theta_vals[-1]:.3f}] with {len(theta_vals)} points")
    print(f"Theta_dot range: [{theta_dot_vals[0]:.3f}, {theta_dot_vals[-1]:.3f}] with {len(theta_dot_vals)} points")
    
    return grid_points, grid_shape, (theta_vals, theta_dot_vals)


def analyze_grid_variance(inferencer, grid_points, num_samples=50, batch_size=500):
    """
    Analyze variance of endpoint predictions across grid points using mega-batch optimization
    
    Uses vectorized mega-batch processing for maximum GPU utilization and performance.
    Processes batch_size * num_samples predictions simultaneously instead of looping.
    
    Args:
        inferencer: Conditional flow matching inference object
        grid_points: Array of (theta, theta_dot) grid points [N, 2]
        num_samples: Number of samples per grid point
        batch_size: Batch size for processing (default: 500 for better GPU utilization)
        
    Returns:
        Dictionary with analysis results including mean, std, var for each grid point
    """
    n_points = len(grid_points)
    
    # Storage for results
    mean_endpoints = np.zeros((n_points, 2))  # [N, 2] - mean (theta, theta_dot) per grid point
    std_endpoints = np.zeros((n_points, 2))   # [N, 2] - std (theta, theta_dot) per grid point  
    var_endpoints = np.zeros((n_points, 2))   # [N, 2] - var (theta, theta_dot) per grid point
    total_std = np.zeros(n_points)            # [N] - magnitude of std vector
    total_var = np.zeros(n_points)            # [N] - magnitude of var vector
    
    print(f"Analyzing {n_points} grid points with {num_samples} samples each...")
    print(f"Total predictions: {n_points * num_samples:,}")
    print(f"Using MEGA-BATCH optimization with batch size: {batch_size}")
    print(f"Mega-batch size per iteration: {batch_size * num_samples:,} predictions")
    
    # Process grid points in batches
    for i in tqdm(range(0, n_points, batch_size), desc="Processing grid"):
        end_idx = min(i + batch_size, n_points)
        batch_points = grid_points[i:end_idx]
        batch_size_actual = len(batch_points)
        
        # MEGA-BATCH OPTIMIZATION: Process all samples for all points in one batch
        # This eliminates the inner loop and provides ~10-100x speedup!
        # Create expanded batch: [batch_size_actual * num_samples, 2]
        mega_batch = np.repeat(batch_points, num_samples, axis=0)
        
        # Single mega-batch prediction call - MUCH faster than looping!
        mega_endpoints = inferencer.predict_endpoint(
            mega_batch,
            num_steps=50,
            method='rk4'
        )  # [batch_size_actual * num_samples, 2]
        
        # Convert to numpy if needed
        if hasattr(mega_endpoints, 'cpu'):
            mega_endpoints = mega_endpoints.cpu().numpy()
        
        # Reshape to [batch_size_actual, num_samples, 2] for statistics computation
        batch_all_endpoints = mega_endpoints.reshape(batch_size_actual, num_samples, 2)
        
        # Compute statistics for this batch
        batch_means = np.mean(batch_all_endpoints, axis=1)  # [batch_size_actual, 2]
        batch_stds = np.std(batch_all_endpoints, axis=1)    # [batch_size_actual, 2]
        batch_vars = np.var(batch_all_endpoints, axis=1)    # [batch_size_actual, 2]
        
        # Store results
        mean_endpoints[i:end_idx] = batch_means
        std_endpoints[i:end_idx] = batch_stds
        var_endpoints[i:end_idx] = batch_vars
        
        # Compute magnitude (Euclidean norm) of std and var vectors
        total_std[i:end_idx] = np.linalg.norm(batch_stds, axis=1)
        total_var[i:end_idx] = np.linalg.norm(batch_vars, axis=1)
    
    # Compute summary statistics
    print(f"\nVariance Analysis Summary:")
    print(f"Mean std (theta): {np.mean(std_endpoints[:, 0]):.6f} ± {np.std(std_endpoints[:, 0]):.6f}")
    print(f"Mean std (theta_dot): {np.mean(std_endpoints[:, 1]):.6f} ± {np.std(std_endpoints[:, 1]):.6f}")
    print(f"Mean total std magnitude: {np.mean(total_std):.6f} ± {np.std(total_std):.6f}")
    print(f"Max total std magnitude: {np.max(total_std):.6f}")
    print(f"Min total std magnitude: {np.min(total_std):.6f}")
    
    return {
        'grid_points': grid_points,
        'mean_endpoints': mean_endpoints,
        'std_endpoints': std_endpoints,
        'var_endpoints': var_endpoints,
        'total_std': total_std,
        'total_var': total_var,
        'num_samples': num_samples
    }


def create_variance_heatmaps(results, grid_shape, grid_ranges, output_dir, std_clip_percentile=95):
    """Create heatmap visualizations of variance analysis
    
    Args:
        std_clip_percentile: Percentile for clipping outlier standard deviations (default: 95)
    """
    
    # Extract data
    total_std = results['total_std']
    total_var = results['total_var']
    std_endpoints = results['std_endpoints']
    var_endpoints = results['var_endpoints']
    theta_vals, theta_dot_vals = grid_ranges
    
    # Clip outlier standard deviations based on percentile
    std_clip_value = np.percentile(total_std, std_clip_percentile)
    var_clip_value = np.percentile(total_var, std_clip_percentile)
    
    total_std_clipped = np.clip(total_std, 0, std_clip_value)
    total_var_clipped = np.clip(total_var, 0, var_clip_value)
    std_theta_clipped = np.clip(std_endpoints[:, 0], 0, np.percentile(std_endpoints[:, 0], std_clip_percentile))
    std_theta_dot_clipped = np.clip(std_endpoints[:, 1], 0, np.percentile(std_endpoints[:, 1], std_clip_percentile))
    var_theta_clipped = np.clip(var_endpoints[:, 0], 0, np.percentile(var_endpoints[:, 0], std_clip_percentile))
    var_theta_dot_clipped = np.clip(var_endpoints[:, 1], 0, np.percentile(var_endpoints[:, 1], std_clip_percentile))
    
    print(f"\n📊 Clipping outliers at {std_clip_percentile}th percentile:")
    print(f"  Total std: max={np.max(total_std):.6f} → clipped to {std_clip_value:.6f}")
    print(f"  Total var: max={np.max(total_var):.6f} → clipped to {var_clip_value:.6f}")
    print(f"  θ std: max={np.max(std_endpoints[:, 0]):.6f} → clipped to {np.percentile(std_endpoints[:, 0], std_clip_percentile):.6f}")
    print(f"  θ̇ std: max={np.max(std_endpoints[:, 1]):.6f} → clipped to {np.percentile(std_endpoints[:, 1], std_clip_percentile):.6f}")
    
    # Reshape arrays for heatmap plotting (using clipped values)
    total_std_grid = total_std_clipped.reshape(grid_shape)
    total_var_grid = total_var_clipped.reshape(grid_shape)
    std_theta_grid = std_theta_clipped.reshape(grid_shape)
    std_theta_dot_grid = std_theta_dot_clipped.reshape(grid_shape)
    var_theta_grid = var_theta_clipped.reshape(grid_shape)
    var_theta_dot_grid = var_theta_dot_clipped.reshape(grid_shape)
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Configure pendulum system for attractors
    config = PendulumConfig()
    
    # Plot 1: Total Standard Deviation Magnitude
    ax = axes[0, 0]
    im1 = ax.imshow(total_std_grid.T, origin='lower', aspect='auto', cmap='viridis',
                    extent=[theta_vals[0], theta_vals[-1], theta_dot_vals[0], theta_dot_vals[-1]])
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_title(f'Total Standard Deviation Magnitude\n({results["num_samples"]} samples per point)')
    
    # Add attractors
    for attractor in config.ATTRACTORS:
        ax.plot(attractor[0], attractor[1], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)
    
    # Set axis ticks to pi multiples
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
    
    plt.colorbar(im1, ax=ax, label='Std Magnitude')
    
    # Plot 2: Total Variance Magnitude  
    ax = axes[0, 1]
    im2 = ax.imshow(total_var_grid.T, origin='lower', aspect='auto', cmap='plasma',
                    extent=[theta_vals[0], theta_vals[-1], theta_dot_vals[0], theta_dot_vals[-1]])
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_title(f'Total Variance Magnitude\n({results["num_samples"]} samples per point)')
    
    # Add attractors
    for attractor in config.ATTRACTORS:
        ax.plot(attractor[0], attractor[1], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
    
    plt.colorbar(im2, ax=ax, label='Var Magnitude')
    
    # Plot 3: Theta Standard Deviation
    ax = axes[0, 2]
    im3 = ax.imshow(std_theta_grid.T, origin='lower', aspect='auto', cmap='coolwarm',
                    extent=[theta_vals[0], theta_vals[-1], theta_dot_vals[0], theta_dot_vals[-1]])
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_title('θ Standard Deviation')
    
    # Add attractors
    for attractor in config.ATTRACTORS:
        ax.plot(attractor[0], attractor[1], 'k*', markersize=15, markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
    
    plt.colorbar(im3, ax=ax, label='θ Std (rad)')
    
    # Plot 4: Theta_dot Standard Deviation
    ax = axes[1, 0]
    im4 = ax.imshow(std_theta_dot_grid.T, origin='lower', aspect='auto', cmap='coolwarm',
                    extent=[theta_vals[0], theta_vals[-1], theta_dot_vals[0], theta_dot_vals[-1]])
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_title('θ̇ Standard Deviation')
    
    # Add attractors
    for attractor in config.ATTRACTORS:
        ax.plot(attractor[0], attractor[1], 'k*', markersize=15, markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
    
    plt.colorbar(im4, ax=ax, label='θ̇ Std (rad/s)')
    
    # Plot 5: Theta Variance
    ax = axes[1, 1]
    im5 = ax.imshow(var_theta_grid.T, origin='lower', aspect='auto', cmap='inferno',
                    extent=[theta_vals[0], theta_vals[-1], theta_dot_vals[0], theta_dot_vals[-1]])
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_title('θ Variance')
    
    # Add attractors
    for attractor in config.ATTRACTORS:
        ax.plot(attractor[0], attractor[1], 'w*', markersize=15, markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
    
    plt.colorbar(im5, ax=ax, label='θ Var (rad²)')
    
    # Plot 6: Theta_dot Variance
    ax = axes[1, 2]
    im6 = ax.imshow(var_theta_dot_grid.T, origin='lower', aspect='auto', cmap='inferno',
                    extent=[theta_vals[0], theta_vals[-1], theta_dot_vals[0], theta_dot_vals[-1]])
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_title('θ̇ Variance')
    
    # Add attractors
    for attractor in config.ATTRACTORS:
        ax.plot(attractor[0], attractor[1], 'w*', markersize=15, markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
    
    plt.colorbar(im6, ax=ax, label='θ̇ Var (rad²/s²)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'conditional_roa_variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create separate high-detail plots
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # High detail total std (using clipped values)
    total_std_grid_detailed = total_std_clipped.reshape(grid_shape)
    im1 = ax1.imshow(total_std_grid_detailed.T, origin='lower', aspect='auto', cmap='viridis',
                     extent=[theta_vals[0], theta_vals[-1], theta_dot_vals[0], theta_dot_vals[-1]])
    ax1.set_xlabel('θ (rad)')
    ax1.set_ylabel('θ̇ (rad/s)')
    ax1.set_title(f'Endpoint Prediction Standard Deviation\nConditional Flow Matching ({results["num_samples"]} samples/point)')
    
    # Add attractors
    for attractor in config.ATTRACTORS:
        ax1.plot(attractor[0], attractor[1], 'r*', markersize=20, markeredgecolor='white', markeredgewidth=3)
    
    ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax1.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax1.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax1.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
    
    cbar1 = plt.colorbar(im1, ax=ax1, label='Standard Deviation Magnitude')
    
    # High detail total var (using clipped values)
    total_var_grid_detailed = total_var_clipped.reshape(grid_shape)
    im2 = ax2.imshow(total_var_grid_detailed.T, origin='lower', aspect='auto', cmap='plasma',
                     extent=[theta_vals[0], theta_vals[-1], theta_dot_vals[0], theta_dot_vals[-1]])
    ax2.set_xlabel('θ (rad)')
    ax2.set_ylabel('θ̇ (rad/s)')
    ax2.set_title(f'Endpoint Prediction Variance\nConditional Flow Matching ({results["num_samples"]} samples/point)')
    
    # Add attractors
    for attractor in config.ATTRACTORS:
        ax2.plot(attractor[0], attractor[1], 'r*', markersize=20, markeredgecolor='white', markeredgewidth=3)
    
    ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax2.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax2.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax2.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
    
    cbar2 = plt.colorbar(im2, ax=ax2, label='Variance Magnitude')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'conditional_roa_variance_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Variance heatmap visualizations saved:")
    print(f"  - conditional_roa_variance_analysis.png (comprehensive 6-panel view)")
    print(f"  - conditional_roa_variance_detailed.png (high-detail std/var maps)")


def save_analysis_data(results, output_dir):
    """Save analysis data to files"""
    
    # Save raw data
    np.savez(
        output_dir / 'conditional_roa_variance_data.npz',
        grid_points=results['grid_points'],
        mean_endpoints=results['mean_endpoints'],
        std_endpoints=results['std_endpoints'],
        var_endpoints=results['var_endpoints'],
        total_std=results['total_std'],
        total_var=results['total_var'],
        num_samples=results['num_samples']
    )
    
    # Save summary report
    with open(output_dir / 'conditional_roa_variance_report.txt', 'w') as f:
        f.write("CONDITIONAL FLOW MATCHING ROA VARIANCE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Parameters:\n")
        f.write(f"  Grid points: {len(results['grid_points']):,}\n")
        f.write(f"  Samples per point: {results['num_samples']}\n")
        f.write(f"  Total predictions: {len(results['grid_points']) * results['num_samples']:,}\n\n")
        
        f.write(f"Variance Statistics:\n")
        std_endpoints = results['std_endpoints']
        var_endpoints = results['var_endpoints']
        total_std = results['total_std']
        total_var = results['total_var']
        
        f.write(f"  Standard Deviation (θ):\n")
        f.write(f"    Mean: {np.mean(std_endpoints[:, 0]):.6f} rad\n")
        f.write(f"    Std:  {np.std(std_endpoints[:, 0]):.6f} rad\n")
        f.write(f"    Min:  {np.min(std_endpoints[:, 0]):.6f} rad\n")
        f.write(f"    Max:  {np.max(std_endpoints[:, 0]):.6f} rad\n\n")
        
        f.write(f"  Standard Deviation (θ̇):\n")
        f.write(f"    Mean: {np.mean(std_endpoints[:, 1]):.6f} rad/s\n")
        f.write(f"    Std:  {np.std(std_endpoints[:, 1]):.6f} rad/s\n")
        f.write(f"    Min:  {np.min(std_endpoints[:, 1]):.6f} rad/s\n")
        f.write(f"    Max:  {np.max(std_endpoints[:, 1]):.6f} rad/s\n\n")
        
        f.write(f"  Total Standard Deviation Magnitude:\n")
        f.write(f"    Mean: {np.mean(total_std):.6f}\n")
        f.write(f"    Std:  {np.std(total_std):.6f}\n")
        f.write(f"    Min:  {np.min(total_std):.6f}\n")
        f.write(f"    Max:  {np.max(total_std):.6f}\n\n")
        
        f.write(f"  Total Variance Magnitude:\n")
        f.write(f"    Mean: {np.mean(total_var):.6f}\n")
        f.write(f"    Std:  {np.std(total_var):.6f}\n")
        f.write(f"    Min:  {np.min(total_var):.6f}\n")
        f.write(f"    Max:  {np.max(total_var):.6f}\n\n")
    
    print(f"\n💾 Analysis data saved:")
    print(f"  - conditional_roa_variance_data.npz (raw numerical data)")
    print(f"  - conditional_roa_variance_report.txt (summary statistics)")


def main():
    parser = argparse.ArgumentParser(description="Dense grid ROA variance analysis for conditional flow matching")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to conditional flow matching checkpoint")
    parser.add_argument("--output_dir", type=str, default="conditional_roa_analysis", help="Output directory")
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid resolution (default: 0.1)")
    parser.add_argument("--num_samples", type=int, default=50, help="Samples per grid point (default: 50)")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for processing (default: 500)")
    parser.add_argument("--std_clip_percentile", type=float, default=95, help="Percentile for clipping outlier standard deviations (default: 95)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use")
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"🎯 Set CUDA_VISIBLE_DEVICES={args.gpu}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.device_count()} devices")
        print(f"📍 Current device: {torch.cuda.current_device()}")
    else:
        print("💻 Running on CPU")
    
    # Load conditional flow matching model
    print(f"Loading conditional flow matching model from: {args.checkpoint}")
    try:
        inferencer = ConditionalFlowMatchingInference(args.checkpoint)
        print("✅ Model loaded successfully!")
        print(f"Model info: {inferencer.get_model_info()}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Starting dense grid ROA variance analysis:")
    print(f"  Resolution: {args.resolution}")
    print(f"  Samples per point: {args.num_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output directory: {output_dir}")
    
    # Create dense grid
    grid_points, grid_shape, grid_ranges = create_dense_grid(resolution=args.resolution)
    
    # Analyze variance across grid
    results = analyze_grid_variance(
        inferencer, 
        grid_points, 
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
    
    # Create visualizations
    create_variance_heatmaps(results, grid_shape, grid_ranges, output_dir, std_clip_percentile=args.std_clip_percentile)
    
    # Save data
    save_analysis_data(results, output_dir)
    
    print(f"\n🎉 Dense grid ROA variance analysis complete!")
    print(f"📁 All results saved to: {output_dir}")
    print(f"\n📈 Key insights:")
    print(f"  • Processed {len(grid_points):,} grid points")
    print(f"  • Generated {len(grid_points) * args.num_samples:,} endpoint predictions")
    print(f"  • Computed variance/std for endpoint prediction uncertainty")
    print(f"  • Created comprehensive heatmap visualizations")
    print(f"  • Identified regions of high/low prediction variability")


if __name__ == "__main__":
    main()