"""
Analyze extreme variance points from conditional ROA analysis.

Loads saved variance analysis data and identifies datapoints with the highest 
and lowest prediction variance, listing their start states.
"""

import argparse
import numpy as np
import os
from pathlib import Path


def load_variance_data(data_file):
    """Load variance analysis data from NPZ file"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Variance data file not found: {data_file}")
    
    print(f"Loading variance analysis data from: {data_file}")
    data = np.load(data_file)
    
    # Extract arrays
    grid_points = data['grid_points']
    total_var = data['total_var'] 
    total_std = data['total_std']
    var_endpoints = data['var_endpoints']
    std_endpoints = data['std_endpoints']
    mean_endpoints = data['mean_endpoints']
    num_samples = data['num_samples'].item()
    
    print(f"✅ Loaded data for {len(grid_points):,} grid points")
    print(f"   Samples per point: {num_samples}")
    print(f"   Total predictions analyzed: {len(grid_points) * num_samples:,}")
    
    return {
        'grid_points': grid_points,
        'total_var': total_var,
        'total_std': total_std, 
        'var_endpoints': var_endpoints,
        'std_endpoints': std_endpoints,
        'mean_endpoints': mean_endpoints,
        'num_samples': num_samples
    }


def find_extreme_variance_points(data, percentile=5):
    """Find points in bottom and top percentiles by variance"""
    
    grid_points = data['grid_points']
    total_var = data['total_var']
    total_std = data['total_std']
    var_endpoints = data['var_endpoints']
    std_endpoints = data['std_endpoints']
    
    # Calculate percentile thresholds
    n_points = len(total_var)
    n_bottom = int(n_points * percentile / 100)
    n_top = int(n_points * percentile / 100)
    
    # Find indices of extreme points
    max_var_idx = np.argmax(total_var)
    min_var_idx = np.argmin(total_var)
    
    # Find bottom and top percentile indices
    sorted_indices = np.argsort(total_var)
    bottom_percentile_indices = sorted_indices[:n_bottom]
    top_percentile_indices = sorted_indices[-n_top:][::-1]  # Reverse for descending order
    
    print(f"\n🔍 EXTREME VARIANCE ANALYSIS ({percentile}% PERCENTILES)")
    print(f"=" * 60)
    print(f"Total points: {n_points:,}")
    print(f"Bottom {percentile}%: {n_bottom:,} points")
    print(f"Top {percentile}%: {n_top:,} points")
    
    # Single max/min points for reference
    print(f"\n📈 ABSOLUTE HIGHEST VARIANCE POINT:")
    print(f"   Index: {max_var_idx}")
    print(f"   Start state: θ = {grid_points[max_var_idx, 0]:.6f} rad ({np.degrees(grid_points[max_var_idx, 0]):.2f}°)")
    print(f"                θ̇ = {grid_points[max_var_idx, 1]:.6f} rad/s ({np.degrees(grid_points[max_var_idx, 1]):.2f}°/s)")
    print(f"   Total variance magnitude: {total_var[max_var_idx]:.8f}")
    
    print(f"\n📉 ABSOLUTE LOWEST VARIANCE POINT:")
    print(f"   Index: {min_var_idx}")
    print(f"   Start state: θ = {grid_points[min_var_idx, 0]:.6f} rad ({np.degrees(grid_points[min_var_idx, 0]):.2f}°)")
    print(f"                θ̇ = {grid_points[min_var_idx, 1]:.6f} rad/s ({np.degrees(grid_points[min_var_idx, 1]):.2f}°/s)")
    print(f"   Total variance magnitude: {total_var[min_var_idx]:.8f}")
    
    # Top percentile start states
    print(f"\n📈 TOP {percentile}% HIGHEST VARIANCE START STATES ({n_top:,} points):")
    for i, idx in enumerate(top_percentile_indices[:20], 1):  # Show first 20
        theta, theta_dot = grid_points[idx]
        print(f"   {i:2d}. θ = {theta:8.6f} rad ({np.degrees(theta):7.2f}°), "
              f"θ̇ = {theta_dot:8.6f} rad/s ({np.degrees(theta_dot):7.2f}°/s), "
              f"var = {total_var[idx]:.8f}")
    if n_top > 20:
        print(f"   ... and {n_top - 20:,} more points")
    
    # Bottom percentile start states
    print(f"\n📉 BOTTOM {percentile}% LOWEST VARIANCE START STATES ({n_bottom:,} points):")
    for i, idx in enumerate(bottom_percentile_indices[:20], 1):  # Show first 20
        theta, theta_dot = grid_points[idx]
        print(f"   {i:2d}. θ = {theta:8.6f} rad ({np.degrees(theta):7.2f}°), "
              f"θ̇ = {theta_dot:8.6f} rad/s ({np.degrees(theta_dot):7.2f}°/s), "
              f"var = {total_var[idx]:.8f}")
    if n_bottom > 20:
        print(f"   ... and {n_bottom - 20:,} more points")
    
    # Summary statistics
    print(f"\n📊 VARIANCE SUMMARY STATISTICS:")
    print(f"   Mean variance: {np.mean(total_var):.8f}")
    print(f"   Std of variance: {np.std(total_var):.8f}")
    print(f"   Min variance: {np.min(total_var):.8f}")
    print(f"   Max variance: {np.max(total_var):.8f}")
    print(f"   Variance range: {np.max(total_var) - np.min(total_var):.8f}")
    print(f"   Variance ratio (max/min): {np.max(total_var) / np.min(total_var):.2f}")
    
    return {
        'max_var_idx': max_var_idx,
        'min_var_idx': min_var_idx,
        'top_percentile_indices': top_percentile_indices,
        'bottom_percentile_indices': bottom_percentile_indices,
        'max_var_point': grid_points[max_var_idx],
        'min_var_point': grid_points[min_var_idx],
        'max_variance': total_var[max_var_idx],
        'min_variance': total_var[min_var_idx],
        'n_top': n_top,
        'n_bottom': n_bottom,
        'percentile': percentile
    }


def save_extreme_points_report(data, extreme_points, output_file):
    """Save detailed report of extreme variance points with full percentile lists"""
    
    grid_points = data['grid_points']
    total_var = data['total_var']
    total_std = data['total_std']
    var_endpoints = data['var_endpoints']
    std_endpoints = data['std_endpoints']
    num_samples = data['num_samples']
    
    percentile = extreme_points['percentile']
    n_top = extreme_points['n_top']
    n_bottom = extreme_points['n_bottom']
    top_indices = extreme_points['top_percentile_indices']
    bottom_indices = extreme_points['bottom_percentile_indices']
    
    with open(output_file, 'w') as f:
        f.write("EXTREME VARIANCE POINTS ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Analysis Parameters:\n")
        f.write(f"  Total grid points: {len(grid_points):,}\n")
        f.write(f"  Samples per point: {num_samples}\n")
        f.write(f"  Total predictions: {len(grid_points) * num_samples:,}\n")
        f.write(f"  Percentile analyzed: {percentile}%\n")
        f.write(f"  Top {percentile}% points: {n_top:,}\n")
        f.write(f"  Bottom {percentile}% points: {n_bottom:,}\n\n")
        
        # Extreme single points for reference
        max_idx = extreme_points['max_var_idx']
        min_idx = extreme_points['min_var_idx']
        
        f.write("ABSOLUTE HIGHEST VARIANCE POINT:\n")
        f.write(f"  Index: {max_idx}\n")
        f.write(f"  Start state: θ = {grid_points[max_idx, 0]:.6f} rad ({np.degrees(grid_points[max_idx, 0]):.2f}°)\n")
        f.write(f"               θ̇ = {grid_points[max_idx, 1]:.6f} rad/s ({np.degrees(grid_points[max_idx, 1]):.2f}°/s)\n")
        f.write(f"  Total variance magnitude: {total_var[max_idx]:.8f}\n\n")
        
        f.write("ABSOLUTE LOWEST VARIANCE POINT:\n")
        f.write(f"  Index: {min_idx}\n")
        f.write(f"  Start state: θ = {grid_points[min_idx, 0]:.6f} rad ({np.degrees(grid_points[min_idx, 0]):.2f}°)\n")
        f.write(f"               θ̇ = {grid_points[min_idx, 1]:.6f} rad/s ({np.degrees(grid_points[min_idx, 1]):.2f}°/s)\n")
        f.write(f"  Total variance magnitude: {total_var[min_idx]:.8f}\n\n")
        
        # Complete lists of percentile points
        f.write(f"TOP {percentile}% HIGHEST VARIANCE START STATES ({n_top:,} points):\n")
        for i, idx in enumerate(top_indices, 1):
            theta, theta_dot = grid_points[idx]
            f.write(f"  {i:4d}. θ = {theta:8.6f} rad ({np.degrees(theta):7.2f}°), "
                   f"θ̇ = {theta_dot:8.6f} rad/s ({np.degrees(theta_dot):7.2f}°/s), "
                   f"var = {total_var[idx]:.8f}\n")
        
        f.write(f"\nBOTTOM {percentile}% LOWEST VARIANCE START STATES ({n_bottom:,} points):\n")
        for i, idx in enumerate(bottom_indices, 1):
            theta, theta_dot = grid_points[idx]
            f.write(f"  {i:4d}. θ = {theta:8.6f} rad ({np.degrees(theta):7.2f}°), "
                   f"θ̇ = {theta_dot:8.6f} rad/s ({np.degrees(theta_dot):7.2f}°/s), "
                   f"var = {total_var[idx]:.8f}\n")
        
        # Statistics
        f.write(f"\nVARIANCE SUMMARY STATISTICS:\n")
        f.write(f"  Mean variance: {np.mean(total_var):.8f}\n")
        f.write(f"  Std of variance: {np.std(total_var):.8f}\n")
        f.write(f"  Min variance: {np.min(total_var):.8f}\n")
        f.write(f"  Max variance: {np.max(total_var):.8f}\n")
        f.write(f"  Variance range: {np.max(total_var) - np.min(total_var):.8f}\n")
        f.write(f"  Variance ratio (max/min): {np.max(total_var) / np.min(total_var):.2f}\n")
        
        # Percentile statistics
        top_vars = total_var[top_indices]
        bottom_vars = total_var[bottom_indices]
        f.write(f"\nPERCENTILE STATISTICS:\n")
        f.write(f"  Top {percentile}% variance range: {np.min(top_vars):.8f} - {np.max(top_vars):.8f}\n")
        f.write(f"  Top {percentile}% variance mean: {np.mean(top_vars):.8f}\n")
        f.write(f"  Bottom {percentile}% variance range: {np.min(bottom_vars):.8f} - {np.max(bottom_vars):.8f}\n")
        f.write(f"  Bottom {percentile}% variance mean: {np.mean(bottom_vars):.8f}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze extreme variance points from conditional ROA analysis")
    parser.add_argument("--data_file", type=str, required=True, 
                       help="Path to variance analysis data file (.npz)")
    parser.add_argument("--output_dir", type=str, default="extreme_variance_analysis",
                       help="Output directory for reports")
    parser.add_argument("--percentile", type=float, default=5.0,
                       help="Percentile for top/bottom analysis (default: 5.0)")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load variance analysis data
        data = load_variance_data(args.data_file)
        
        # Find extreme variance points
        extreme_points = find_extreme_variance_points(data, percentile=args.percentile)
        
        # Save detailed report
        report_file = output_dir / "extreme_variance_points_report.txt"
        save_extreme_points_report(data, extreme_points, report_file)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        print(f"\n🎉 Extreme variance analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())