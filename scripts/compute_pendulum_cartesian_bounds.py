#!/usr/bin/env python3
"""
Compute data bounds for Pendulum Cartesian trajectories

State representation: (x, y, ẋ, ẏ) where (x,y) is position on unit circle

Usage:
    python scripts/compute_pendulum_cartesian_bounds.py --data_dir /path/to/trajectories --output bounds.pkl
"""
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse


def compute_bounds(data_dir: str, num_files: int = None, output_file: str = None):
    """
    Compute min/max bounds for Pendulum Cartesian state variables

    Args:
        data_dir: Directory containing trajectory files
        num_files: Number of files to process (None = all)
        output_file: Path to save pickle file (optional)

    Returns:
        Dictionary with bounds and statistics
    """
    data_dir = Path(data_dir)

    # Find all trajectory files
    trajectory_files = sorted(list(data_dir.glob("sequence_*.txt")))

    if not trajectory_files:
        raise ValueError(f"No trajectory files found in {data_dir}")

    print(f"📂 Found {len(trajectory_files)} trajectory files")

    if num_files is not None:
        trajectory_files = trajectory_files[:num_files]
        print(f"   Processing first {num_files} files")

    # Initialize bounds for 4D state: [x, y, x_dot, y_dot]
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    x_dot_min, x_dot_max = float('inf'), float('-inf')
    y_dot_min, y_dot_max = float('inf'), float('-inf')

    # Track position norms for verification
    position_norms = []

    total_states = 0

    print("🔍 Computing bounds...")
    for file_path in tqdm(trajectory_files, desc="Processing trajectories"):
        try:
            # Load trajectory: x, y, x_dot, y_dot
            data = np.loadtxt(file_path, delimiter=',')

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Update bounds
            x_min = min(x_min, data[:, 0].min())
            x_max = max(x_max, data[:, 0].max())

            y_min = min(y_min, data[:, 1].min())
            y_max = max(y_max, data[:, 1].max())

            x_dot_min = min(x_dot_min, data[:, 2].min())
            x_dot_max = max(x_dot_max, data[:, 2].max())

            y_dot_min = min(y_dot_min, data[:, 3].min())
            y_dot_max = max(y_dot_max, data[:, 3].max())

            # Verify position is on unit circle
            norms = np.sqrt(data[:, 0]**2 + data[:, 1]**2)
            position_norms.extend(norms.tolist())

            total_states += len(data)

        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue

    # Analyze position norms
    position_norms = np.array(position_norms)
    norm_mean = position_norms.mean()
    norm_std = position_norms.std()
    norm_min = position_norms.min()
    norm_max = position_norms.max()

    # Package results
    bounds_data = {
        'bounds': {
            'x': {
                'min': float(x_min),
                'max': float(x_max)
            },
            'y': {
                'min': float(y_min),
                'max': float(y_max)
            },
            'x_dot': {
                'min': float(x_dot_min),
                'max': float(x_dot_max)
            },
            'y_dot': {
                'min': float(y_dot_min),
                'max': float(y_dot_max)
            }
        },
        'statistics': {
            'total_files_processed': len(trajectory_files),
            'total_states_analyzed': total_states,
            'files_requested': num_files,
            'data_directory': str(data_dir),
            'position_norm_mean': float(norm_mean),
            'position_norm_std': float(norm_std),
            'position_norm_min': float(norm_min),
            'position_norm_max': float(norm_max)
        },
        'ranges': {
            'x': float(x_max - x_min),
            'y': float(y_max - y_min),
            'x_dot': float(x_dot_max - x_dot_min),
            'y_dot': float(y_dot_max - y_dot_min)
        }
    }

    # Print results
    print("\n" + "="*80)
    print("📊 Computed Bounds")
    print("="*80)
    print(f"Files processed: {len(trajectory_files)}")
    print(f"Total states: {total_states:,}")
    print()
    print("State Bounds (in state vector order):")
    print(f"  [0] Position x:       [{x_min:.6f}, {x_max:.6f}]  range={x_max-x_min:.6f}")
    print(f"  [1] Position y:       [{y_min:.6f}, {y_max:.6f}]  range={y_max-y_min:.6f}")
    print(f"  [2] Velocity ẋ:       [{x_dot_min:.6f}, {x_dot_max:.6f}]  range={x_dot_max-x_dot_min:.6f}")
    print(f"  [3] Velocity ẏ:       [{y_dot_min:.6f}, {y_dot_max:.6f}]  range={y_dot_max-y_dot_min:.6f}")
    print()
    print("Position constraint verification (should be ~1.0 for unit circle):")
    print(f"  ||(x,y)|| mean: {norm_mean:.6f}")
    print(f"  ||(x,y)|| std:  {norm_std:.6f}")
    print(f"  ||(x,y)|| min:  {norm_min:.6f}")
    print(f"  ||(x,y)|| max:  {norm_max:.6f}")
    print()
    print("Symmetric limits (for normalization):")
    print(f"  x_limit:       ±{max(abs(x_min), abs(x_max)):.6f}")
    print(f"  y_limit:       ±{max(abs(y_min), abs(y_max)):.6f}")
    print(f"  x_dot_limit:   ±{max(abs(x_dot_min), abs(x_dot_max)):.6f}")
    print(f"  y_dot_limit:   ±{max(abs(y_dot_min), abs(y_dot_max)):.6f}")
    print()

    # Save to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(bounds_data, f)

        print(f"💾 Saved bounds to: {output_path}")

    return bounds_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Pendulum Cartesian data bounds")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/common/users/shared/pracsys/genMoPlan/data_trajectories/pendulum_cartesian_50k/trajectories",
        help="Directory containing trajectory files"
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=None,
        help="Number of files to process (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/common/users/dm1487/arcmg_datasets/pendulum_cartesian_50k/pendulum_cartesian_data_bounds.pkl",
        help="Output pickle file path"
    )

    args = parser.parse_args()

    compute_bounds(
        data_dir=args.data_dir,
        num_files=args.num_files,
        output_file=args.output
    )
