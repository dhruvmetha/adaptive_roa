#!/usr/bin/env python3
"""Compute data bounds for Mountain Car system.

Analyzes trajectory files to compute per-dimension normalization bounds.

Usage:
    python scripts/compute_mountain_car_bounds.py \
        --data_dir /path/to/trajectories \
        --num_files 1000 \
        --output /path/to/bounds.pkl
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import glob


def compute_bounds(data_dir: str, num_files: int, output_file: str):
    """Compute bounds for Mountain Car data.

    Mountain Car state: [position, velocity]
    Both dimensions are Euclidean (ℝ²)

    Args:
        data_dir: Directory containing trajectory files
        num_files: Number of files to analyze
        output_file: Output pickle file path
    """
    print("=" * 80)
    print("Computing Mountain Car Bounds")
    print("=" * 80)
    print()
    print(f"Data directory: {data_dir}")
    print(f"Number of files: {num_files}")
    print(f"Output file: {output_file}")
    print()

    # Find trajectory files
    trajectory_files = sorted(glob.glob(f"{data_dir}/sequence_*.txt"))

    if len(trajectory_files) == 0:
        raise FileNotFoundError(f"No trajectory files found in {data_dir}")

    print(f"Found {len(trajectory_files)} trajectory files")

    # Limit to num_files
    if num_files > 0:
        trajectory_files = trajectory_files[:num_files]
        print(f"Analyzing first {num_files} files")
    else:
        print(f"Analyzing all {len(trajectory_files)} files")

    print()

    # Initialize tracking arrays
    all_positions = []
    all_velocities = []

    # Load data from trajectory files
    print("Loading trajectory data...")
    for traj_file in tqdm(trajectory_files):
        try:
            # Load trajectory: each line is [position, velocity]
            try:
                traj = np.loadtxt(traj_file)
            except:
                traj = np.loadtxt(traj_file, delimiter=',')

            if traj.ndim == 1:
                traj = traj.reshape(1, -1)

            assert traj.shape[1] == 2, f"Expected 2 columns, got {traj.shape[1]}"

            # Collect data
            all_positions.extend(traj[:, 0])
            all_velocities.extend(traj[:, 1])

        except Exception as e:
            print(f"\n⚠️  Error loading {traj_file}: {e}")
            continue

    # Convert to numpy arrays
    positions = np.array(all_positions)
    velocities = np.array(all_velocities)

    print()
    print(f"Total samples: {len(positions):,}")
    print()

    # Compute statistics
    print("Computing bounds...")
    print()

    # Per-dimension bounds
    bounds = {}

    # Dimension 0: Position
    pos_min = positions.min()
    pos_max = positions.max()
    pos_mean = positions.mean()
    pos_std = positions.std()
    pos_limit = max(abs(pos_min), abs(pos_max))

    bounds[0] = {
        'min': float(pos_min),
        'max': float(pos_max),
        'mean': float(pos_mean),
        'std': float(pos_std),
        'limit': float(pos_limit)
    }

    print(f"Dimension 0 (Position):")
    print(f"  Min: {pos_min:.6f}")
    print(f"  Max: {pos_max:.6f}")
    print(f"  Mean: {pos_mean:.6f}")
    print(f"  Std: {pos_std:.6f}")
    print(f"  Symmetric limit: ±{pos_limit:.6f}")
    print()

    # Dimension 1: Velocity
    vel_min = velocities.min()
    vel_max = velocities.max()
    vel_mean = velocities.mean()
    vel_std = velocities.std()
    vel_limit = max(abs(vel_min), abs(vel_max))

    bounds[1] = {
        'min': float(vel_min),
        'max': float(vel_max),
        'mean': float(vel_mean),
        'std': float(vel_std),
        'limit': float(vel_limit)
    }

    print(f"Dimension 1 (Velocity):")
    print(f"  Min: {vel_min:.6f}")
    print(f"  Max: {vel_max:.6f}")
    print(f"  Mean: {vel_mean:.6f}")
    print(f"  Std: {vel_std:.6f}")
    print(f"  Symmetric limit: ±{vel_limit:.6f}")
    print()

    # Compute ranges
    ranges = {
        0: float(pos_max - pos_min),
        1: float(vel_max - vel_min)
    }

    # Package results
    bounds_data = {
        'bounds': bounds,
        'ranges': ranges,
        'statistics': {
            'num_files': len(trajectory_files),
            'num_samples': int(len(positions)),
            'state_dim': 2
        },
        'summary': {
            'position_limit': float(pos_limit),
            'velocity_limit': float(vel_limit)
        },
        'dimension_names': ['position', 'velocity']
    }

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to pickle
    with open(output_file, 'wb') as f:
        pickle.dump(bounds_data, f)

    print(f"✅ Bounds saved to: {output_file}")
    print()
    print("Summary:")
    print(f"  Position limit: ±{pos_limit:.6f}")
    print(f"  Velocity limit: ±{vel_limit:.6f}")
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compute bounds for Mountain Car data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/common/users/shared/pracsys/genMoPlan/data_trajectories/mountain_car_power_0p0008/trajectories",
        help="Directory containing trajectory files"
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=1000,
        help="Number of files to analyze (0 for all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/common/users/dm1487/arcmg_datasets/mountain_car_power_0p0008/mountain_car_data_bounds.pkl",
        help="Output pickle file"
    )

    args = parser.parse_args()

    compute_bounds(args.data_dir, args.num_files, args.output)


if __name__ == "__main__":
    main()
