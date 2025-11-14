#!/usr/bin/env python3
"""
Compute data bounds for Mountain Car trajectories

Usage:
    python scripts/compute_mountain_car_bounds.py --data_dir /path/to/trajectories --output bounds.pkl
"""
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse


def compute_bounds(data_dir: str, num_files: int = None, output_file: str = None):
    """
    Compute min/max bounds for Mountain Car state variables

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

    # Initialize bounds
    position_min, position_max = float('inf'), float('-inf')
    velocity_min, velocity_max = float('inf'), float('-inf')

    total_states = 0

    print("🔍 Computing bounds...")
    for file_path in tqdm(trajectory_files, desc="Processing trajectories"):
        try:
            # Load trajectory: position, velocity
            data = np.loadtxt(file_path, delimiter=',')

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Update bounds
            position_min = min(position_min, data[:, 0].min())
            position_max = max(position_max, data[:, 0].max())

            velocity_min = min(velocity_min, data[:, 1].min())
            velocity_max = max(velocity_max, data[:, 1].max())

            total_states += len(data)

        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue

    # Package results
    bounds_data = {
        'bounds': {
            'position': {
                'min': float(position_min),
                'max': float(position_max)
            },
            'velocity': {
                'min': float(velocity_min),
                'max': float(velocity_max)
            }
        },
        'statistics': {
            'total_files_processed': len(trajectory_files),
            'total_states_analyzed': total_states,
            'files_requested': num_files,
            'data_directory': str(data_dir)
        },
        'ranges': {
            'position': float(position_max - position_min),
            'velocity': float(velocity_max - velocity_min)
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
    print(f"  [0] Position:    [{position_min:.6f}, {position_max:.6f}]  range={position_max-position_min:.6f}")
    print(f"  [1] Velocity:    [{velocity_min:.6f}, {velocity_max:.6f}]  range={velocity_max-velocity_min:.6f}")
    print()
    print("Symmetric limits (for normalization):")
    print(f"  position_limit: ±{max(abs(position_min), abs(position_max)):.6f}")
    print(f"  velocity_limit: ±{max(abs(velocity_min), abs(velocity_max)):.6f}")
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
    parser = argparse.ArgumentParser(description="Compute Mountain Car data bounds")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/common/users/shared/pracsys/genMoPlan/data_trajectories/mountain_car_power_0p0008/trajectories",
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
        default="/common/users/dm1487/arcmg_datasets/mountain_car/mountain_car_data_bounds.pkl",
        help="Output pickle file path"
    )

    args = parser.parse_args()

    compute_bounds(
        data_dir=args.data_dir,
        num_files=args.num_files,
        output_file=args.output
    )
