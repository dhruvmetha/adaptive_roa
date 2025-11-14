#!/usr/bin/env python3
"""
Compute data bounds for Pendulum Cartesian trajectories

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

    # Initialize bounds for 4D state: [x, y, vx, vy]
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    vx_min, vx_max = float('inf'), float('-inf')
    vy_min, vy_max = float('inf'), float('-inf')

    total_states = 0

    print("🔍 Computing bounds...")
    for file_path in tqdm(trajectory_files, desc="Processing trajectories"):
        try:
            # Load trajectory: x, y, vx, vy
            data = np.loadtxt(file_path, delimiter=',')

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Update bounds for each dimension
            x_min = min(x_min, data[:, 0].min())
            x_max = max(x_max, data[:, 0].max())

            y_min = min(y_min, data[:, 1].min())
            y_max = max(y_max, data[:, 1].max())

            vx_min = min(vx_min, data[:, 2].min())
            vx_max = max(vx_max, data[:, 2].max())

            vy_min = min(vy_min, data[:, 3].min())
            vy_max = max(vy_max, data[:, 3].max())

            total_states += len(data)

        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue

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
            'vx': {
                'min': float(vx_min),
                'max': float(vx_max)
            },
            'vy': {
                'min': float(vy_min),
                'max': float(vy_max)
            }
        },
        'statistics': {
            'total_files_processed': len(trajectory_files),
            'total_states_analyzed': total_states,
            'files_requested': num_files,
            'data_directory': str(data_dir)
        },
        'ranges': {
            'x': float(x_max - x_min),
            'y': float(y_max - y_min),
            'vx': float(vx_max - vx_min),
            'vy': float(vy_max - vy_min)
        }
    }

    # Print results
    print("\n" + "="*80)
    print("📊 Computed Bounds")
    print("="*80)
    print(f"Files processed: {len(trajectory_files)}")
    print(f"Total states: {total_states:,}")
    print()
    print("State Bounds (in state vector order: [x, y, vx, vy]):")
    print(f"  [0] X Position:   [{x_min:.6f}, {x_max:.6f}]  range={x_max-x_min:.6f}")
    print(f"  [1] Y Position:   [{y_min:.6f}, {y_max:.6f}]  range={y_max-y_min:.6f}")
    print(f"  [2] X Velocity:   [{vx_min:.6f}, {vx_max:.6f}]  range={vx_max-vx_min:.6f}")
    print(f"  [3] Y Velocity:   [{vy_min:.6f}, {vy_max:.6f}]  range={vy_max-vy_min:.6f}")
    print()
    print("Symmetric limits (for normalization):")
    print(f"  x_limit:  ±{max(abs(x_min), abs(x_max)):.6f}")
    print(f"  y_limit:  ±{max(abs(y_min), abs(y_max)):.6f}")
    print(f"  vx_limit: ±{max(abs(vx_min), abs(vx_max)):.6f}")
    print(f"  vy_limit: ±{max(abs(vy_min), abs(vy_max)):.6f}")
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
        default="/common/users/dm1487/arcmg_datasets/pendulum_cartesian/pendulum_cartesian_data_bounds.pkl",
        help="Output pickle file path"
    )

    args = parser.parse_args()

    compute_bounds(
        data_dir=args.data_dir,
        num_files=args.num_files,
        output_file=args.output
    )
