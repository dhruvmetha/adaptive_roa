#!/usr/bin/env python3
"""
Compute data bounds for CartPole DeepMind Control Suite trajectories

Key Differences from Regular CartPole:
- Theta is UNWRAPPED in raw data (can exceed ±π, up to ±130 rad)
- Higher velocity bounds (ẋ ~±9, θ̇ ~±19 vs regular ~±5)
- Computes bounds on raw unwrapped theta for reference
- During data loading, theta will be wrapped to [-π, π] using arctan2

Usage:
    python scripts/compute_cartpole_dmcontrol_bounds.py \
        --data_dir /common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_dm_control/trajectories \
        --output /common/users/dm1487/arcmg_datasets/cartpole_dmcontrol/cartpole_dmcontrol_data_bounds.pkl
"""
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse


def compute_bounds(data_dir: str, num_files: int = None, output_file: str = None):
    """
    Compute min/max bounds for CartPole DM Control state variables

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
    x_min, x_max = float('inf'), float('-inf')
    theta_min, theta_max = float('inf'), float('-inf')
    x_dot_min, x_dot_max = float('inf'), float('-inf')
    theta_dot_min, theta_dot_max = float('inf'), float('-inf')

    # Also track wrapped theta bounds for reference
    theta_wrapped_min, theta_wrapped_max = float('inf'), float('-inf')

    total_states = 0

    print("🔍 Computing bounds...")
    for file_path in tqdm(trajectory_files, desc="Processing trajectories"):
        try:
            # Load trajectory: x, theta, x_dot, theta_dot
            data = np.loadtxt(file_path, delimiter=',')

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Update bounds for raw (unwrapped) values
            x_min = min(x_min, data[:, 0].min())
            x_max = max(x_max, data[:, 0].max())

            theta_min = min(theta_min, data[:, 1].min())
            theta_max = max(theta_max, data[:, 1].max())

            x_dot_min = min(x_dot_min, data[:, 2].min())
            x_dot_max = max(x_dot_max, data[:, 2].max())

            theta_dot_min = min(theta_dot_min, data[:, 3].min())
            theta_dot_max = max(theta_dot_max, data[:, 3].max())

            # Also compute wrapped theta bounds for reference
            theta_wrapped = np.arctan2(np.sin(data[:, 1]), np.cos(data[:, 1]))
            theta_wrapped_min = min(theta_wrapped_min, theta_wrapped.min())
            theta_wrapped_max = max(theta_wrapped_max, theta_wrapped.max())

            total_states += len(data)

        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue

    # Package results
    bounds_data = {
        'bounds': {
            0: {  # x (cart position)
                'min': float(x_min),
                'max': float(x_max),
                'limit': float(max(abs(x_min), abs(x_max)))
            },
            1: {  # theta (pole angle - UNWRAPPED)
                'min': float(theta_min),
                'max': float(theta_max),
                'limit': float(max(abs(theta_min), abs(theta_max))),
                'wrapped_min': float(theta_wrapped_min),
                'wrapped_max': float(theta_wrapped_max),
                'note': 'Raw theta is UNWRAPPED. Will be wrapped to [-π, π] during data loading.'
            },
            2: {  # x_dot (cart velocity)
                'min': float(x_dot_min),
                'max': float(x_dot_max),
                'limit': float(max(abs(x_dot_min), abs(x_dot_max)))
            },
            3: {  # theta_dot (angular velocity)
                'min': float(theta_dot_min),
                'max': float(theta_dot_max),
                'limit': float(max(abs(theta_dot_min), abs(theta_dot_max)))
            }
        },
        'statistics': {
            'total_files_processed': len(trajectory_files),
            'total_states_analyzed': total_states,
            'files_requested': num_files,
            'data_directory': str(data_dir)
        },
        'ranges': {
            0: float(x_max - x_min),
            1: float(theta_max - theta_min),
            2: float(x_dot_max - x_dot_min),
            3: float(theta_dot_max - theta_dot_min)
        },
        'dimension_names': ['x', 'theta', 'x_dot', 'theta_dot'],
        'manifold': 'Product(input_dim=4, manifolds=[(Euclidean(), 1), (FlatTorus(), 1), (Euclidean(), 2)])',
        'manifold_description': 'ℝ × S¹ × ℝ² (same as regular CartPole)',
        'system_name': 'CartPole DeepMind Control Suite'
    }

    # Print results
    print("\n" + "="*80)
    print("📊 CartPole DM Control - Computed Bounds")
    print("="*80)
    print(f"Files processed: {len(trajectory_files)}")
    print(f"Total states: {total_states:,}")
    print()
    print("State Bounds (in state vector order [x, θ, ẋ, θ̇]):")
    print(f"  [0] Cart position (x):      [{x_min:.6f}, {x_max:.6f}]  range={x_max-x_min:.6f}")
    print(f"  [1] Pole angle (θ):         [{theta_min:.6f}, {theta_max:.6f}]  range={theta_max-theta_min:.6f} (UNWRAPPED)")
    print(f"      → Wrapped to [-π, π]:   [{theta_wrapped_min:.6f}, {theta_wrapped_max:.6f}]")
    print(f"  [2] Cart velocity (ẋ):      [{x_dot_min:.6f}, {x_dot_max:.6f}]  range={x_dot_max-x_dot_min:.6f}")
    print(f"  [3] Angular velocity (θ̇):   [{theta_dot_min:.6f}, {theta_dot_max:.6f}]  range={theta_dot_max-theta_dot_min:.6f}")
    print()
    print("Symmetric limits (for normalization):")
    print(f"  cart_limit:             ±{max(abs(x_min), abs(x_max)):.6f}")
    print(f"  velocity_limit:         ±{max(abs(x_dot_min), abs(x_dot_max)):.6f}")
    print(f"  angular_velocity_limit: ±{max(abs(theta_dot_min), abs(theta_dot_max)):.6f}")
    print(f"  angle_limit:            ±π (after wrapping to [-π, π])")
    print()
    print("Manifold Structure:")
    print(f"  ℝ × S¹ × ℝ² (Euclidean × FlatTorus × Euclidean)")
    print(f"  Same as regular CartPole, theta wrapped during loading")
    print()

    # Save to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(bounds_data, f)

        print(f"💾 Saved bounds to: {output_path}")
        print()

    return bounds_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CartPole DM Control data bounds")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_dmcontrol/trajectories",
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
        default="/common/users/dm1487/arcmg_datasets/cartpole_dmcontrol/cartpole_dmcontrol_data_bounds.pkl",
        help="Output pickle file path"
    )

    args = parser.parse_args()

    compute_bounds(
        data_dir=args.data_dir,
        num_files=args.num_files,
        output_file=args.output
    )
