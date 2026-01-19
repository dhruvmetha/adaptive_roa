#!/usr/bin/env python3
"""
Get data bounds for CartPole trajectories from dataset_description.json

Usage:
    python scripts/compute_cartpole_bounds.py --data_dir /path/to/cartpole_pybullet
    python scripts/compute_cartpole_bounds.py --data_dir /path/to/cartpole_pybullet --output bounds.pkl
"""
import json
import pickle
from pathlib import Path
import argparse


def get_bounds(data_dir: str, output_file: str = None):
    """
    Get min/max bounds for CartPole state variables from dataset_description.json

    Args:
        data_dir: Directory containing dataset_description.json
        output_file: Path to save pickle file (optional)

    Returns:
        Dictionary with bounds and statistics
    """
    data_dir = Path(data_dir)
    json_path = data_dir / "dataset_description.json"

    if not json_path.exists():
        raise FileNotFoundError(f"dataset_description.json not found in {data_dir}")

    with open(json_path) as f:
        dataset_info = json.load(f)

    achieved = dataset_info["achieved_bounds"]
    stats = dataset_info.get("dataset_statistics", {})

    # Extract bounds
    x_min, x_max = achieved["x"]["min"], achieved["x"]["max"]
    theta_min, theta_max = achieved["theta"]["min"], achieved["theta"]["max"]
    x_dot_min, x_dot_max = achieved["x_dot"]["min"], achieved["x_dot"]["max"]
    theta_dot_min, theta_dot_max = achieved["theta_dot"]["min"], achieved["theta_dot"]["max"]

    # Package results
    bounds_data = {
        'bounds': {
            'x': {'min': x_min, 'max': x_max},
            'theta': {'min': theta_min, 'max': theta_max},
            'x_dot': {'min': x_dot_min, 'max': x_dot_max},
            'theta_dot': {'min': theta_dot_min, 'max': theta_dot_max}
        },
        'statistics': {
            'total_trajectories': stats.get('total_trajectories'),
            'successful_trajectories': stats.get('successful_trajectories', {}).get('count'),
            'failed_trajectories': stats.get('failed_trajectories', {}).get('count'),
            'data_directory': str(data_dir)
        },
        'ranges': {
            'x': x_max - x_min,
            'theta': theta_max - theta_min,
            'x_dot': x_dot_max - x_dot_min,
            'theta_dot': theta_dot_max - theta_dot_min
        }
    }

    # Print results
    print("=" * 80)
    print("CartPole Data Bounds (from dataset_description.json)")
    print("=" * 80)
    print(f"Source: {json_path}")
    print(f"Total trajectories: {stats.get('total_trajectories', 'N/A'):,}")
    print()
    print("State Bounds (in state vector order):")
    print(f"  [0] Cart position (x):      [{x_min:.6f}, {x_max:.6f}]  range={x_max-x_min:.6f}")
    print(f"  [1] Pole angle (theta):     [{theta_min:.6f}, {theta_max:.6f}]  range={theta_max-theta_min:.6f}")
    print(f"  [2] Cart velocity (x_dot):  [{x_dot_min:.6f}, {x_dot_max:.6f}]  range={x_dot_max-x_dot_min:.6f}")
    print(f"  [3] Angular vel (theta_dot):[{theta_dot_min:.6f}, {theta_dot_max:.6f}]  range={theta_dot_max-theta_dot_min:.6f}")
    print()
    print("Symmetric limits (for normalization):")
    print(f"  cart_limit:             +/-{max(abs(x_min), abs(x_max)):.6f}")
    print(f"  velocity_limit:         +/-{max(abs(x_dot_min), abs(x_dot_max)):.6f}")
    print(f"  angular_velocity_limit: +/-{max(abs(theta_dot_min), abs(theta_dot_max)):.6f}")
    print(f"  angle_limit:            +/-pi (after wrapping)")
    print()

    # Save to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(bounds_data, f)

        print(f"Saved bounds to: {output_path}")

    return bounds_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get CartPole data bounds from dataset_description.json")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_pybullet",
        help="Directory containing dataset_description.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pickle file path (optional)"
    )

    args = parser.parse_args()

    get_bounds(
        data_dir=args.data_dir,
        output_file=args.output
    )
