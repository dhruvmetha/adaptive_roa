#!/usr/bin/env python3
"""
Test conversion of 5 trajectory files to verify correctness.
"""

import numpy as np
from pathlib import Path
import shutil


def polar_to_cartesian(theta, theta_dot):
    """
    Convert polar pendulum state (θ, θ̇) to Cartesian (x, y, ẋ, ẏ).

    Convention: θ = 0 is vertical UP (pendulum pointing upward)

    Args:
        theta: Angle from vertical UP (radians)
        theta_dot: Angular velocity (rad/s)

    Returns:
        Tuple of (x, y, x_dot, y_dot)
    """
    # Position (θ=0 is vertical up: x=0, y=1)
    x = np.sin(theta)
    y = np.cos(theta)

    # Velocity (time derivatives)
    x_dot = theta_dot * np.cos(theta)
    y_dot = -theta_dot * np.sin(theta)

    return x, y, x_dot, y_dot


def convert_trajectory_file(input_path, output_path):
    """
    Convert a single trajectory file from polar to Cartesian coordinates.

    Args:
        input_path: Path to input trajectory file (θ, θ̇ format)
        output_path: Path to output trajectory file (x, y, ẋ, ẏ format)
    """
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            # Parse line: "theta,theta_dot"
            line = line.strip()
            if not line:
                continue

            coords = line.split(',')
            if len(coords) != 2:
                continue

            theta = float(coords[0])
            theta_dot = float(coords[1])

            # Convert to Cartesian
            x, y, x_dot, y_dot = polar_to_cartesian(theta, theta_dot)

            # Write in same format
            f_out.write(f"{x:.6f},{y:.6f},{x_dot:.6f},{y_dot:.6f}\n")


# Test with 5 files
source_dir = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/pendulum_lqr_50k")
target_dir = Path("/common/users/dm1487/arcmg_datasets/pendulum_cartesian_50k_test")

# Create target directories
target_dir.mkdir(parents=True, exist_ok=True)
(target_dir / "trajectories").mkdir(parents=True, exist_ok=True)

# Convert first 5 trajectory files
traj_source = source_dir / "trajectories"
traj_target = target_dir / "trajectories"

test_files = [f"sequence_{i}.txt" for i in range(5)]

print("Converting 5 test files...")
for filename in test_files:
    input_file = traj_source / filename
    output_file = traj_target / filename

    if input_file.exists():
        convert_trajectory_file(input_file, output_file)
        print(f"  ✓ Converted {filename}")
    else:
        print(f"  ✗ File not found: {filename}")

print("\nDone! Test files saved to:", target_dir)
