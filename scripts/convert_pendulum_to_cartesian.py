#!/usr/bin/env python3
"""
Convert pendulum dataset from polar (θ, θ̇) to Cartesian (x, y, ẋ, ẏ) coordinates.

For a unit-length pendulum with pivot at origin:
- Position: x = sin(θ), y = -cos(θ)
- Velocity: ẋ = θ̇·cos(θ), ẏ = θ̇·sin(θ)

This maintains the constraint x² + y² = 1 for all states.
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
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

# def cartesian_to_polar(x, y, x_dot, y_dot):
#     """
#     Convert Cartesian (x, y, ẋ, ẏ) to polar (θ, θ̇).
#     """
#     theta = np.arctan2(y, x)
#     theta_dot = (x_dot * np.cos(theta) - y_dot * np.sin(theta)) / (x**2 + y**2)
#     return theta, theta_dot


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


def convert_roa_labels_file(input_path, output_path):
    """
    Convert ROA labels file from polar to Cartesian.

    Input format: theta, theta_dot, label
    Output format: x, y, x_dot, y_dot, label

    Args:
        input_path: Path to input ROA labels file (3 columns)
        output_path: Path to output ROA labels file (5 columns)
    """
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            if len(values) != 3:
                continue

            theta, theta_dot, label = float(values[0]), float(values[1]), int(values[2])

            # Convert to Cartesian
            x, y, x_dot, y_dot = polar_to_cartesian(theta, theta_dot)

            data.append([x, y, x_dot, y_dot, label])

    # Save as comma-separated values
    with open(output_path, 'w') as f:
        for row in data:
            f.write(f"{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]}\n")


def convert_endpoint_file(input_path, output_path):
    """
    Convert endpoint dataset file from polar to Cartesian.

    Input format: theta_start theta_dot_start theta_end theta_dot_end
    Output format: x_start y_start x_dot_start y_dot_start x_end y_end x_dot_end y_dot_end

    Args:
        input_path: Path to input endpoint file (4 columns)
        output_path: Path to output endpoint file (8 columns)
    """
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) != 4:
                continue

            theta_start, theta_dot_start, theta_end, theta_dot_end = map(float, values)

            # Convert start state
            x_start, y_start, x_dot_start, y_dot_start = polar_to_cartesian(
                theta_start, theta_dot_start
            )

            # Convert end state
            x_end, y_end, x_dot_end, y_dot_end = polar_to_cartesian(
                theta_end, theta_dot_end
            )

            data.append([
                x_start, y_start, x_dot_start, y_dot_start,
                x_end, y_end, x_dot_end, y_dot_end
            ])

    # Save as space-separated values
    np.savetxt(output_path, data, fmt='%.6f')


def convert_dataset(source_dir, target_dir, convert_trajectories=True, convert_endpoints=True):
    """
    Convert entire pendulum dataset from polar to Cartesian coordinates.

    Args:
        source_dir: Source directory with polar coordinate data
        target_dir: Target directory for Cartesian coordinate data
        convert_trajectories: Whether to convert trajectory files
        convert_endpoints: Whether to convert endpoint files
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directory structure
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"Converting dataset from {source_path} to {target_path}")
    print(f"  Converting trajectories: {convert_trajectories}")
    print(f"  Converting endpoints: {convert_endpoints}")

    # Convert trajectory files
    if convert_trajectories:
        traj_source = source_path / "trajectories"
        traj_target = target_path / "trajectories"

        if traj_source.exists():
            traj_target.mkdir(parents=True, exist_ok=True)

            trajectory_files = sorted(traj_source.glob("sequence_*.txt"))
            print(f"\nConverting {len(trajectory_files)} trajectory files...")

            for traj_file in tqdm(trajectory_files):
                output_file = traj_target / traj_file.name
                convert_trajectory_file(traj_file, output_file)

    # Convert endpoint datasets
    if convert_endpoints:
        # Look for endpoint dataset files
        endpoint_patterns = [
            "incremental_endpoint_dataset",
            "sampled_endpoint_dataset",
            "*endpoint*.txt"
        ]

        for pattern in endpoint_patterns:
            for endpoint_source in source_path.glob(pattern):
                if endpoint_source.is_dir():
                    # Handle endpoint dataset directory
                    endpoint_target = target_path / endpoint_source.name
                    endpoint_target.mkdir(parents=True, exist_ok=True)

                    endpoint_files = list(endpoint_source.glob("*.txt"))
                    if endpoint_files:
                        print(f"\nConverting {len(endpoint_files)} endpoint files from {endpoint_source.name}...")

                        for ep_file in tqdm(endpoint_files):
                            output_file = endpoint_target / ep_file.name
                            convert_endpoint_file(ep_file, output_file)

                elif endpoint_source.is_file() and endpoint_source.suffix == '.txt':
                    # Handle individual endpoint file
                    output_file = target_path / endpoint_source.name
                    print(f"\nConverting endpoint file: {endpoint_source.name}")
                    convert_endpoint_file(endpoint_source, output_file)

    # Convert ROA labels file if it exists
    roa_labels_file = source_path / "roa_labels.txt"
    if roa_labels_file.exists():
        print(f"\nConverting ROA labels file: roa_labels.txt")
        convert_roa_labels_file(roa_labels_file, target_path / "roa_labels.txt")

    # Copy other files (images, metadata, etc.) that don't need conversion
    print("\nCopying non-trajectory files...")
    files_to_skip = ['trajectories', 'roa_labels.txt', 'roa_labels_700k_DO_NOT_USE_DEATH_INSIDE.txt']
    for item in source_path.iterdir():
        if item.is_file():
            # Copy all files except those we're converting or explicitly skipping
            if item.name not in files_to_skip:
                shutil.copy2(item, target_path / item.name)
                print(f"  Copied: {item.name}")

    print(f"\n✓ Conversion complete! Dataset saved to: {target_path}")

    # Print summary
    if convert_trajectories and (target_path / "trajectories").exists():
        num_traj = len(list((target_path / "trajectories").glob("*.txt")))
        print(f"  Trajectories: {num_traj} files")

    if convert_endpoints:
        for pattern in ["*endpoint*"]:
            endpoint_dirs = list(target_path.glob(pattern))
            for ep_dir in endpoint_dirs:
                if ep_dir.is_dir():
                    num_ep = len(list(ep_dir.glob("*.txt")))
                    print(f"  {ep_dir.name}: {num_ep} files")


def main():
    parser = argparse.ArgumentParser(
        description="Convert pendulum dataset from polar to Cartesian coordinates"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/common/users/shared/pracsys/genMoPlan/data_trajectories/pendulum_lqr_50k",
        help="Source directory with polar coordinate data"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="/common/users/dm1487/arcmg_datasets/pendulum_cartesian_50k",
        help="Target directory for Cartesian coordinate data"
    )
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Skip converting trajectory files"
    )
    parser.add_argument(
        "--no-endpoints",
        action="store_true",
        help="Skip converting endpoint files"
    )

    args = parser.parse_args()

    convert_dataset(
        args.source,
        args.target,
        convert_trajectories=not args.no_trajectories,
        convert_endpoints=not args.no_endpoints
    )


if __name__ == "__main__":
    main()
