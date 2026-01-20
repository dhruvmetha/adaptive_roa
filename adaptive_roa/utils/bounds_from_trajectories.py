"""
Utility for computing state bounds from trajectory files.

This provides a fallback mechanism when dataset_description.json is not accessible
(e.g., due to file permissions). Instead of reading pre-computed bounds, we sample
trajectory files and compute bounds on-the-fly.

Usage:
    from adaptive_roa.utils.bounds_from_trajectories import compute_bounds_from_trajectories

    bounds = compute_bounds_from_trajectories(
        trajectories_dir="/path/to/trajectories",
        state_dim=4,
        max_files=1000
    )
    # bounds = {'min': [min_0, min_1, ...], 'max': [max_0, max_1, ...]}
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import random


def compute_bounds_from_trajectories(
    trajectories_dir: Union[str, Path],
    state_dim: int,
    max_files: int = 1000,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Compute state bounds by sampling trajectory files.

    Args:
        trajectories_dir: Path to directory containing trajectory files
        state_dim: Dimension of state vector (e.g., 4 for CartPole, 2 for Pendulum)
        max_files: Maximum number of trajectory files to sample
        seed: Random seed for reproducible sampling
        verbose: Whether to print progress

    Returns:
        Dictionary with 'min' and 'max' lists for each state dimension

    Example:
        >>> bounds = compute_bounds_from_trajectories("./trajectories", state_dim=4)
        >>> bounds
        {'min': [-6.0, -3.14, -5.0, -5.0], 'max': [6.0, 3.14, 5.0, 5.0]}
    """
    trajectories_dir = Path(trajectories_dir)

    if not trajectories_dir.exists():
        raise FileNotFoundError(f"Trajectories directory not found: {trajectories_dir}")

    # Find all trajectory files (common patterns)
    trajectory_files = set()
    for pattern in ["*.txt", "*.csv"]:
        trajectory_files.update(trajectories_dir.glob(pattern))
    trajectory_files = list(trajectory_files)

    if not trajectory_files:
        raise ValueError(f"No trajectory files found in {trajectories_dir}")

    # Sample files if we have more than max_files
    random.seed(seed)
    if len(trajectory_files) > max_files:
        trajectory_files = random.sample(trajectory_files, max_files)

    if verbose:
        print(f"Computing bounds from {len(trajectory_files)} trajectory files...")

    # Initialize bounds
    mins = np.full(state_dim, np.inf)
    maxs = np.full(state_dim, -np.inf)

    files_processed = 0
    for traj_file in trajectory_files:
        try:
            # Try to read the trajectory file (handle both space and comma delimiters)
            try:
                data = np.loadtxt(traj_file, delimiter=',')
            except ValueError:
                # Fall back to whitespace delimiter
                data = np.loadtxt(traj_file)

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Handle different file formats
            # Common format: each row is a state, or each row is [state, action, ...]
            if data.shape[1] >= state_dim:
                states = data[:, :state_dim]
                mins = np.minimum(mins, states.min(axis=0))
                maxs = np.maximum(maxs, states.max(axis=0))
                files_processed += 1

        except Exception as e:
            # Skip files that can't be parsed
            continue

    if files_processed == 0:
        raise ValueError(f"Could not parse any trajectory files in {trajectories_dir}")

    if verbose:
        print(f"Successfully processed {files_processed} trajectory files")
        print(f"Computed bounds:")
        for i in range(state_dim):
            print(f"  [dim {i}]: [{mins[i]:.4f}, {maxs[i]:.4f}]")

    return {
        'min': mins.tolist(),
        'max': maxs.tolist()
    }


def get_symmetric_limits(bounds: Dict[str, List[float]]) -> List[float]:
    """
    Convert min/max bounds to symmetric limits (max of abs values).

    Args:
        bounds: Dictionary with 'min' and 'max' lists

    Returns:
        List of symmetric limits for each dimension

    Example:
        >>> bounds = {'min': [-6.0, -3.0], 'max': [5.0, 4.0]}
        >>> get_symmetric_limits(bounds)
        [6.0, 4.0]
    """
    mins = np.array(bounds['min'])
    maxs = np.array(bounds['max'])
    return np.maximum(np.abs(mins), np.abs(maxs)).tolist()


def compute_cartpole_bounds(trajectories_dir: Union[str, Path], max_files: int = 1000) -> Dict:
    """
    Compute CartPole-specific bounds from trajectories.

    CartPole state: [x, theta, x_dot, theta_dot]

    Returns:
        Dictionary with cart_limit, angle_limit, velocity_limit, angular_velocity_limit
    """
    bounds = compute_bounds_from_trajectories(trajectories_dir, state_dim=4, max_files=max_files)
    limits = get_symmetric_limits(bounds)

    return {
        'cart_limit': limits[0],
        'angle_limit': np.pi,  # Always wrap to [-pi, pi]
        'velocity_limit': limits[2],
        'angular_velocity_limit': limits[3],
        'achieved_bounds': {
            'x': {'min': bounds['min'][0], 'max': bounds['max'][0]},
            'theta': {'min': bounds['min'][1], 'max': bounds['max'][1]},
            'x_dot': {'min': bounds['min'][2], 'max': bounds['max'][2]},
            'theta_dot': {'min': bounds['min'][3], 'max': bounds['max'][3]}
        }
    }


def compute_pendulum_bounds(trajectories_dir: Union[str, Path], max_files: int = 1000) -> Dict:
    """
    Compute Pendulum-specific bounds from trajectories.

    Pendulum state: [theta, theta_dot]

    Returns:
        Dictionary with angle_limit, angular_velocity_limit
    """
    bounds = compute_bounds_from_trajectories(trajectories_dir, state_dim=2, max_files=max_files)
    limits = get_symmetric_limits(bounds)

    return {
        'angle_limit': np.pi,  # Always wrap to [-pi, pi]
        'angular_velocity_limit': limits[1],
        'achieved_bounds': {
            'theta': {'min': bounds['min'][0], 'max': bounds['max'][0]},
            'theta_dot': {'min': bounds['min'][1], 'max': bounds['max'][1]}
        }
    }


def compute_mountain_car_bounds(trajectories_dir: Union[str, Path], max_files: int = 1000) -> Dict:
    """
    Compute Mountain Car-specific bounds from trajectories.

    Mountain Car state: [position, velocity]

    Returns:
        Dictionary with position_limit, velocity_limit
    """
    bounds = compute_bounds_from_trajectories(trajectories_dir, state_dim=2, max_files=max_files)
    limits = get_symmetric_limits(bounds)

    return {
        'position_limit': limits[0],
        'velocity_limit': limits[1],
        'achieved_bounds': {
            'position': {'min': bounds['min'][0], 'max': bounds['max'][0]},
            'velocity': {'min': bounds['min'][1], 'max': bounds['max'][1]}
        }
    }


def compute_pendulum_cartesian_bounds(trajectories_dir: Union[str, Path], max_files: int = 1000) -> Dict:
    """
    Compute Pendulum Cartesian-specific bounds from trajectories.

    Pendulum Cartesian state: [x, y, vx, vy]

    Returns:
        Dictionary with position_limit, velocity_limit
    """
    bounds = compute_bounds_from_trajectories(trajectories_dir, state_dim=4, max_files=max_files)
    limits = get_symmetric_limits(bounds)

    return {
        'x_limit': limits[0],
        'y_limit': limits[1],
        'vx_limit': limits[2],
        'vy_limit': limits[3],
        'achieved_bounds': {
            'x': {'min': bounds['min'][0], 'max': bounds['max'][0]},
            'y': {'min': bounds['min'][1], 'max': bounds['max'][1]},
            'vx': {'min': bounds['min'][2], 'max': bounds['max'][2]},
            'vy': {'min': bounds['min'][3], 'max': bounds['max'][3]}
        }
    }
