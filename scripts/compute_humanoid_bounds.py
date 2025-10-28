#!/usr/bin/env python3
"""
Compute data bounds for Humanoid trajectories

Usage:
    python scripts/compute_humanoid_bounds.py --data_dir /path/to/trajectories --output bounds.pkl
"""
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse


def compute_bounds(data_dir: str, num_files: int = None, output_file: str = None):
    """
    Compute min/max bounds for Humanoid state variables

    Manifold structure: ℝ³⁴ × S² × ℝ³⁰
    - Dims 0-33: Euclidean (34 dims)
    - Dims 34-36: Sphere (3 dims) - 3D unit vector
    - Dims 37-66: Euclidean (30 dims)

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

    # Initialize bounds for Euclidean dimensions
    euclidean1_bounds = {i: {'min': float('inf'), 'max': float('-inf')} for i in range(34)}
    euclidean2_bounds = {i: {'min': float('inf'), 'max': float('-inf')} for i in range(37, 67)}

    # Sphere dimensions (3D unit vector - track range for reference only)
    sphere_bounds = {34: {'min': float('inf'), 'max': float('-inf')},
                     35: {'min': float('inf'), 'max': float('-inf')},
                     36: {'min': float('inf'), 'max': float('-inf')}}

    total_states = 0

    print("🔍 Computing bounds...")
    for file_path in tqdm(trajectory_files, desc="Processing trajectories"):
        try:
            # Load trajectory: 67-dimensional state
            data = np.loadtxt(file_path, delimiter=',')

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Update Euclidean bounds (dims 0-33)
            for i in range(34):
                euclidean1_bounds[i]['min'] = min(euclidean1_bounds[i]['min'], data[:, i].min())
                euclidean1_bounds[i]['max'] = max(euclidean1_bounds[i]['max'], data[:, i].max())

            # Update sphere bounds (dims 34-36) - for reference only
            for i in range(34, 37):
                sphere_bounds[i]['min'] = min(sphere_bounds[i]['min'], data[:, i].min())
                sphere_bounds[i]['max'] = max(sphere_bounds[i]['max'], data[:, i].max())

            # Update Euclidean bounds (dims 37-66)
            for i in range(37, 67):
                euclidean2_bounds[i]['min'] = min(euclidean2_bounds[i]['min'], data[:, i].min())
                euclidean2_bounds[i]['max'] = max(euclidean2_bounds[i]['max'], data[:, i].max())

            total_states += len(data)

        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue

    # Compute global Euclidean limits
    all_euclidean_mins = [euclidean1_bounds[i]['min'] for i in range(34)] + \
                         [euclidean2_bounds[i]['min'] for i in range(37, 67)]
    all_euclidean_maxs = [euclidean1_bounds[i]['max'] for i in range(34)] + \
                         [euclidean2_bounds[i]['max'] for i in range(37, 67)]

    global_euclidean_min = min(all_euclidean_mins)
    global_euclidean_max = max(all_euclidean_maxs)
    euclidean_limit = max(abs(global_euclidean_min), abs(global_euclidean_max))

    # Package results
    bounds_data = {
        'bounds': {
            'euclidean1': euclidean1_bounds,  # Dims 0-33
            'sphere': sphere_bounds,          # Dims 34-36 (unit vector)
            'euclidean2': euclidean2_bounds,  # Dims 37-66
        },
        'limits': {
            'euclidean_limit': float(euclidean_limit),  # Symmetric limit for all Euclidean dims
        },
        'statistics': {
            'total_files_processed': len(trajectory_files),
            'total_states_analyzed': total_states,
            'files_requested': num_files,
            'data_directory': str(data_dir),
            'state_dimension': 67
        },
        'summary': {
            'euclidean_global_min': float(global_euclidean_min),
            'euclidean_global_max': float(global_euclidean_max),
            'euclidean_range': float(global_euclidean_max - global_euclidean_min),
        }
    }

    # Print results
    print("\n" + "="*80)
    print("📊 Computed Bounds")
    print("="*80)
    print(f"Files processed: {len(trajectory_files)}")
    print(f"Total states: {total_states:,}")
    print()
    print("Manifold Structure: ℝ³⁴ × S² × ℝ³⁰")
    print()
    print("Euclidean Dimensions (0-33, 37-66):")
    print(f"  Global min: {global_euclidean_min:.6f}")
    print(f"  Global max: {global_euclidean_max:.6f}")
    print(f"  Range: {global_euclidean_max - global_euclidean_min:.6f}")
    print(f"  Symmetric limit: ±{euclidean_limit:.6f}")
    print()
    print("Sphere Dimensions (34-36) - 3D unit vector:")
    for i in range(34, 37):
        min_val = sphere_bounds[i]['min']
        max_val = sphere_bounds[i]['max']
        print(f"  [{i}] component_{i-34}: [{min_val:.6f}, {max_val:.6f}]  range={max_val-min_val:.6f}")
    print()
    print("Normalization Strategy:")
    print(f"  Euclidean dims: normalize by ±{euclidean_limit:.6f}")
    print(f"  Sphere dims: keep as-is (already unit norm)")
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
    parser = argparse.ArgumentParser(description="Compute Humanoid data bounds")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/common/users/shared/pracsys/genMoPlan/data_trajectories/humanoid_get_up/trajectories",
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
        default="/common/users/dm1487/arcmg_datasets/humanoid_get_up/humanoid_data_bounds.pkl",
        help="Output pickle file path"
    )

    args = parser.parse_args()

    compute_bounds(
        data_dir=args.data_dir,
        num_files=args.num_files,
        output_file=args.output
    )
