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

    # Initialize bounds for Euclidean dimensions only (skip sphere dims 34-36)
    # Sphere dimensions are always unit norm, no normalization needed
    dimension_bounds = {}
    for i in range(67):
        if 34 <= i <= 36:
            continue  # Skip sphere dimensions
        dimension_bounds[i] = {'min': float('inf'), 'max': float('-inf')}

    total_states = 0

    print("🔍 Computing bounds...")
    for file_path in tqdm(trajectory_files, desc="Processing trajectories"):
        try:
            # Load trajectory: 67-dimensional state
            data = np.loadtxt(file_path, delimiter=',')

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Update bounds for Euclidean dimensions only (skip sphere 34-36)
            for i in range(67):
                if 34 <= i <= 36:
                    continue  # Skip sphere dimensions
                dimension_bounds[i]['min'] = min(dimension_bounds[i]['min'], data[:, i].min())
                dimension_bounds[i]['max'] = max(dimension_bounds[i]['max'], data[:, i].max())

            total_states += len(data)

        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue

    # Package results (per-dimension like CartPole, only Euclidean dims)
    bounds_data = {
        'bounds': dimension_bounds,  # Only Euclidean dimensions (no sphere 34-36)
        'statistics': {
            'total_files_processed': len(trajectory_files),
            'total_states_analyzed': total_states,
            'files_requested': num_files,
            'data_directory': str(data_dir),
            'state_dimension': 67,
            'euclidean_dimensions': 64,  # 34 + 30
            'sphere_dimensions': 3       # dims 34-36 (no bounds stored)
        },
        'ranges': {
            i: float(dimension_bounds[i]['max'] - dimension_bounds[i]['min'])
            for i in dimension_bounds.keys()
        }
    }

    # Print results
    print("\n" + "="*80)
    print("📊 Computed Bounds (Per-Dimension)")
    print("="*80)
    print(f"Files processed: {len(trajectory_files)}")
    print(f"Total states: {total_states:,}")
    print()
    print("Manifold Structure: ℝ³⁴ × S² × ℝ³⁰ (67-dimensional state)")
    print()

    # Print Euclidean dimensions (0-33)
    print("Euclidean Block 1 (dims 0-33):")
    for i in range(min(5, 34)):  # Show first 5 dimensions
        min_val = dimension_bounds[i]['min']
        max_val = dimension_bounds[i]['max']
        print(f"  [{i:2d}] euclidean1_{i:2d}: [{min_val:8.3f}, {max_val:8.3f}]  range={max_val-min_val:8.3f}")
    if 34 > 5:
        print(f"  ... ({34-5} more dimensions)")
    print()

    # Print Sphere dimensions (34-36) - no bounds computed
    print("Sphere (dims 34-36) - 3D unit vector:")
    print("  NO BOUNDS COMPUTED (always unit norm, no normalization needed)")
    print()

    # Print Euclidean dimensions (37-66)
    print("Euclidean Block 2 (dims 37-66):")
    for i in range(37, min(42, 67)):  # Show first 5 dimensions
        min_val = dimension_bounds[i]['min']
        max_val = dimension_bounds[i]['max']
        print(f"  [{i:2d}] euclidean2_{i-37:2d}: [{min_val:8.3f}, {max_val:8.3f}]  range={max_val-min_val:8.3f}")
    if 67 > 42:
        print(f"  ... ({67-42} more dimensions)")
    print()

    print("Per-dimension bounds stored for Euclidean dims only")
    print("  64 Euclidean dimensions: keys [0-33, 37-66]")
    print("  3 Sphere dimensions [34-36]: NO bounds (always unit norm)")
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
