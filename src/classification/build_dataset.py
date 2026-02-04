#!/usr/bin/env python3
"""
Build classification dataset from trajectory data.

Uses shuffled_indices_0.txt and shuffled_labels_0.txt.
Expands trajectories to (state, label) pairs.

Usage:
    python src/classification/build_dataset.py \
        --system quadrotor2d \
        --num-trajectories 1000 \
        --output data/classification/quadrotor2d/train.txt \
        --balance
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

np.random.seed(42)

SYSTEM_CONFIGS = {
    'quadrotor2d': {
        'base_path': '/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor2D_rl',
        'state_dim': 6,
    },
    'quadrotor3d': {
        'base_path': '/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor3D_lqr',
        'state_dim': 13,
    },
}


def main():
    parser = argparse.ArgumentParser(description='Build classification dataset')
    parser.add_argument('--system', type=str, required=True,
                        choices=['quadrotor2d', 'quadrotor3d'])
    parser.add_argument('--num-trajectories', type=int, required=True,
                        help='Number of trajectories to use')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting index (default: 0)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path')
    parser.add_argument('--balance', action='store_true',
                        help='Balance success/failure samples')
    args = parser.parse_args()

    config = SYSTEM_CONFIGS[args.system]
    base_path = Path(config['base_path'])
    state_dim = config['state_dim']

    # Load indices and labels
    indices_file = base_path / 'train_test_splits' / 'shuffled_indices_0.txt'
    labels_file = base_path / 'train_test_splits' / 'shuffled_labels_0.txt'

    with open(indices_file, 'r') as f:
        indices = [line.strip() for line in f.readlines()][args.start:args.start + args.num_trajectories]
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()][args.start:args.start + args.num_trajectories]

    print(f"Using trajectories [{args.start}:{args.start + args.num_trajectories}] = {len(indices)} trajectories")
    print(f"  Success: {sum(labels)}, Failure: {len(labels) - sum(labels)}")

    # Expand trajectories
    traj_dir = base_path / 'trajectories'
    success_samples = []
    failure_samples = []

    for fname, label in tqdm(zip(indices, labels), total=len(indices)):
        traj_path = traj_dir / fname
        try:
            with open(traj_path, 'r') as f:
                lines = f.readlines()
        except:
            continue

        for line in lines[:-1]:  # Exclude terminal state
            if not line.strip():
                continue
            values = [float(x) for x in line.strip().split(',')]
            if len(values) != state_dim:
                continue
            sample = values + [label]
            if label == 1:
                success_samples.append(sample)
            else:
                failure_samples.append(sample)

    print(f"Expanded: {len(success_samples)} success, {len(failure_samples)} failure")

    # Balance if requested
    if args.balance:
        min_count = min(len(success_samples), len(failure_samples))
        print(f"Balancing to {min_count} per class")
        np.random.shuffle(success_samples)
        np.random.shuffle(failure_samples)
        all_samples = success_samples[:min_count] + failure_samples[:min_count]
    else:
        all_samples = success_samples + failure_samples

    np.random.shuffle(all_samples)
    print(f"Final: {len(all_samples)} samples")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in all_samples:
            state_str = ' '.join(f'{x:.6f}' for x in sample[:-1])
            f.write(f'{state_str} {int(sample[-1])}\n')

    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
