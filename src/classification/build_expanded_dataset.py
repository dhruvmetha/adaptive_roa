#!/usr/bin/env python3
"""
Build expanded classification dataset from trajectory data.

Reads shuffled_indices_X.txt and shuffled_labels_X.txt files,
expands each trajectory to (state, label) pairs where every state
along the trajectory gets the trajectory's outcome label.

Usage:
    # Quadrotor 2D - train split (indices 0-7)
    python src/classification/build_expanded_dataset.py \
        --system quadrotor2d \
        --split train \
        --indices 0 1 2 3 4 5 6 7

    # Quadrotor 2D - val split (index 8)
    python src/classification/build_expanded_dataset.py \
        --system quadrotor2d \
        --split val \
        --indices 8

    # Quadrotor 2D - test split (index 9)
    python src/classification/build_expanded_dataset.py \
        --system quadrotor2d \
        --split test \
        --indices 9
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)


# Data paths for each system
SYSTEM_CONFIGS = {
    'quadrotor2d': {
        'base_path': '/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor2D_rl',
        'state_dim': 6,
        'delimiter': ',',
    },
    'quadrotor3d': {
        'base_path': '/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor3D_lqr',
        'state_dim': 13,
        'delimiter': ',',
    },
}


def load_indices_and_labels(base_path: Path, index: int):
    """Load shuffled indices and labels for a given split index."""
    indices_file = base_path / 'train_test_splits' / f'shuffled_indices_{index}.txt'
    labels_file = base_path / 'train_test_splits' / f'shuffled_labels_{index}.txt'

    with open(indices_file, 'r') as f:
        indices = [line.strip() for line in f.readlines()]

    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]

    assert len(indices) == len(labels), f"Mismatch: {len(indices)} indices vs {len(labels)} labels"

    return indices, labels


def load_trajectory(traj_path: Path, delimiter: str = ','):
    """Load a single trajectory file."""
    states = []
    with open(traj_path, 'r') as f:
        for line in f:
            if line.strip():
                values = [float(x) for x in line.strip().split(delimiter)]
                states.append(values)
    return states


def build_expanded_dataset(
    system: str,
    split_indices: list,
    output_dir: Path,
    split_name: str,
    balance: bool = True,
    exclude_last_state: bool = True,
):
    """
    Build expanded dataset from trajectory files.

    Args:
        system: System name ('quadrotor2d' or 'quadrotor3d')
        split_indices: List of split indices to use (e.g., [0, 1, 2])
        output_dir: Directory to save output files
        split_name: Name of the split ('train', 'val', 'test')
        balance: Whether to balance success/failure samples
        exclude_last_state: Whether to exclude the final state of each trajectory
    """
    config = SYSTEM_CONFIGS[system]
    base_path = Path(config['base_path'])
    state_dim = config['state_dim']
    delimiter = config['delimiter']
    traj_dir = base_path / 'trajectories'

    print(f"\n{'='*70}")
    print(f"Building {split_name} dataset for {system}")
    print(f"{'='*70}")
    print(f"Using split indices: {split_indices}")
    print(f"Trajectory directory: {traj_dir}")

    # Collect all trajectories and labels from specified splits
    all_traj_files = []
    all_labels = []

    for idx in split_indices:
        indices, labels = load_indices_and_labels(base_path, idx)
        print(f"  Split {idx}: {len(indices)} trajectories")
        all_traj_files.extend([traj_dir / fname for fname in indices])
        all_labels.extend(labels)

    print(f"\nTotal trajectories: {len(all_traj_files)}")
    print(f"  Success (label=1): {sum(all_labels)}")
    print(f"  Failure (label=0): {len(all_labels) - sum(all_labels)}")

    # Expand trajectories to (state, label) pairs
    success_samples = []
    failure_samples = []

    for traj_file, label in tqdm(zip(all_traj_files, all_labels),
                                   total=len(all_traj_files),
                                   desc="Expanding trajectories"):
        try:
            states = load_trajectory(traj_file, delimiter)
        except Exception as e:
            print(f"Warning: Failed to load {traj_file}: {e}")
            continue

        if len(states) == 0:
            continue

        # Exclude last state if requested (it's the terminal state)
        if exclude_last_state and len(states) > 1:
            states = states[:-1]

        # Add all states with the trajectory label
        for state in states:
            if len(state) != state_dim:
                print(f"Warning: State dim mismatch in {traj_file}: {len(state)} vs {state_dim}")
                continue

            sample = state + [label]  # Append label to state

            if label == 1:
                success_samples.append(sample)
            else:
                failure_samples.append(sample)

    print(f"\nExpanded samples:")
    print(f"  Success: {len(success_samples)}")
    print(f"  Failure: {len(failure_samples)}")

    # Balance if requested
    if balance and split_name != 'test':
        min_count = min(len(success_samples), len(failure_samples))
        print(f"\nBalancing to {min_count} samples per class...")

        np.random.shuffle(success_samples)
        np.random.shuffle(failure_samples)

        success_samples = success_samples[:min_count]
        failure_samples = failure_samples[:min_count]

        all_samples = success_samples + failure_samples
        np.random.shuffle(all_samples)
    else:
        all_samples = success_samples + failure_samples
        np.random.shuffle(all_samples)

    print(f"\nFinal dataset: {len(all_samples)} samples")

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{split_name}.txt'

    print(f"Writing to: {output_file}")
    with open(output_file, 'w') as f:
        for sample in tqdm(all_samples, desc="Writing"):
            # Format: state values (space-separated) + label
            state_str = ' '.join(f'{x:.6f}' for x in sample[:-1])
            label = int(sample[-1])
            f.write(f'{state_str} {label}\n')

    print(f"Done! Saved {len(all_samples)} samples to {output_file}")

    return len(all_samples)


def main():
    parser = argparse.ArgumentParser(description='Build expanded classification dataset')
    parser.add_argument('--system', type=str, required=True,
                        choices=['quadrotor2d', 'quadrotor3d'],
                        help='System type')
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'val', 'test'],
                        help='Dataset split name')
    parser.add_argument('--indices', type=int, nargs='+', required=True,
                        help='Split indices to use (e.g., 0 1 2 3)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data/classification/{system})')
    parser.add_argument('--no-balance', action='store_true',
                        help='Disable class balancing')
    parser.add_argument('--include-last-state', action='store_true',
                        help='Include the final (terminal) state of each trajectory')
    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        output_dir = Path(f'data/classification/{args.system}')
    else:
        output_dir = Path(args.output_dir)

    build_expanded_dataset(
        system=args.system,
        split_indices=args.indices,
        output_dir=output_dir,
        split_name=args.split,
        balance=not args.no_balance,
        exclude_last_state=not args.include_last_state,
    )


if __name__ == '__main__':
    main()
