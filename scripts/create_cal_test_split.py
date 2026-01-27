#!/usr/bin/env python3
"""
Create held-out calibration and test set splits from eval_states.txt.

Splits eval_states.txt into:
- cal_set.txt: Fixed number (e.g., 1000) for held-out q_hat calibration
- test_set.txt: Remaining for held-out evaluation

This ensures proper conformal prediction guarantees by calibrating
q_hat on data not used for training.
"""

import numpy as np
from pathlib import Path
import argparse


def create_split(eval_states_file: str, output_dir: str = None,
                 cal_size: int = 1000, seed: int = 42):
    """
    Split eval_states.txt into calibration and test sets.

    Args:
        eval_states_file: Path to eval_states.txt
        output_dir: Output directory (defaults to same as eval_states_file)
        cal_size: Number of samples for calibration set (default 1000)
        seed: Random seed for reproducibility
    """
    eval_path = Path(eval_states_file)
    if output_dir is None:
        output_dir = eval_path.parent
    else:
        output_dir = Path(output_dir)

    print(f"Loading {eval_path}...")

    # Load all lines
    with open(eval_path, 'r') as f:
        lines = f.readlines()

    n_total = len(lines)
    print(f"Total lines: {n_total}")

    # Check label distribution
    labels = []
    for line in lines:
        parts = line.strip().split(',')
        labels.append(int(parts[-1]))
    labels = np.array(labels)

    n_success = np.sum(labels == 1)
    n_failure = np.sum(labels == 0)  # In raw data, 0 = failure
    print(f"Label distribution (raw format: 1=success, 0=failure):")
    print(f"  Success (1): {n_success} ({n_success/n_total:.1%})")
    print(f"  Failure (0): {n_failure} ({n_failure/n_total:.1%})")

    # Shuffle indices
    np.random.seed(seed)
    indices = np.random.permutation(n_total)

    # Split with fixed cal_size
    n_cal = min(cal_size, n_total)
    n_test = n_total - n_cal

    cal_indices = indices[:n_cal]
    test_indices = indices[n_cal:]

    print(f"\nSplit with cal_size={cal_size}, seed={seed}:")
    print(f"  Calibration set: {n_cal}")
    print(f"  Test set: {n_test}")

    # Write calibration set
    cal_file = output_dir / "cal_set.txt"
    print(f"\nWriting {cal_file}...")
    with open(cal_file, 'w') as f:
        for idx in cal_indices:
            f.write(lines[idx])

    # Check cal set label distribution
    cal_labels = labels[cal_indices]
    print(f"  Cal set labels: {np.sum(cal_labels==1)} success, {np.sum(cal_labels==0)} failure")

    # Write test set
    test_file = output_dir / "test_set.txt"
    print(f"Writing {test_file}...")
    with open(test_file, 'w') as f:
        for idx in test_indices:
            f.write(lines[idx])

    # Check test set label distribution
    test_labels = labels[test_indices]
    print(f"  Test set labels: {np.sum(test_labels==1)} success, {np.sum(test_labels==0)} failure")

    print("\nDone!")
    print(f"  {cal_file}")
    print(f"  {test_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split eval_states.txt into cal/test sets")
    parser.add_argument("eval_states_file", help="Path to eval_states.txt")
    parser.add_argument("--output-dir", help="Output directory (default: same as input)")
    parser.add_argument("--cal-size", type=int, default=1000, help="Calibration set size (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    create_split(args.eval_states_file, args.output_dir, args.cal_size, args.seed)
