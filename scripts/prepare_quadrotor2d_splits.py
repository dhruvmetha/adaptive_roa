"""
Prepare data splits for the Quadrotor 2D adaptive sampling pipeline.

Generates:
1. shuffled_labels_*.txt - Labels aligned with shuffled_indices files
2. cal_set.txt - 1000 randomly sampled lines from eval_states.txt
3. test_set.txt - Remaining lines from eval_states.txt

Usage:
    python scripts/prepare_quadrotor2d_splits.py
"""
import numpy as np
from pathlib import Path

DATA_DIR = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor2D_rl")
SPLITS_DIR = DATA_DIR / "train_test_splits"
CAL_SET_SIZE = 1000
SEED = 42


def build_label_lookup(trajectory_labels_file: Path) -> dict:
    """Build filename -> label mapping from trajectory_labels.txt."""
    lookup = {}
    with open(trajectory_labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filename, label = line.rsplit(',', 1)
            lookup[filename.strip()] = int(label.strip())
    print(f"Loaded {len(lookup)} trajectory labels")
    return lookup


def generate_shuffled_labels(label_lookup: dict):
    """Generate shuffled_labels_*.txt files aligned with shuffled_indices_*.txt."""
    # Find all shuffled_indices files
    indices_files = sorted(SPLITS_DIR.glob("shuffled_indices_*.txt"))
    all_indices_file = SPLITS_DIR / "all_shuffled_indices.txt"

    files_to_process = list(indices_files)
    if all_indices_file.exists():
        files_to_process.append(all_indices_file)

    for indices_file in files_to_process:
        # Determine output filename
        name = indices_file.stem.replace("shuffled_indices", "shuffled_labels")
        labels_file = SPLITS_DIR / f"{name}.txt"

        # Read filenames from indices file
        with open(indices_file, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]

        # Look up labels
        labels = []
        missing = 0
        for fname in filenames:
            label = label_lookup.get(fname)
            if label is None:
                missing += 1
                labels.append(0)  # Default to failure if missing
            else:
                # Map: 1 -> 1 (success), 0 -> 0 (out-of-bounds/failure), -1 -> 0 (timeout/failure)
                labels.append(1 if label == 1 else 0)

        # Write labels file
        with open(labels_file, 'w') as f:
            for label in labels:
                f.write(f"{label}\n")

        n_success = sum(1 for l in labels if l == 1)
        n_failure = sum(1 for l in labels if l == 0)
        print(f"  {labels_file.name}: {len(labels)} labels "
              f"({n_success} success, {n_failure} failure)"
              f"{f', {missing} missing' if missing else ''}")


def generate_cal_test_split():
    """Split eval_states.txt into cal_set.txt (1000 random) and test_set.txt (rest)."""
    eval_states_file = DATA_DIR / "eval_states.txt"

    with open(eval_states_file, 'r') as f:
        lines = f.readlines()

    n_total = len(lines)
    print(f"\neval_states.txt: {n_total} lines")

    # Randomly sample 1000 indices for calibration
    rng = np.random.RandomState(SEED)
    all_indices = np.arange(n_total)
    rng.shuffle(all_indices)

    cal_indices = set(all_indices[:CAL_SET_SIZE].tolist())
    test_indices = set(all_indices[CAL_SET_SIZE:].tolist())

    # Write cal_set.txt
    cal_file = DATA_DIR / "cal_set.txt"
    with open(cal_file, 'w') as f:
        for i in sorted(cal_indices):
            f.write(lines[i])
    print(f"cal_set.txt: {len(cal_indices)} lines")

    # Write test_set.txt
    test_file = DATA_DIR / "test_set.txt"
    with open(test_file, 'w') as f:
        for i in sorted(test_indices):
            f.write(lines[i])
    print(f"test_set.txt: {len(test_indices)} lines")


def main():
    print("=" * 60)
    print("Preparing Quadrotor 2D data splits")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Splits directory: {SPLITS_DIR}")
    print()

    # Step 1: Generate shuffled labels
    print("Step 1: Generating shuffled_labels files...")
    label_lookup = build_label_lookup(DATA_DIR / "trajectory_labels.txt")
    generate_shuffled_labels(label_lookup)

    # Step 2: Generate cal/test split
    print("\nStep 2: Generating cal_set.txt and test_set.txt...")
    generate_cal_test_split()

    print("\nDone!")


if __name__ == "__main__":
    main()
