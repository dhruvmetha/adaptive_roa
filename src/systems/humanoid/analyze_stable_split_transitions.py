#!/usr/bin/env python3
"""
Analyze transition statistics for stable split datasets.
Computes: success->success, success->failure, failure->success, failure->failure
"""

import numpy as np
from pathlib import Path
from collections import defaultdict

def check_success(state):
    """Check if a state meets success criteria."""
    # Success criteria indices from the dataset description:
    # - head_height (index 21) >= 1.4
    # - torso_z (index 36) >= 0.9
    # - horizontal speed sqrt(vx^2 + vy^2) where vx=index 37, vy=index 38, <= 0.2

    head_height = state[21]
    torso_z = state[36]
    vx = state[37]
    vy = state[38]
    horizontal_speed = np.sqrt(vx**2 + vy**2)

    return (head_height >= 1.4) and (torso_z >= 0.9) and (horizontal_speed <= 0.2)

def analyze_dataset(dataset_path, dataset_name):
    """Analyze transition statistics for a dataset."""
    traj_dir = Path(dataset_path) / "split_trajectories"

    # Counters for transitions
    transitions = {
        'success_to_success': 0,
        'success_to_failure': 0,
        'failure_to_success': 0,
        'failure_to_failure': 0,
    }

    # Get all trajectory files
    traj_files = sorted(traj_dir.glob("*.txt"))

    print(f"\n{'='*70}")
    print(f"Analyzing: {dataset_name}")
    print(f"{'='*70}")
    print(f"Total trajectory segments: {len(traj_files)}")

    for traj_file in traj_files:
        # Read trajectory
        traj = np.loadtxt(traj_file, delimiter=',')

        if len(traj.shape) == 1:
            # Single timestep trajectory, skip
            continue

        # Get initial and final states
        initial_state = traj[0, :]
        final_state = traj[-1, :]

        # Check success criteria
        initial_success = check_success(initial_state)
        final_success = check_success(final_state)

        # Categorize transition
        if initial_success and final_success:
            transitions['success_to_success'] += 1
        elif initial_success and not final_success:
            transitions['success_to_failure'] += 1
        elif not initial_success and final_success:
            transitions['failure_to_success'] += 1
        else:  # not initial_success and not final_success
            transitions['failure_to_failure'] += 1

    # Compute percentages
    total = sum(transitions.values())

    print(f"\nTransition Statistics:")
    print(f"-" * 70)
    print(f"Success -> Success: {transitions['success_to_success']:6d} ({100*transitions['success_to_success']/total:5.2f}%)")
    print(f"Success -> Failure: {transitions['success_to_failure']:6d} ({100*transitions['success_to_failure']/total:5.2f}%)")
    print(f"Failure -> Success: {transitions['failure_to_success']:6d} ({100*transitions['failure_to_success']/total:5.2f}%)")
    print(f"Failure -> Failure: {transitions['failure_to_failure']:6d} ({100*transitions['failure_to_failure']/total:5.2f}%)")
    print(f"-" * 70)
    print(f"Total segments:     {total:6d} (100.00%)")

    return transitions, total

if __name__ == "__main__":
    # Analyze both datasets
    medium_results = analyze_dataset(
        "/common/users/dm1487/arcmg_datasets/humanoid_get_up_medium_stable_split",
        "Humanoid Get Up Medium Stable Split"
    )

    slow_results = analyze_dataset(
        "/common/users/dm1487/arcmg_datasets/humanoid_get_up_slow_stable_split",
        "Humanoid Get Up Slow Stable Split"
    )

    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}\n")
