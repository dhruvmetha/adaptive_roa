#!/usr/bin/env python3
"""
Create humanoid reach and stable_split datasets from any source humanoid trajectory data.

Usage:
    python create_humanoid_datasets.py humanoid_get_up_slow
    python create_humanoid_datasets.py humanoid_get_up_medium
    python create_humanoid_datasets.py <dataset_name>

Or specify custom source path:
    python create_humanoid_datasets.py <dataset_name> --source /path/to/source

The script will:
1. Create <dataset_name>_reach (truncated at first success)
2. Create <dataset_name>_stable_split (recursive splitting at every state transition F↔S)
3. Generate roa_labels.txt, shuffled_indices.txt, and dataset_description.json for each
"""

import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from src.systems.humanoid import HumanoidSystem

# Success criteria thresholds (configurable via command line)
DEFAULT_H_THR = 1.4         # Head height threshold (meters)
DEFAULT_TZ_THR = 0.9        # Torso vertical z-component threshold
DEFAULT_SPEED_THR = 0.2     # Horizontal speed threshold (m/s)


class HumanoidDatasetCreator:
    """Creates reach and stable_split datasets from humanoid trajectory data."""

    def __init__(self, dataset_name, source_dir=None, output_base_dir=None,
                 h_thr=DEFAULT_H_THR, tz_thr=DEFAULT_TZ_THR, speed_thr=DEFAULT_SPEED_THR,
                 create_reach=True, create_stable_split=True,
                 create_roa_labels=True, create_shuffled_indices=True, create_description=True,
                 only_shuffled_indices=False):
        """
        Initialize dataset creator.

        Args:
            dataset_name: Name of the dataset (e.g., 'humanoid_get_up_slow')
            source_dir: Source directory path (if None, uses default location)
            output_base_dir: Base output directory (if None, uses default location)
            h_thr: Head height threshold
            tz_thr: Torso vertical z threshold
            speed_thr: Horizontal speed threshold
            create_reach: Whether to create reach dataset
            create_stable_split: Whether to create stable_split dataset
            create_roa_labels: Whether to create roa_labels.txt
            create_shuffled_indices: Whether to create shuffled_indices.txt
            create_description: Whether to create dataset_description.json
            only_shuffled_indices: Only regenerate shuffled_indices.txt from existing files
        """
        self.dataset_name = dataset_name
        self.h_thr = h_thr
        self.tz_thr = tz_thr
        self.speed_thr = speed_thr

        # Initialize HumanoidSystem with the success thresholds
        # This ensures consistency between dataset creation and model evaluation
        self.system = HumanoidSystem(
            head_height_threshold=h_thr,
            torso_z_threshold=tz_thr,
            speed_threshold=speed_thr
        )

        # Toggles for what to create
        self.only_shuffled_indices = only_shuffled_indices

        # If only_shuffled_indices is True, override other settings
        if only_shuffled_indices:
            self.create_reach = create_reach
            self.create_stable_split = create_stable_split
            self.create_roa_labels = False
            self.create_shuffled_indices = True
            self.create_description = False
        else:
            self.create_reach = create_reach
            self.create_stable_split = create_stable_split
            self.create_roa_labels = create_roa_labels
            self.create_shuffled_indices = create_shuffled_indices
            self.create_description = create_description

        # Set source directory
        if source_dir is None:
            self.source_dir = Path(f"/common/users/shared/pracsys/genMoPlan/data_trajectories/{dataset_name}")
        else:
            self.source_dir = Path(source_dir)

        # Set output directories
        if output_base_dir is None:
            output_base_dir = Path("/common/users/dm1487/arcmg_datasets")
        else:
            output_base_dir = Path(output_base_dir)

        self.reach_dir = output_base_dir / f"{dataset_name}_reach"
        self.stable_dir = output_base_dir / f"{dataset_name}_stable_split"

        print(f"Source: {self.source_dir}")
        print(f"Reach output: {self.reach_dir}")
        print(f"Stable split output: {self.stable_dir}")

    def check_success_criteria(self, state):
        """
        Check if a single state meets success criteria using HumanoidSystem.

        This delegates to the system's is_in_attractor() method to ensure
        consistency with model evaluation and training.

        Args:
            state: Numpy array or list representing the state (67 dimensions expected)

        Returns:
            bool: True if state meets success criteria, False otherwise
        """
        # Convert to numpy array if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        # Validate state dimensions
        if len(state) < 39:
            raise ValueError(f"State vector too short: expected ≥39 dimensions, got {len(state)}")

        # Use system's is_in_attractor method for consistency
        # This handles both numpy arrays and torch tensors automatically
        return self.system.is_in_attractor(state)

    def load_trajectory(self, traj_file):
        """Load a full trajectory from file."""
        with open(traj_file, 'r') as f:
            lines = f.readlines()

        trajectory = []
        for line in lines:
            state = np.array([float(x) for x in line.strip().split(', ')])
            trajectory.append(state)

        return np.array(trajectory)

    def save_trajectory(self, trajectory, output_file):
        """Save trajectory to file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for state in trajectory:
                state_str = ', '.join([f"{val}" for val in state])
                f.write(f"{state_str}\n")

    def find_success_point(self, trajectory):
        """
        Find first timestep where success criteria are met.

        Returns:
            int or None: First timestep index where success achieved, or None if never reached
        """
        for i, state in enumerate(trajectory):
            if self.check_success_criteria(state):
                return i
        return None

    def process_trajectory_reach(self, source_file, dest_file):
        """
        Process trajectory for REACH dataset: truncate at first success point.

        Returns:
            tuple: (success_label, start_state)
        """
        trajectory = self.load_trajectory(source_file)
        start_state = trajectory[0]

        # Find first success point
        success_idx = self.find_success_point(trajectory)

        if success_idx is not None:
            # Truncate at success point
            truncated_trajectory = trajectory[:success_idx + 1]
            success_label = 1
        else:
            # Keep full trajectory
            truncated_trajectory = trajectory
            success_label = 0

        # Save truncated trajectory
        self.save_trajectory(truncated_trajectory, dest_file)

        return success_label, start_state

    def process_trajectory_recursive(self, trajectory, output_dir, base_name, part_counter):
        """
        Recursively split trajectory at every state transition (F→S or S→F).

        Example: FFFSSFFSSS → FFFS, SSF, FFS, SSS

        Returns the next part_counter value to use.
        """
        if len(trajectory) <= 1:
            # Base case: single state or empty
            output_file = output_dir / f"{base_name}_part{part_counter}.txt"
            self.save_trajectory(trajectory, output_file)
            return part_counter + 1

        # Determine success status of first state
        current_status = self.check_success_criteria(trajectory[0])

        # Find next transition (change in success status)
        transition_idx = None
        for i in range(1, len(trajectory)):
            next_status = self.check_success_criteria(trajectory[i])
            if next_status != current_status:
                transition_idx = i
                break

        if transition_idx is None:
            # No transitions found - save entire trajectory
            output_file = output_dir / f"{base_name}_part{part_counter}.txt"
            self.save_trajectory(trajectory, output_file)
            return part_counter + 1

        # Found transition - save from start to transition point (inclusive)
        chunk = trajectory[:transition_idx + 1]
        output_file = output_dir / f"{base_name}_part{part_counter}.txt"
        self.save_trajectory(chunk, output_file)

        # Recursively process from transition point onwards (overlapping)
        remaining = trajectory[transition_idx:]
        return self.process_trajectory_recursive(remaining, output_dir, base_name, part_counter + 1)

    def create_reach_dataset(self):
        """Create the reach dataset (truncated at first success)."""
        print("\n" + "="*70)
        print(f"CREATING {self.dataset_name.upper()}_REACH DATASET")
        print("="*70)
        print()

        # Create directory structure
        self.reach_dir.mkdir(parents=True, exist_ok=True)
        reach_traj_dir = self.reach_dir / "trajectories"

        # If only regenerating shuffled indices, skip trajectory processing
        if self.only_shuffled_indices:
            print("Mode: Only regenerating shuffled_indices.txt from existing files")
            print()

            if not reach_traj_dir.exists():
                print(f"Error: {reach_traj_dir} does not exist!")
                print("Cannot regenerate shuffled_indices without existing trajectories.")
                return

            # Get existing trajectory files
            traj_files = sorted(reach_traj_dir.glob("sequence_*.txt"))
            if not traj_files:
                print(f"Error: No trajectory files found in {reach_traj_dir}")
                return

            print(f"Found {len(traj_files)} existing trajectory files")

            # Create shuffled indices from existing files
            print("Creating shuffled_indices.txt...")
            filenames = [f.name for f in traj_files]
            np.random.seed(42)
            np.random.shuffle(filenames)
            with open(self.reach_dir / "shuffled_indices.txt", 'w') as f:
                for filename in filenames:
                    f.write(f"{filename}\n")

            print(f"✓ Created shuffled_indices.txt with {len(filenames)} entries")
            print()
            return

        # Normal mode: process trajectories
        reach_traj_dir.mkdir(exist_ok=True)
        source_traj_dir = self.source_dir / "trajectories"

        # Get all trajectory files
        traj_files = sorted(source_traj_dir.glob("sequence_*.txt"))
        print(f"Found {len(traj_files)} trajectories to process")
        print()

        # Process all trajectories
        print("Processing trajectories (truncating at first success)...")
        roa_data = []
        success_count = 0
        failure_count = 0

        for traj_file in tqdm(traj_files, desc="Processing"):
            idx = int(traj_file.stem.split('_')[1])
            dest_file = reach_traj_dir / f"sequence_{idx}.txt"

            success_label, start_state = self.process_trajectory_reach(traj_file, dest_file)

            roa_data.append((start_state, success_label))

            if success_label == 1:
                success_count += 1
            else:
                failure_count += 1

        # Write roa_labels.txt
        if self.create_roa_labels:
            print("\nWriting roa_labels.txt...")
            with open(self.reach_dir / "roa_labels.txt", 'w') as f:
                for start_state, label in roa_data:
                    state_str = ', '.join([f"{val:.15f}" for val in start_state])
                    f.write(f"{state_str}, {label}\n")
        else:
            print("\nSkipping roa_labels.txt (disabled)")

        # Create shuffled indices
        if self.create_shuffled_indices:
            print("Creating shuffled_indices.txt...")
            indices = np.arange(len(traj_files))
            np.random.seed(42)
            np.random.shuffle(indices)
            with open(self.reach_dir / "shuffled_indices.txt", 'w') as f:
                for idx in indices:
                    f.write(f"sequence_{idx}.txt\n")
        else:
            print("Skipping shuffled_indices.txt (disabled)")

        # Create dataset description
        if self.create_description:
            print("Creating dataset_description.json...")
            description = {
                "dataset_name": f"{self.dataset_name.replace('_', ' ').title()} Reach Dataset (Truncated at Success)",
                "description": f"Reachability dataset derived from {self.dataset_name} trajectories. Trajectories are truncated at the FIRST timestep where success criteria are met.",
                "source_dataset": self.dataset_name,
                "source_location": str(self.source_dir),
                "processing": "Trajectories truncated at first success point; full trajectories kept for failures",

                "success_condition": {
                    "type": "composite",
                    "criteria": [
                        {
                            "metric": "head_height",
                            "index": 21,
                            "threshold": self.h_thr,
                            "operator": ">=",
                            "description": f"Head height must be at least {self.h_thr}m"
                        },
                        {
                            "metric": "torso_vertical_z",
                            "index": 36,
                            "threshold": self.tz_thr,
                            "operator": ">=",
                            "description": f"Torso must be nearly upright (z-component >= {self.tz_thr})"
                        },
                        {
                            "metric": "com_horizontal_speed",
                            "indices": [37, 38],
                            "threshold": self.speed_thr,
                            "operator": "<=",
                            "description": f"Center of mass horizontal speed must be low (sqrt(vx^2 + vy^2) <= {self.speed_thr} m/s)"
                        }
                    ],
                    "description": f"Success requires ALL three conditions: head_height >= {self.h_thr}, torso_z >= {self.tz_thr}, and horizontal speed <= {self.speed_thr} m/s"
                },

                "dataset_statistics": {
                    "total_trajectories": len(traj_files),
                    "successful_trajectories": {
                        "count": success_count,
                        "percentage": 100 * success_count / len(traj_files),
                        "description": "Trajectories that ever reached success criteria"
                    },
                    "failed_trajectories": {
                        "count": failure_count,
                        "percentage": 100 * failure_count / len(traj_files),
                        "description": "Trajectories that never reached success criteria"
                    }
                },

                "state_space": {
                    "total_dimensions": 67,
                    "description": "Same as source humanoid dataset"
                },

                "additional_files": {
                    "roa_labels.txt": {
                        "description": "Reachability labels - maps starting state to success label",
                        "format": "67 comma-separated initial state values followed by label (0 or 1)"
                    },
                    "shuffled_indices.txt": {
                        "description": "Shuffled indices for train/test splits (seed=42)",
                        "format": "One trajectory filename per line (e.g., 'sequence_1234.txt')"
                    }
                }
            }

            with open(self.reach_dir / "dataset_description.json", 'w') as f:
                json.dump(description, f, indent=2)
        else:
            print("Skipping dataset_description.json (disabled)")

        # Print statistics
        print()
        print("="*70)
        print("REACH DATASET COMPLETE")
        print("="*70)
        print(f"Total trajectories: {len(traj_files)}")
        print(f"Successes: {success_count} ({100*success_count/len(traj_files):.2f}%)")
        print(f"Failures:  {failure_count} ({100*failure_count/len(traj_files):.2f}%)")
        print(f"Output: {self.reach_dir}")
        print()

    def create_stable_split_dataset(self):
        """Create the stable split dataset (recursive splitting at every state transition)."""
        print("="*70)
        print(f"CREATING {self.dataset_name.upper()}_STABLE_SPLIT DATASET")
        print("="*70)
        print()

        # Create directory structure
        self.stable_dir.mkdir(parents=True, exist_ok=True)
        split_traj_dir = self.stable_dir / "split_trajectories"

        # If only regenerating shuffled indices, skip trajectory processing
        if self.only_shuffled_indices:
            print("Mode: Only regenerating shuffled_indices.txt from existing files")
            print()

            if not split_traj_dir.exists():
                print(f"Error: {split_traj_dir} does not exist!")
                print("Cannot regenerate shuffled_indices without existing split trajectories.")
                return

            # Get existing split trajectory files
            split_files = sorted(split_traj_dir.glob("sequence_*_part*.txt"))
            if not split_files:
                print(f"Error: No split trajectory files found in {split_traj_dir}")
                return

            print(f"Found {len(split_files)} existing split trajectory files")

            # Create shuffled indices from existing split files
            print("Creating shuffled_indices.txt...")
            filenames = [f.name for f in split_files]
            np.random.seed(42)
            np.random.shuffle(filenames)
            with open(self.stable_dir / "shuffled_indices.txt", 'w') as f:
                for filename in filenames:
                    f.write(f"{filename}\n")

            print(f"✓ Created shuffled_indices.txt with {len(filenames)} entries")
            print(f"  (Each split part listed individually)")
            print()
            return

        # Normal mode: process trajectories
        split_traj_dir.mkdir(exist_ok=True)
        source_traj_dir = self.source_dir / "trajectories"

        # Get all trajectory files
        traj_files = sorted(source_traj_dir.glob("sequence_*.txt"))
        print(f"Found {len(traj_files)} trajectories to process")
        print()

        # Process all trajectories with recursive splitting at every state transition
        print("Processing trajectories with recursive splitting (at every F↔S transition)...")
        roa_data = []
        total_splits = 0
        all_split_files = []  # Track all split trajectory filenames

        for traj_file in tqdm(traj_files, desc="Splitting"):
            idx = int(traj_file.stem.split('_')[1])
            trajectory = self.load_trajectory(traj_file)

            # Store original trajectory info for ROA labels
            start_state = trajectory[0]
            final_state = trajectory[-1]
            success_label = 1 if self.check_success_criteria(final_state) else 0
            roa_data.append((start_state, success_label))

            # Process recursively to create splits
            base_name = f"sequence_{idx}"
            num_parts = self.process_trajectory_recursive(trajectory, split_traj_dir, base_name, 1)
            total_splits += (num_parts - 1)

            # Track all split filenames for this trajectory
            for part_num in range(1, num_parts):
                all_split_files.append(f"{base_name}_part{part_num}.txt")

        # Write roa_labels.txt (based on original final states)
        if self.create_roa_labels:
            print("\nWriting roa_labels.txt...")
            with open(self.stable_dir / "roa_labels.txt", 'w') as f:
                for start_state, label in roa_data:
                    state_str = ', '.join([f"{val:.15f}" for val in start_state])
                    f.write(f"{state_str}, {label}\n")
        else:
            print("\nSkipping roa_labels.txt (disabled)")

        # Create shuffled indices
        # NOTE: For stable_split, this lists ALL split trajectory parts individually
        if self.create_shuffled_indices:
            print("Creating shuffled_indices.txt...")
            # Shuffle the actual split filenames (each part is independent)
            shuffled_files = all_split_files.copy()
            np.random.seed(42)
            np.random.shuffle(shuffled_files)
            with open(self.stable_dir / "shuffled_indices.txt", 'w') as f:
                for filename in shuffled_files:
                    f.write(f"{filename}\n")
        else:
            print("Skipping shuffled_indices.txt (disabled)")

        # Count success/failure
        success_count = sum(1 for _, label in roa_data if label == 1)
        failure_count = len(roa_data) - success_count

        # Create dataset description
        if self.create_description:
            print("Creating dataset_description.json...")
            description = {
                "dataset_name": f"{self.dataset_name.replace('_', ' ').title()} Stable Split Dataset (State Transition Splitting)",
                "description": f"Dataset with {self.dataset_name} trajectories recursively split at EVERY state transition (F↔S). Creates multiple overlapping trajectory segments capturing both approaches to and departures from success.",
                "source_dataset": self.dataset_name,
                "source_location": str(self.source_dir),
                "processing": "Trajectories recursively split at every state transition (both F→S and S→F)",

                "success_condition": {
                    "type": "composite",
                    "criteria": [
                        {
                            "metric": "head_height",
                            "index": 21,
                            "threshold": self.h_thr,
                            "operator": ">=",
                            "description": f"Head height must be at least {self.h_thr}m"
                        },
                        {
                            "metric": "torso_vertical_z",
                            "index": 36,
                            "threshold": self.tz_thr,
                            "operator": ">=",
                            "description": f"Torso must be nearly upright (z-component >= {self.tz_thr})"
                        },
                        {
                            "metric": "com_horizontal_speed",
                            "indices": [37, 38],
                            "threshold": self.speed_thr,
                            "operator": "<=",
                            "description": f"Center of mass horizontal speed must be low (sqrt(vx^2 + vy^2) <= {self.speed_thr} m/s)"
                        }
                    ],
                    "description": f"Success requires ALL three conditions: head_height >= {self.h_thr}, torso_z >= {self.tz_thr}, and horizontal speed <= {self.speed_thr} m/s"
                },

                "dataset_statistics": {
                    "original_trajectories": len(traj_files),
                    "split_trajectories": total_splits,
                    "total_split_files": len(all_split_files),
                    "average_splits_per_trajectory": total_splits / len(traj_files),
                    "roa_labels_based_on_final_state": {
                        "successful": success_count,
                        "failed": failure_count,
                        "note": "Labels based on whether ORIGINAL trajectory's final state meets success criteria"
                    }
                },

                "state_space": {
                    "total_dimensions": 67,
                    "description": "Same as source humanoid dataset"
                },

                "additional_files": {
                    "roa_labels.txt": {
                        "description": "Labels based on ORIGINAL trajectory final states",
                        "format": "67 comma-separated initial state values followed by label (0 or 1)"
                    },
                    "shuffled_indices.txt": {
                        "description": "Shuffled indices for train/test splits (seed=42) - lists ALL split trajectory parts individually",
                        "format": "One split trajectory filename per line (e.g., 'sequence_1234_part1.txt', 'sequence_1234_part2.txt')",
                        "note": "Each part is treated as an independent trajectory segment. All split parts are shuffled together.",
                        "total_entries": len(all_split_files) if self.create_shuffled_indices else "N/A"
                    },
                    "split_trajectories/": {
                        "description": "Directory containing all split trajectory segments",
                        "naming": "sequence_{i}_part{j}.txt where i is original index, j is part number"
                    }
                }
            }

            with open(self.stable_dir / "dataset_description.json", 'w') as f:
                json.dump(description, f, indent=2)
        else:
            print("Skipping dataset_description.json (disabled)")

        # Print statistics
        print()
        print("="*70)
        print("STABLE SPLIT DATASET COMPLETE")
        print("="*70)
        print(f"Original trajectories: {len(traj_files)}")
        print(f"Split trajectories created: {total_splits}")
        print(f"Average splits per trajectory: {total_splits / len(traj_files):.2f}")
        if self.create_shuffled_indices:
            print(f"Shuffled indices entries: {len(all_split_files)} (each split part listed individually)")
        print(f"ROA labels (based on final state):")
        print(f"  Successes: {success_count} ({100*success_count/len(traj_files):.2f}%)")
        print(f"  Failures:  {failure_count} ({100*failure_count/len(traj_files):.2f}%)")
        print(f"Output: {self.stable_dir}")
        print()

    def create_all(self):
        """Create both reach and stable_split datasets."""
        print("\n" + "="*70)
        print(f"HUMANOID DATASET CREATION: {self.dataset_name.upper()}")
        print("="*70)
        print()
        print(f"Source: {self.source_dir}")

        # Show what will be created
        outputs = []
        if self.create_reach:
            outputs.append(f"Reach dataset: {self.reach_dir}")
        if self.create_stable_split:
            outputs.append(f"Stable split dataset: {self.stable_dir}")

        if outputs:
            print("Outputs:")
            for i, output in enumerate(outputs, 1):
                print(f"  {i}. {output}")
        else:
            print("WARNING: No datasets will be created (both reach and stable_split disabled)")
            return

        print()
        print("Components:")
        print(f"  - Trajectories: {'✓' if (self.create_reach or self.create_stable_split) else '✗'}")
        print(f"  - ROA labels: {'✓' if self.create_roa_labels else '✗'}")
        print(f"  - Shuffled indices: {'✓' if self.create_shuffled_indices else '✗'}")
        print(f"  - Description JSON: {'✓' if self.create_description else '✗'}")
        print()
        print("Success criteria:")
        print(f"  - head_height >= {self.h_thr}m (index 21)")
        print(f"  - torso_vertical_z >= {self.tz_thr} (index 36)")
        print(f"  - horizontal_speed <= {self.speed_thr} m/s (indices 37-38)")
        print()

        # Verify source exists
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {self.source_dir}")

        source_traj_dir = self.source_dir / "trajectories"
        if not source_traj_dir.exists():
            raise FileNotFoundError(f"Source trajectories directory does not exist: {source_traj_dir}")

        # Create datasets
        if self.create_reach:
            self.create_reach_dataset()
        else:
            print("Skipping reach dataset (disabled)")
            print()

        if self.create_stable_split:
            self.create_stable_split_dataset()
        else:
            print("Skipping stable split dataset (disabled)")
            print()

        print("="*70)
        print("DATASET CREATION COMPLETE")
        print("="*70)
        print()


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create humanoid reach and stable_split datasets from trajectory data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process humanoid_get_up_slow (all components enabled by default)
  python create_humanoid_datasets.py humanoid_get_up_slow

  # Process humanoid_get_up_medium
  python create_humanoid_datasets.py humanoid_get_up_medium

  # Only regenerate shuffled_indices.txt from existing trajectories (FAST)
  python create_humanoid_datasets.py humanoid_get_up --only-shuffled-indices
  python create_humanoid_datasets.py humanoid_get_up --only-shuffled-indices --skip-reach

  # Create only reach dataset (skip stable_split)
  python create_humanoid_datasets.py humanoid_get_up_slow --skip-stable-split

  # Create only stable_split dataset (skip reach)
  python create_humanoid_datasets.py humanoid_get_up_slow --skip-reach

  # Create trajectories only (skip metadata files)
  python create_humanoid_datasets.py humanoid_get_up_slow --skip-roa-labels --skip-shuffled-indices --skip-description

  # Use custom source path
  python create_humanoid_datasets.py my_dataset --source /path/to/source

  # Use custom thresholds
  python create_humanoid_datasets.py humanoid_get_up_slow --head-height 1.5 --torso-z 0.95
        """
    )

    parser.add_argument('dataset_name', type=str,
                       help='Name of the dataset (e.g., humanoid_get_up_slow)')
    parser.add_argument('--source', type=str, default=None,
                       help='Source directory path (default: /common/users/shared/pracsys/genMoPlan/data_trajectories/{dataset_name})')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output base directory (default: /common/users/dm1487/arcmg_datasets)')
    parser.add_argument('--head-height', type=float, default=DEFAULT_H_THR,
                       help=f'Head height threshold (default: {DEFAULT_H_THR})')
    parser.add_argument('--torso-z', type=float, default=DEFAULT_TZ_THR,
                       help=f'Torso vertical z threshold (default: {DEFAULT_TZ_THR})')
    parser.add_argument('--speed', type=float, default=DEFAULT_SPEED_THR,
                       help=f'Horizontal speed threshold (default: {DEFAULT_SPEED_THR})')

    # Toggles for what to create (default: all enabled)
    parser.add_argument('--skip-reach', action='store_true',
                       help='Skip creating reach dataset')
    parser.add_argument('--skip-stable-split', action='store_true',
                       help='Skip creating stable_split dataset')
    parser.add_argument('--skip-roa-labels', action='store_true',
                       help='Skip creating roa_labels.txt')
    parser.add_argument('--skip-shuffled-indices', action='store_true',
                       help='Skip creating shuffled_indices.txt')
    parser.add_argument('--skip-description', action='store_true',
                       help='Skip creating dataset_description.json')

    # Special mode: only regenerate shuffled indices
    parser.add_argument('--only-shuffled-indices', action='store_true',
                       help='Only regenerate shuffled_indices.txt from existing trajectory files (fast, no reprocessing)')

    args = parser.parse_args()

    # Create dataset creator
    creator = HumanoidDatasetCreator(
        dataset_name=args.dataset_name,
        source_dir=args.source,
        output_base_dir=args.output_dir,
        h_thr=args.head_height,
        tz_thr=args.torso_z,
        speed_thr=args.speed,
        create_reach=not args.skip_reach,
        create_stable_split=not args.skip_stable_split,
        create_roa_labels=not args.skip_roa_labels,
        create_shuffled_indices=not args.skip_shuffled_indices,
        create_description=not args.skip_description,
        only_shuffled_indices=args.only_shuffled_indices
    )

    # Create all datasets
    creator.create_all()


if __name__ == "__main__":
    main()
