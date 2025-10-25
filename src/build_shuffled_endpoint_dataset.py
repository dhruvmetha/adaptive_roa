import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm
import random
import os

np.random.seed(42)
random.seed(42)

@hydra.main(config_path="../configs", config_name="build_shuffled_endpoint_dataset.yaml")
def main(cfg: DictConfig) -> None:
    system = hydra.utils.instantiate(cfg.system)
    data_dirs = Path(cfg.data_dirs)
    dest_dir = Path(cfg.dest_dir)
    shuffled_idxs_file = Path(cfg.shuffled_idxs_file)
    type = cfg.type
    increment = cfg.increment
    start = cfg.start
    end = cfg.end
    use_fixed_attractors = cfg.get('use_fixed_attractors', False)
    attractor_radius = cfg.get('attractor_radius', 0.1)  # Default radius for classification
    balance_dataset = cfg.get('balance_dataset', False)  # Balance success/failure for train/val

    # Get system name
    system_name = system.name if hasattr(system, 'name') else 'unknown'

    with open(shuffled_idxs_file, 'r') as f:
        shuffled_idxs = [os.path.join(data_dirs, line.strip()) for line in f.readlines()][start:end]

    print(len(shuffled_idxs))
    print(shuffled_idxs)

    if use_fixed_attractors:
        print(f"Using fixed attractor mode:")
        print(f"  Target attractor: 1 (success), -1 (failure)")
        print(f"  Attractor radius: {attractor_radius}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    trajectory_files = [shuffled_idxs[i] for i in range(len(shuffled_idxs))]

    print(f"Found {len(trajectory_files)} trajectory files")
    endpoint_data = []
    success_endpoints = []
    failure_endpoints = []
    success_count = 0
    failure_count = 0

    for traj_file in tqdm(trajectory_files, desc="Processing trajectories"):
        with open(traj_file, 'r') as f:
            lines = f.readlines()

        trajectory = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split(',')))
                trajectory.append(values)

        if len(trajectory) == 0:
            continue

        if use_fixed_attractors:
            # Determine success/failure using system's is_in_attractor() criteria
            final_state = np.array(trajectory[-1])

            # Use system.is_in_attractor() logic with configurable radius
            is_success = system.is_in_attractor(final_state, radius=attractor_radius)

            # Scalar attractor: 1 for success, -1 for failure
            attractor = 1 if is_success else -1

            if is_success:
                success_count += 1
            else:
                failure_count += 1
        else:
            # Use the final point of the trajectory as the endpoint
            endpoint = np.array(trajectory[-1])
            attractor = None

        # Create endpoint pairs for all points in trajectory
        trajectory_endpoints = []
        for i in range(len(trajectory) - 1):  # Exclude final point as start
            start_state = np.array(trajectory[i])
            if use_fixed_attractors:
                # Format: [start_state, attractor] where attractor is 1 or -1
                trajectory_endpoints.append([*start_state, attractor])
            else:
                # Format: [start_state, endpoint] (multi-dimensional endpoint)
                trajectory_endpoints.append([*start_state, *endpoint])
            if type == "test":
                break

        # Store in separate lists for balancing or add directly
        if use_fixed_attractors and balance_dataset and type != "test":
            if is_success:
                success_endpoints.extend(trajectory_endpoints)
            else:
                failure_endpoints.extend(trajectory_endpoints)
        else:
            endpoint_data.extend(trajectory_endpoints)

    # Balance dataset for train/val splits if requested
    if use_fixed_attractors and balance_dataset and type != "test":
        min_count = min(len(success_endpoints), len(failure_endpoints))
        print(f"\n🔀 Balancing dataset:")
        print(f"  Success endpoints: {len(success_endpoints)}")
        print(f"  Failure endpoints: {len(failure_endpoints)}")
        print(f"  Sampling {min_count} from each class")

        # Randomly sample equal numbers from each class
        np.random.shuffle(success_endpoints)
        np.random.shuffle(failure_endpoints)
        endpoint_data = success_endpoints[:min_count] + failure_endpoints[:min_count]

        # Shuffle the combined dataset
        np.random.shuffle(endpoint_data)
        print(f"  Final balanced dataset: {len(endpoint_data)} endpoints")

    output_file = dest_dir / f"{increment}_endpoint_dataset.txt"
    print(f"Writing {len(endpoint_data)} endpoint pairs to file...")
    with open(output_file, 'w') as f:
        for endpoint in tqdm(endpoint_data, desc="Writing endpoints"):
            if use_fixed_attractors:
                # Format: start_state + attractor (scalar: 1 or -1)
                values = endpoint[:-1]  # All except attractor
                attractor = int(endpoint[-1])  # Attractor as int (1 or -1)
                f.write(' '.join(map(str, values)) + f' {attractor}\n')
            else:
                # Format: start_state + endpoint (multi-dimensional)
                f.write(' '.join(map(str, endpoint)) + '\n')

    print(f"\nBuilt endpoint dataset with {len(endpoint_data)} endpoint pairs")
    print(f"Saved to: {output_file}")

    if use_fixed_attractors:
        total = success_count + failure_count
        if total > 0:
            success_pct = (success_count / total) * 100
            failure_pct = (failure_count / total) * 100
            print(f"\n📊 Trajectory Classification:")
            print(f"  Total trajectories: {total}")
            print(f"  Success: {success_count} ({success_pct:.1f}%)")
            print(f"  Failure: {failure_count} ({failure_pct:.1f}%)")

if __name__ == "__main__":
    main()