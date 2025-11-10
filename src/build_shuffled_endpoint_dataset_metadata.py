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
    attractor_radius = cfg.get('attractor_radius', 0.1)  # Default radius for classification
    first_last_only = cfg.get('first_last_only', False)
    
    success_endpoint_count = 0
    failure_endpoint_count = 0
    

    # Get system name
    system_name = system.name if hasattr(system, 'name') else 'unknown'
    
    print(f"System name: {system_name}")

    with open(shuffled_idxs_file, 'r') as f:
        trajectory_files = [os.path.join(data_dirs, line.strip()) for line in f.readlines()][start:end]

    print(f"Classifying trajectories using system attractor:")
    print(f"  Target attractor: 1 (success), -1 (failure)")
    print(f"  Attractor radius: {attractor_radius}")

    dest_dir.mkdir(parents=True, exist_ok=True)


    print(f"Found {len(trajectory_files)} trajectory files")
    success_trajectories = []
    failure_trajectories = []

    # First pass: classify trajectories and separate by success/failure
    for traj_file in tqdm(trajectory_files, desc="Classifying trajectories"):
        traj = np.loadtxt(traj_file, delimiter=",")
        num_states = traj.shape[0]

        final_state = traj[-1]
        is_success = system.is_in_attractor(final_state)

        if is_success:
            success_trajectories.append((traj_file, num_states))
            success_endpoint_count += num_states - 1
        else:
            failure_trajectories.append((traj_file, num_states))
            failure_endpoint_count += num_states - 1

    success_count = len(success_trajectories)
    failure_count = len(failure_trajectories)

    # Second pass: create endpoint data from selected trajectories
    endpoint_data = []

    for traj_file, num_states in tqdm(success_trajectories, desc="Processing success trajectories"):
        for i in range(num_states - 1):  # Exclude final point as start
            endpoint_data.append([traj_file, i, num_states - 1, True])
            if first_last_only:
                break
            if type == "test":
                break

    for traj_file, num_states in tqdm(failure_trajectories, desc="Processing failure trajectories"):
        for i in range(num_states - 1):  # Exclude final point as start
            endpoint_data.append([traj_file, i, num_states - 1, False])
            if first_last_only:
                break
            if type == "test":
                break

    # Shuffle the combined endpoint data
    random.shuffle(endpoint_data)

    output_file = dest_dir / f"{increment}_endpoint_dataset.txt"
    print(f"Writing {len(endpoint_data)} endpoint metadata to file...")
    with open(output_file, 'w') as f:
        for endpoint in tqdm(endpoint_data, desc="Writing endpoints"):
            # Format: file_path start_idx end_idx label
            file_path, start_idx, end_idx, is_success = endpoint
            f.write(f'{file_path} {start_idx} {end_idx} {int(is_success)}\n')

    print(f"\nBuilt endpoint metadata dataset with {len(endpoint_data)} endpoint pairs")
    print(f"Saved to: {output_file}")

    total = success_count + failure_count
    if total > 0:
        success_pct = (success_count / total) * 100
        failure_pct = (failure_count / total) * 100
        print(f"\n📊 Trajectory Classification:")
        print(f"  Total trajectories: {total}")
        print(f"  Success: {success_count} ({success_pct:.1f}%)")
        print(f"  Failure: {failure_count} ({failure_pct:.1f}%)")
        
    print(f"Success endpoint count: {success_endpoint_count}")
    print(f"Failure endpoint count: {failure_endpoint_count}")
    print(f"Total endpoint count: {success_endpoint_count + failure_endpoint_count}")
    print(f"Success endpoint percentage: {success_endpoint_count / (success_endpoint_count + failure_endpoint_count) * 100:.1f}%")
    print(f"Failure endpoint percentage: {failure_endpoint_count / (success_endpoint_count + failure_endpoint_count) * 100:.1f}%")

if __name__ == "__main__":
    main()
