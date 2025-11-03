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

    # Get system name
    system_name = system.name if hasattr(system, 'name') else 'unknown'

    with open(shuffled_idxs_file, 'r') as f:
        trajectory_files = [os.path.join(data_dirs, line.strip()) for line in f.readlines()][start:end]

    print(f"Classifying trajectories using system attractor:")
    print(f"  Target attractor: 1 (success), -1 (failure)")
    print(f"  Attractor radius: {attractor_radius}")

    dest_dir.mkdir(parents=True, exist_ok=True)


    print(f"Found {len(trajectory_files)} trajectory files")
    endpoint_data = []
    success_count = 0
    failure_count = 0

    for traj_file in tqdm(trajectory_files, desc="Processing trajectories"):
        
        
        traj = np.loadtxt(traj_file, delimiter=",")
        num_states = traj.shape[0]
        
        final_state = traj[-1]
        is_success = system.is_in_attractor(final_state, radius=attractor_radius)
        
        
        if is_success:
            success_count += 1
        else:
            failure_count += 1

        # Create endpoint metadata for all points in trajectory
        # Format: [file_path, start_idx, end_idx]
        for i in range(num_states - 1):  # Exclude final point as start
            endpoint_data.append([traj_file, i, traj.shape[0] - 1, is_success])
            if type == "test":
                break

    output_file = dest_dir / f"{increment}_endpoint_dataset.txt"
    print(f"Writing {len(endpoint_data)} endpoint metadata to file...")
    with open(output_file, 'w') as f:
        for endpoint in tqdm(endpoint_data, desc="Writing endpoints"):
            A
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

if __name__ == "__main__":
    main()
