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
    with open(shuffled_idxs_file, 'r') as f:
        shuffled_idxs = [os.path.join(data_dirs, line.strip()) for line in f.readlines()][start:end]
        
    print(len(shuffled_idxs))
    print(shuffled_idxs)
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    trajectory_files = [shuffled_idxs[i] for i in range(len(shuffled_idxs))]
    
    print(f"Found {len(trajectory_files)} trajectory files")
    endpoint_data = []
    
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
            
        # Use the final point of the trajectory as the endpoint
        final_state = np.array(trajectory[-1])
        
        # Create endpoint pairs for all points in trajectory pointing to final state
        for i in range(len(trajectory) - 1):  # Exclude final point as start
            start_state = np.array(trajectory[i])
            endpoint_data.append([*start_state, *final_state])
            if type == "test":
                break
    
    output_file = dest_dir / f"{increment}_endpoint_dataset.txt"
    print(f"Writing {len(endpoint_data)} endpoint pairs to file...")
    with open(output_file, 'w') as f:
        for endpoint in tqdm(endpoint_data, desc="Writing endpoints"):
            f.write(' '.join(map(str, endpoint)) + '\n')
    
    print(f"Built endpoint dataset with {len(endpoint_data)} trajectories")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()