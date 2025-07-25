import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm

@hydra.main(config_path="../configs", config_name="build_endpoint_dataset.yaml")
def main(cfg: DictConfig) -> None:
    system = hydra.utils.instantiate(cfg.system)
    data_dirs = Path(cfg.data_dirs)
    dest_dir = Path(cfg.dest_dir)
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    trajectory_files = list(data_dirs.glob("*.txt"))
    
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
            
        start_state = np.array(trajectory[0])
        
        first_attractor_state = None
        for state_values in trajectory:
            state = np.array(state_values)
            if system.is_in_attractor(state):
                first_attractor_state = state
                break
        
        if first_attractor_state is not None:
            endpoint_data.append([*start_state, *first_attractor_state])
    
    output_file = dest_dir / "endpoint_dataset.txt"
    print(f"Writing {len(endpoint_data)} endpoint pairs to file...")
    with open(output_file, 'w') as f:
        for endpoint in tqdm(endpoint_data, desc="Writing endpoints"):
            f.write(' '.join(map(str, endpoint)) + '\n')
    
    print(f"Built endpoint dataset with {len(endpoint_data)} trajectories")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()