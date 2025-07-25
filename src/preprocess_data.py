from tqdm import tqdm
import hydra
import os
import numpy as np
import random

def get_positive_samples(traj, num_samples):
    pos_samples = []
    for _ in range(num_samples):
        start_idx, end_idx = np.random.randint(0, len(traj), size=2)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        pos_samples.append([*traj[start_idx], *traj[end_idx], 1.0])
    return pos_samples

def get_negative_samples(data, num_samples):
    neg_samples = []
    pbar = tqdm(range(num_samples), desc="Generating negative samples")
    for _ in pbar:
        idx1, idx2 = np.random.randint(0, len(data), size=2)
        neg_samples.append([*data[idx1].tolist(), *data[idx2].tolist(), 0.0])
    return neg_samples

def build_reachability_data(data_dirs, samples_per_traj, dest_folder, train_ratio, valid_ratio):
    datafiles = []
    for data_dir in data_dirs:
        for file in os.listdir(data_dir):
            if file.endswith(".txt"):
                datafiles.append(os.path.join(data_dir, file))
                
    random.shuffle(datafiles)
                
    traj_data = []
    pos_samples = []
    neg_samples = []
    for datafile in tqdm(datafiles[:10000]):
        data = []
        with open(datafile, "r") as f:
            data = f.readlines()
        if len(data) == 0:
            continue
        record = False
        traj = []
        for line in data:
            if record:
                traj.append([float(x) for x in line.strip().split(" ") if x != ""][:4])
            if line.strip() == "":
                record = True
        # get positive samples
        pos_samples.extend(get_positive_samples(traj, samples_per_traj))
        traj_data.append(traj)
        
    traj_data = np.array(traj_data)
    
    # get negative samples
    flattended_traj_data = np.vstack(traj_data)
    np.random.shuffle(flattended_traj_data)
    neg_samples.extend(get_negative_samples(flattended_traj_data, len(pos_samples)))
        
    all_samples = pos_samples + neg_samples
    random.shuffle(all_samples)
    
    train_samples = all_samples[:int(len(all_samples) * train_ratio)]
    valid_samples = all_samples[int(len(all_samples) * train_ratio):int(len(all_samples) * (train_ratio + valid_ratio))]
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    with open(os.path.join(dest_folder, "train.txt"), "w") as f:
        pbar = tqdm(train_samples, desc="Writing train samples")
        for sample in pbar:
            f.write(f"{sample[0]} {sample[1]} {sample[2]} {sample[3]} {sample[4]} {sample[5]} {sample[6]} {sample[7]} {sample[8]}\n")
    
    with open(os.path.join(dest_folder, "valid.txt"), "w") as f:
        pbar = tqdm(valid_samples, desc="Writing valid samples")
        for sample in pbar:
            f.write(f"{sample[0]} {sample[1]} {sample[2]} {sample[3]} {sample[4]} {sample[5]} {sample[6]} {sample[7]} {sample[8]}\n")

@hydra.main(config_path="../configs", config_name="preprocess_data.yaml")
def main(cfg):
    build_reachability_data(cfg.data_dirs, cfg.samples_per_traj, cfg.dest_folder, cfg.train_ratio, cfg.valid_ratio)

if __name__ == "__main__":
    main()