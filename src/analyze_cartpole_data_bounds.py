#!/usr/bin/env python3
"""
Analyze CartPole trajectory data to find actual data bounds and save them for reuse

NOTE: Updated for NEW state format [x, theta, x_dot, theta_dot] (theta moved from index 2 to 1)
"""
import os
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse


def analyze_trajectory_file(file_path):
    """Analyze a single trajectory file and return min/max values"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        states = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split(',')))
                if len(values) == 4:  # [x, theta, x_dot, theta_dot] - NEW FORMAT
                    x, theta, x_dot, theta_dot = values
                    states.append([x, theta, x_dot, theta_dot])
        
        if len(states) == 0:
            return None
            
        states = np.array(states)
        return {
            'min': np.min(states, axis=0),
            'max': np.max(states, axis=0),
            'count': len(states)
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def analyze_all_cartpole_data(data_dir, output_path, max_files=None):
    """
    Analyze all CartPole trajectory files to find global data bounds
    
    Args:
        data_dir: Directory containing trajectory files
        output_path: Path to save the bounds pickle file
        max_files: Maximum number of files to process (None for all)
    """
    data_dir = Path(data_dir)
    trajectory_files = list(data_dir.glob("sequence_*.txt"))
    
    if max_files:
        trajectory_files = trajectory_files[:max_files]
    
    print(f"Found {len(trajectory_files)} trajectory files")
    print(f"Processing up to {max_files if max_files else 'all'} files...")
    
    # Initialize bounds tracking
    global_min = np.array([float('inf')] * 4)  # [x, theta, x_dot, theta_dot] - NEW FORMAT
    global_max = np.array([float('-inf')] * 4)
    total_states = 0
    processed_files = 0
    
    # Process each trajectory file
    for traj_file in tqdm(trajectory_files, desc="Analyzing trajectories"):
        result = analyze_trajectory_file(traj_file)
        if result is not None:
            # Update global bounds
            global_min = np.minimum(global_min, result['min'])
            global_max = np.maximum(global_max, result['max'])
            total_states += result['count']
            processed_files += 1
    
    # Calculate statistics
    bounds_data = {
        'bounds': {
            'x': {'min': float(global_min[0]), 'max': float(global_max[0])},        # Index 0: cart position
            'theta': {'min': float(global_min[1]), 'max': float(global_max[1])},    # Index 1: pole angle  
            'x_dot': {'min': float(global_min[2]), 'max': float(global_max[2])},    # Index 2: cart velocity
            'theta_dot': {'min': float(global_min[3]), 'max': float(global_max[3])} # Index 3: angular velocity
        },
        'statistics': {
            'total_files_processed': processed_files,
            'total_states_analyzed': total_states,
            'files_requested': max_files,
            'data_directory': str(data_dir)
        },
        'ranges': {
            'x': float(global_max[0] - global_min[0]),        # Index 0: cart position
            'theta': float(global_max[1] - global_min[1]),    # Index 1: pole angle
            'x_dot': float(global_max[2] - global_min[2]),    # Index 2: cart velocity  
            'theta_dot': float(global_max[3] - global_min[3]) # Index 3: angular velocity
        }
    }
    
    # Save to pickle file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(bounds_data, f)
    
    # Print results
    print(f"\n=== CartPole Data Bounds Analysis ===")
    print(f"Processed {processed_files} files with {total_states:,} total states")
    print(f"\nData Bounds (NEW FORMAT: [x, theta, x_dot, theta_dot]):")
    print(f"  x (cart position):     [{global_min[0]:8.3f}, {global_max[0]:8.3f}] (range: {bounds_data['ranges']['x']:.3f})")
    print(f"  theta (pole angle):    [{global_min[1]:8.3f}, {global_max[1]:8.3f}] (range: {bounds_data['ranges']['theta']:.3f})")
    print(f"  x_dot (cart velocity): [{global_min[2]:8.3f}, {global_max[2]:8.3f}] (range: {bounds_data['ranges']['x_dot']:.3f})")
    print(f"  theta_dot (angular vel): [{global_min[3]:8.3f}, {global_max[3]:8.3f}] (range: {bounds_data['ranges']['theta_dot']:.3f})")
    
    print(f"\nSaved bounds to: {output_path}")
    print(f"\n✅ Updated for NEW state format: [x, theta, x_dot, theta_dot]")
    
    return bounds_data


def load_cartpole_bounds(bounds_path):
    """Load CartPole bounds from pickle file"""
    with open(bounds_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CartPole data bounds")
    parser.add_argument("--data_dir", 
                       default="/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/trajectories",
                       help="Directory containing trajectory files")
    parser.add_argument("--output", 
                       default="/common/users/dm1487/arcmg_datasets/cartpole/cartpole_data_bounds.pkl",
                       help="Output pickle file path")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process (None for all)")
    parser.add_argument("--load_test", action="store_true",
                       help="Test loading the saved bounds file")
    
    args = parser.parse_args()
    
    if args.load_test:
        if os.path.exists(args.output):
            print("Testing bounds loading...")
            bounds = load_cartpole_bounds(args.output)
            print("Successfully loaded bounds:")
            print(bounds)
        else:
            print(f"Bounds file not found: {args.output}")
    else:
        analyze_all_cartpole_data(args.data_dir, args.output, args.max_files)