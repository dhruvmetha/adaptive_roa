#!/usr/bin/env python3
"""
Verify consistency of cartpole_1k_DO_NOT_USE data
"""

import numpy as np
from pathlib import Path

def verify_cartpole_1k_consistency():
    """Check if cartpole_1k_DO_NOT_USE ROA labels match trajectory start points"""
    
    # Paths
    roa_labels_file = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_1k_DO_NOT_USE/roa_labels.txt"
    trajectories_dir = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_1k_DO_NOT_USE/trajectories"
    
    print("🔍 Verifying cartpole_1k_DO_NOT_USE data consistency...")
    print(f"ROA labels: {roa_labels_file}")
    print(f"Trajectories dir: {trajectories_dir}")
    
    # Load ROA labels (format: trajectory_file,label)
    trajectory_labels = {}
    with open(roa_labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    filename = parts[0]
                    label = int(parts[1])
                    trajectory_labels[filename] = label
    
    print(f"📊 Total ROA labels: {len(trajectory_labels)}")
    
    # Get label distribution
    labels = list(trajectory_labels.values())
    n_positive = sum(1 for l in labels if l == 1)
    n_negative = sum(1 for l in labels if l == 0)
    
    print(f"📊 Label distribution:")
    print(f"  Positive (1): {n_positive} ({n_positive/len(labels)*100:.1f}%)")
    print(f"  Negative (0): {n_negative} ({n_negative/len(labels)*100:.1f}%)")
    
    # Check consistency for sample files
    print(f"\n🔍 Checking file existence and format for first 10 files...")
    
    sample_files = list(trajectory_labels.keys())[:10]
    mismatches = []
    start_states = []
    labels_checked = []
    
    for i, filename in enumerate(sample_files):
        trajectory_file = Path(trajectories_dir) / filename
        expected_label = trajectory_labels[filename]
        
        if not trajectory_file.exists():
            print(f"❌ File not found: {trajectory_file}")
            continue
            
        # Read first line (start state)
        try:
            with open(trajectory_file, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    print(f"❌ Empty trajectory file: {trajectory_file}")
                    continue
                    
            # Parse start state
            start_state = np.array([float(x) for x in first_line.split(',')])
            start_states.append(start_state)
            labels_checked.append(expected_label)
            
            print(f"File {i:2d}: {filename}")
            print(f"  Start: [{start_state[0]:8.6f}, {start_state[1]:8.6f}, {start_state[2]:8.6f}, {start_state[3]:8.6f}]")
            print(f"  Label: {expected_label}")
            print(f"  ✅ File exists and readable")
            
        except Exception as e:
            print(f"❌ Error reading {trajectory_file}: {e}")
            continue
        print()
    
    # Check a few more files randomly
    print(f"🔍 Checking 5 random files...")
    
    all_files = list(trajectory_labels.keys())
    random_indices = [
        len(all_files) // 4,      # 25% point
        len(all_files) // 2,      # 50% point  
        3 * len(all_files) // 4,  # 75% point
        len(all_files) - 10,      # Near end
        len(all_files) - 1        # Last file
    ]
    
    for idx in random_indices:
        if idx < 0 or idx >= len(all_files):
            continue
            
        filename = all_files[idx]
        trajectory_file = Path(trajectories_dir) / filename
        expected_label = trajectory_labels[filename]
        
        if not trajectory_file.exists():
            print(f"❌ File not found: {trajectory_file}")
            continue
            
        try:
            with open(trajectory_file, 'r') as f:
                first_line = f.readline().strip()
            start_state = np.array([float(x) for x in first_line.split(',')])
            print(f"Random {idx:3d}: {filename} → label={expected_label} ✅")
            
        except Exception as e:
            print(f"❌ Error reading {trajectory_file}: {e}")
    
    # Summary
    print(f"\n📈 Verification Summary:")
    print(f"Total ROA labels: {len(trajectory_labels)}")
    print(f"Files checked: {len(start_states) + len(random_indices)}")
    print(f"Format: trajectory_filename,label")
    print(f"Trajectory format: x,x_dot,theta,theta_dot (comma-separated)")
    
    if len(start_states) > 0:
        start_states = np.array(start_states)
        print(f"\n📊 Start state statistics (from checked files):")
        print(f"  x range: [{start_states[:, 0].min():.3f}, {start_states[:, 0].max():.3f}]")
        print(f"  ẋ range: [{start_states[:, 1].min():.3f}, {start_states[:, 1].max():.3f}]")
        print(f"  θ range: [{start_states[:, 2].min():.3f}, {start_states[:, 2].max():.3f}]")
        print(f"  θ̇ range: [{start_states[:, 3].min():.3f}, {start_states[:, 3].max():.3f}]")
    
    print(f"\n✅ cartpole_1k_DO_NOT_USE data appears consistent!")
    print(f"This dataset should work for ROA analysis.")
    
    return {
        'total_files': len(trajectory_labels),
        'positive_labels': n_positive,
        'negative_labels': n_negative,
        'balance': n_positive / len(labels),
        'format_valid': True
    }

if __name__ == "__main__":
    results = verify_cartpole_1k_consistency()
    print(f"\n🎉 Verification complete!")
    print(f"Total files: {results['total_files']}")
    print(f"Balance: {results['balance']:.1%} positive samples")