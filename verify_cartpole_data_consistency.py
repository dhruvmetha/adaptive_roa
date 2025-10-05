#!/usr/bin/env python3
"""
Verify that trajectory start points from shuffled files match ROA labels
"""

import numpy as np
from pathlib import Path

def verify_cartpole_data_consistency():
    """Check if trajectory start points match ROA labels"""
    
    # Paths
    roa_labels_file = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/roa_labels.txt"
    shuffled_indices_file = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/shuffled_indices.txt"
    trajectories_dir = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/trajectories"
    
    print("🔍 Verifying CartPole data consistency...")
    print(f"ROA labels: {roa_labels_file}")
    print(f"Shuffled indices: {shuffled_indices_file}")
    print(f"Trajectories dir: {trajectories_dir}")
    
    # Load ROA labels
    roa_data = np.loadtxt(roa_labels_file)
    print(f"\n📊 ROA labels shape: {roa_data.shape}")
    
    # Load shuffled indices
    with open(shuffled_indices_file, 'r') as f:
        trajectory_files = [line.strip() for line in f.readlines()]
    
    print(f"📊 Number of trajectory files: {len(trajectory_files)}")
    
    # Verify counts match
    assert len(roa_data) == len(trajectory_files), f"Mismatch: {len(roa_data)} ROA labels vs {len(trajectory_files)} trajectory files"
    
    print(f"✅ Counts match: {len(roa_data)} samples")
    
    # Check first few trajectory files
    print(f"\n🔍 Checking consistency for first 10 samples...")
    
    mismatches = []
    
    for i in range(min(10, len(trajectory_files))):
        # ROA label start state
        roa_start = roa_data[i, :4]  # First 4 columns: [x, ẋ, θ, θ̇]
        roa_label = int(roa_data[i, 4])  # Last column: label
        
        # Load trajectory file
        trajectory_file = Path(trajectories_dir) / trajectory_files[i]
        
        if not trajectory_file.exists():
            print(f"❌ File not found: {trajectory_file}")
            continue
            
        # Read first line of trajectory (start state)
        with open(trajectory_file, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                print(f"❌ Empty trajectory file: {trajectory_file}")
                continue
                
        try:
            traj_start = np.array([float(x) for x in first_line.split(',')])
        except ValueError:
            print(f"❌ Invalid format in {trajectory_file}: {first_line}")
            continue
            
        # Compare start states (allow small tolerance for floating point)
        tolerance = 1e-10
        state_diff = np.abs(roa_start - traj_start)
        max_diff = np.max(state_diff)
        
        print(f"Sample {i:2d}:")
        print(f"  ROA:  [{roa_start[0]:8.6f}, {roa_start[1]:8.6f}, {roa_start[2]:8.6f}, {roa_start[3]:8.6f}] label={roa_label}")
        print(f"  Traj: [{traj_start[0]:8.6f}, {traj_start[1]:8.6f}, {traj_start[2]:8.6f}, {traj_start[3]:8.6f}]")
        print(f"  Max diff: {max_diff:.2e} {'✅' if max_diff < tolerance else '❌'}")
        
        if max_diff >= tolerance:
            mismatches.append({
                'index': i,
                'file': trajectory_files[i],
                'roa_start': roa_start,
                'traj_start': traj_start,
                'max_diff': max_diff
            })
        print()
    
    # Check a few more samples randomly
    print(f"🔍 Checking 5 random samples from middle and end...")
    
    random_indices = [
        len(trajectory_files) // 4,      # 25% point
        len(trajectory_files) // 2,      # 50% point  
        3 * len(trajectory_files) // 4,  # 75% point
        len(trajectory_files) - 10,      # Near end
        len(trajectory_files) - 1        # Last sample
    ]
    
    for i in random_indices:
        if i < 0 or i >= len(trajectory_files):
            continue
            
        # ROA label start state
        roa_start = roa_data[i, :4]
        roa_label = int(roa_data[i, 4])
        
        # Load trajectory file
        trajectory_file = Path(trajectories_dir) / trajectory_files[i]
        
        if not trajectory_file.exists():
            print(f"❌ File not found: {trajectory_file}")
            continue
            
        # Read first line of trajectory
        with open(trajectory_file, 'r') as f:
            first_line = f.readline().strip()
            
        try:
            traj_start = np.array([float(x) for x in first_line.split(',')])
        except ValueError:
            print(f"❌ Invalid format in {trajectory_file}: {first_line}")
            continue
            
        # Compare
        tolerance = 1e-10
        state_diff = np.abs(roa_start - traj_start)
        max_diff = np.max(state_diff)
        
        print(f"Sample {i:3d}: Max diff = {max_diff:.2e} {'✅' if max_diff < tolerance else '❌'}")
        
        if max_diff >= tolerance:
            mismatches.append({
                'index': i,
                'file': trajectory_files[i],
                'roa_start': roa_start,
                'traj_start': traj_start,
                'max_diff': max_diff
            })
    
    # Summary
    print(f"\n📈 Verification Summary:")
    print(f"Total samples checked: {min(10, len(trajectory_files)) + len(random_indices)}")
    print(f"Mismatches found: {len(mismatches)}")
    
    if len(mismatches) == 0:
        print("✅ All checked samples match! Data is consistent.")
        return True
    else:
        print("❌ Found mismatches:")
        for mismatch in mismatches:
            print(f"  Index {mismatch['index']}: {mismatch['file']} (max diff: {mismatch['max_diff']:.2e})")
        return False

if __name__ == "__main__":
    is_consistent = verify_cartpole_data_consistency()
    if is_consistent:
        print(f"\n🎉 Data consistency verified! ROA labels match trajectory start points.")
    else:
        print(f"\n⚠️  Data inconsistency detected! Please investigate mismatches.")