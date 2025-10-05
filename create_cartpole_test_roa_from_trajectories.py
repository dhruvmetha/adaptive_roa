#!/usr/bin/env python3
"""
Create CartPole 80-20 split using shuffled indices and extract test ROA labels from trajectories
"""

import numpy as np
from pathlib import Path

def create_cartpole_test_roa_from_trajectories():
    """
    Use shuffled indices to create 80-20 split, then extract start states from 20% test trajectories
    """
    
    # Paths - source data (consistent dataset)
    consistent_roa_labels = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_1k_DO_NOT_USE/roa_labels.txt"
    consistent_trajectories_dir = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_1k_DO_NOT_USE/trajectories"
    
    # Original shuffled indices (for split determination)  
    original_shuffled_indices = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/shuffled_indices.txt"
    
    # Output directory - save in cartpole_1k_DO_NOT_USE directory  
    output_dir = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_1k_DO_NOT_USE")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Creating CartPole test ROA labels from trajectory start states...")
    print(f"Consistent ROA labels: {consistent_roa_labels}")
    print(f"Consistent trajectories: {consistent_trajectories_dir}")
    print(f"Original shuffled indices: {original_shuffled_indices}")
    print(f"Output directory: {output_dir}")
    
    # Load consistent ROA labels (trajectory_filename,label format)
    trajectory_labels = {}
    with open(consistent_roa_labels, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    filename = parts[0]
                    label = int(parts[1])
                    trajectory_labels[filename] = label
    
    print(f"📊 Loaded {len(trajectory_labels)} consistent ROA labels")
    
    # Load original shuffled indices to determine order
    with open(original_shuffled_indices, 'r') as f:
        shuffled_sequence_files = [line.strip() for line in f.readlines()]
    
    print(f"📊 Loaded {len(shuffled_sequence_files)} shuffled indices")
    
    # Map sequence files to trajectory files (sequence_X.txt → trajectory_Y.txt)
    # The original shuffled indices are sequence_X.txt, but consistent data uses trajectory_Y.txt
    # We need to find the mapping
    
    # Extract sequence numbers from shuffled indices
    sequence_numbers = []
    for seq_file in shuffled_sequence_files:
        # Extract number from sequence_X.txt
        try:
            seq_num = int(seq_file.replace('sequence_', '').replace('.txt', ''))
            sequence_numbers.append(seq_num)
        except ValueError:
            print(f"⚠️  Could not parse sequence number from: {seq_file}")
            continue
    
    print(f"📊 Extracted {len(sequence_numbers)} sequence numbers")
    print(f"Sequence number range: {min(sequence_numbers)} to {max(sequence_numbers)}")
    
    # Create mapping: assume sequence_X.txt maps to trajectory_X.txt in consistent dataset
    available_trajectories = []
    available_labels = []
    missing_trajectories = []
    
    for i, seq_num in enumerate(sequence_numbers):
        trajectory_filename = f"trajectory_{seq_num}.txt"
        trajectory_path = Path(consistent_trajectories_dir) / trajectory_filename
        
        if trajectory_filename in trajectory_labels and trajectory_path.exists():
            available_trajectories.append(trajectory_filename)
            available_labels.append(trajectory_labels[trajectory_filename])
        else:
            missing_trajectories.append(trajectory_filename)
    
    print(f"📊 Mapping results:")
    print(f"  Available trajectories: {len(available_trajectories)}")
    print(f"  Missing trajectories: {len(missing_trajectories)}")
    
    if len(missing_trajectories) > 0:
        print(f"⚠️  Missing trajectories (first 10): {missing_trajectories[:10]}")
    
    # Create 80-20 split from available trajectories (maintaining shuffled order)
    total_available = len(available_trajectories)
    split_point = int(0.8 * total_available)
    
    train_trajectories = available_trajectories[:split_point]
    test_trajectories = available_trajectories[split_point:]
    
    train_labels = available_labels[:split_point]
    test_labels = available_labels[split_point:]
    
    print(f"\\n📊 Split Statistics:")
    print(f"Total available: {total_available}")
    print(f"Train: {len(train_trajectories)} (80%)")
    print(f"Test: {len(test_trajectories)} (20%)")
    
    # Analyze test set balance
    test_positive = sum(1 for label in test_labels if label == 1)
    test_negative = sum(1 for label in test_labels if label == 0)
    
    print(f"\\n🎯 Test Set Label Balance:")
    print(f"Positive samples (label=1): {test_positive} ({test_positive/len(test_labels)*100:.1f}%)")
    print(f"Negative samples (label=0): {test_negative} ({test_negative/len(test_labels)*100:.1f}%)")
    
    # Extract start states from test trajectory files
    print(f"\\n🔄 Extracting start states from test trajectories...")
    
    test_start_states = []
    valid_test_labels = []
    failed_extractions = []
    
    for i, (trajectory_filename, label) in enumerate(zip(test_trajectories, test_labels)):
        trajectory_path = Path(consistent_trajectories_dir) / trajectory_filename
        
        try:
            # Read first line of trajectory (start state)
            with open(trajectory_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    failed_extractions.append(f"{trajectory_filename}: empty file")
                    continue
            
            # Parse start state: x,x_dot,theta,theta_dot
            start_state = [float(x) for x in first_line.split(',')]
            if len(start_state) != 4:
                failed_extractions.append(f"{trajectory_filename}: invalid format")
                continue
                
            test_start_states.append(start_state)
            valid_test_labels.append(label)
            
        except Exception as e:
            failed_extractions.append(f"{trajectory_filename}: {str(e)}")
            continue
    
    print(f"✅ Successfully extracted {len(test_start_states)} start states")
    if len(failed_extractions) > 0:
        print(f"⚠️  Failed extractions: {len(failed_extractions)}")
        for failure in failed_extractions[:5]:  # Show first 5 failures
            print(f"  - {failure}")
    
    # Create test ROA labels in [x, ẋ, θ, θ̇, label] format
    test_roa_data = []
    for start_state, label in zip(test_start_states, valid_test_labels):
        roa_row = start_state + [label]  # [x, ẋ, θ, θ̇, label]
        test_roa_data.append(roa_row)
    
    test_roa_data = np.array(test_roa_data)
    
    # Save test ROA labels
    test_roa_file = output_dir / "test_roa_labels_from_trajectories.txt"
    np.savetxt(test_roa_file, test_roa_data, fmt='%.16e %.16e %.16e %.16e %i')
    
    # Save test trajectory filenames for reference
    test_trajectories_file = output_dir / "test_trajectory_filenames.txt"
    with open(test_trajectories_file, 'w') as f:
        for filename in test_trajectories[:len(test_start_states)]:  # Only successful ones
            f.write(f"{filename}\\n")
    
    print(f"\\n✅ Files saved:")
    print(f"📁 Test ROA labels: {test_roa_file} ({len(test_roa_data)} samples)")
    print(f"📁 Test trajectory files: {test_trajectories_file}")
    
    # Display sample data
    print(f"\\n🔍 Sample Test ROA Data (first 5 rows):")
    print("   x          ẋ          θ          θ̇          label")
    for i in range(min(5, len(test_roa_data))):
        row = test_roa_data[i]
        print(f"   {row[0]:8.3f}  {row[1]:8.3f}  {row[2]:8.3f}  {row[3]:8.3f}     {int(row[4])}")
    
    # Final statistics
    final_positive = sum(1 for label in valid_test_labels if label == 1)
    final_negative = sum(1 for label in valid_test_labels if label == 0)
    
    print(f"\\n📈 Final Test Set Statistics:")
    print(f"Total samples: {len(test_roa_data)}")
    print(f"Positive (success): {final_positive} ({final_positive/len(test_roa_data)*100:.1f}%)")
    print(f"Negative (failure): {final_negative} ({final_negative/len(test_roa_data)*100:.1f}%)")
    
    # Analyze start state ranges
    if len(test_roa_data) > 0:
        print(f"\\n📊 Test Start State Ranges:")
        print(f"  x: [{test_roa_data[:, 0].min():.3f}, {test_roa_data[:, 0].max():.3f}]")
        print(f"  ẋ: [{test_roa_data[:, 1].min():.3f}, {test_roa_data[:, 1].max():.3f}]")
        print(f"  θ: [{test_roa_data[:, 2].min():.3f}, {test_roa_data[:, 2].max():.3f}]")
        print(f"  θ̇: [{test_roa_data[:, 3].min():.3f}, {test_roa_data[:, 3].max():.3f}]")
    
    return {
        'total_available': total_available,
        'train_count': len(train_trajectories),
        'test_count': len(test_start_states),
        'test_positive': final_positive,
        'test_negative': final_negative,
        'test_balance': final_positive / len(test_roa_data) if len(test_roa_data) > 0 else 0,
        'failed_extractions': len(failed_extractions)
    }

if __name__ == "__main__":
    results = create_cartpole_test_roa_from_trajectories()
    print(f"\\n🎉 CartPole test ROA labels created successfully!")
    print(f"Test samples: {results['test_count']}")
    print(f"Test balance: {results['test_balance']:.1%} positive samples")
    print(f"Ready for ROA analysis!")