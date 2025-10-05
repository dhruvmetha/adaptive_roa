#!/usr/bin/env python3
"""
Simple approach: Use existing ROA labels and shuffled indices to create 80-20 split
"""

import numpy as np
from pathlib import Path

def create_cartpole_test_roa_simple():
    """
    Use shuffled indices to create 80-20 split from existing ROA labels
    """
    
    # Paths
    existing_roa_labels = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_1k_DO_NOT_USE/roa_labels.txt"
    original_shuffled_indices = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/shuffled_indices.txt"
    
    # Output directory
    output_dir = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_1k_DO_NOT_USE")
    
    print("🔄 Creating CartPole test ROA labels using existing labels + shuffled indices...")
    print(f"Existing ROA labels: {existing_roa_labels}")
    print(f"Shuffled indices: {original_shuffled_indices}")
    
    # Load existing ROA labels (trajectory_filename,label format)
    trajectory_labels = {}
    with open(existing_roa_labels, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    filename = parts[0]
                    label = int(parts[1])
                    trajectory_labels[filename] = label
    
    print(f"📊 Loaded {len(trajectory_labels)} existing ROA labels")
    
    # Load shuffled indices to determine order
    with open(original_shuffled_indices, 'r') as f:
        shuffled_sequence_files = [line.strip() for line in f.readlines()]
    
    print(f"📊 Loaded {len(shuffled_sequence_files)} shuffled indices")
    
    # Map sequence_X.txt to trajectory_X.txt and collect corresponding labels
    ordered_labels = []
    ordered_filenames = []
    
    for seq_file in shuffled_sequence_files:
        # Extract sequence number: sequence_X.txt → X
        try:
            seq_num = int(seq_file.replace('sequence_', '').replace('.txt', ''))
            trajectory_filename = f"trajectory_{seq_num}.txt"
            
            if trajectory_filename in trajectory_labels:
                ordered_labels.append(trajectory_labels[trajectory_filename])
                ordered_filenames.append(trajectory_filename)
            else:
                print(f"⚠️  Missing trajectory: {trajectory_filename}")
        except ValueError:
            print(f"⚠️  Could not parse: {seq_file}")
    
    print(f"📊 Successfully mapped {len(ordered_labels)} trajectories")
    
    # Create 80-20 split maintaining shuffled order
    total_samples = len(ordered_labels)
    split_point = int(0.8 * total_samples)
    
    train_labels = ordered_labels[:split_point]
    test_labels = ordered_labels[split_point:]
    
    train_filenames = ordered_filenames[:split_point]
    test_filenames = ordered_filenames[split_point:]
    
    print(f"\\n📊 Split Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Train: {len(train_labels)} (80%)")
    print(f"Test: {len(test_labels)} (20%)")
    
    # Analyze test balance
    test_positive = sum(1 for label in test_labels if label == 1)
    test_negative = sum(1 for label in test_labels if label == 0)
    
    print(f"\\n🎯 Test Set Label Balance:")
    print(f"Positive samples (label=1): {test_positive} ({test_positive/len(test_labels)*100:.1f}%)")
    print(f"Negative samples (label=0): {test_negative} ({test_negative/len(test_labels)*100:.1f}%)")
    
    # Save test labels as simple filename,label format (for endpoint generation)
    test_labels_file = output_dir / "test_roa_labels_simple.txt"
    with open(test_labels_file, 'w') as f:
        for filename, label in zip(test_filenames, test_labels):
            f.write(f"{filename},{label}\n")
    
    print(f"\\n✅ Test ROA labels saved:")
    print(f"📁 {test_labels_file} ({len(test_labels)} samples)")
    print(f"Format: trajectory_filename,label")
    
    # Show sample
    print(f"\\n🔍 Sample Test Data (first 5 rows):")
    for i in range(min(5, len(test_filenames))):
        print(f"  {test_filenames[i]},{test_labels[i]}")
    
    return {
        'total_samples': total_samples,
        'test_count': len(test_labels),
        'test_positive': test_positive,
        'test_negative': test_negative,
        'test_balance': test_positive / len(test_labels)
    }

if __name__ == "__main__":
    results = create_cartpole_test_roa_simple()
    print(f"\\n🎉 Simple CartPole test ROA labels created!")
    print(f"Test samples: {results['test_count']}")
    print(f"Balance: {results['test_balance']:.1%} positive")
    print(f"Ready for endpoint generation!")