#!/usr/bin/env python3
"""
Create 80-20 split for CartPole data from shuffled indices
"""

import numpy as np
from pathlib import Path

def create_cartpole_splits():
    """Create 80-20 train/test split for CartPole data"""
    
    # Paths
    source_roa_labels = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/roa_labels.txt"
    source_shuffled_indices = "/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole/shuffled_indices.txt"
    
    output_dir = Path("/common/users/dm1487/arcmg_datasets/cartpole")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Creating CartPole 80-20 data split...")
    print(f"Source ROA labels: {source_roa_labels}")
    print(f"Source shuffled indices: {source_shuffled_indices}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    roa_data = np.loadtxt(source_roa_labels)
    with open(source_shuffled_indices, 'r') as f:
        indices = [line.strip() for line in f.readlines()]
    
    total_samples = len(roa_data)
    assert len(indices) == total_samples, f"Mismatch: {len(roa_data)} labels vs {len(indices)} indices"
    
    # Create 80-20 split (directly from order - top 80%, final 20%)
    split_point = int(0.8 * total_samples)
    
    train_indices = list(range(split_point))
    test_indices = list(range(split_point, total_samples))
    
    print(f"\n📊 Split Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {len(train_indices)} (80%)")
    print(f"Test samples: {len(test_indices)} (20%)")
    
    # Split data
    train_roa_data = roa_data[train_indices]
    test_roa_data = roa_data[test_indices]
    
    train_file_indices = [indices[i] for i in train_indices]
    test_file_indices = [indices[i] for i in test_indices]
    
    # Analyze label balance in test set
    test_labels = test_roa_data[:, 4].astype(int)  # Last column is label
    test_positive = np.sum(test_labels == 1)
    test_negative = np.sum(test_labels == 0)
    
    print(f"\n🎯 Test Set Label Balance:")
    print(f"Positive samples (label=1): {test_positive} ({test_positive/len(test_labels)*100:.1f}%)")
    print(f"Negative samples (label=0): {test_negative} ({test_negative/len(test_labels)*100:.1f}%)")
    
    # Analyze train set balance too
    train_labels = train_roa_data[:, 4].astype(int)
    train_positive = np.sum(train_labels == 1)
    train_negative = np.sum(train_labels == 0)
    
    print(f"\n📈 Train Set Label Balance:")
    print(f"Positive samples (label=1): {train_positive} ({train_positive/len(train_labels)*100:.1f}%)")
    print(f"Negative samples (label=0): {train_negative} ({train_negative/len(train_labels)*100:.1f}%)")
    
    # Save train split
    train_roa_file = output_dir / "train_roa_labels.txt"
    train_indices_file = output_dir / "train_shuffled_indices.txt"
    
    np.savetxt(train_roa_file, train_roa_data, fmt='%.16e %.16e %.16e %.16e %i')
    with open(train_indices_file, 'w') as f:
        for idx in train_file_indices:
            f.write(f"{idx}\n")
    
    # Save test split  
    test_roa_file = output_dir / "test_roa_labels.txt"
    test_indices_file = output_dir / "test_shuffled_indices.txt"
    
    np.savetxt(test_roa_file, test_roa_data, fmt='%.16e %.16e %.16e %.16e %i')
    with open(test_indices_file, 'w') as f:
        for idx in test_file_indices:
            f.write(f"{idx}\n")
    
    print(f"\n✅ Files saved:")
    print(f"📁 Train split:")
    print(f"   - {train_roa_file} ({len(train_roa_data)} samples)")
    print(f"   - {train_indices_file} ({len(train_file_indices)} indices)")
    print(f"📁 Test split:")
    print(f"   - {test_roa_file} ({len(test_roa_data)} samples)")
    print(f"   - {test_indices_file} ({len(test_file_indices)} indices)")
    
    # Display sample test data
    print(f"\n🔍 Sample Test Data (first 5 rows):")
    print("   x          ẋ          θ          θ̇          label")
    for i in range(min(5, len(test_roa_data))):
        row = test_roa_data[i]
        print(f"   {row[0]:8.3f}  {row[1]:8.3f}  {row[2]:8.3f}  {row[3]:8.3f}     {int(row[4])}")
    
    return {
        'total_samples': total_samples,
        'train_samples': len(train_indices),
        'test_samples': len(test_indices),
        'test_positive': test_positive,
        'test_negative': test_negative,
        'test_balance': test_positive / len(test_labels),
        'train_positive': train_positive,
        'train_negative': train_negative,
        'train_balance': train_positive / len(train_labels)
    }

if __name__ == "__main__":
    results = create_cartpole_splits()
    print(f"\n🎉 CartPole 80-20 split created successfully!")
    print(f"Test set balance: {results['test_balance']:.1%} positive samples")