#!/usr/bin/env python3
"""
Compare the legacy vs unified inference systems to identify differences
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np

# Import both inference systems
from src.inference_circular_flow_matching import CircularFlowMatchingInference as LegacyInference
from src.flow_matching.circular import CircularFlowMatchingInference as UnifiedInference

def compare_inference_systems():
    """Compare the two inference systems on the same inputs"""
    
    print("COMPARING LEGACY vs UNIFIED INFERENCE SYSTEMS")
    print("=" * 60)
    
    # Load model with both systems
    checkpoint_path = "outputs/circular_flow_matching/checkpoints/epoch=189-step=6650-val_loss=0.000308.ckpt"
    
    print("Loading models...")
    legacy_inferencer = LegacyInference(checkpoint_path)
    unified_inferencer = UnifiedInference(checkpoint_path)
    print("✓ Both models loaded")
    
    # Test with a few sample states
    test_states = torch.tensor([
        [0.0, 0.0],      # Origin
        [1.0, 1.0],      # Quadrant 1
        [-1.0, -1.0],    # Quadrant 3
        [2.0, 0.0],      # Near right attractor
        [-2.0, 0.0],     # Near left attractor
    ], dtype=torch.float32)
    
    print(f"\nTest states:")
    for i, state in enumerate(test_states):
        print(f"  {i}: θ={state[0]:.3f}, θ̇={state[1]:.3f}")
    
    print(f"\nPredicting endpoints...")
    
    # Get predictions from both systems
    with torch.no_grad():
        legacy_predictions = legacy_inferencer.predict_endpoint(test_states)
        unified_predictions = unified_inferencer.predict_endpoint(test_states)
    
    print(f"\nCOMPARISON RESULTS:")
    print("=" * 60)
    print(f"{'State':<10} {'Legacy θ':<10} {'Legacy θ̇':<10} {'Unified θ':<10} {'Unified θ̇':<10} {'Diff θ':<10} {'Diff θ̇':<10}")
    print("-" * 60)
    
    for i in range(len(test_states)):
        legacy_pred = legacy_predictions[i]
        unified_pred = unified_predictions[i]
        
        diff_theta = abs(legacy_pred[0] - unified_pred[0])
        diff_theta_dot = abs(legacy_pred[1] - unified_pred[1])
        
        print(f"{i:<10} {legacy_pred[0]:<10.4f} {legacy_pred[1]:<10.4f} {unified_pred[0]:<10.4f} {unified_pred[1]:<10.4f} {diff_theta:<10.4f} {diff_theta_dot:<10.4f}")
    
    # Check if predictions are similar
    legacy_np = legacy_predictions.cpu().numpy()
    unified_np = unified_predictions.cpu().numpy()
    
    max_diff = np.max(np.abs(legacy_np - unified_np))
    mean_diff = np.mean(np.abs(legacy_np - unified_np))
    
    print(f"\nDifference Statistics:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print("✓ Systems produce very similar results")
        return True
    else:
        print("✗ Systems produce significantly different results")
        return False

def test_with_ground_truth_samples():
    """Test with actual samples from ground truth data"""
    
    print(f"\n" + "=" * 60)
    print("TESTING WITH GROUND TRUTH SAMPLES")
    print("=" * 60)
    
    # Load a few samples from ground truth
    ground_truth_data = []
    with open('/common/users/dm1487/arcmg_datasets/pendulum_lqr/pendulum_roa.txt', 'r') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0 and len(ground_truth_data) < 5:  # Get 5 samples
                parts = line.strip().split()
                if len(parts) >= 4:
                    serial, q, q_dot, label = int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
                    ground_truth_data.append([q, q_dot, label])
    
    print("Ground truth samples:")
    for i, (q, q_dot, label) in enumerate(ground_truth_data):
        print(f"  {i}: θ={q:.3f}, θ̇={q_dot:.3f}, label={label}")
    
    # Convert to tensor
    test_states = torch.tensor([[q, q_dot] for q, q_dot, _ in ground_truth_data], dtype=torch.float32)
    
    # Load models
    checkpoint_path = "outputs/circular_flow_matching/checkpoints/epoch=189-step=6650-val_loss=0.000308.ckpt"
    legacy_inferencer = LegacyInference(checkpoint_path)
    unified_inferencer = UnifiedInference(checkpoint_path)
    
    # Get predictions
    with torch.no_grad():
        legacy_predictions = legacy_inferencer.predict_endpoint(test_states)
        unified_predictions = unified_inferencer.predict_endpoint(test_states)
    
    print(f"\nPredictions comparison:")
    print("Sample | GT Label | Legacy Final           | Unified Final          | Legacy Dist | Unified Dist")
    print("-" * 100)
    
    from src.systems.pendulum_config import PendulumConfig
    config = PendulumConfig()
    
    for i in range(len(ground_truth_data)):
        _, _, gt_label = ground_truth_data[i]
        
        legacy_final = legacy_predictions[i].cpu().numpy()
        unified_final = unified_predictions[i].cpu().numpy()
        
        # Calculate distance to center attractor
        legacy_dist = np.linalg.norm(legacy_final - config.ATTRACTORS[0])
        unified_dist = np.linalg.norm(unified_final - config.ATTRACTORS[0])
        
        print(f"{i:6d} | {gt_label:8d} | ({legacy_final[0]:6.3f}, {legacy_final[1]:6.3f}) | ({unified_final[0]:6.3f}, {unified_final[1]:6.3f}) | {legacy_dist:11.4f} | {unified_dist:12.4f}")

def main():
    # Compare basic functionality
    systems_similar = compare_inference_systems()
    
    # Test with ground truth samples
    test_with_ground_truth_samples()
    
    print(f"\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    if systems_similar:
        print("The inference systems produce similar results.")
        print("The issue may be in:")
        print("  - How I'm processing the outputs")
        print("  - How I'm mapping predictions to attractors")
        print("  - How I'm handling ground truth labels")
    else:
        print("The inference systems produce different results!")
        print("Key differences to investigate:")
        print("  - Normalization/denormalization")
        print("  - Coordinate system handling")
        print("  - State space bounds")
        print("  - Model loading/architecture")

if __name__ == "__main__":
    main()