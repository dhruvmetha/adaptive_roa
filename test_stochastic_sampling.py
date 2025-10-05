#!/usr/bin/env python3
"""
Quick test to verify LCFM stochastic sampling is working
"""

import torch
import numpy as np
import sys
import os
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier')

from src.flow_matching.latent_conditional.inference import LatentConditionalFlowMatchingInference

def test_stochastic_sampling():
    """Test if LCFM produces different outputs for identical inputs"""
    
    # Test parameters
    checkpoint_path = "/path/to/checkpoint.ckpt"  # You'll need to provide this
    test_start_state = torch.tensor([[0.0, 0.1]], dtype=torch.float32)  # Single start state
    num_samples = 5
    
    print("🧪 Testing LCFM stochastic sampling...")
    print(f"Start state: {test_start_state.numpy()}")
    print(f"Number of samples: {num_samples}")
    
    try:
        # Load inference model
        inferencer = LatentConditionalFlowMatchingInference(checkpoint_path)
        print("✅ Model loaded successfully")
        
        # Test Method 1: Mega-batch approach (as used in evaluation)
        print(f"\n📊 Method 1: Mega-batch approach")
        mega_batch = test_start_state.repeat(num_samples, 1)  # [5, 2]
        print(f"Mega-batch shape: {mega_batch.shape}")
        print(f"Mega-batch content:\n{mega_batch.numpy()}")
        
        endpoints_mega = inferencer.predict_endpoint(mega_batch, num_steps=50)
        print(f"Endpoints shape: {endpoints_mega.shape}")
        print(f"Endpoints:\n{endpoints_mega.cpu().numpy()}")
        
        # Check if endpoints are different
        endpoints_np = endpoints_mega.cpu().numpy()
        unique_endpoints = np.unique(endpoints_np, axis=0)
        print(f"Number of unique endpoints: {len(unique_endpoints)} / {num_samples}")
        
        # Test Method 2: Multiple separate calls
        print(f"\n📊 Method 2: Multiple separate calls")
        endpoints_separate = []
        for i in range(num_samples):
            endpoint = inferencer.predict_endpoint(test_start_state, num_steps=50)
            endpoints_separate.append(endpoint.cpu().numpy())
            print(f"Sample {i+1}: {endpoint.cpu().numpy().flatten()}")
        
        endpoints_separate = np.array(endpoints_separate).squeeze()
        unique_separate = np.unique(endpoints_separate, axis=0)
        print(f"Number of unique endpoints: {len(unique_separate)} / {num_samples}")
        
        # Compare methods
        print(f"\n🔍 Comparison:")
        print(f"Mega-batch produces {len(unique_endpoints)} unique results")
        print(f"Separate calls produce {len(unique_separate)} unique results")
        
        if len(unique_endpoints) == num_samples and len(unique_separate) == num_samples:
            print("✅ STOCHASTIC SAMPLING IS WORKING CORRECTLY!")
        elif len(unique_endpoints) == 1 or len(unique_separate) == 1:
            print("❌ STOCHASTIC SAMPLING IS NOT WORKING - ALL IDENTICAL!")
        else:
            print("⚠️ PARTIAL STOCHASTIC SAMPLING - SOME DUPLICATES")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to provide a valid checkpoint path")

if __name__ == "__main__":
    test_stochastic_sampling()