#!/usr/bin/env python3
"""
Demo script for latent conditional flow matching

This script demonstrates the new latent variable functionality that enables
controllable multi-modality in conditional flow matching models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.flow_matching.conditional.flow_matcher import ConditionalFlowMatcher
from src.flow_matching.conditional.inference import ConditionalFlowMatchingInference
from src.model.conditional_unet1d import ConditionalUNet1D
from src.flow_matching.base.config import FlowMatchingConfig

def create_latent_conditional_model(latent_dim=8):
    """Create a conditional flow matching model with latent support"""
    
    # Create UNet with expanded condition dimension to accommodate latent
    condition_dim = 3 + latent_dim  # start_state (3) + latent (latent_dim)
    unet = ConditionalUNet1D(
        input_dim=3,
        condition_dim=condition_dim,
        output_dim=3,
        hidden_dims=[64, 128, 256],
        time_emb_dim=128
    )
    
    # Create flow matcher with latent support
    config = FlowMatchingConfig()
    model = ConditionalFlowMatcher(
        model=unet,
        optimizer=None,  # Not needed for demo
        scheduler=None,  # Not needed for demo
        config=config,
        latent_dim=latent_dim
    )
    
    return model

def demo_latent_sampling():
    """Demonstrate latent variable sampling and controllability"""
    print("=== Latent Conditional Flow Matching Demo ===")
    
    # Create model with 8-dimensional latent space
    latent_dim = 8
    model = create_latent_conditional_model(latent_dim)
    
    print(f"Model created with latent dimension: {latent_dim}")
    print(f"Model uses latent: {model.use_latent}")
    
    # Create some dummy start states
    batch_size = 4
    start_states = torch.randn(batch_size, 3)  # Embedded states
    
    # Demo 1: Random latent sampling (default behavior)
    print("\n--- Demo 1: Random Latent Sampling ---")
    random_latent = model.sample_latent(batch_size, 'cpu')
    print(f"Random latent shape: {random_latent.shape}")
    print(f"Random latent values:\n{random_latent}")
    
    # Demo 2: Controlled latent variables
    print("\n--- Demo 2: Controlled Latent Variables ---")
    
    # Define different "modes" manually
    mode_1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # "High energy mode"
    mode_2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # "Low energy mode"
    mode_3 = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # "Oscillatory mode"
    
    print(f"Mode 1 (High energy): {mode_1}")
    print(f"Mode 2 (Low energy): {mode_2}")
    print(f"Mode 3 (Oscillatory): {mode_3}")
    
    # Demo 3: Forward pass with different latents
    print("\n--- Demo 3: Forward Pass with Different Latents ---")
    
    # Create dummy inputs
    x_t = torch.randn(1, 3)  # Current state
    t = torch.tensor([0.5])  # Time
    condition = torch.randn(1, 3)  # Start state
    
    print(f"Input state shape: {x_t.shape}")
    print(f"Time: {t}")
    print(f"Condition shape: {condition.shape}")
    
    # Test forward pass with different latents
    with torch.no_grad():
        # Mode 1 prediction
        latent_1 = mode_1.unsqueeze(0)  # Add batch dimension
        velocity_1 = model.forward(x_t, t, condition, latent=latent_1)
        
        # Mode 2 prediction  
        latent_2 = mode_2.unsqueeze(0)
        velocity_2 = model.forward(x_t, t, condition, latent=latent_2)
        
        # Random latent prediction
        latent_random = model.sample_latent(1, 'cpu')
        velocity_random = model.forward(x_t, t, condition, latent=latent_random)
        
        print(f"Velocity with Mode 1: {velocity_1}")
        print(f"Velocity with Mode 2: {velocity_2}")
        print(f"Velocity with Random: {velocity_random}")
        
        # Show they're different
        diff_1_2 = torch.norm(velocity_1 - velocity_2)
        diff_1_random = torch.norm(velocity_1 - velocity_random)
        print(f"L2 difference (Mode1 vs Mode2): {diff_1_2:.4f}")
        print(f"L2 difference (Mode1 vs Random): {diff_1_random:.4f}")

def demo_inference_api():
    """Demonstrate the inference API with latent support"""
    print("\n\n=== Inference API Demo ===")
    
    # Note: This would normally load from a checkpoint
    print("Note: This demo creates a model from scratch.")
    print("In practice, you would load from a trained checkpoint:")
    print("  inferencer = ConditionalFlowMatchingInference('checkpoint.ckpt')")
    
    # Create a dummy model for demonstration
    model = create_latent_conditional_model(latent_dim=4)
    
    # Simulate the inference API
    print("\n--- Simulated Inference Usage ---")
    
    # Example start state in original coordinates
    start_state = torch.tensor([np.pi/4, 1.5])  # (theta, theta_dot)
    print(f"Start state (θ, θ̇): {start_state}")
    
    # Example latent variables for different modes
    latent_conservative = torch.tensor([1.0, 0.0, 0.0, 0.0])
    latent_aggressive = torch.tensor([0.0, 1.0, 0.0, 0.0]) 
    latent_chaotic = torch.tensor([0.0, 0.0, 1.0, 0.0])
    
    print("\nDifferent latent modes:")
    print(f"  Conservative: {latent_conservative}")
    print(f"  Aggressive: {latent_aggressive}")
    print(f"  Chaotic: {latent_chaotic}")
    
    print("\nUsage examples:")
    print("  # Random endpoint (existing behavior)")
    print("  endpoint = inferencer.predict_endpoint(start_state)")
    print("")
    print("  # Controlled endpoints with specific latents")
    print("  conservative_end = inferencer.predict_endpoint(start_state, latent=latent_conservative)")
    print("  aggressive_end = inferencer.predict_endpoint(start_state, latent=latent_aggressive)")
    print("")
    print("  # Multiple samples with same latent (controlled diversity)")
    print("  samples = inferencer.predict_multiple_samples(start_state, num_samples=10, latent=fixed_latent)")

def demo_backward_compatibility():
    """Demonstrate that existing code still works unchanged"""
    print("\n\n=== Backward Compatibility Demo ===")
    
    # Create model WITHOUT latent support (existing behavior)
    unet = ConditionalUNet1D(
        input_dim=3,
        condition_dim=3,  # Original condition dimension
        output_dim=3
    )
    
    config = FlowMatchingConfig()
    model_no_latent = ConditionalFlowMatcher(
        model=unet,
        optimizer=None,
        scheduler=None,
        config=config,
        latent_dim=None  # No latent support
    )
    
    print(f"Model without latent support created")
    print(f"Model uses latent: {model_no_latent.use_latent}")
    print(f"Latent dim: {model_no_latent.latent_dim}")
    
    # Test that existing API still works
    x_t = torch.randn(2, 3)
    t = torch.tensor([0.3, 0.7])
    condition = torch.randn(2, 3)
    
    with torch.no_grad():
        # Old API still works
        velocity_old = model_no_latent.forward(x_t, t, condition)
        
        # New API with latent=None also works
        velocity_new = model_no_latent.forward(x_t, t, condition, latent=None)
        
        print(f"Old API output shape: {velocity_old.shape}")
        print(f"New API output shape: {velocity_new.shape}")
        print(f"Outputs identical: {torch.allclose(velocity_old, velocity_new)}")

if __name__ == "__main__":
    demo_latent_sampling()
    demo_inference_api() 
    demo_backward_compatibility()
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("Your conditional flow matching model now supports:")
    print("  ✓ Gaussian latent variable sampling")
    print("  ✓ Controllable multi-modality")
    print("  ✓ Backward compatibility")
    print("  ✓ Extended inference API")
    print("="*60)