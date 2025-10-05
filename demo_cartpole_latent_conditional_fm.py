#!/usr/bin/env python3
"""
Demo script for CartPole Latent Conditional Flow Matching inference
"""
import torch
import numpy as np
from pathlib import Path

def demo_cartpole_lcfm_inference():
    """
    Demo CartPole LCFM inference capabilities
    
    Note: This demo shows the intended usage. It requires a trained model.
    """
    print("🤖 CartPole Latent Conditional Flow Matching Demo")
    print("=" * 60)
    
    # Simulate what would happen with a trained model
    print("\n📋 Demo Overview:")
    print("This demo shows how CartPole LCFM inference works:")
    print("1. Load trained model from timestamped folder")
    print("2. Sample start states for conditioning")
    print("3. Integrate on ℝ² × S¹ × ℝ manifold from noise to endpoints")
    print("4. Analyze predictions vs ground truth")
    
    print("\n⚠️  Note: This demo requires a trained CartPole LCFM model.")
    print("   Train first using: python src/flow_matching/cartpole_latent_conditional/train.py")
    
    # Show the intended usage pattern
    print("\n💻 Usage Pattern:")
    print("""
# 1. Load trained model
from src.flow_matching.cartpole_latent_conditional.inference import CartPoleLatentConditionalFlowMatchingInference

inferencer = CartPoleLatentConditionalFlowMatchingInference(
    folder_path="outputs/cartpole_latent_conditional_fm/2024-XX-XX_XX-XX-XX"
)

# 2. Define test start states [x, theta, x_dot, theta_dot] - NEW FORMAT!
start_states = torch.tensor([
    [0.0, 0.1, 0.0, 0.0],      # Slightly tilted pole (theta=0.1)
    [0.5, -0.1, 0.0, 0.0],     # Displaced cart, tilted pole
    [0.0, 0.0, 1.0, 0.0],      # Moving cart, upright pole
    [-0.3, 0.2, -0.5, -0.1],   # Complex initial state
])

# 3. Predict endpoints (stochastic due to latent variable)
predicted_endpoints = inferencer.predict_endpoint(start_states, num_samples=5)

# 4. Generate full trajectories
trajectories = inferencer.predict_trajectory(start_states, num_steps=100)

# 5. Analyze results
print(f"Start states shape: {start_states.shape}")
print(f"Predicted endpoints shape: {predicted_endpoints.shape}")  # [4*5, 4]
print(f"Trajectory length: {len(trajectories)}")  # 101 steps
    """)
    
    print("\n🔧 Key Technical Details:")
    print("1. **Manifold**: ℝ² × S¹ × ℝ (cart pos, pole angle, cart vel, angular vel)")
    print("2. **Embedding**: [x, θ, ẋ, θ̇] → [x_norm, sin θ, cos θ, ẋ_norm, θ̇_norm]")
    print("3. **Integration**: Uses TheseusIntegrator for proper manifold integration")
    print("4. **Angle Handling**: Angles wrapped to [-π,π] before embedding")
    print("5. **Velocity Prediction**: Model predicts 4D tangent space velocities")
    print("6. **Noise Sampling**: Samples from actual data bounds")
    
    print("\n🎯 Expected Behavior:")
    print("- Start states condition the flow")
    print("- Noisy initial conditions flow to predicted endpoints") 
    print("- Multiple samples per start state (stochastic latent)")
    print("- Integration respects CartPole manifold structure")
    print("- Successful states should flow to balanced configurations")
    
    return True

def analyze_manifold_integration():
    """Analyze how integration works on ℝ² × S¹ × ℝ manifold"""
    
    print("\n🧮 Manifold Integration Analysis")
    print("=" * 40)
    
    print("\n📐 Manifold Structure: ℝ² × S¹ × ℝ")
    print("- ℝ (cart_position): Standard Euler integration")  
    print("- ℝ (cart_velocity): Standard Euler integration")
    print("- S¹ (pole_angle): SO(2) integration via Theseus")
    print("- ℝ (angular_velocity): Standard Euler integration")
    
    print("\n⚙️  Integration Process:")
    print("1. Model predicts velocity in tangent space: v = [dx/dt, dẋ/dt, dθ/dt, dθ̇/dt]")
    print("2. TheseusIntegrator decomposes by manifold component:")
    print("   - Linear components (x, ẋ, θ̇): x_{t+1} = x_t + v_t * dt")
    print("   - Angular component (θ): Uses proper SO(2) exponential map")
    print("3. Proper angle wrapping maintained throughout integration")
    
    print("\n🔄 Flow Matching Process:")
    print("1. Sample noisy initial state x_0 ~ Noise(bounds)")
    print("2. Sample latent variable z ~ N(0, I)")
    print("3. Embed start state as condition c = Embed(start_state)")
    print("4. Integrate ODE: dx/dt = f(x_t, t, z, c) from t=0 to t=1")
    print("5. Final state x_1 should match target endpoint distribution")
    
    print("\n✨ Key Advantages:")
    print("- Respects CartPole geometry (no gimbal lock)")
    print("- Proper circular interpolation for pole angles")
    print("- Handles multi-turn rotations correctly")
    print("- Maintains manifold constraints during integration")

if __name__ == "__main__":
    demo_cartpole_lcfm_inference()
    analyze_manifold_integration()
    
    print(f"\n🚀 Ready to train and test CartPole LCFM!")
    print(f"   Next steps:")
    print(f"   1. Train: python src/flow_matching/cartpole_latent_conditional/train.py")
    print(f"   2. Test: Use the inference module on trained checkpoints")
    print(f"   3. Analyze: Compare predictions vs ground truth endpoints")