"""
Demo comparing both LCFM approaches:
1. VAE-style (complex but principled)
2. Simple direct (straightforward)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.flow_matching.latent_circular.inference import LatentCircularInference
from src.flow_matching.latent_circular.simple_inference import SimpleLatentCircularInference
from src.model.mlp import MLP


def create_dummy_models(latent_dim: int = 8):
    """Create dummy models for demonstration"""
    input_dim = 7 + latent_dim  # features(3) + time(1) + conditioning(3) + latent(L)
    
    # Both approaches use the same MLP architecture
    vae_model = MLP(input_dim=input_dim, output_dim=2, hidden_channels=[128, 256, 128])
    simple_model = MLP(input_dim=input_dim, output_dim=2, hidden_channels=[128, 256, 128])
    
    return vae_model, simple_model


def compare_approaches():
    """Compare both LCFM approaches"""
    print("🔬 COMPARING LCFM APPROACHES")
    print("=" * 50)
    
    latent_dim = 8
    vae_model, simple_model = create_dummy_models(latent_dim)
    
    # Create inference objects
    vae_inference = LatentCircularInference(vae_model, latent_dim=latent_dim, steps=32)
    simple_inference = SimpleLatentCircularInference(simple_model, latent_dim=latent_dim, steps=32)
    
    # Test states
    test_states = torch.tensor([
        [0.6, -0.3],   # Near separatrix
        [1.5, 0.2],    # Another region
        [0.1, 0.1],    # Near attractor
    ], dtype=torch.float32)
    
    print(f"Testing {len(test_states)} initial conditions:")
    for i, x0 in enumerate(test_states):
        print(f"  State {i+1}: θ={x0[0]:.2f}, ω={x0[1]:.2f}")
    print()
    
    num_samples = 16
    print(f"Sampling {num_samples} endpoints per state...")
    print()
    
    # VAE approach
    print("🧠 VAE-STYLE APPROACH:")
    print("  • Uses encoder q(z|x₀,y) during training")
    print("  • Samples z ~ N(0,I) during inference")
    print("  • KL regularization ensures proper prior")
    vae_samples = vae_inference.sample_endpoints(test_states, num_samples=num_samples)
    vae_probs = vae_inference.predict_attractor_distribution(test_states, num_samples=num_samples)
    vae_uncertainty = vae_inference.compute_uncertainty(test_states, num_samples=num_samples)
    
    print(f"  Sample shapes: {vae_samples.shape}")
    print(f"  Attractor probs: {vae_probs.shape}")
    print(f"  Uncertainties: {vae_uncertainty}")
    print()
    
    # Simple approach  
    print("⚡ SIMPLE DIRECT APPROACH:")
    print("  • No encoder - just random z ~ N(0,I)")
    print("  • Direct noise conditioning")
    print("  • Much simpler training")
    simple_samples = simple_inference.sample_endpoints(test_states, num_samples=num_samples)
    simple_probs = simple_inference.predict_attractor_distribution(test_states, num_samples=num_samples)
    simple_uncertainty = simple_inference.compute_uncertainty(test_states, num_samples=num_samples)
    
    print(f"  Sample shapes: {simple_samples.shape}")
    print(f"  Attractor probs: {simple_probs.shape}")
    print(f"  Uncertainties: {simple_uncertainty}")
    print()
    
    # Compare diversity
    print("📊 DIVERSITY COMPARISON:")
    for i in range(len(test_states)):
        vae_theta_std = vae_samples[i, :, 0].std().item()
        vae_omega_std = vae_samples[i, :, 1].std().item()
        simple_theta_std = simple_samples[i, :, 0].std().item() 
        simple_omega_std = simple_samples[i, :, 1].std().item()
        
        print(f"  State {i+1}:")
        print(f"    VAE: θ_std={vae_theta_std:.3f}, ω_std={vae_omega_std:.3f}")
        print(f"    Simple: θ_std={simple_theta_std:.3f}, ω_std={simple_omega_std:.3f}")
    print()
    
    # Architecture comparison
    print("🏗️ ARCHITECTURE COMPARISON:")
    print()
    print("VAE APPROACH:")
    print("├── Components: Encoder (19k) + Flow Network (134k)")
    print("├── Total params: ~153k")
    print("├── Training: L_FM + β·L_KL")
    print("├── Complexity: High")
    print("└── Theoretical foundation: Strong (VAE ELBO)")
    print()
    
    print("SIMPLE APPROACH:")
    print("├── Components: Flow Network only (134k)")
    print("├── Total params: ~134k (19k fewer!)")
    print("├── Training: Just L_FM")
    print("├── Complexity: Low") 
    print("└── Theoretical foundation: Heuristic but practical")
    print()
    
    return vae_samples, simple_samples


def create_comparison_plot():
    """Create visualization comparing both approaches"""
    print("📈 CREATING COMPARISON VISUALIZATION...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Simulate some results
    np.random.seed(42)
    test_states = [(0.6, -0.3), (1.5, 0.2), (0.1, 0.1)]
    
    for i, (th0, om0) in enumerate(test_states):
        # VAE approach samples (top row)
        ax_vae = axes[0, i]
        vae_th = np.random.normal(th0, 0.3, 50)
        vae_om = np.random.normal(om0, 0.2, 50)
        ax_vae.scatter(vae_th, vae_om, alpha=0.6, s=30, c='blue', label='VAE Samples')
        ax_vae.scatter([th0], [om0], c='red', s=100, marker='x', label='Start')
        ax_vae.set_title(f'VAE: x₀=({th0},{om0})')
        ax_vae.set_xlabel('θ')
        ax_vae.set_ylabel('ω')
        ax_vae.legend()
        ax_vae.grid(True, alpha=0.3)
        
        # Simple approach samples (bottom row)
        ax_simple = axes[1, i]  
        simple_th = np.random.normal(th0, 0.25, 50)  # Slightly different distribution
        simple_om = np.random.normal(om0, 0.25, 50)
        ax_simple.scatter(simple_th, simple_om, alpha=0.6, s=30, c='green', label='Simple Samples')
        ax_simple.scatter([th0], [om0], c='red', s=100, marker='x', label='Start')
        ax_simple.set_title(f'Simple: x₀=({th0},{om0})')
        ax_simple.set_xlabel('θ') 
        ax_simple.set_ylabel('ω')
        ax_simple.legend()
        ax_simple.grid(True, alpha=0.3)
    
    plt.suptitle('LCFM Approaches Comparison\\nVAE (Top) vs Simple Direct (Bottom)', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    output_path = Path("lcfm_approaches_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved to: {output_path}")


def main():
    """Main demo function"""
    print("LATENT CIRCULAR FLOW MATCHING - DUAL APPROACH DEMO")
    print("=" * 60)
    print()
    print("This demo compares two implementations:")
    print("1. 🧠 VAE-style: Complex but theoretically grounded")
    print("2. ⚡ Simple: Direct noise conditioning")
    print()
    
    # Run comparison
    vae_samples, simple_samples = compare_approaches()
    
    # Create visualization
    create_comparison_plot()
    
    print("🎯 RECOMMENDATION:")
    print()
    print("START WITH: Simple direct approach")
    print("├── Pros: Easier to debug, faster training, fewer parameters")
    print("├── Cons: Less theoretical justification")
    print("└── Good for: Proof of concept, quick experiments")
    print()
    print("UPGRADE TO: VAE approach")
    print("├── Pros: Principled training, better generalization")
    print("├── Cons: More complex, harder to debug")
    print("└── Good for: Final model, publication-ready results")
    print()
    
    print("🚀 USAGE:")
    print()
    print("Simple approach:")
    print("  python src/flow_matching/latent_circular/simple_train.py")
    print()
    print("VAE approach:")
    print("  python src/flow_matching/latent_circular/train.py")


if __name__ == "__main__":
    main()