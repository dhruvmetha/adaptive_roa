"""
Demo script for Latent Circular Flow Matching (LCFM)
Demonstrates endpoint sampling and attractor prediction
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.flow_matching.latent_circular.inference import LatentCircularInference
from src.model.mlp import MLP


def load_trained_model(checkpoint_path: str, latent_dim: int = 8) -> LatentCircularInference:
    """
    Load a trained LCFM model from checkpoint
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        latent_dim: Latent dimension used during training
        
    Returns:
        Configured inference object
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration from checkpoint
    # Note: This assumes the model configuration is saved in hyperparameters
    input_dim = 7 + latent_dim  # features(3) + time(1) + conditioning(3) + latent(L)
    output_dim = 2
    hidden_channels = [128, 256, 256, 128]  # Default from config
    
    # Initialize model architecture
    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_channels=hidden_channels)
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict']['model'])
    
    # Create inference object
    inference = LatentCircularInference(model, latent_dim=latent_dim, steps=64)
    
    return inference


def demo_endpoint_sampling():
    """Demo endpoint sampling for different initial conditions"""
    print("=== Latent Circular Flow Matching Demo ===\\n")
    
    # For demo purposes, create a dummy model (replace with actual trained model)
    latent_dim = 8
    input_dim = 7 + latent_dim
    model = MLP(input_dim=input_dim, output_dim=2, hidden_channels=[128, 256, 256, 128])
    
    # Create inference object
    inference = LatentCircularInference(model, latent_dim=latent_dim, steps=64)
    
    # Test initial conditions near separatrix (high uncertainty regions)
    test_states = torch.tensor([
        [0.6, -0.3],   # Near separatrix
        [1.5, 0.2],    # Another uncertain region
        [0.1, 0.1],    # Near stable point
        [3.0, -0.5],   # Near other stable point
    ], dtype=torch.float32)
    
    print(f"Testing {len(test_states)} initial conditions:")
    for i, x0 in enumerate(test_states):
        print(f"  State {i+1}: θ={x0[0]:.2f}, ω={x0[1]:.2f}")
    print()
    
    # Sample endpoints
    num_samples = 64
    print(f"Sampling {num_samples} endpoints per initial condition...")
    samples = inference.sample_endpoints(test_states, num_samples=num_samples)
    
    # Predict attractor distributions
    print("Computing attractor probability distributions...")
    attractor_probs = inference.predict_attractor_distribution(test_states, num_samples=num_samples)
    
    # Compute uncertainty
    uncertainties = inference.compute_uncertainty(test_states, num_samples=num_samples)
    
    # Display results
    print("\\n=== Results ===")
    attractor_names = ["Attractor 1 (0, 0)", "Attractor 2 (π, 0)"]
    
    for i in range(len(test_states)):
        print(f"\\nState {i+1}: θ={test_states[i,0]:.2f}, ω={test_states[i,1]:.2f}")
        print(f"  Uncertainty (entropy): {uncertainties[i]:.3f}")
        print("  Attractor probabilities:")
        for j, name in enumerate(attractor_names):
            print(f"    {name}: {attractor_probs[i,j]:.3f}")
        
        # Sample statistics
        theta_samples = samples[i, :, 0]
        omega_samples = samples[i, :, 1]
        print(f"  Endpoint samples:")
        print(f"    θ range: [{theta_samples.min():.2f}, {theta_samples.max():.2f}]")
        print(f"    ω range: [{omega_samples.min():.2f}, {omega_samples.max():.2f}]")
        print(f"    θ std: {theta_samples.std():.3f}, ω std: {omega_samples.std():.3f}")


def create_visualization_plots():
    """Create visualization plots for LCFM results"""
    print("\\n=== Creating Visualization Plots ===")
    
    # This would typically use real trained model results
    # For demo, create synthetic data
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Sample endpoints in phase space
    ax = axes[0, 0]
    theta = np.linspace(-np.pi, np.pi, 100)
    omega = np.linspace(-2, 2, 100)
    
    # Simulate some sample endpoints
    np.random.seed(42)
    for i, (th0, om0) in enumerate([(0.6, -0.3), (-1.2, 0.4)]):
        samples_th = np.random.normal(th0, 0.5, 50)
        samples_om = np.random.normal(om0, 0.3, 50)
        ax.scatter(samples_th, samples_om, alpha=0.6, s=20, label=f'x₀=({th0},{om0})')
    
    ax.set_xlabel('θ (radians)')
    ax.set_ylabel('ω (rad/s)')
    ax.set_title('Endpoint Samples from LCFM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Attractor probabilities
    ax = axes[0, 1]
    states = ['State 1', 'State 2', 'State 3', 'State 4']
    prob_att1 = [0.8, 0.3, 0.9, 0.1]
    prob_att2 = [0.2, 0.7, 0.1, 0.9]
    
    x = np.arange(len(states))
    width = 0.35
    ax.bar(x - width/2, prob_att1, width, label='Attractor 1', alpha=0.7)
    ax.bar(x + width/2, prob_att2, width, label='Attractor 2', alpha=0.7)
    
    ax.set_ylabel('Probability')
    ax.set_title('Attractor Probability Distributions')
    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty analysis
    ax = axes[1, 0]
    uncertainties = [0.2, 0.8, 0.1, 0.3]
    bars = ax.bar(states, uncertainties, color='orange', alpha=0.7)
    ax.set_ylabel('Entropy (Uncertainty)')
    ax.set_title('Prediction Uncertainty by State')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, uncertainties):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    
    # Plot 4: Loss curves (synthetic)
    ax = axes[1, 1]
    epochs = np.arange(1, 101)
    fm_loss = 2.0 * np.exp(-epochs/20) + 0.1
    kl_loss = 0.5 * np.exp(-epochs/30) + 0.02
    total_loss = fm_loss + kl_loss
    
    ax.plot(epochs, total_loss, 'b-', label='Total Loss', linewidth=2)
    ax.plot(epochs, fm_loss, 'r--', label='FM Loss', alpha=0.7)
    ax.plot(epochs, kl_loss, 'g--', label='KL Loss', alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("lcfm_demo_results.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {output_path}")


def main():
    """Main demo function"""
    print("Latent Circular Flow Matching (LCFM) Demo")
    print("=" * 50)
    
    # Run endpoint sampling demo
    demo_endpoint_sampling()
    
    # Create visualization plots
    create_visualization_plots()
    
    print("\\n=== Usage Instructions ===")
    print("To use LCFM with a trained model:")
    print("1. Train model: python src/flow_matching/latent_circular/train.py")
    print("2. Load checkpoint: inference = load_trained_model('checkpoint.ckpt')")
    print("3. Sample endpoints: samples = inference.sample_endpoints(x0, num_samples=64)")
    print("4. Get probabilities: probs = inference.predict_attractor_distribution(x0)")
    print("5. Measure uncertainty: entropy = inference.compute_uncertainty(x0)")
    
    print("\\n=== Key Features ===")
    print("• Produces *distributions* of endpoints, not single predictions")
    print("• Handles circular topology properly with S¹×R bridges")
    print("• VAE-style latent variables for uncertainty quantification")
    print("• Automatic attractor classification and probability estimation")
    print("• Entropy-based uncertainty measurement")
    print("• Efficient batch processing for multiple initial conditions")


if __name__ == "__main__":
    main()