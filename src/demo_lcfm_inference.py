"""
Demo script showing how to run inference with trained LCFM model
"""
import torch
import functools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.flow_matching.latent_circular.inference import LatentCircularInference
from src.model.mlp import MLP


def load_trained_lcfm_model(checkpoint_path: str, latent_dim: int = 8):
    """
    Load trained LCFM model from checkpoint
    
    Args:
        checkpoint_path: Path to Lightning checkpoint (.ckpt file)
        latent_dim: Latent dimension used during training
        
    Returns:
        LatentCircularInference object ready for use
    """
    # Model architecture (must match training config)
    input_dim = 7 + latent_dim  # features(3) + time(1) + conditioning(3) + latent(L)
    model = MLP(
        input_dim=input_dim, 
        output_dim=2, 
        hidden_channels=[128, 256, 256, 128]
    )
    
    # Load checkpoint (PyTorch 2.6+: allowlist safe globals or disable weights_only)
    try:
        from torch.serialization import add_safe_globals  # type: ignore
        add_safe_globals([functools.partial])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception:
        # Fallback to explicit unsafe (but trusted) load
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model weights (Lightning adds "model." prefix)
    model_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            # Remove "model." prefix
            new_key = key[6:]  # Remove "model."
            model_state_dict[new_key] = value
    
    # Load weights into model
    model.load_state_dict(model_state_dict)
    
    # Move model to device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create inference object
    inference = LatentCircularInference(
        model=model,
        latent_dim=latent_dim,
        steps=64  # Integration steps
    )
    
    print(f"✅ Loaded LCFM model from {checkpoint_path}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Latent dimension: {latent_dim}")
    
    return inference


def demo_inference_examples():
    """Demo inference with example initial conditions"""
    print("\\n🎯 INFERENCE EXAMPLES")
    print("=" * 30)
    
    # For demo, create a dummy model (replace with actual checkpoint loading)
    print("⚠️  Using dummy model for demo (replace with trained model)")
    latent_dim = 8
    input_dim = 7 + latent_dim
    model = MLP(input_dim=input_dim, output_dim=2, hidden_channels=[128, 256, 128])
    inference = LatentCircularInference(model, latent_dim=latent_dim, steps=32)
    
    # Test initial conditions
    test_states = torch.tensor([
        [0.6, -0.3],   # Near separatrix
        [1.5, 0.2],    # Another region  
        [0.1, 0.1],    # Near attractor
        [-1.2, -0.8],  # Different quadrant
    ], dtype=torch.float32)
    
    print(f"\\n📍 Testing {len(test_states)} initial conditions:")
    for i, x0 in enumerate(test_states):
        print(f"   State {i+1}: θ={x0[0]:.2f}, ω={x0[1]:.2f}")
    
    # Sample endpoints
    num_samples = 64
    print(f"\\n🎲 Sampling {num_samples} endpoints per state...")
    samples = inference.sample_endpoints(test_states, num_samples=num_samples)
    
    # Predict attractor distributions
    print("🎯 Computing attractor probabilities...")
    probs = inference.predict_attractor_distribution(test_states, num_samples=num_samples)
    
    # Compute uncertainties
    print("📊 Computing uncertainties...")
    uncertainties = inference.compute_uncertainty(test_states, num_samples=num_samples)
    
    # Display results
    print("\\n📋 RESULTS:")
    print("-" * 50)
    attractor_names = ["Attractor 1 (0, 0)", "Attractor 2 (π, 0)"]
    
    for i in range(len(test_states)):
        print(f"\\n🔍 State {i+1}: θ={test_states[i,0]:.2f}, ω={test_states[i,1]:.2f}")
        print(f"   Uncertainty (entropy): {uncertainties[i]:.3f}")
        print(f"   Attractor probabilities:")
        for j, name in enumerate(attractor_names):
            print(f"     • {name}: {probs[i,j]:.3f}")
        
        # Sample statistics
        theta_samples = samples[i, :, 0]
        omega_samples = samples[i, :, 1]
        print(f"   Endpoint sample statistics:")
        print(f"     • θ ∈ [{theta_samples.min():.2f}, {theta_samples.max():.2f}], std={theta_samples.std():.3f}")
        print(f"     • ω ∈ [{omega_samples.min():.2f}, {omega_samples.max():.2f}], std={omega_samples.std():.3f}")
    
    return samples, probs, uncertainties


def visualize_results(test_states, samples, probs, uncertainties):
    """Create visualization of inference results"""
    print("\\n📈 Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Endpoint samples for each initial condition
    ax = axes[0, 0]
    colors = ['blue', 'red', 'green', 'orange']
    
    for i in range(len(test_states)):
        theta_samples = samples[i, :, 0]
        omega_samples = samples[i, :, 1]
        
        # Initial state
        ax.scatter(test_states[i, 0], test_states[i, 1], 
                  c=colors[i], s=100, marker='x', 
                  label=f'Start {i+1}')
        
        # Endpoint samples
        ax.scatter(theta_samples, omega_samples, 
                  c=colors[i], alpha=0.4, s=15)
    
    ax.set_xlabel('θ (radians)')
    ax.set_ylabel('ω (rad/s)')
    ax.set_title('LCFM Endpoint Samples')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Attractor probabilities
    ax = axes[0, 1]
    state_labels = [f'State {i+1}' for i in range(len(test_states))]
    x = np.arange(len(state_labels))
    width = 0.35
    
    ax.bar(x - width/2, probs[:, 0], width, label='Attractor 1', alpha=0.7)
    ax.bar(x + width/2, probs[:, 1], width, label='Attractor 2', alpha=0.7)
    
    ax.set_ylabel('Probability')
    ax.set_title('Attractor Probability Distributions')
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty analysis
    ax = axes[1, 0]
    bars = ax.bar(state_labels, uncertainties, color='purple', alpha=0.7)
    ax.set_ylabel('Entropy (Uncertainty)')
    ax.set_title('Prediction Uncertainty')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, uncertainties):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Plot 4: Sample diversity
    ax = axes[1, 1]
    theta_stds = [samples[i, :, 0].std().item() for i in range(len(test_states))]
    omega_stds = [samples[i, :, 1].std().item() for i in range(len(test_states))]
    
    ax.bar(x - width/2, theta_stds, width, label='θ std', alpha=0.7)
    ax.bar(x + width/2, omega_stds, width, label='ω std', alpha=0.7)
    
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Endpoint Sample Diversity')
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('LCFM Inference Results', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    output_path = Path("lcfm_inference_results.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Results visualization saved to: {output_path}")


def main():
    """Main demo function"""
    print("🚀 LCFM INFERENCE DEMO")
    print("=" * 25)
    print()
    print("This script demonstrates how to:")
    print("1. Load a trained LCFM model from checkpoint")
    print("2. Run inference on initial conditions")
    print("3. Analyze endpoint distributions")
    print("4. Quantify prediction uncertainty")
    
    # Run inference examples
    samples, probs, uncertainties = demo_inference_examples()
    
    # Create visualization
    test_states = torch.tensor([
        [0.6, -0.3], [1.5, 0.2], [0.1, 0.1], [-1.2, -0.8]
    ])
    visualize_results(test_states, samples, probs, uncertainties)
    
    print("\\n📝 REAL USAGE:")
    print("-" * 30)
    print("# Replace dummy model with actual trained model:")
    print("inference = load_trained_lcfm_model('outputs/latent_circular_fm/checkpoints/best.ckpt')")
    print("x0 = torch.tensor([[0.6, -0.3]])  # Your initial state")
    print("samples = inference.sample_endpoints(x0, num_samples=100)")
    print("probs = inference.predict_attractor_distribution(x0)")
    print("uncertainty = inference.compute_uncertainty(x0)")
    
    print("\\n🎯 Key capabilities:")
    print("• Endpoint distribution sampling (uncertainty quantification)")
    print("• Attractor basin probability estimation")
    print("• Entropy-based uncertainty measurement")
    print("• Batch processing for multiple initial conditions")


if __name__ == "__main__":
    main()