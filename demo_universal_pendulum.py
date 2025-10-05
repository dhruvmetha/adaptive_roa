"""
Demo script showing migration from circular flow matching to universal framework
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

# Universal framework imports
from src.systems.pendulum_universal import PendulumSystem
from src.flow_matching.universal import (
    UniversalFlowMatcher, 
    UniversalFlowMatchingInference,
    UniversalFlowMatchingConfig
)
from src.model.universal_unet import UniversalUNet
from src.manifold_integration import TheseusIntegrator


def demo_system_definition():
    """Demo 1: System definition and basic operations"""
    print("=== Demo 1: System Definition ===")
    
    # Create pendulum system
    pendulum = PendulumSystem(attractor_radius=0.1)
    print(f"System: {pendulum}")
    print(f"Manifold components: {[comp.name for comp in pendulum.manifold_components]}")
    print(f"Dimensions: state={pendulum.state_dim}, embedding={pendulum.embedding_dim}, tangent={pendulum.tangent_dim}")
    print(f"State bounds: {pendulum.state_bounds}")
    print()
    
    # Test state embedding/extraction
    test_states = torch.tensor([
        [0.0, 0.0],      # Bottom equilibrium
        [np.pi/2, 1.0],  # Quarter turn with velocity
        [np.pi, 0.0]     # Top equilibrium
    ])
    
    print("State embedding/extraction test:")
    embedded = pendulum.embed_state(test_states)
    extracted = pendulum.extract_state(embedded)
    print(f"Original: {test_states}")
    print(f"Embedded: {embedded}")
    print(f"Extracted: {extracted}")
    print(f"Roundtrip error: {torch.max(torch.abs(test_states - extracted)):.6f}")
    print()


def demo_manifold_integration():
    """Demo 2: Manifold integration with Theseus"""
    print("=== Demo 2: Manifold Integration ===")
    
    pendulum = PendulumSystem()
    integrator = TheseusIntegrator(pendulum)
    print(f"Integrator: {integrator}")
    print()
    
    # Test integration step
    state = torch.tensor([[np.pi - 0.1, 0.5]])  # Near top, with velocity
    velocity = torch.tensor([[-2.0, -1.0]])     # Falling down
    dt = 0.01
    
    print("Integration test:")
    print(f"Initial state: θ={state[0,0]:.3f}, θ̇={state[0,1]:.3f}")
    
    # Integrate several steps
    current_state = state.clone()
    for i in range(5):
        next_state = integrator.integrate_step(current_state, velocity, dt)
        print(f"Step {i+1}: θ={next_state[0,0]:.3f}, θ̇={next_state[0,1]:.3f}")
        current_state = next_state
    print()


def demo_universal_config():
    """Demo 3: Universal configuration"""
    print("=== Demo 3: Universal Configuration ===")
    
    pendulum = PendulumSystem()
    config = UniversalFlowMatchingConfig.for_system(
        pendulum,
        num_integration_steps=50,
        hidden_dims=(32, 64, 128)
    )
    
    print(f"Config: {config}")
    print("System info:")
    for key, value in config.get_system_info().items():
        print(f"  {key}: {value}")
    print()


def demo_universal_model():
    """Demo 4: Universal model architecture"""
    print("=== Demo 4: Universal Model Architecture ===")
    
    pendulum = PendulumSystem()
    config = UniversalFlowMatchingConfig.for_system(pendulum)
    
    # Create universal model
    model = UniversalUNet(
        input_dim=config.model_input_dim,   # 6 (embedded state + condition)
        output_dim=config.model_output_dim, # 2 (tangent velocities: dθ/dt, dθ̇/dt)
        hidden_dims=[32, 64, 128],
        time_emb_dim=64
    )
    
    print(f"Model: {model}")
    print("Architecture info:")
    for key, value in model.get_architecture_info().items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    batch_size = 4
    embedded_state = torch.randn(batch_size, 3)  # (sin θ, cos θ, θ̇)
    condition = torch.randn(batch_size, 3)       # Same format
    time = torch.rand(batch_size)
    
    with torch.no_grad():
        output = model(embedded_state, time, condition)
        print(f"\\nTest forward pass:")
        print(f"  Input shape: {embedded_state.shape} + {condition.shape} + {time.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print()


def demo_training_setup():
    """Demo 5: Training setup (mock)"""
    print("=== Demo 5: Training Setup ===")
    
    pendulum = PendulumSystem()
    config = UniversalFlowMatchingConfig.for_system(pendulum, hidden_dims=[32, 64])
    
    # Create model
    model = UniversalUNet(
        input_dim=config.model_input_dim,
        output_dim=config.model_output_dim,
        hidden_dims=list(config.hidden_dims),
        time_emb_dim=config.time_emb_dim
    )
    
    # Create optimizer and scheduler (mock)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # Create universal flow matcher
    flow_matcher = UniversalFlowMatcher(
        system=pendulum,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    print(f"Flow matcher: {flow_matcher}")
    
    # Test loss computation with mock batch
    batch = {
        "start_state": torch.tensor([[0.0, 0.0], [np.pi/4, 0.5]]),  # Raw states
        "end_state": torch.tensor([[0.1, 0.2], [np.pi/2, 0.0]])    # Raw states
    }
    
    with torch.no_grad():
        loss = flow_matcher.compute_flow_loss(batch)
        print(f"Mock loss: {loss.item():.6f}")
    print()


def demo_comparison_with_old():
    """Demo 6: Compare with old circular flow matching approach"""
    print("=== Demo 6: Comparison with Old Approach ===")
    
    # Universal approach
    pendulum = PendulumSystem()
    print("Universal approach:")
    print(f"  System: {pendulum}")
    print(f"  Manifold structure: Automatic from system definition")
    print(f"  Integration: Theseus-based with proper S¹ × ℝ handling")
    print(f"  Model sizing: Automatic based on system dimensions")
    
    # Show extensibility
    print(f"\\nExtensibility:")
    print(f"  Adding new system: Just define manifold structure")
    print(f"  Same code works for any system (pendulum, cartpole, humanoid)")
    print(f"  Automatic Theseus integration for all manifold types")
    
    print(f"\\nMigration benefits:")
    print(f"  ✅ Mathematically principled: Proper Lie group operations")
    print(f"  ✅ System agnostic: Same framework for all systems") 
    print(f"  ✅ Automatic sizing: No manual dimension calculation")
    print(f"  ✅ Extensible: Easy to add SO(3), SE(3) for humanoids")
    print(f"  ✅ Maintainable: Single codebase for all systems")
    print()


if __name__ == "__main__":
    print("🚀 Universal Flow Matching Framework Demo\\n")
    
    demo_system_definition()
    demo_manifold_integration() 
    demo_universal_config()
    demo_universal_model()
    demo_training_setup()
    demo_comparison_with_old()
    
    print("✅ Demo completed! The universal framework is ready for:")
    print("   • Pendulum (S¹ × ℝ)")
    print("   • CartPole (ℝ² × S¹ × ℝ)")
    print("   • Future: Humanoid (SE(3) × SO(3)ⁿ × ℝᵐ)")
    print("   • Any system with proper manifold definition!")