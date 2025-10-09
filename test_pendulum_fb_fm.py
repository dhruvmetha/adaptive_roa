#!/usr/bin/env python3
"""
Quick test script to verify the Facebook FM refactored Pendulum implementation

This script tests:
1. Manifold operations (logmap, expmap, projx)
2. GeodesicProbPath interpolation
3. Flow matcher initialization
4. Loss computation
5. Inference capability
"""
import torch
import sys
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier/flow_matching')

from src.utils.fb_manifolds import PendulumManifold
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from src.flow_matching.latent_conditional.flow_matcher_fb import LatentConditionalFlowMatcher
from src.model.latent_conditional_unet1d import LatentConditionalUNet1D
from src.systems.pendulum_lcfm import PendulumSystemLCFM
import math


def test_manifold():
    """Test PendulumManifold operations"""
    print("\n" + "="*80)
    print("TEST 1: PendulumManifold Operations")
    print("="*80)

    manifold = PendulumManifold()

    # Test geodesic from -π to +π (should go through boundary)
    x = torch.tensor([[-math.pi + 0.1, 0.5]])
    y = torch.tensor([[math.pi - 0.1, 0.5]])

    # Logmap should give short path
    tangent = manifold.logmap(x, y)
    print(f"✓ Logmap from near -π to near +π:")
    print(f"  Tangent: {tangent.numpy()}")
    print(f"  Expected: Small angular distance (~0.2) through boundary")

    # Expmap should wrap correctly
    halfway = manifold.expmap(x, tangent * 0.5)
    print(f"✓ Expmap halfway point:")
    print(f"  Result: {halfway.numpy()}")
    print(f"  Expected: Near ±π (boundary)")

    # Projection should wrap to [-π, π]
    out_of_range = torch.tensor([[3.5, 0.0]])
    wrapped = manifold.projx(out_of_range)
    print(f"✓ Projx wrapping:")
    print(f"  Input: {out_of_range[0, 0]:.2f}")
    print(f"  Output: {wrapped[0, 0]:.2f}")
    print(f"  Expected: In range [-π, π]")

    assert -math.pi <= wrapped[0, 0] <= math.pi, "Projection failed!"
    print("✅ Manifold operations PASSED\n")


def test_geodesic_path():
    """Test GeodesicProbPath"""
    print("="*80)
    print("TEST 2: GeodesicProbPath Interpolation")
    print("="*80)

    manifold = PendulumManifold()
    path = GeodesicProbPath(CondOTScheduler(), manifold)

    # Sample some points
    x_0 = torch.tensor([[0.0, 0.5], [math.pi/2, -0.3]])
    x_1 = torch.tensor([[math.pi, 0.8], [-math.pi/2, 0.2]])
    t = torch.tensor([0.5, 0.3])

    # Get path sample
    path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

    print(f"✓ Input shapes:")
    print(f"  x_0: {x_0.shape}")
    print(f"  x_1: {x_1.shape}")
    print(f"  t: {t.shape}")

    print(f"✓ Output shapes:")
    print(f"  x_t: {path_sample.x_t.shape}")
    print(f"  dx_t: {path_sample.dx_t.shape}")

    print(f"✓ Sample values:")
    print(f"  x_t[0]: {path_sample.x_t[0].numpy()}")
    print(f"  dx_t[0]: {path_sample.dx_t[0].numpy()}")

    # Check that x_t is on manifold (angles wrapped)
    assert torch.all(torch.abs(path_sample.x_t[:, 0]) <= math.pi), "Angles not wrapped!"

    print("✅ GeodesicProbPath PASSED\n")


def test_flow_matcher():
    """Test LatentConditionalFlowMatcher"""
    print("="*80)
    print("TEST 3: LatentConditionalFlowMatcher (Facebook FM Version)")
    print("="*80)

    # Create system
    system = PendulumSystemLCFM()

    # Create small model for testing
    model = LatentConditionalUNet1D(
        embedded_dim=3,  # (sin θ, cos θ, θ̇)
        latent_dim=2,
        condition_dim=3,
        time_emb_dim=64,
        hidden_dims=[64, 128, 64],
        output_dim=2,
        use_input_embeddings=False
    )

    print(f"✓ Model created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create flow matcher (no optimizer/scheduler needed for test)
    flow_matcher = LatentConditionalFlowMatcher(
        system=system,
        model=model,
        optimizer=None,
        scheduler=None,
        config={},
        latent_dim=2
    )

    print(f"✓ Flow matcher created successfully")
    print(f"  Manifold: {type(flow_matcher.manifold).__name__}")
    print(f"  Path: {type(flow_matcher.path).__name__}")

    # Test loss computation with dummy batch
    batch = {
        "start_state_original": torch.randn(4, 2) * 0.5,  # Small random angles
        "end_state_original": torch.randn(4, 2) * 0.5,
    }

    # Wrap angles to [-π, π]
    batch["start_state_original"][:, 0] = torch.atan2(
        torch.sin(batch["start_state_original"][:, 0]),
        torch.cos(batch["start_state_original"][:, 0])
    )
    batch["end_state_original"][:, 0] = torch.atan2(
        torch.sin(batch["end_state_original"][:, 0]),
        torch.cos(batch["end_state_original"][:, 0])
    )

    flow_matcher.eval()  # Set to eval mode
    with torch.no_grad():
        loss = flow_matcher.compute_flow_loss(batch)

    print(f"✓ Loss computation:")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss shape: {loss.shape}")

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() >= 0, "Loss is negative!"

    print("✅ Flow matcher PASSED\n")


def test_inference():
    """Test inference with RiemannianODESolver"""
    print("="*80)
    print("TEST 4: Inference with RiemannianODESolver")
    print("="*80)

    # Create system and model
    system = PendulumSystemLCFM()
    model = LatentConditionalUNet1D(
        embedded_dim=3, latent_dim=2, condition_dim=3,
        time_emb_dim=64, hidden_dims=[64, 128, 64],
        output_dim=2
    )

    flow_matcher = LatentConditionalFlowMatcher(
        system=system, model=model,
        optimizer=None, scheduler=None,
        config={}, latent_dim=2
    )

    # Test predict_endpoint
    start_states = torch.tensor([[0.5, 0.2], [-0.3, 0.1]])

    print(f"✓ Testing inference methods:")

    # Test Euler
    predictions_euler = flow_matcher.predict_endpoint(
        start_states, num_steps=50, method="euler"
    )
    print(f"  Euler method: {predictions_euler.shape}")

    # Test RK4
    predictions_rk4 = flow_matcher.predict_endpoint(
        start_states, num_steps=50, method="rk4"
    )
    print(f"  RK4 method: {predictions_rk4.shape}")

    # Test Midpoint
    predictions_midpoint = flow_matcher.predict_endpoint(
        start_states, num_steps=50, method="midpoint"
    )
    print(f"  Midpoint method: {predictions_midpoint.shape}")

    # Check outputs are on manifold
    assert torch.all(torch.abs(predictions_euler[:, 0]) <= math.pi), "Euler: angles not wrapped!"
    assert torch.all(torch.abs(predictions_rk4[:, 0]) <= math.pi), "RK4: angles not wrapped!"
    assert torch.all(torch.abs(predictions_midpoint[:, 0]) <= math.pi), "Midpoint: angles not wrapped!"

    print(f"✓ Sample predictions:")
    print(f"  Euler: {predictions_euler[0].numpy()}")
    print(f"  RK4: {predictions_rk4[0].numpy()}")
    print(f"  Midpoint: {predictions_midpoint[0].numpy()}")

    print("✅ Inference PASSED\n")


def main():
    print("\n" + "="*80)
    print("🧪 TESTING FACEBOOK FM PENDULUM IMPLEMENTATION")
    print("="*80)

    try:
        test_manifold()
        test_geodesic_path()
        test_flow_matcher()
        test_inference()

        print("="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\n🎉 The Facebook FM refactored implementation is working correctly!")
        print("\nYou can now:")
        print("  1. Train using: python src/flow_matching/latent_conditional/train.py")
        print("  2. Use any ODE solver: method='euler', 'rk4', or 'midpoint'")
        print("  3. Automatic geodesic interpolation and velocity computation!")
        print("\n")

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
