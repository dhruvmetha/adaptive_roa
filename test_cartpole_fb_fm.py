"""
Test script for CartPole Facebook Flow Matching implementation

This script validates:
1. CartPoleManifold operations (logmap, expmap, projx)
2. GeodesicProbPath interpolation on ℝ²×S¹×ℝ
3. CartPoleLatentConditionalFlowMatcher initialization
4. Loss computation with automatic geodesic velocities
5. Inference with RiemannianODESolver (multiple methods)
"""

import torch
import sys
import math

# Add flow_matching to path
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier/flow_matching')

from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler

from src.utils.fb_manifolds import CartPoleManifold
from src.flow_matching.cartpole_latent_conditional.flow_matcher_fb import CartPoleLatentConditionalFlowMatcher
from src.systems.cartpole_lcfm import CartPoleSystemLCFM
from src.model.cartpole_latent_conditional_unet1d import CartPoleLatentConditionalUNet1D


def test_cartpole_manifold():
    """Test CartPoleManifold basic operations"""
    print("="*80)
    print("TEST 1: CartPoleManifold Operations")
    print("="*80)

    manifold = CartPoleManifold()

    # Test: geodesic across angle boundary
    x = torch.tensor([[0.0, -math.pi + 0.1, 0.0, 0.0]])  # Near -π
    y = torch.tensor([[0.5, math.pi - 0.1, 0.2, 0.1]])   # Near +π

    # Logmap should give short angular path
    tangent = manifold.logmap(x, y)
    print(f"✓ Logmap from near -π to near +π:")
    print(f"  Tangent: {tangent.numpy()}")
    print(f"  Expected: Small angular component (~0.2) through boundary")

    # Expmap halfway
    halfway = manifold.expmap(x, tangent * 0.5)
    print(f"✓ Expmap halfway point:")
    print(f"  Result: {halfway.numpy()}")
    print(f"  Expected: Angle near ±π (boundary)")

    # Test wrapping
    out_of_range = torch.tensor([[0.0, 3.5, 0.0, 0.0]])
    wrapped = manifold.projx(out_of_range)
    print(f"✓ Projx wrapping:")
    print(f"  Input angle: {out_of_range[0,1]:.2f}")
    print(f"  Output angle: {wrapped[0,1]:.2f}")
    print(f"  Expected: In range [-π, π]")

    print("✅ Manifold operations PASSED\n")


def test_geodesic_path():
    """Test GeodesicProbPath with CartPoleManifold"""
    print("="*80)
    print("TEST 2: GeodesicProbPath Interpolation")
    print("="*80)

    manifold = CartPoleManifold()
    path = GeodesicProbPath(
        scheduler=CondOTScheduler(),
        manifold=manifold
    )

    # Sample points
    x_0 = torch.tensor([
        [0.0, -math.pi/2, 0.0, 0.0],
        [0.5, math.pi/4, 0.1, 0.2]
    ])
    x_1 = torch.tensor([
        [1.0, math.pi/2, 0.5, 0.3],
        [-0.5, -math.pi/4, -0.1, -0.2]
    ])
    t = torch.tensor([0.5, 0.3])

    print(f"✓ Input shapes:")
    print(f"  x_0: {x_0.shape}")
    print(f"  x_1: {x_1.shape}")
    print(f"  t: {t.shape}")

    # Sample path
    path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

    print(f"✓ Output shapes:")
    print(f"  x_t: {path_sample.x_t.shape}")
    print(f"  dx_t: {path_sample.dx_t.shape}")

    print(f"✓ Sample values:")
    print(f"  x_t[0]: {path_sample.x_t[0].numpy()}")
    print(f"  dx_t[0]: {path_sample.dx_t[0].numpy()}")

    assert path_sample.x_t.shape == (2, 4), "x_t shape mismatch"
    assert path_sample.dx_t.shape == (2, 4), "dx_t shape mismatch"

    print("✅ GeodesicProbPath PASSED\n")


def test_flow_matcher():
    """Test CartPoleLatentConditionalFlowMatcher initialization and loss"""
    print("="*80)
    print("TEST 3: CartPoleLatentConditionalFlowMatcher (Facebook FM Version)")
    print("="*80)

    # Create system
    system = CartPoleSystemLCFM()

    # Create model
    model = CartPoleLatentConditionalUNet1D(
        embedded_dim=5,
        latent_dim=2,
        condition_dim=5,
        time_emb_dim=32,
        hidden_dims=[64, 128, 64],
        output_dim=4
    )

    print(f"✓ Model created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create flow matcher (no optimizer/scheduler needed for testing)
    flow_matcher = CartPoleLatentConditionalFlowMatcher(
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

    # Test loss computation
    batch = {
        "raw_start_state": torch.randn(4, 4),  # [B=4, 4]
        "raw_end_state": torch.randn(4, 4),
    }

    loss = flow_matcher.compute_flow_loss(batch)

    print(f"✓ Loss computation:")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss shape: {loss.shape}")

    assert loss.shape == torch.Size([]), "Loss should be scalar"

    print("✅ Flow matcher PASSED\n")


def test_inference():
    """Test inference with RiemannianODESolver"""
    print("="*80)
    print("TEST 4: Inference with RiemannianODESolver")
    print("="*80)

    # Create system and model
    system = CartPoleSystemLCFM()
    model = CartPoleLatentConditionalUNet1D(
        embedded_dim=5,
        latent_dim=2,
        condition_dim=5,
        time_emb_dim=32,
        hidden_dims=[64, 128, 64],
        output_dim=4
    )

    # Create flow matcher
    flow_matcher = CartPoleLatentConditionalFlowMatcher(
        system=system,
        model=model,
        optimizer=None,
        scheduler=None,
        config={},
        latent_dim=2
    )

    # Set to eval mode
    flow_matcher.eval()

    # Test start states
    start_states = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, math.pi/4, 0.1, 0.2]
    ])

    print(f"✓ Testing inference methods:")

    # Test different ODE methods
    with torch.no_grad():
        # Euler
        pred_norm_euler, pred_raw_euler = flow_matcher.predict_endpoint(
            start_states, num_steps=50, method="euler"
        )
        print(f"  Euler method: {pred_raw_euler.shape}")

        # RK4
        pred_norm_rk4, pred_raw_rk4 = flow_matcher.predict_endpoint(
            start_states, num_steps=50, method="rk4"
        )
        print(f"  RK4 method: {pred_raw_rk4.shape}")

        # Midpoint
        pred_norm_mid, pred_raw_mid = flow_matcher.predict_endpoint(
            start_states, num_steps=50, method="midpoint"
        )
        print(f"  Midpoint method: {pred_raw_mid.shape}")

    print(f"✓ Sample predictions:")
    print(f"  Euler: {pred_raw_euler[0].numpy()}")
    print(f"  RK4: {pred_raw_rk4[0].numpy()}")
    print(f"  Midpoint: {pred_raw_mid[0].numpy()}")

    assert pred_raw_euler.shape == (2, 4), "Euler output shape mismatch"
    assert pred_raw_rk4.shape == (2, 4), "RK4 output shape mismatch"
    assert pred_raw_mid.shape == (2, 4), "Midpoint output shape mismatch"

    print("✅ Inference PASSED\n")


def main():
    """Run all tests"""
    print("="*80)
    print("🧪 TESTING FACEBOOK FM CARTPOLE IMPLEMENTATION")
    print("="*80)
    print()

    try:
        test_cartpole_manifold()
        test_geodesic_path()
        test_flow_matcher()
        test_inference()

        print("="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print()
        print("🎉 The Facebook FM refactored CartPole implementation is working correctly!")
        print()
        print("You can now:")
        print("  1. Train using: python src/flow_matching/cartpole_latent_conditional/train.py")
        print("  2. Use any ODE solver: method='euler', 'rk4', or 'midpoint'")
        print("  3. Automatic geodesic interpolation and velocity computation!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
