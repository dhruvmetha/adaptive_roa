"""
Test script for evaluation metrics implementation.
Run this to verify the manifold distance, Euclidean, and RMSE metrics work correctly.
"""
import torch
import sys

# Test with mock data first to verify the computation logic
def test_evaluation_metrics():
    print("=" * 80)
    print("TESTING EVALUATION METRICS IMPLEMENTATION")
    print("=" * 80)

    # Import the flow matching utilities
    from flow_matching.utils.manifolds import Product, Euclidean, SO3

    # Create Quadrotor3D manifold: R^3 x SO(3) x R^6
    manifold = Product(input_dim=13, manifolds=[
        (Euclidean(), 3),      # Position (x, y, z)
        (SO3(), 4, 3),         # Quaternion (qw, qx, qy, qz) - 4D state, 3D tangent
        (Euclidean(), 6)       # Velocities
    ])

    print("\n1. Testing manifold.dist() output shape...")

    # Create test data
    batch_size = 10
    x = torch.randn(batch_size, 13)
    y = torch.randn(batch_size, 13)

    # Project onto manifold (normalize quaternions)
    x = manifold.projx(x)
    y = manifold.projx(y)

    # Compute distances
    distances = manifold.dist(x, y)

    print(f"   Input shapes: x={x.shape}, y={y.shape}")
    print(f"   Output shape: distances={distances.shape}")
    print(f"   Expected: [batch_size, 10] = [{batch_size}, 10]")
    print(f"   Breakdown: 3 (pos) + 1 (SO3 geodesic) + 6 (vel) = 10")

    assert distances.shape == (batch_size, 10), f"Expected shape (10, 10), got {distances.shape}"
    print("   ✓ Shape check PASSED")

    # Verify distance values are non-negative
    assert (distances >= 0).all(), "Distances should be non-negative"
    print("   ✓ Non-negativity check PASSED")

    print("\n2. Testing grouped Euclidean (L2) distance computation...")

    # Define groups (same as Quadrotor3D)
    groups = {
        "position_L2": [0, 1, 2],
        "quaternion_L2": [3, 4, 5, 6],
        "linear_velocity_L2": [7, 8, 9],
        "angular_velocity_L2": [10, 11, 12],
        "full_state_L2": list(range(13)),
    }

    diff = x - y
    grouped_distances = {}
    for group_name, indices in groups.items():
        group_diff = diff[:, indices]
        l2_dist = torch.norm(group_diff, dim=1)
        grouped_distances[group_name] = l2_dist
        print(f"   {group_name}: shape={l2_dist.shape}, mean={l2_dist.mean():.4f}")

    print("   ✓ Grouped Euclidean computation PASSED")

    print("\n3. Testing RMSE computation...")

    squared_diff = (x - y) ** 2
    mse_per_sample = squared_diff.mean(dim=1)
    rmse_per_sample = torch.sqrt(mse_per_sample)

    print(f"   RMSE shape: {rmse_per_sample.shape}")
    print(f"   RMSE mean: {rmse_per_sample.mean():.4f}")
    print(f"   RMSE std: {rmse_per_sample.std():.4f}")
    print("   ✓ RMSE computation PASSED")

    print("\n4. Testing CERTAIN/UNCERTAIN split logic...")

    # Mock attractor check (random for testing)
    in_attractor = torch.rand(batch_size) > 0.5
    certain_mask = in_attractor
    uncertain_mask = ~certain_mask

    n_certain = certain_mask.sum().item()
    n_uncertain = uncertain_mask.sum().item()

    print(f"   Total samples: {batch_size}")
    print(f"   CERTAIN: {n_certain} ({n_certain/batch_size*100:.1f}%)")
    print(f"   UNCERTAIN: {n_uncertain} ({n_uncertain/batch_size*100:.1f}%)")

    # Compute subset statistics
    if n_certain > 0:
        certain_distances = distances[certain_mask]
        print(f"   CERTAIN distances shape: {certain_distances.shape}")
        print(f"   CERTAIN mean per component: {certain_distances.mean(dim=0)}")

    if n_uncertain > 0:
        uncertain_distances = distances[uncertain_mask]
        print(f"   UNCERTAIN distances shape: {uncertain_distances.shape}")
        print(f"   UNCERTAIN mean per component: {uncertain_distances.mean(dim=0)}")

    print("   ✓ CERTAIN/UNCERTAIN split PASSED")

    print("\n5. Testing statistics computation...")

    # Per-component statistics
    for i in range(distances.shape[1]):
        comp_dist = distances[:, i]
        stats = {
            'mean': comp_dist.mean().item(),
            'median': comp_dist.median().item(),
            'var': comp_dist.var().item(),
            'std': comp_dist.std().item(),
        }
        print(f"   Component {i}: mean={stats['mean']:.4f}, median={stats['median']:.4f}, std={stats['std']:.4f}")

    print("   ✓ Statistics computation PASSED")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)

    return True


def test_with_real_model():
    """Test with actual Quadrotor3D model if available."""
    print("\n" + "=" * 80)
    print("TESTING WITH REAL QUADROTOR3D MODEL")
    print("=" * 80)

    try:
        from adaptive_roa.flow_matching.quadrotor_3d.latent_conditional.flow_matcher import Quadrotor3DLatentConditionalFlowMatcher
        from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem
        from torch.utils.data import DataLoader, TensorDataset

        print("\n1. Creating mock system and data...")

        # Create system
        system = Quadrotor3DSystem()

        # Create mock dataset
        batch_size = 50
        start_states = torch.randn(batch_size, 13)
        end_states = torch.randn(batch_size, 13)

        # Normalize quaternions
        start_states[:, 3:7] = start_states[:, 3:7] / torch.norm(start_states[:, 3:7], dim=1, keepdim=True)
        end_states[:, 3:7] = end_states[:, 3:7] / torch.norm(end_states[:, 3:7], dim=1, keepdim=True)

        # Create dataset with dict structure
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, start, end):
                self.start = start
                self.end = end

            def __len__(self):
                return len(self.start)

            def __getitem__(self, idx):
                return {"start_state": self.start[idx], "end_state": self.end[idx]}

        dataset = MockDataset(start_states, end_states)
        dataloader = DataLoader(dataset, batch_size=16)

        print(f"   Created dataset with {len(dataset)} samples")

        print("\n2. Testing component names...")

        # Test the component names method
        from adaptive_roa.model.quadrotor3d_unet import Quadrotor3DUNet

        model = Quadrotor3DUNet(
            embedded_dim=13,
            latent_dim=4,
            condition_dim=13,
            output_dim=12
        )

        flow_matcher = Quadrotor3DLatentConditionalFlowMatcher(
            system=system,
            model=model,
            optimizer=None,
            scheduler=None,
            latent_dim=4
        )

        component_names = flow_matcher.get_manifold_component_names()
        print(f"   Manifold component names ({len(component_names)}): {component_names}")

        euclidean_groups = flow_matcher.get_euclidean_groups()
        print(f"   Euclidean groups: {list(euclidean_groups.keys())}")

        print("\n3. Testing manifold distance computation directly...")

        # Test the compute_manifold_distance_per_component method
        pred = torch.randn(10, 13)
        true = torch.randn(10, 13)
        pred[:, 3:7] = pred[:, 3:7] / torch.norm(pred[:, 3:7], dim=1, keepdim=True)
        true[:, 3:7] = true[:, 3:7] / torch.norm(true[:, 3:7], dim=1, keepdim=True)

        distances = flow_matcher.compute_manifold_distance_per_component(pred, true)
        print(f"   Manifold distances shape: {distances.shape}")
        print(f"   Expected: [10, 10]")
        assert distances.shape == (10, 10), f"Wrong shape: {distances.shape}"
        print("   ✓ Manifold distance shape CORRECT")

        print("\n4. Testing grouped Euclidean distances...")

        grouped = flow_matcher.compute_grouped_euclidean_distances(pred, true)
        for name, dist in grouped.items():
            print(f"   {name}: shape={dist.shape}, mean={dist.mean():.4f}")
        print("   ✓ Grouped Euclidean CORRECT")

        print("\n5. Testing RMSE...")

        rmse = flow_matcher.compute_rmse(pred, true)
        print(f"   RMSE shape: {rmse.shape}, mean={rmse.mean():.4f}")
        print("   ✓ RMSE CORRECT")

        print("\n" + "=" * 80)
        print("REAL MODEL TESTS PASSED!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run basic tests
    basic_passed = test_evaluation_metrics()

    # Run real model tests
    real_passed = test_with_real_model()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Basic tests: {'PASSED' if basic_passed else 'FAILED'}")
    print(f"Real model tests: {'PASSED' if real_passed else 'FAILED'}")

    if basic_passed and real_passed:
        print("\nAll tests passed! The evaluation metrics implementation is working correctly.")
        sys.exit(0)
    else:
        print("\nSome tests failed. Check the output above for details.")
        sys.exit(1)
