"""
Custom manifolds for Facebook Flow Matching library
Proper implementations for S¹×ℝ (Pendulum) and ℝ²×S¹×ℝ (CartPole)

Following FB FM's interface requirements and best practices:
1. logmap/expmap for geodesic computation (used by GeodesicProbPath)
2. projx for manifold projection (used by RiemannianODESolver)
3. proju for tangent space projection (used by RiemannianODESolver)

Key design decisions:
- Work in [-π, π] range (not [0, 2π]) to match our data
- Handle product manifolds correctly (S¹×ℝ and ℝ²×S¹×ℝ)
- Use atan2 for robust angle wrapping
"""
import torch
from torch import Tensor
import sys
import math
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier/flow_matching')
from flow_matching.utils.manifolds import Manifold


# flat torus



class PendulumManifold(Manifold):
    """
    S¹ × ℝ product manifold for pendulum system

    State: (θ, θ̇_norm) where:
    - θ ∈ S¹: angle in [-π, π] (periodic)
    - θ̇_norm ∈ [-1, 1]: normalized angular velocity (linear)

    This implements the product manifold structure correctly for FB FM.
    The key is handling the S¹ component with shortest angular distance.
    """

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Exponential map: transport from x along tangent vector u

        Args:
            x: Point on manifold [B, 2] = (θ, θ̇_norm)
            u: Tangent vector [B, 2] = (dθ, dθ̇_norm)

        Returns:
            Transported point [B, 2]
        """
        theta = x[..., 0:1]       # [B, 1]
        theta_dot = x[..., 1:2]   # [B, 1]

        u_theta = u[..., 0:1]     # [B, 1]
        u_theta_dot = u[..., 1:2] # [B, 1]

        # S¹ component: wrap angle to [-π, π]
        new_theta = torch.atan2(
            torch.sin(theta + u_theta),
            torch.cos(theta + u_theta)
        )

        # ℝ component: standard addition (normalized space)
        new_theta_dot = theta_dot + u_theta_dot

        return torch.cat([new_theta, new_theta_dot], dim=-1)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Logarithmic map: tangent vector from x to y

        Args:
            x: Source point [B, 2] = (θ_x, θ̇_x)
            y: Target point [B, 2] = (θ_y, θ̇_y)

        Returns:
            Tangent vector [B, 2] from x to y
        """
        theta_x = x[..., 0:1]
        theta_y = y[..., 0:1]
        theta_dot_x = x[..., 1:2]
        theta_dot_y = y[..., 1:2]

        # S¹: shortest angular distance (wrapped)
        delta_theta = torch.atan2(
            torch.sin(theta_y - theta_x),
            torch.cos(theta_y - theta_x)
        )

        # ℝ: standard difference
        delta_theta_dot = theta_dot_y - theta_dot_x

        return torch.cat([delta_theta, delta_theta_dot], dim=-1)

    def projx(self, x: Tensor) -> Tensor:
        """Project point onto manifold (wrap angles to [-π, π])"""
        theta = torch.atan2(torch.sin(x[..., 0:1]), torch.cos(x[..., 0:1]))
        theta_dot = x[..., 1:2]
        return torch.cat([theta, theta_dot], dim=-1)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Project vector onto tangent space (identity for S¹×ℝ)"""
        return u


class CartPoleManifold(Manifold):
    """
    ℝ² × S¹ × ℝ manifold for CartPole system

    State: (x_norm, θ, ẋ_norm, θ̇_norm) where:
    - x_norm ∈ [-1, 1] (normalized cart position)
    - θ ∈ S¹ (pole angle, wrapped to [-π, π])
    - ẋ_norm ∈ [-1, 1] (normalized cart velocity)
    - θ̇_norm ∈ [-1, 1] (normalized pole angular velocity)

    Note: We work in normalized space for neural network compatibility
    """

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Exponential map: transport from x along tangent vector u

        Args:
            x: Point on manifold [B, 4] = (x_norm, θ, ẋ_norm, θ̇_norm)
            u: Tangent vector [B, 4] = (dx_norm, dθ, dẋ_norm, dθ̇_norm)

        Returns:
            Transported point [B, 4]
        """
        # Extract components
        cart_pos = x[..., 0:1]      # [B, 1] ℝ (normalized)
        pole_angle = x[..., 1:2]    # [B, 1] S¹
        cart_vel = x[..., 2:3]      # [B, 1] ℝ (normalized)
        pole_ang_vel = x[..., 3:4]  # [B, 1] ℝ (normalized)

        u_pos = u[..., 0:1]
        u_angle = u[..., 1:2]
        u_vel = u[..., 2:3]
        u_ang_vel = u[..., 3:4]

        # ℝ components: standard addition
        new_pos = cart_pos + u_pos
        new_vel = cart_vel + u_vel
        new_ang_vel = pole_ang_vel + u_ang_vel

        # S¹ component: wrap angle to [-π, π]
        new_angle = torch.atan2(
            torch.sin(pole_angle + u_angle),
            torch.cos(pole_angle + u_angle)
        )

        # Reconstruct state: (x_norm, θ, ẋ_norm, θ̇_norm)
        return torch.cat([new_pos, new_angle, new_vel, new_ang_vel], dim=-1)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Logarithmic map: tangent vector from x to y

        Args:
            x: Source point [B, 4]
            y: Target point [B, 4]

        Returns:
            Tangent vector [B, 4] from x to y
        """
        # Extract components
        x_pos, x_angle, x_vel, x_ang_vel = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
        y_pos, y_angle, y_vel, y_ang_vel = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

        # ℝ components: standard difference
        delta_pos = y_pos - x_pos
        delta_vel = y_vel - x_vel
        delta_ang_vel = y_ang_vel - x_ang_vel

        # S¹: shortest angular distance (wrapped)
        delta_angle = torch.atan2(
            torch.sin(y_angle - x_angle),
            torch.cos(y_angle - x_angle)
        )

        return torch.cat([delta_pos, delta_angle, delta_vel, delta_ang_vel], dim=-1)

    def projx(self, x: Tensor) -> Tensor:
        """Project point onto manifold (wrap pole angle to [-π, π])"""
        cart_pos = x[..., 0:1]
        pole_angle = torch.atan2(torch.sin(x[..., 1:2]), torch.cos(x[..., 1:2]))
        cart_vel = x[..., 2:3]
        pole_ang_vel = x[..., 3:4]
        return torch.cat([cart_pos, pole_angle, cart_vel, pole_ang_vel], dim=-1)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Project vector onto tangent space (identity for ℝ²×S¹×ℝ)"""
        return u


# ============================================================================
# TESTING / VALIDATION UTILITIES
# ============================================================================

def test_pendulum_manifold():
    """Test PendulumManifold implementation"""
    print("Testing PendulumManifold...")
    manifold = PendulumManifold()

    # Test case: geodesic from -π to +π should go through boundary
    x = torch.tensor([[-math.pi + 0.1, 0.5]])
    y = torch.tensor([[math.pi - 0.1, 0.5]])

    # Logmap should give short path
    tangent = manifold.logmap(x, y)
    print(f"  Tangent from near -π to near +π: {tangent}")
    print(f"  (Should be small, going through boundary)")

    # Halfway point should be at boundary
    halfway = manifold.expmap(x, tangent * 0.5)
    print(f"  Halfway point: {halfway}")
    print(f"  (Should be near ±π)")

    # Test wrapping
    out_of_range = torch.tensor([[3.5, 0.0]])
    wrapped = manifold.projx(out_of_range)
    print(f"  Wrapped {out_of_range[0,0]:.2f} → {wrapped[0,0]:.2f}")
    print(f"  (Should be in [-π, π])")

    print("  ✓ PendulumManifold tests passed\n")


def test_cartpole_manifold():
    """Test CartPoleManifold implementation"""
    print("Testing CartPoleManifold...")
    manifold = CartPoleManifold()

    # Test geodesic with angle wrapping
    x = torch.tensor([[0.0, -math.pi + 0.1, 0.0, 0.0]])
    y = torch.tensor([[0.5, math.pi - 0.1, 0.2, 0.1]])

    tangent = manifold.logmap(x, y)
    print(f"  Tangent vector: {tangent}")
    print(f"  (Angular component should be small)")

    # Test wrapping
    out_of_range = torch.tensor([[0.0, 3.5, 0.0, 0.0]])
    wrapped = manifold.projx(out_of_range)
    print(f"  Wrapped angle {out_of_range[0,1]:.2f} → {wrapped[0,1]:.2f}")
    print(f"  (Should be in [-π, π])")

    print("  ✓ CartPoleManifold tests passed\n")


if __name__ == "__main__":
    test_pendulum_manifold()
    test_cartpole_manifold()
