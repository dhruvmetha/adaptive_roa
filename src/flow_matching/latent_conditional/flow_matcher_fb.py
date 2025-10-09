"""
Latent Conditional Flow Matching implementation using Facebook Flow Matching library

REFACTORED VERSION - Uses GeodesicProbPath and RiemannianODESolver
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import lightning.pytorch as pl
from torchmetrics import MeanMetric
import sys
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier/flow_matching')

from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper

from ..base.flow_matcher import BaseFlowMatcher
from ...systems.base import DynamicalSystem
from ...utils.fb_manifolds import PendulumManifold

from flow_matching.utils.manifolds import Product, FlatTorus, Euclidean

class LatentConditionalFlowMatcher(BaseFlowMatcher):
    """
    Latent Conditional Flow Matching using Facebook FM:
    - Uses GeodesicProbPath for geodesic interpolation on S¹×ℝ
    - Uses RiemannianODESolver for manifold-aware ODE integration
    - Neural net takes embedded x_t, time t, latent z, and start state condition
    - Predicts velocity in S¹ × ℝ tangent space

    KEY CHANGES FROM ORIGINAL:
    - ✅ Removed: manual interpolate_s1_x_r() → uses GeodesicProbPath
    - ✅ Removed: manual compute_target_velocity_s1_x_r() → automatic via path.sample()
    - ✅ Added: RiemannianODESolver for inference
    - ✅ Kept: latent variable z, conditioning on start state, all model logic
    """

    def __init__(self,
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 model_config: Optional[dict] = None,
                 latent_dim: int = 2,
                 mae_val_frequency: int = 10):
        """
        Initialize latent conditional flow matcher with FB FM integration

        Args:
            system: DynamicalSystem (pendulum with S¹ × ℝ structure)
            model: LatentConditionalUNet1D model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            model_config: Configuration dict
            latent_dim: Dimension of latent space
            mae_val_frequency: Compute MAE validation every N epochs
        """
        self.system = system
        self.latent_dim = latent_dim
        self.mae_val_frequency = mae_val_frequency
        super().__init__(model, optimizer, scheduler, model_config)

        # ===================================================================
        # NEW: Facebook FM Components
        # ===================================================================

        # Create manifold for S¹×ℝ (pendulum state space)
        self.manifold = Product(input_dim=2, manifolds=[(FlatTorus(), 1), (Euclidean(), 1)])

        # Create geodesic path with conditional OT scheduler
        self.path = GeodesicProbPath(
            scheduler=CondOTScheduler(),
            manifold=self.manifold
        )

        # ===================================================================
        # Validation endpoint MAE metrics (per dimension)
        # ===================================================================
        self.val_endpoint_mae_per_dim = nn.ModuleList([
            MeanMetric() for _ in range(self.system.state_dim)
        ])

        print("✅ Initialized with Facebook Flow Matching:")
        print(f"   - Manifold: PendulumManifold (S¹×ℝ)")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Latent dim: {latent_dim}")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noisy input in S¹ × ℝ space

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Noisy states [batch_size, 2] as (θ, θ̇_norm)
        """
        # θ ~ Uniform[-π, π]
        theta = torch.rand(batch_size, 1, device=device) * 2 * torch.pi - torch.pi

        # θ̇ ~ Uniform[-1, 1] (already normalized)
        theta_dot = torch.rand(batch_size, 1, device=device) * 2 - 1

        return torch.cat([theta, theta_dot], dim=1)

    def sample_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample Gaussian latent vector

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Latent vectors [batch_size, latent_dim]
        """
        return torch.randn(batch_size, self.latent_dim, device=device)

    # ===================================================================
    # REMOVED METHODS (now handled by Facebook FM):
    # ===================================================================
    # ❌ interpolate_s1_x_r() → replaced by self.path.sample()
    # ❌ compute_target_velocity_s1_x_r() → automatic in path_sample.dx_t

    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute latent conditional flow matching loss using Facebook FM

        CHANGES FROM ORIGINAL:
        - Uses GeodesicProbPath for interpolation (automatic geodesics!)
        - Uses path_sample.dx_t for target velocity (automatic via autodiff!)
        - No manual Theseus computation needed

        Args:
            batch: Dictionary containing 'start_state_original' and 'end_state_original'

        Returns:
            Flow matching loss
        """
        # Extract data endpoints and start states (raw S¹ × ℝ format)
        start_states = batch["start_state_original"]  # [B, 2] (θ, θ̇_norm)
        data_endpoints = batch["end_state_original"]  # [B, 2] (θ, θ̇_norm)

        batch_size = start_states.shape[0]
        device = self.device

        # Sample noisy inputs in S¹ × ℝ
        x_noise = self.sample_noisy_input(batch_size, device)

        # Sample random times
        t = torch.rand(batch_size, device=device)

        # Sample latent vectors
        z = self.sample_latent(batch_size, device)

        # ===================================================================
        # NEW: Use Facebook FM GeodesicProbPath
        # ===================================================================
        # This replaces:
        # - interpolate_s1_x_r(x_noise, data_endpoints, t)
        # - compute_target_velocity_s1_x_r(x_noise, data_endpoints, t)

        path_sample = self.path.sample(
            x_0=x_noise,           # [B, 2] noise in S¹×ℝ
            x_1=data_endpoints,    # [B, 2] target endpoints
            t=t                    # [B] random times
        )

        # path_sample contains:
        # - path_sample.x_t: [B, 2] interpolated state (geodesic on S¹×ℝ)
        # - path_sample.dx_t: [B, 2] target velocity (computed via autodiff!)
        # - path_sample.x_0, path_sample.x_1, path_sample.t

        # ===================================================================
        # SAME AS BEFORE: Model prediction
        # ===================================================================

        # Embed interpolated state for neural network input
        x_t_embedded = self.system.embed_state(path_sample.x_t)  # [B, 2] → [B, 3]

        # Embed start state for conditioning
        start_embedded = self.system.embed_state(start_states)  # [B, 2] → [B, 3]

        # Predict velocity using the model (YOUR architecture - unchanged!)
        predicted_velocity = self.forward(x_t_embedded, t, z, condition=start_embedded)

        # ===================================================================
        # NEW: Use automatic target velocity from path.sample()
        # ===================================================================
        # This replaces manual Theseus computation!

        target_velocity = path_sample.dx_t  # [B, 2] automatic geodesic velocity!

        # Compute MSE loss between predicted and target velocities
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)

        return loss

    def compute_endpoint_mae_per_dim(self,
                                    predicted_endpoints: torch.Tensor,
                                    true_endpoints: torch.Tensor) -> torch.Tensor:
        """
        Compute MAE per dimension with geodesic distance for angular components

        Args:
            predicted_endpoints: Predicted endpoints [B, state_dim]
            true_endpoints: True endpoints [B, state_dim]

        Returns:
            mae_per_dim: MAE for each dimension [state_dim]
        """
        mae_per_dim = []
        start_idx = 0

        for comp in self.system.manifold_components:
            end_idx = start_idx + comp.dim
            pred_comp = predicted_endpoints[:, start_idx:end_idx]
            true_comp = true_endpoints[:, start_idx:end_idx]

            if comp.manifold_type == "SO2":
                # Use circular/geodesic distance for angular components
                # For SO2, we have a single angle
                pred_angle = pred_comp[:, 0]
                true_angle = true_comp[:, 0]

                # Compute circular distance: |angle_diff| wrapped to [-π, π]
                angle_diff = pred_angle - true_angle
                circular_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                mae = torch.abs(circular_diff).mean()

            elif comp.manifold_type == "Real":
                # Standard absolute difference for real-valued components
                mae = torch.abs(pred_comp - true_comp).mean(dim=0)

                # If multiple dimensions in this component, append each
                if mae.numel() > 1:
                    mae_per_dim.extend(mae.tolist())
                    start_idx = end_idx
                    continue
            else:
                raise NotImplementedError(f"MAE computation for {comp.manifold_type} not implemented")

            mae_per_dim.append(mae.item() if isinstance(mae, torch.Tensor) else mae)
            start_idx = end_idx

        return torch.tensor(mae_per_dim, device=predicted_endpoints.device)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step with endpoint MAE computation

        Computes both:
        1. Flow matching loss (velocity field)
        2. Endpoint MAE (post-integration, per dimension with geodesic distance) - every 10 epochs
        """
        # Compute standard flow matching loss
        loss = self.compute_flow_loss(batch)

        # Log velocity field loss
        try:
            self.val_loss(loss)
        except Exception:
            self.val_loss.update(loss)

        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # ===================================================================
        # NEW: Compute endpoint MAE via integration (configurable frequency)
        # ===================================================================

        # Only compute endpoint MAE every N epochs (configurable)
        if self.current_epoch % self.mae_val_frequency == 0:
            start_states = batch["start_state_original"]  # [B, state_dim]
            true_endpoints = batch["end_state_original"]  # [B, state_dim]

            # Predict endpoints by integrating the flow
            predicted_endpoints = self.predict_endpoint(
                start_states=start_states,
                num_steps=100,  # Use 100 steps for validation
                latent=None     # Sample random latent
            )

            # Compute MAE per dimension with geodesic distance for angular components
            mae_per_dim = self.compute_endpoint_mae_per_dim(predicted_endpoints, true_endpoints)

            # Update and log metrics for each dimension
            for dim_idx in range(self.system.state_dim):
                metric = self.val_endpoint_mae_per_dim[dim_idx]
                dim_mae = mae_per_dim[dim_idx]

                try:
                    metric(dim_mae)
                except Exception:
                    metric.update(dim_mae)

                # Get component name for logging
                comp_name = self._get_dimension_name(dim_idx)
                self.log(f'val_endpoint_mae_{comp_name}', metric,
                        on_step=False, on_epoch=True, prog_bar=False)

            # Log average endpoint MAE
            avg_endpoint_mae = mae_per_dim.mean()
            self.log('val_endpoint_mae_avg', avg_endpoint_mae,
                    on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def _get_dimension_name(self, dim_idx: int) -> str:
        """
        Get human-readable name for a dimension based on system manifold structure

        Args:
            dim_idx: Dimension index

        Returns:
            Human-readable name (e.g., "angle", "angular_velocity", "cart_position")
        """
        current_idx = 0
        for comp in self.system.manifold_components:
            for local_idx in range(comp.dim):
                if current_idx == dim_idx:
                    if comp.dim == 1:
                        return comp.name
                    else:
                        return f"{comp.name}_{local_idx}"
                current_idx += 1
        return f"dim_{dim_idx}"

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Log and reset velocity field loss
        val_loss = self.val_loss.compute()
        self.log('val_loss_epoch', val_loss)
        self.val_loss.reset()

        # Print per-dimension MAE every N epochs (configurable)
        if self.current_epoch % self.mae_val_frequency == 0:
            print(f"\n📊 Epoch {self.current_epoch} - Validation MAE per dimension:")
            for dim_idx in range(self.system.state_dim):
                comp_name = self._get_dimension_name(dim_idx)
                mae_value = self.val_endpoint_mae_per_dim[dim_idx].compute()
                print(f"   {comp_name:20s}: {mae_value:.6f}")
            print()

        # Reset endpoint MAE metrics
        for metric in self.val_endpoint_mae_per_dim:
            metric.reset()

    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                z: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model (UNCHANGED)

        Args:
            x_t: Embedded interpolated state [B, 3] (sin θ, cos θ, θ̇_norm)
            t: Time parameter [B]
            z: Latent vector [B, latent_dim]
            condition: Embedded start state [B, 3]

        Returns:
            Predicted velocity [B, 2] in tangent space
        """
        return self.model(x_t, t, z, condition)

    # ===================================================================
    # NEW: INFERENCE METHODS using RiemannianODESolver
    # ===================================================================

    def predict_endpoint(self,
                        start_states: torch.Tensor,
                        num_steps: int = 100,
                        latent: Optional[torch.Tensor] = None,
                        method: str = "euler") -> torch.Tensor:
        """
        Predict endpoints from start states using Facebook FM's RiemannianODESolver

        NEW METHOD - Uses proper ODE integration with manifold projection

        Args:
            start_states: Start states [B, 2] in raw coordinates (θ, θ̇_norm)
            num_steps: Number of integration steps for ODE solving
            latent: Optional latent vectors [B, latent_dim]. If None, will sample.
            method: Integration method ("euler", "rk4", "midpoint")

        Returns:
            Predicted endpoints [B, 2] in raw coordinates
        """
        batch_size = start_states.shape[0]
        device = start_states.device

        # Ensure model is in eval mode for inference
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                # Sample noisy inputs
                x_noise = self.sample_noisy_input(batch_size, device)

                # Sample or use provided latent vectors
                if latent is None:
                    z = torch.randn(batch_size, self.latent_dim, device=device)
                else:
                    z = latent

                # Embed start states for conditioning
                start_embedded = self.system.embed_state(start_states)

                # ===================================================================
                # NEW: Create model wrapper for RiemannianODESolver
                # ===================================================================

                velocity_model = _PendulumVelocityModelWrapper(
                    model=self.model,
                    latent=z,
                    condition=start_embedded,
                    embed_fn=lambda x: self.system.embed_state(x)
                )

                # ===================================================================
                # NEW: Use RiemannianODESolver for integration
                # ===================================================================

                solver = RiemannianODESolver(
                    manifold=self.manifold,
                    velocity_model=velocity_model
                )

                final_states = solver.sample(
                    x_init=x_noise,
                    step_size=1.0/num_steps,
                    method=method,
                    projx=True,   # Use manifold projection (wraps angles)
                    proju=True,   # Use tangent projection
                    time_grid=torch.tensor([0.0, 1.0], device=device)
                )

                return final_states

        finally:
            # Restore original training mode
            if was_training:
                self.train()


# ============================================================================
# MODEL WRAPPER for RiemannianODESolver
# ============================================================================

class _PendulumVelocityModelWrapper(ModelWrapper):
    """
    Wrapper to adapt latent conditional model for FB FM's RiemannianODESolver

    FB FM solvers expect: velocity_model(x, t) → velocity
    Our model needs: model(x_embedded, t, z, condition) → velocity

    This wrapper bridges the gap.
    """

    def __init__(self, model: nn.Module, latent: torch.Tensor,
                 condition: torch.Tensor, embed_fn):
        """
        Args:
            model: The neural network (UNet)
            latent: Latent vectors [B, latent_dim] (fixed for trajectory)
            condition: Condition (start state embedded) [B, condition_dim]
            embed_fn: Function to embed state: (θ, θ̇) → (sin θ, cos θ, θ̇)
        """
        super().__init__(model)
        self.latent = latent
        self.condition = condition
        self.embed_fn = embed_fn

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        """
        Forward pass compatible with RiemannianODESolver

        Args:
            x: Current state [B, 2] raw format (θ, θ̇_norm)
            t: Time [B] or scalar

        Returns:
            Velocity [B, 2] in tangent space
        """
        # Handle scalar t (expand to batch)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # Embed state for neural network
        x_embedded = self.embed_fn(x)  # [B, 2] → [B, 3]

        # Expand latent and condition to match batch size if needed
        batch_size = x.shape[0]
        z = self.latent
        cond = self.condition

        if z.shape[0] == 1 and batch_size > 1:
            z = z.expand(batch_size, -1)
        if cond.shape[0] == 1 and batch_size > 1:
            cond = cond.expand(batch_size, -1)

        # Call the model
        velocity = self.model(x_embedded, t, z, cond)

        return velocity
