"""
Abstract base class for flow matching models
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import MeanMetric
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import hydra

from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper

from .config import FlowMatchingConfig


class LatentConditionalVelocityWrapper(ModelWrapper):
    """
    Wrapper to adapt latent conditional model for Facebook FM's RiemannianODESolver

    FB FM solvers expect: velocity_model(x, t) → velocity
    Our model needs: model(x_embedded, t, z, condition) → velocity

    This wrapper bridges the gap by handling:
    - State embedding via embed_fn
    - Latent and condition broadcasting
    - Model invocation with correct arguments
    """

    def __init__(
        self, model: nn.Module, latent: torch.Tensor, condition: torch.Tensor, embed_fn
    ):
        """
        Args:
            model: The neural network (UNet)
            latent: Latent vectors [B, latent_dim] (fixed for trajectory)
            condition: Condition (start state embedded) [B, condition_dim]
            embed_fn: Function to embed state for model input
        """
        super().__init__(model)
        self.latent = latent
        self.condition = condition
        self.embed_fn = embed_fn

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        """
        Forward pass compatible with RiemannianODESolver

        Args:
            x: Current state [B, state_dim] (normalized or raw depending on system)
            t: Time [B] or scalar

        Returns:
            Velocity [B, state_dim] in tangent space
        """
        # Handle scalar t (expand to batch)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # Embed state for neural network
        x_embedded = self.embed_fn(x)

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


class BaseFlowMatcher(pl.LightningModule, ABC):
    """
    Abstract base class for latent conditional flow matching Lightning modules

    Provides common functionality for:
    - Training and validation loops
    - Latent variable sampling
    - MAE per-dimension tracking
    - Facebook Flow Matching integration (GeodesicProbPath)

    Subclasses must implement system-specific methods for:
    - Manifold creation
    - Noisy input sampling
    - Flow loss computation
    - Endpoint prediction
    """

    def __init__(
        self,
        system,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        model_config: Optional[FlowMatchingConfig] = None,
        latent_dim: int = 2,
        mae_val_frequency: int = 10,
        use_loss_weights: bool = False,
        clamp_noise: bool = True,
        zero_latent: bool = False,
        val_error_log_file: Optional[str] = None,
        noise_scale: float = 1.0,
    ):
        """
        Initialize base flow matcher

        Args:
            system: DynamicalSystem instance (pendulum, cartpole, etc.)
            model: Neural network model (UNet, etc.)
            optimizer: Optimizer configuration
            scheduler: LR scheduler configuration
            model_config: Flow matching configuration
            latent_dim: Dimension of latent variable z
            mae_val_frequency: Compute endpoint MAE every N epochs
            use_loss_weights: If True, weight loss by normalization limits
            clamp_noise: If True, clamp noise to [-1, 1] to prevent ODE divergence
            zero_latent: If True, use zero latent vectors instead of random sampling
            val_error_log_file: Path to text file for logging validation errors (None = no file logging)
            noise_scale: Scale factor for noise in sample_noisy_input (0-1, default 1.0)
        """
        super().__init__()

        # Store system and model
        self.system = system
        self.model = model
        self.config = model_config or FlowMatchingConfig()
        self.latent_dim = latent_dim
        self.mae_val_frequency = mae_val_frequency
        self.use_loss_weights = use_loss_weights
        self.clamp_noise = clamp_noise
        self.zero_latent = zero_latent
        self.val_error_log_file = val_error_log_file
        self.noise_scale = noise_scale

        # Store optimizer and scheduler configs (will be instantiated in configure_optimizers)
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        # Metrics tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # Storage for collecting validation errors for percentile computation
        # These are lists that get reset each validation epoch
        self._val_abs_errors_buffer = []  # Will hold [B, manifold_dist_dim] tensors

        # Facebook FM components (subclass creates manifold)
        self.manifold = self._create_manifold()
        # Distance manifold is always the true system manifold (for proper geodesic distances)
        # Even when use_manifold=False for training, distances should use true manifold
        self.distance_manifold = self._create_distance_manifold()

        # MAE metrics per dimension for endpoint prediction
        # Must be initialized AFTER manifold creation since manifold.dist() may return
        # fewer dimensions than state_dim (e.g., SO3 returns 1 geodesic distance, not 4)
        self._manifold_dist_dim = self._get_manifold_dist_dim()
        self.val_endpoint_mae_per_dim = nn.ModuleList(
            [MeanMetric() for _ in range(self._manifold_dist_dim)]
        )
        self.path = GeodesicProbPath(
            scheduler=CondOTScheduler(), manifold=self.manifold
        )

        # Loss weights for weighted MSE (proportional to normalization limits)
        if use_loss_weights:
            loss_weights = self.system.get_loss_weights()
            self.register_buffer("loss_weights", loss_weights)
            print(f"📊 Loss weights enabled: {loss_weights.tolist()}")
        else:
            self.loss_weights = None

        # Save hyperparameters (exclude model and optimizer/scheduler to avoid pickle issues)
        self.save_hyperparameters(ignore=["model", "optimizer", "scheduler", "system"])
        # Persist dataset location explicitly so strict checkpoint restore does not
        # need to infer or fallback to Hydra config files.
        self.hparams["system_dataset_dir"] = getattr(system, "dataset_dir", None)

    def sample_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample Gaussian latent vector (or return zeros if zero_latent=True)

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Latent vectors [batch_size, latent_dim]
        """
        if self.zero_latent:
            return torch.zeros(batch_size, self.latent_dim, device=device)
        return torch.randn(batch_size, self.latent_dim, device=device)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model

        Args:
            x_t: Embedded interpolated state [batch_size, embedded_dim]
            t: Time parameter [batch_size]
            z: Latent vector [batch_size, latent_dim]
            condition: Embedded start state [batch_size, condition_dim]

        Returns:
            Predicted velocity [batch_size, state_dim]
        """
        return self.model(x_t, t, z, condition)

    def compute_evaluation_metrics(
        self,
        predicted_endpoints: torch.Tensor,
        true_endpoints: torch.Tensor,
        aggregation: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """
        DEPRECATED: Use evaluate_with_manifold_metrics() instead.

        Compute comprehensive evaluation metrics for final state error.

        WARNING: This method uses Euclidean absolute differences which are
        incorrect for manifold components (SO3 quaternions, S1 angles).
        For proper geodesic distances, use evaluate_with_manifold_metrics().

        Args:
            predicted_endpoints: Predicted endpoints [B, state_dim]
            true_endpoints: True endpoints [B, state_dim]
            aggregation: "mean" or "median" - how to aggregate over samples

        Returns:
            Dictionary with metrics:
            - 'abs_errors': Raw absolute errors [B, state_dim]
            - 'mae_per_dim': MAE for each dimension [state_dim]
            - 'mae_per_dim_names': List of dimension names
            - 'mae_mean_over_dims': Mean of per-dim MAE (scalar)
            - 'mae_median_over_dims': Median of per-dim MAE (scalar)
            - 'overall_mae': Overall MAE across all samples and dims (scalar)
            - 'overall_median_ae': Overall median absolute error (scalar)
            - 'per_sample_mae': MAE for each sample [B]
            - 'per_sample_median_ae': Median AE for each sample [B]
        """
        import warnings
        warnings.warn(
            "compute_evaluation_metrics() uses Euclidean distances which are incorrect "
            "for manifold components (SO3, S1). Use evaluate_with_manifold_metrics() "
            "for proper geodesic distances.",
            DeprecationWarning,
            stacklevel=2
        )
        # Compute absolute errors: [B, state_dim]
        abs_errors = torch.abs(predicted_endpoints - true_endpoints)

        # 1. Per-dimension MAE (aggregate over samples using mean or median)
        if aggregation == "mean":
            mae_per_dim = abs_errors.mean(dim=0)  # [state_dim]
        elif aggregation == "median":
            mae_per_dim = abs_errors.median(dim=0).values  # [state_dim]
        else:
            raise ValueError(
                f"aggregation must be 'mean' or 'median', got {aggregation}"
            )

        # Get dimension names
        mae_per_dim_names = [
            self._get_dimension_name(i) for i in range(self.system.state_dim)
        ]

        # 2. Aggregate over dimensions
        mae_mean_over_dims = mae_per_dim.mean()  # scalar
        mae_median_over_dims = mae_per_dim.median()  # scalar

        # 3. Overall statistics
        overall_mae = abs_errors.mean()  # scalar (all samples, all dims)
        overall_median_ae = abs_errors.median()  # scalar

        # Per-sample statistics (aggregate over dims for each sample)
        per_sample_mae = abs_errors.mean(dim=1)  # [B]
        per_sample_median_ae = abs_errors.median(dim=1).values  # [B]

        return {
            "abs_errors": abs_errors,
            "mae_per_dim": mae_per_dim,
            "mae_per_dim_names": mae_per_dim_names,
            "mae_mean_over_dims": mae_mean_over_dims,
            "mae_median_over_dims": mae_median_over_dims,
            "overall_mae": overall_mae,
            "overall_median_ae": overall_median_ae,
            "per_sample_mae": per_sample_mae,
            "per_sample_median_ae": per_sample_median_ae,
            "aggregation": aggregation,
        }

    def evaluate_on_dataset(
        self,
        dataloader,
        num_steps: int = 100,
        aggregation: str = "mean",
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Use evaluate_with_manifold_metrics() instead.

        Evaluate the model on a full dataset (e.g., evaluation/test set).

        WARNING: This method uses Euclidean absolute differences which are
        incorrect for manifold components (SO3 quaternions, S1 angles).
        For proper geodesic distances, use evaluate_with_manifold_metrics().

        Args:
            dataloader: DataLoader providing batches with 'start_state' and 'end_state'
            num_steps: Number of ODE integration steps
            aggregation: "mean" or "median" for per-dim aggregation
            device: Device to run evaluation on

        Returns:
            Dictionary with comprehensive evaluation metrics:
            - Per-dimension MAE with names
            - Mean/Median over dimensions
            - Overall dataset statistics
            - Per-sample errors for analysis
        """
        import warnings
        warnings.warn(
            "evaluate_on_dataset() uses Euclidean distances which are incorrect "
            "for manifold components (SO3, S1). Use evaluate_with_manifold_metrics() "
            "for proper geodesic distances.",
            DeprecationWarning,
            stacklevel=2
        )
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        all_abs_errors = []
        all_per_sample_mae = []

        with torch.no_grad():
            for batch in dataloader:
                start_states = self._get_start_states(batch).to(device)
                true_endpoints = self._get_end_states(batch).to(device)

                # Predict endpoints
                predicted_endpoints = self.predict_endpoint(
                    start_states=start_states, num_steps=num_steps, latent=None
                )

                # Compute absolute errors for this batch
                abs_errors = torch.abs(predicted_endpoints - true_endpoints)
                all_abs_errors.append(abs_errors)

                # Per-sample MAE
                per_sample_mae = abs_errors.mean(dim=1)
                all_per_sample_mae.append(per_sample_mae)

        # Concatenate all batches
        all_abs_errors = torch.cat(all_abs_errors, dim=0)  # [N, state_dim]
        all_per_sample_mae = torch.cat(all_per_sample_mae, dim=0)  # [N]

        N = all_abs_errors.shape[0]

        # 1. Per-dimension MAE (aggregate over all samples)
        if aggregation == "mean":
            mae_per_dim = all_abs_errors.mean(dim=0)  # [state_dim]
        else:
            mae_per_dim = all_abs_errors.median(dim=0).values

        mae_per_dim_names = [
            self._get_dimension_name(i) for i in range(self.system.state_dim)
        ]

        # 2. Aggregate over dimensions
        mae_mean_over_dims = mae_per_dim.mean()
        mae_median_over_dims = mae_per_dim.median()

        # 3. Overall statistics
        overall_mae = all_abs_errors.mean()
        overall_median_ae = all_abs_errors.median()

        # Per-sample statistics
        per_sample_mean = all_per_sample_mae.mean()
        per_sample_median = all_per_sample_mae.median()
        per_sample_std = all_per_sample_mae.std()

        return {
            # Per-dimension metrics
            "mae_per_dim": mae_per_dim.cpu(),
            "mae_per_dim_names": mae_per_dim_names,
            "mae_per_dim_dict": {
                name: mae_per_dim[i].item() for i, name in enumerate(mae_per_dim_names)
            },
            # Aggregated over dimensions
            "mae_mean_over_dims": mae_mean_over_dims.item(),
            "mae_median_over_dims": mae_median_over_dims.item(),
            # Overall dataset statistics
            "overall_mae": overall_mae.item(),
            "overall_median_ae": overall_median_ae.item(),
            # Per-sample statistics
            "per_sample_mae_mean": per_sample_mean.item(),
            "per_sample_mae_median": per_sample_median.item(),
            "per_sample_mae_std": per_sample_std.item(),
            # Raw data for further analysis
            "all_per_sample_mae": all_per_sample_mae.cpu(),
            "num_samples": N,
            "aggregation": aggregation,
        }

    def print_evaluation_report(self, metrics: Dict[str, Any]) -> None:
        """
        Print a formatted evaluation report.

        Args:
            metrics: Dictionary from evaluate_on_dataset()
        """
        print("\n" + "=" * 70)
        print("📊 EVALUATION REPORT")
        print("=" * 70)
        print(f"Samples evaluated: {metrics['num_samples']}")
        print(f"Aggregation method: {metrics['aggregation']}")
        print()

        print("📈 Per-Dimension MAE:")
        print("-" * 40)
        for name, mae in metrics["mae_per_dim_dict"].items():
            print(f"  {name:12s}: {mae:.6f}")
        print()

        print("📊 Aggregated Over Dimensions:")
        print("-" * 40)
        print(f"  Mean(MAE per dim):   {metrics['mae_mean_over_dims']:.6f}")
        print(f"  Median(MAE per dim): {metrics['mae_median_over_dims']:.6f}")
        print()

        print("🎯 Overall Dataset Statistics:")
        print("-" * 40)
        print(f"  Overall MAE:       {metrics['overall_mae']:.6f}")
        print(f"  Overall Median AE: {metrics['overall_median_ae']:.6f}")
        print()

        print("📉 Per-Sample Statistics:")
        print("-" * 40)
        print(f"  Mean of sample MAEs:   {metrics['per_sample_mae_mean']:.6f}")
        print(f"  Median of sample MAEs: {metrics['per_sample_mae_median']:.6f}")
        print(f"  Std of sample MAEs:    {metrics['per_sample_mae_std']:.6f}")
        print("=" * 70 + "\n")

    @abstractmethod
    def _create_manifold(self):
        """
        Create Facebook FM manifold for this system (used for training).

        When use_manifold=False, this may return Euclidean manifold.

        Returns:
            Product manifold (e.g., S¹×ℝ for pendulum, ℝ²×S¹×ℝ for cartpole)
        """
        pass

    def _create_distance_manifold(self):
        """
        Create manifold for distance computation (always true system manifold).

        This is used for geodesic distance calculations regardless of use_manifold setting.
        Override in subclasses that support use_manifold=False to return the true
        system manifold even when training uses Euclidean.

        By default, returns the same as _create_manifold() for backward compatibility.

        Returns:
            Product manifold representing the true system geometry
        """
        return self.manifold

    def _get_manifold_dist_dim(self) -> int:
        """
        Get the output dimension of distance_manifold.dist().

        Uses distance_manifold (true system manifold) not training manifold,
        so distances are always computed with correct geodesic structure.

        For Product manifolds, the distance output may differ from state_dim:
        - Euclidean(n) returns n per-dimension distances
        - SO3(4, 3) returns 1 geodesic distance (not 4)
        - SE3(7, 6) returns 4 distances (3 per-axis translation + 1 SO3 geodesic angle)
        - FlatTorus/S1 returns 1 geodesic distance per circle

        This is used to correctly initialize per-dimension validation metrics.

        Returns:
            Number of distance components returned by distance_manifold.dist()
        """
        # Import manifold types for isinstance checks
        try:
            from flow_matching.utils.manifolds import SO3, SE3, FlatTorus
        except ImportError:
            SO3 = None
            SE3 = None
            FlatTorus = None

        # For Product manifolds, compute based on component structure
        # Use distance_manifold (true system manifold) not training manifold
        if hasattr(self.distance_manifold, 'manifolds') and hasattr(self.distance_manifold, 'dimensions'):
            total_dim = 0
            manifolds = self.distance_manifold.manifolds
            dimensions = self.distance_manifold.dimensions

            for i, m in enumerate(manifolds):
                # SO3 and FlatTorus return single geodesic distance, not per-dim
                if SO3 is not None and isinstance(m, SO3):
                    total_dim += 1  # Single geodesic angle
                elif SE3 is not None and isinstance(m, SE3):
                    total_dim += 4  # 3 per-axis translation + 1 SO3 geodesic angle
                elif FlatTorus is not None and isinstance(m, FlatTorus):
                    total_dim += 1  # Single geodesic distance
                else:
                    # Euclidean and other manifolds return per-dimension distances
                    total_dim += dimensions[i]
            return total_dim
        else:
            # Non-product manifold (e.g., pure Euclidean)
            return self.system.state_dim

    @abstractmethod
    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample Gaussian noise directly in normalized space.

        Returns noise that is ready to use without further normalization.
        For manifolds with circular/spherical components, the noise should be
        projected onto the manifold using self.manifold.projx().

        IMPORTANT: Subclasses should respect self.clamp_noise. If True, clamp
        the noise to [-1, 1] before manifold projection to prevent extreme values
        that can cause ODE integration to diverge during endpoint prediction.

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Noisy states [batch_size, state_dim] in normalized space
        """
        pass

    @abstractmethod
    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract start states from batch

        Standard key: "start_state"

        Args:
            batch: Batch dictionary with "start_state" key

        Returns:
            Start states [batch_size, state_dim]
        """
        pass

    @abstractmethod
    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract end states from batch

        Standard key: "end_state"

        Args:
            batch: Batch dictionary with "end_state" key

        Returns:
            End states [batch_size, state_dim]
        """
        pass

    @abstractmethod
    def _get_dimension_name(self, dim_idx: int) -> str:
        """
        Get human-readable name for dimension

        Args:
            dim_idx: Dimension index

        Returns:
            Dimension name (e.g., "angle", "cart_position")
        """
        pass

    @abstractmethod
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state for model input (optional, can be no-op)

        Args:
            state: Raw state [B, state_dim]

        Returns:
            Normalized state [B, state_dim]
        """
        pass

    @abstractmethod
    def embed_state_for_model(self, state: torch.Tensor) -> torch.Tensor:
        """
        Embed state for neural network input

        Args:
            state: State [B, state_dim] (normalized or raw)

        Returns:
            Embedded state [B, embedded_dim]
        """
        pass

    def _prepare_model_inputs(
        self,
        batch_size: int,
        start_states: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Common preparation logic for both training and inference

        Handles the shared pattern of:
        1. Sampling and normalizing noisy inputs
        2. Handling latent vectors (sample or use provided)
        3. Normalizing and embedding start states

        Args:
            batch_size: Number of samples
            start_states: Start states [batch_size, state_dim]
            latent: Optional latent vectors [batch_size, latent_dim]. If None, will sample.
            device: Device for tensor creation. If None, uses self.device

        Returns:
            Tuple of:
                x_noise_normalized: Normalized noisy input [batch_size, state_dim]
                z: Latent vectors [batch_size, latent_dim]
                start_embedded: Embedded start state [batch_size, condition_dim]
                start_normalized: Normalized start state [batch_size, state_dim]
        """
        if device is None:
            device = self.device

        # Sample noisy inputs (already in normalized space)
        x_noise_normalized = self.sample_noisy_input(batch_size, device)

        # Handle latent vectors
        if latent is None:
            z = self.sample_latent(batch_size, device)
        else:
            z = latent

        # Normalize and embed start states
        start_normalized = self.normalize_state(start_states)
        start_normalized_embedded = self.embed_state_for_model(start_normalized)

        return x_noise_normalized, z, start_normalized_embedded, start_normalized

    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute latent conditional flow matching loss using Facebook FM

        Unified implementation for all systems using:
        - GeodesicProbPath for interpolation (automatic geodesics!)
        - path_sample.dx_t for target velocity (automatic via autodiff!)

        Args:
            batch: Dictionary containing batch data

        Returns:
            Flow matching loss
        """
        # Extract data endpoints and start states
        start_states = self._get_start_states(batch)
        data_endpoints = self._get_end_states(batch)

        batch_size = start_states.shape[0]
        device = self.device

        # Common preparation logic
        x_noise_normalized, z, start_normalized_embedded, _ = (
            self._prepare_model_inputs(
                batch_size=batch_size,
                start_states=start_states,
                latent=None,  # Always sample fresh latents for training
                device=device,
            )
        )

        # Sample random times
        t = torch.rand(batch_size, device=device)

        # Normalize data endpoints
        data_normalized = self.normalize_state(data_endpoints)

        # Use Facebook FM GeodesicProbPath for geodesic interpolation
        path_sample = self.path.sample(
            x_0=x_noise_normalized,  # Normalized noise
            x_1=data_normalized,  # Normalized target endpoints
            t=t,  # Random times
        )

        # Embed interpolated state for neural network input
        x_t_embedded = self.embed_state_for_model(path_sample.x_t)

        # Predict velocity using the model
        predicted_velocity = self.forward(
            x_t_embedded, t, z, condition=start_normalized_embedded
        )

        # Use automatic target velocity from path.sample()
        target_velocity = path_sample.dx_t

        # Compute MSE loss between predicted and target velocities
        if self.use_loss_weights and self.loss_weights is not None:
            # Weighted MSE: mean(weights * (pred - target)^2)
            # weights shape: [tangent_dim], velocity shape: [batch, tangent_dim]
            normalized_loss_weights = self.loss_weights / (self.loss_weights.mean() + 1e-12)
            squared_error = (predicted_velocity - target_velocity) ** 2
            weighted_error = normalized_loss_weights.unsqueeze(0) * squared_error
            loss = weighted_error.mean()
        else:
            loss = nn.functional.mse_loss(predicted_velocity, target_velocity)

        return loss

    def predict_endpoint(
        self,
        start_states: torch.Tensor,
        num_steps: int = 100,
        latent: Optional[torch.Tensor] = None,
        method: str = "euler",
    ) -> torch.Tensor:
        """
        Predict endpoints from start states using Facebook FM's RiemannianODESolver

        Unified implementation using:
        - Proper normalization
        - State embedding for model input
        - Manifold-aware ODE integration

        Args:
            start_states: Start states [B, state_dim] in raw coordinates
            num_steps: Number of integration steps for ODE solving
            latent: Optional latent vectors [B, latent_dim]. If None, will sample.
            method: Integration method ("euler_riemannian", "euler", "rk4", "midpoint")

        Returns:
            Predicted endpoints [B, state_dim] in raw coordinates
        """
        batch_size = start_states.shape[0]
        device = start_states.device

        # Ensure model is in eval mode for inference
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                # Common preparation logic
                x_noise_normalized, z, start_normalized_embedded, start_normalized = (
                    self._prepare_model_inputs(
                        batch_size=batch_size,
                        start_states=start_states,
                        latent=latent,  # Use provided or sample
                        device=device,
                    )
                )

                # Create model wrapper for RiemannianODESolver
                velocity_model = LatentConditionalVelocityWrapper(
                    model=self.model,
                    latent=z,
                    condition=start_normalized_embedded,
                    embed_fn=self.embed_state_for_model,
                )

                # Use RiemannianODESolver for integration
                solver = RiemannianODESolver(
                    manifold=self.manifold, velocity_model=velocity_model
                )

                final_states_normalized = solver.sample(
                    x_init=x_noise_normalized,
                    step_size=1.0 / num_steps,
                    method=method,
                    projx=True,  # Use manifold projection (wraps angles)
                    proju=True,  # Use tangent projection
                    time_grid=torch.tensor([0.0, 1.0], device=device),
                )

                # Denormalize back to raw coordinates
                final_states_raw = self.denormalize_state(final_states_normalized)

                return final_states_raw

        finally:
            # Restore original training mode
            if was_training:
                self.train()

    def refine_endpoints(
        self,
        invalid_endpoints: torch.Tensor,
        start_states: torch.Tensor,
        t_range: tuple = (0.7, 0.9),
        num_steps: int = 100,
        latent: Optional[torch.Tensor] = None,
        method: str = "euler",
    ) -> torch.Tensor:
        """
        Refine invalid endpoints by re-running ODE from a late timestep t_start to 1.0.

        For endpoints that landed in the separatrix (label=0), treat them as
        partially-converged predictions at some t close to 1. The velocity field
        at t ~ 0.8 expects mostly-data-like inputs with small noise — which matches
        the distribution of "almost converged" invalid endpoints. Integrating from
        t_start → 1.0 applies the final correction the model learned during training.

        Args:
            invalid_endpoints: Endpoints classified as invalid [B, state_dim] in raw coords
            start_states: Original start states for conditioning [B, state_dim] in raw coords
            t_range: Range (t_min, t_max) to uniformly sample t_start from
            num_steps: Number of ODE integration steps for the [t_start, 1.0] interval
            latent: Optional latent vectors [B, latent_dim]. If None, will sample fresh.
            method: Integration method ("euler", "rk4", "midpoint")

        Returns:
            Refined endpoints [B, state_dim] in raw coordinates
        """
        batch_size = invalid_endpoints.shape[0]
        device = invalid_endpoints.device

        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                # Sample random t_start ~ U[t_min, t_max]
                t_start = torch.empty(1, device=device).uniform_(t_range[0], t_range[1]).item()

                # Normalize invalid endpoints to use as x_init
                x_init = self.normalize_state(invalid_endpoints)

                # Prepare latent and condition (same as predict_endpoint)
                if latent is None:
                    z = self.sample_latent(batch_size, device)
                else:
                    z = latent

                start_normalized = self.normalize_state(start_states)
                start_embedded = self.embed_state_for_model(start_normalized)

                # Create velocity model wrapper
                velocity_model = LatentConditionalVelocityWrapper(
                    model=self.model,
                    latent=z,
                    condition=start_embedded,
                    embed_fn=self.embed_state_for_model,
                )

                # ODE integration from t_start → 1.0
                solver = RiemannianODESolver(
                    manifold=self.manifold, velocity_model=velocity_model
                )

                step_size = (1.0 - t_start) / num_steps

                refined_normalized = solver.sample(
                    x_init=x_init,
                    step_size=step_size,
                    method=method,
                    projx=True,
                    proju=True,
                    time_grid=torch.tensor([t_start, 1.0], device=device),
                )

                return self.denormalize_state(refined_normalized)

        finally:
            if was_training:
                self.train()

    @abstractmethod
    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state from model space back to raw coordinates

        Args:
            normalized_state: Normalized state [B, state_dim]

        Returns:
            Raw state [B, state_dim]
        """
        pass

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step - common across all variants"""
        loss = self.compute_flow_loss(batch)

        # Log metrics with compatibility for different TorchMetrics versions
        try:
            # TorchMetrics >= 1.2 style
            self.train_loss(loss)
        except Exception:
            # Fallback for older TorchMetrics versions
            self.train_loss.update(loss)

        self.log(
            "train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step with optional endpoint MAE computation"""
        loss = self.compute_flow_loss(batch)

        # Log metrics with compatibility for different TorchMetrics versions
        try:
            # TorchMetrics >= 1.2 style
            self.val_loss(loss)
        except Exception:
            # Fallback for older TorchMetrics versions
            self.val_loss.update(loss)

        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute endpoint MAE every N epochs
        if self.current_epoch % self.mae_val_frequency == 0:
            # Extract states from batch
            start_states = self._get_start_states(batch)
            true_endpoints = self._get_end_states(batch)

            # Predict endpoints
            with torch.no_grad():
                predicted_endpoints = self.predict_endpoint(
                    start_states=start_states, num_steps=100, latent=None
                )

            # Compute geodesic errors per component using manifold distance [B, manifold_dist_dim]
            # This properly handles circular/manifold components (e.g., angles on S¹, quaternions on SO(3))
            # Note: manifold.dist() returns per-component distances, not per-state-dimension
            # e.g., Quadrotor3D: 13D state → 10D distances (SE3 returns 4: 3 translation + 1 geodesic)
            pred_normalized = self.normalize_state(predicted_endpoints)
            true_normalized = self.normalize_state(true_endpoints)
            geodesic_errors = self.manifold.dist(pred_normalized, true_normalized)

            # Store for percentile computation at epoch end
            self._val_abs_errors_buffer.append(geodesic_errors.detach().cpu())

            # Compute MAE per component (for backward compatibility with logging)
            mae_per_dim = geodesic_errors.mean(dim=0)

            # Update metrics - use manifold dist dim, not state dim
            # Get component names for logging (may differ from state dimension names)
            component_names = self.get_manifold_component_names()
            for dim_idx in range(self._manifold_dist_dim):
                try:
                    self.val_endpoint_mae_per_dim[dim_idx](mae_per_dim[dim_idx])
                except Exception:
                    self.val_endpoint_mae_per_dim[dim_idx].update(mae_per_dim[dim_idx])

                # Log individual component MAE
                comp_name = component_names[dim_idx] if dim_idx < len(component_names) else f"dim_{dim_idx}"
                self.log(
                    f"val_endpoint_mae_{comp_name}",
                    self.val_endpoint_mae_per_dim[dim_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        return loss

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Instantiate optimizer from config
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.parameters()
        )

        # Handle scheduler if present
        if self.scheduler_config is not None:
            scheduler = hydra.utils.instantiate(
                self.scheduler_config, optimizer=optimizer
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }

        # Return just the optimizer if no scheduler
        return optimizer

    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Log epoch metrics
        self.log("train_loss_epoch", self.train_loss.compute())
        self.train_loss.reset()

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Log epoch metrics
        val_loss = self.val_loss.compute()
        self.log("val_loss_epoch", val_loss)
        self.val_loss.reset()

        # Print comprehensive error statistics every N epochs
        if self.current_epoch % self.mae_val_frequency == 0 and len(self._val_abs_errors_buffer) > 0:
            # Concatenate all collected absolute errors [N, state_dim]
            all_abs_errors = torch.cat(self._val_abs_errors_buffer, dim=0)
            n_samples = all_abs_errors.shape[0]

            # Compute per-component statistics (using manifold dist dimension, not state dim)
            # For each component: compute mean, P50, P90, P99 over all samples
            mean_per_dim = all_abs_errors.mean(dim=0)  # [manifold_dist_dim]
            p50_per_dim = torch.quantile(all_abs_errors, 0.50, dim=0)  # [manifold_dist_dim]
            p90_per_dim = torch.quantile(all_abs_errors, 0.90, dim=0)  # [manifold_dist_dim]
            p99_per_dim = torch.quantile(all_abs_errors, 0.99, dim=0)  # [manifold_dist_dim]

            # Compute overall state error (L2 norm across components for each sample)
            overall_errors = torch.norm(all_abs_errors, dim=1)  # [N]
            overall_mean = overall_errors.mean().item()
            overall_p50 = torch.quantile(overall_errors, 0.50).item()
            overall_p90 = torch.quantile(overall_errors, 0.90).item()
            overall_p99 = torch.quantile(overall_errors, 0.99).item()

            # Get component names for display
            component_names = self.get_manifold_component_names()

            # Print header
            print(f"\n{'='*80}")
            print(f"📊 Epoch {self.current_epoch} - Validation Error Statistics (n={n_samples})")
            print(f"{'='*80}")

            # Print per-component table
            print(f"\n{'Per-Component Geodesic Errors:'}")
            print(f"{'Component':<20} {'Mean':>12} {'P50':>12} {'P90':>12} {'P99':>12}")
            print(f"{'-'*68}")

            for dim_idx in range(self._manifold_dist_dim):
                comp_name = component_names[dim_idx] if dim_idx < len(component_names) else f"dim_{dim_idx}"
                print(f"{comp_name:<20} {mean_per_dim[dim_idx]:>12.6f} {p50_per_dim[dim_idx]:>12.6f} "
                      f"{p90_per_dim[dim_idx]:>12.6f} {p99_per_dim[dim_idx]:>12.6f}")

            # Print overall state error (L2 norm)
            print(f"\n{'Overall State Error (L2 norm):'}")
            print(f"{'Statistic':<20} {'Value':>12}")
            print(f"{'-'*32}")
            print(f"{'Mean':<20} {overall_mean:>12.6f}")
            print(f"{'P50 (Median)':<20} {overall_p50:>12.6f}")
            print(f"{'P90':<20} {overall_p90:>12.6f}")
            print(f"{'P99':<20} {overall_p99:>12.6f}")
            print(f"{'='*80}\n")

            # Log to text file if specified
            if self.val_error_log_file is not None:
                import os
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.val_error_log_file), exist_ok=True) if os.path.dirname(self.val_error_log_file) else None

                # Write header if file doesn't exist
                write_header = not os.path.exists(self.val_error_log_file)

                with open(self.val_error_log_file, 'a') as f:
                    if write_header:
                        # Header: epoch, n_samples, per-component stats, overall stats
                        dim_headers = []
                        for dim_idx in range(self._manifold_dist_dim):
                            comp_name = component_names[dim_idx] if dim_idx < len(component_names) else f"dim_{dim_idx}"
                            for stat in ['mean', 'p50', 'p90', 'p99']:
                                dim_headers.append(f"{comp_name}_{stat}")
                        overall_headers = ['overall_mean', 'overall_p50', 'overall_p90', 'overall_p99']
                        header = '\t'.join(['epoch', 'n_samples'] + dim_headers + overall_headers)
                        f.write(header + '\n')

                    # Data row
                    dim_values = []
                    for dim_idx in range(self._manifold_dist_dim):
                        dim_values.extend([
                            f"{mean_per_dim[dim_idx].item():.6f}",
                            f"{p50_per_dim[dim_idx].item():.6f}",
                            f"{p90_per_dim[dim_idx].item():.6f}",
                            f"{p99_per_dim[dim_idx].item():.6f}",
                        ])
                    overall_values = [
                        f"{overall_mean:.6f}",
                        f"{overall_p50:.6f}",
                        f"{overall_p90:.6f}",
                        f"{overall_p99:.6f}",
                    ]
                    row = '\t'.join([str(self.current_epoch), str(n_samples)] + dim_values + overall_values)
                    f.write(row + '\n')

        # Clear the error buffer for next epoch
        self._val_abs_errors_buffer = []

        # Reset endpoint MAE metrics
        for metric in self.val_endpoint_mae_per_dim:
            metric.reset()

    # ===================================================================
    # COMPREHENSIVE EVALUATION WITH MANIFOLD DISTANCE
    # ===================================================================

    def compute_manifold_distance_per_component(
        self, predicted_endpoints: torch.Tensor, true_endpoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-component geodesic distance using distance_manifold.dist().

        Uses distance_manifold (true system manifold) for proper geodesic distances,
        even when training uses Euclidean manifold (use_manifold=False).

        For Quadrotor3D: Returns 10 values (SE3: 3 translation + 1 SO3 geodesic; + 6 Euclidean)
        For Pendulum: Returns 2 values (1 circular + 1 Euclidean)
        For CartPole: Returns 4 values

        Args:
            predicted_endpoints: [B, state_dim] predicted endpoints
            true_endpoints: [B, state_dim] true endpoints

        Returns:
            [B, num_components] geodesic distances per component
        """
        # Normalize states for manifold distance computation
        pred_normalized = self.normalize_state(predicted_endpoints)
        true_normalized = self.normalize_state(true_endpoints)

        # Use distance_manifold (true system manifold) for proper geodesic distances
        distances = self.distance_manifold.dist(pred_normalized, true_normalized)

        return distances

    def compute_grouped_euclidean_distances(
        self, predicted_endpoints: torch.Tensor, true_endpoints: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute grouped Euclidean (L2) distances for meaningful error metrics.

        Groups related state components and computes L2 norm for each group.
        For example, position (x,y,z) → single L2 distance instead of 3 separate values.

        Override get_euclidean_groups() in subclasses to define groupings.

        Args:
            predicted_endpoints: [B, state_dim] predicted endpoints
            true_endpoints: [B, state_dim] true endpoints

        Returns:
            Dictionary mapping group names to L2 distances [B]
        """
        groups = self.get_euclidean_groups()
        diff = predicted_endpoints - true_endpoints

        grouped_distances = {}
        for group_name, indices in groups.items():
            # Compute L2 norm for this group of dimensions
            group_diff = diff[:, indices]  # [B, len(indices)]
            l2_dist = torch.norm(group_diff, dim=1)  # [B]
            grouped_distances[group_name] = l2_dist

        return grouped_distances

    def get_euclidean_groups(self) -> Dict[str, list]:
        """
        Define groups of state dimensions for Euclidean distance computation.

        Override in subclasses to define system-specific groupings.
        Each group will have its L2 norm computed as a single metric.

        Returns:
            Dictionary mapping group names to lists of dimension indices
        """
        # Default: treat all dimensions as one group
        return {"state": list(range(self.system.state_dim))}

    def compute_rmse(
        self, predicted_endpoints: torch.Tensor, true_endpoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Root Mean Square Error across all dimensions.

        RMSE = sqrt(mean((pred - true)²))

        Args:
            predicted_endpoints: [B, state_dim]
            true_endpoints: [B, state_dim]

        Returns:
            RMSE per sample [B]
        """
        squared_diff = (predicted_endpoints - true_endpoints) ** 2
        mse_per_sample = squared_diff.mean(dim=1)  # Mean over dimensions
        rmse_per_sample = torch.sqrt(mse_per_sample)
        return rmse_per_sample

    def get_manifold_component_names(self) -> list:
        """
        Get names for each manifold distance component.

        Override in subclasses for system-specific names.

        Returns:
            List of component names matching manifold.dist() output
        """
        # Default implementation - subclasses should override
        num_components = sum(
            (
                1
                if hasattr(m, "dist") and not hasattr(m, "state_dims")
                else getattr(m, "state_dims", [1])[0] if hasattr(m, "state_dims") else 1
            )
            for m in self.manifold.manifolds
        )
        return [f"component_{i}" for i in range(num_components)]

    def evaluate_with_manifold_metrics(
        self,
        dataloader,
        num_steps: int = 100,
        device: Optional[torch.device] = None,
        attractor_radius: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation using proper manifold geodesic distances.

        Computes:
        - Per-component MAE using manifold.dist() (e.g., 10 metrics for Quadrotor3D)
        - CERTAIN/UNCERTAIN split based on whether PREDICTED endpoint is in attractor
        - Variance, mean, median statistics for each subset

        Args:
            dataloader: DataLoader with 'start_state' and 'end_state'
            num_steps: ODE integration steps
            device: Evaluation device
            attractor_radius: Radius for attractor classification

        Returns:
            Comprehensive evaluation dictionary with:
            - Per-component metrics (geodesic distances)
            - Full dataset statistics
            - CERTAIN subset statistics (predicted in attractor)
            - UNCERTAIN subset statistics (predicted not in attractor)
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # Collect all data
        all_predicted = []
        all_true = []
        all_start = []

        with torch.no_grad():
            for batch in dataloader:
                start_states = self._get_start_states(batch).to(device)
                true_endpoints = self._get_end_states(batch).to(device)

                # Predict endpoints
                predicted_endpoints = self.predict_endpoint(
                    start_states=start_states, num_steps=num_steps, latent=None
                )

                all_predicted.append(predicted_endpoints)
                all_true.append(true_endpoints)
                all_start.append(start_states)

        # Concatenate all batches
        all_predicted = torch.cat(all_predicted, dim=0)  # [N, state_dim]
        all_true = torch.cat(all_true, dim=0)  # [N, state_dim]
        all_start = torch.cat(all_start, dim=0)  # [N, state_dim]

        N = all_predicted.shape[0]

        # Compute manifold distances for all samples
        # Shape: [N, num_components] (e.g., [N, 10] for Quadrotor3D)
        manifold_distances = self.compute_manifold_distance_per_component(
            all_predicted, all_true
        )
        num_components = manifold_distances.shape[1]
        component_names = self.get_manifold_component_names()

        # Ensure we have the right number of names
        if len(component_names) != num_components:
            component_names = [f"component_{i}" for i in range(num_components)]

        # Classify predictions: CERTAIN if predicted endpoint is in attractor
        in_attractor = self.system.is_in_attractor(
            all_predicted, radius=attractor_radius
        )
        if isinstance(in_attractor, bool):
            in_attractor = torch.tensor([in_attractor], device=device)
        if not isinstance(in_attractor, torch.Tensor):
            in_attractor = torch.tensor(in_attractor, device=device)

        certain_mask = in_attractor  # CERTAIN: predicted endpoint in attractor
        uncertain_mask = ~certain_mask  # UNCERTAIN: predicted endpoint NOT in attractor

        n_certain = certain_mask.sum().item()
        n_uncertain = uncertain_mask.sum().item()

        # ===================================================================
        # Compute statistics for FULL dataset
        # ===================================================================
        full_stats = self._compute_subset_stats(
            manifold_distances, component_names, "full"
        )

        # ===================================================================
        # Compute statistics for CERTAIN subset
        # ===================================================================
        if n_certain > 0:
            certain_distances = manifold_distances[certain_mask]
            certain_stats = self._compute_subset_stats(
                certain_distances, component_names, "certain"
            )
        else:
            certain_stats = self._create_empty_stats(component_names, "certain")

        # ===================================================================
        # Compute statistics for UNCERTAIN subset
        # ===================================================================
        if n_uncertain > 0:
            uncertain_distances = manifold_distances[uncertain_mask]
            uncertain_stats = self._compute_subset_stats(
                uncertain_distances, component_names, "uncertain"
            )
        else:
            uncertain_stats = self._create_empty_stats(component_names, "uncertain")

        # ===================================================================
        # Aggregated metrics (mean over all components)
        # ===================================================================
        # Per-sample mean distance (aggregate over components)
        per_sample_mean_dist = manifold_distances.mean(dim=1)  # [N]

        aggregated_stats = {
            # Full dataset aggregated
            "aggregated_mean": per_sample_mean_dist.mean().item(),
            "aggregated_median": per_sample_mean_dist.median().item(),
            "aggregated_var": per_sample_mean_dist.var().item() if N > 1 else 0.0,
            "aggregated_std": per_sample_mean_dist.std().item() if N > 1 else 0.0,
        }

        # CERTAIN aggregated
        if n_certain > 0:
            certain_agg = per_sample_mean_dist[certain_mask]
            aggregated_stats["aggregated_certain_mean"] = certain_agg.mean().item()
            aggregated_stats["aggregated_certain_median"] = certain_agg.median().item()
            aggregated_stats["aggregated_certain_var"] = (
                certain_agg.var().item() if n_certain > 1 else 0.0
            )
            aggregated_stats["aggregated_certain_std"] = (
                certain_agg.std().item() if n_certain > 1 else 0.0
            )
        else:
            aggregated_stats["aggregated_certain_mean"] = float("nan")
            aggregated_stats["aggregated_certain_median"] = float("nan")
            aggregated_stats["aggregated_certain_var"] = float("nan")
            aggregated_stats["aggregated_certain_std"] = float("nan")

        # UNCERTAIN aggregated
        if n_uncertain > 0:
            uncertain_agg = per_sample_mean_dist[uncertain_mask]
            aggregated_stats["aggregated_uncertain_mean"] = uncertain_agg.mean().item()
            aggregated_stats["aggregated_uncertain_median"] = (
                uncertain_agg.median().item()
            )
            aggregated_stats["aggregated_uncertain_var"] = (
                uncertain_agg.var().item() if n_uncertain > 1 else 0.0
            )
            aggregated_stats["aggregated_uncertain_std"] = (
                uncertain_agg.std().item() if n_uncertain > 1 else 0.0
            )
        else:
            aggregated_stats["aggregated_uncertain_mean"] = float("nan")
            aggregated_stats["aggregated_uncertain_median"] = float("nan")
            aggregated_stats["aggregated_uncertain_var"] = float("nan")
            aggregated_stats["aggregated_uncertain_std"] = float("nan")

        # ===================================================================
        # Grouped Euclidean (L2) distances
        # ===================================================================
        grouped_distances = self.compute_grouped_euclidean_distances(
            all_predicted, all_true
        )
        group_names = list(grouped_distances.keys())

        # Compute statistics for each group
        euclidean_stats = {
            "group_names": group_names,
            "full": {},
            "certain": {},
            "uncertain": {},
        }

        for group_name, group_dist in grouped_distances.items():
            # Full dataset
            euclidean_stats["full"][group_name] = {
                "mean": group_dist.mean().item(),
                "median": group_dist.median().item(),
                "var": group_dist.var().item() if N > 1 else 0.0,
                "std": group_dist.std().item() if N > 1 else 0.0,
            }

            # CERTAIN subset
            if n_certain > 0:
                certain_group = group_dist[certain_mask]
                euclidean_stats["certain"][group_name] = {
                    "mean": certain_group.mean().item(),
                    "median": certain_group.median().item(),
                    "var": certain_group.var().item() if n_certain > 1 else 0.0,
                    "std": certain_group.std().item() if n_certain > 1 else 0.0,
                }
            else:
                euclidean_stats["certain"][group_name] = {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "var": float("nan"),
                    "std": float("nan"),
                }

            # UNCERTAIN subset
            if n_uncertain > 0:
                uncertain_group = group_dist[uncertain_mask]
                euclidean_stats["uncertain"][group_name] = {
                    "mean": uncertain_group.mean().item(),
                    "median": uncertain_group.median().item(),
                    "var": uncertain_group.var().item() if n_uncertain > 1 else 0.0,
                    "std": uncertain_group.std().item() if n_uncertain > 1 else 0.0,
                }
            else:
                euclidean_stats["uncertain"][group_name] = {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "var": float("nan"),
                    "std": float("nan"),
                }

        # ===================================================================
        # RMSE (Root Mean Square Error)
        # ===================================================================
        rmse_per_sample = self.compute_rmse(all_predicted, all_true)  # [N]

        rmse_stats = {
            # Full dataset
            "full_mean": rmse_per_sample.mean().item(),
            "full_median": rmse_per_sample.median().item(),
            "full_var": rmse_per_sample.var().item() if N > 1 else 0.0,
            "full_std": rmse_per_sample.std().item() if N > 1 else 0.0,
        }

        # CERTAIN RMSE
        if n_certain > 0:
            certain_rmse = rmse_per_sample[certain_mask]
            rmse_stats["certain_mean"] = certain_rmse.mean().item()
            rmse_stats["certain_median"] = certain_rmse.median().item()
            rmse_stats["certain_var"] = (
                certain_rmse.var().item() if n_certain > 1 else 0.0
            )
            rmse_stats["certain_std"] = (
                certain_rmse.std().item() if n_certain > 1 else 0.0
            )
        else:
            rmse_stats["certain_mean"] = float("nan")
            rmse_stats["certain_median"] = float("nan")
            rmse_stats["certain_var"] = float("nan")
            rmse_stats["certain_std"] = float("nan")

        # UNCERTAIN RMSE
        if n_uncertain > 0:
            uncertain_rmse = rmse_per_sample[uncertain_mask]
            rmse_stats["uncertain_mean"] = uncertain_rmse.mean().item()
            rmse_stats["uncertain_median"] = uncertain_rmse.median().item()
            rmse_stats["uncertain_var"] = (
                uncertain_rmse.var().item() if n_uncertain > 1 else 0.0
            )
            rmse_stats["uncertain_std"] = (
                uncertain_rmse.std().item() if n_uncertain > 1 else 0.0
            )
        else:
            rmse_stats["uncertain_mean"] = float("nan")
            rmse_stats["uncertain_median"] = float("nan")
            rmse_stats["uncertain_var"] = float("nan")
            rmse_stats["uncertain_std"] = float("nan")

        # ===================================================================
        # Compile final results
        # ===================================================================
        return {
            # Dataset info
            "num_samples": N,
            "num_certain": n_certain,
            "num_uncertain": n_uncertain,
            "certain_fraction": n_certain / N if N > 0 else 0.0,
            "uncertain_fraction": n_uncertain / N if N > 0 else 0.0,
            # Component names and count
            "num_components": num_components,
            "component_names": component_names,
            # Per-component statistics (from manifold.dist)
            "full": full_stats,
            "certain": certain_stats,
            "uncertain": uncertain_stats,
            # Aggregated statistics (mean over components)
            "aggregated": aggregated_stats,
            # Grouped Euclidean (L2) distances
            "euclidean": euclidean_stats,
            # RMSE statistics
            "rmse": rmse_stats,
            # Raw data for further analysis
            "raw_manifold_distances": manifold_distances.cpu(),
            "raw_per_sample_mean": per_sample_mean_dist.cpu(),
            "raw_rmse": rmse_per_sample.cpu(),
            "raw_grouped_euclidean": {k: v.cpu() for k, v in grouped_distances.items()},
            "certain_mask": certain_mask.cpu(),
            "attractor_radius": attractor_radius,
        }

    def _compute_subset_stats(
        self, distances: torch.Tensor, component_names: list, subset_name: str
    ) -> Dict[str, Any]:
        """
        Compute statistics for a subset of manifold distances.

        Args:
            distances: [N_subset, num_components] distances for subset
            component_names: List of component names
            subset_name: Name of subset ("full", "certain", "uncertain")

        Returns:
            Dictionary with per-component and aggregate statistics
        """
        N = distances.shape[0]

        stats = {
            "n_samples": N,
            "subset_name": subset_name,
        }

        # Per-component statistics
        per_component = {}
        for i, name in enumerate(component_names):
            comp_dist = distances[:, i]
            per_component[name] = {
                "mean": comp_dist.mean().item(),
                "median": comp_dist.median().item(),
                "var": comp_dist.var().item() if N > 1 else 0.0,
                "std": comp_dist.std().item() if N > 1 else 0.0,
                "min": comp_dist.min().item(),
                "max": comp_dist.max().item(),
            }

        stats["per_component"] = per_component

        # Create convenient arrays for printing
        stats["mae_per_component"] = [
            per_component[name]["mean"] for name in component_names
        ]
        stats["median_per_component"] = [
            per_component[name]["median"] for name in component_names
        ]
        stats["var_per_component"] = [
            per_component[name]["var"] for name in component_names
        ]
        stats["std_per_component"] = [
            per_component[name]["std"] for name in component_names
        ]

        return stats

    def _create_empty_stats(
        self, component_names: list, subset_name: str
    ) -> Dict[str, Any]:
        """Create empty statistics dictionary when subset has no samples."""
        stats = {
            "n_samples": 0,
            "subset_name": subset_name,
            "per_component": {
                name: {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "var": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
                for name in component_names
            },
            "mae_per_component": [float("nan")] * len(component_names),
            "median_per_component": [float("nan")] * len(component_names),
            "var_per_component": [float("nan")] * len(component_names),
            "std_per_component": [float("nan")] * len(component_names),
        }
        return stats

    def print_manifold_evaluation_report(self, metrics: Dict[str, Any]) -> None:
        """
        Print comprehensive evaluation report with manifold metrics.

        Args:
            metrics: Dictionary from evaluate_with_manifold_metrics()
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION REPORT (Manifold Geodesic Distances)")
        print("=" * 80)
        print(f"Total samples: {metrics['num_samples']}")
        print(
            f"  CERTAIN (predicted in attractor):   {metrics['num_certain']} ({metrics['certain_fraction']*100:.1f}%)"
        )
        print(
            f"  UNCERTAIN (predicted not in attractor): {metrics['num_uncertain']} ({metrics['uncertain_fraction']*100:.1f}%)"
        )
        print(f"Attractor radius: {metrics['attractor_radius']}")
        print(f"Number of manifold components: {metrics['num_components']}")
        print()

        # Per-component table header
        component_names = metrics["component_names"]
        print("-" * 80)
        print("PER-COMPONENT GEODESIC DISTANCES (MAE = Mean over samples)")
        print("-" * 80)
        print(
            f"{'Component':<20} {'FULL Mean':>12} {'CERTAIN Mean':>14} {'UNCERTAIN Mean':>16}"
        )
        print("-" * 80)

        full_stats = metrics["full"]
        certain_stats = metrics["certain"]
        uncertain_stats = metrics["uncertain"]

        for name in component_names:
            full_mae = full_stats["per_component"][name]["mean"]
            certain_mae = certain_stats["per_component"][name]["mean"]
            uncertain_mae = uncertain_stats["per_component"][name]["mean"]

            print(
                f"{name:<20} {full_mae:>12.6f} {certain_mae:>14.6f} {uncertain_mae:>16.6f}"
            )

        print("-" * 80)
        print()

        # Detailed statistics for each subset
        for subset_name, subset_stats in [
            ("FULL", full_stats),
            ("CERTAIN", certain_stats),
            ("UNCERTAIN", uncertain_stats),
        ]:
            print(f"\n{'='*40}")
            print(f"{subset_name} SUBSET (n={subset_stats['n_samples']})")
            print(f"{'='*40}")

            if subset_stats["n_samples"] == 0:
                print("  No samples in this subset")
                continue

            print(
                f"{'Component':<20} {'Mean':>10} {'Median':>10} {'Std':>10} {'Var':>12}"
            )
            print("-" * 62)

            for name in component_names:
                comp = subset_stats["per_component"][name]
                print(
                    f"{name:<20} {comp['mean']:>10.6f} {comp['median']:>10.6f} "
                    f"{comp['std']:>10.6f} {comp['var']:>12.8f}"
                )

        # Aggregated metrics
        agg = metrics["aggregated"]
        print(f"\n{'='*80}")
        print("AGGREGATED METRICS (Mean over all components per sample)")
        print("=" * 80)
        print(f"{'Subset':<20} {'Mean':>12} {'Median':>12} {'Std':>12} {'Var':>14}")
        print("-" * 70)
        print(
            f"{'FULL':<20} {agg['aggregated_mean']:>12.6f} {agg['aggregated_median']:>12.6f} "
            f"{agg['aggregated_std']:>12.6f} {agg['aggregated_var']:>14.8f}"
        )
        print(
            f"{'CERTAIN':<20} {agg['aggregated_certain_mean']:>12.6f} {agg['aggregated_certain_median']:>12.6f} "
            f"{agg['aggregated_certain_std']:>12.6f} {agg['aggregated_certain_var']:>14.8f}"
        )
        print(
            f"{'UNCERTAIN':<20} {agg['aggregated_uncertain_mean']:>12.6f} {agg['aggregated_uncertain_median']:>12.6f} "
            f"{agg['aggregated_uncertain_std']:>12.6f} {agg['aggregated_uncertain_var']:>14.8f}"
        )

        # Grouped Euclidean (L2) distances
        if "euclidean" in metrics:
            euc = metrics["euclidean"]
            print(f"\n{'='*80}")
            print("GROUPED EUCLIDEAN (L2) DISTANCES")
            print("=" * 80)
            print(
                f"{'Group':<25} {'FULL Mean':>12} {'CERTAIN Mean':>14} {'UNCERTAIN Mean':>16}"
            )
            print("-" * 70)

            for group_name in euc["group_names"]:
                full_mean = euc["full"][group_name]["mean"]
                certain_mean = euc["certain"][group_name]["mean"]
                uncertain_mean = euc["uncertain"][group_name]["mean"]
                print(
                    f"{group_name:<25} {full_mean:>12.6f} {certain_mean:>14.6f} {uncertain_mean:>16.6f}"
                )

            print("-" * 70)
            print("\nDetailed Euclidean statistics:")
            for subset_name in ["full", "certain", "uncertain"]:
                print(f"\n  {subset_name.upper()}:")
                for group_name in euc["group_names"]:
                    stats = euc[subset_name][group_name]
                    print(
                        f"    {group_name}: mean={stats['mean']:.6f}, median={stats['median']:.6f}, "
                        f"std={stats['std']:.6f}, var={stats['var']:.8f}"
                    )

        # RMSE statistics
        if "rmse" in metrics:
            rmse = metrics["rmse"]
            print(f"\n{'='*80}")
            print("RMSE (Root Mean Square Error)")
            print("=" * 80)
            print(f"{'Subset':<20} {'Mean':>12} {'Median':>12} {'Std':>12} {'Var':>14}")
            print("-" * 70)
            print(
                f"{'FULL':<20} {rmse['full_mean']:>12.6f} {rmse['full_median']:>12.6f} "
                f"{rmse['full_std']:>12.6f} {rmse['full_var']:>14.8f}"
            )
            print(
                f"{'CERTAIN':<20} {rmse['certain_mean']:>12.6f} {rmse['certain_median']:>12.6f} "
                f"{rmse['certain_std']:>12.6f} {rmse['certain_var']:>14.8f}"
            )
            print(
                f"{'UNCERTAIN':<20} {rmse['uncertain_mean']:>12.6f} {rmse['uncertain_median']:>12.6f} "
                f"{rmse['uncertain_std']:>12.6f} {rmse['uncertain_var']:>14.8f}"
            )

        print("=" * 80 + "\n")
