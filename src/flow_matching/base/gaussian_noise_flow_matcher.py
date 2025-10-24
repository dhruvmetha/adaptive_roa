"""
Base class for Gaussian Noise Flow Matching (system-agnostic)

Provides common functionality for Gaussian-perturbed flow matching WITHOUT:
- Latent variables z
- Conditioning on start state

WITH:
- Gaussian-perturbed initial states: x₀ ~ N(start_state, σ²I)
- Simplified model: f(x_t, t) → velocity
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import lightning.pytorch as pl
from torchmetrics import MeanMetric
from abc import ABC, abstractmethod
import sys
import hydra
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier/flow_matching')

from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper

from src.systems.base import DynamicalSystem


class SimpleVelocityWrapper(ModelWrapper):
    """
    Simplified velocity wrapper for models WITHOUT latent variables or conditioning

    Facebook FM solvers expect: velocity_model(x, t) → velocity
    Our simplified model provides: model(x_embedded, t) → velocity

    This wrapper handles:
    - State embedding via embed_fn
    - NO latent broadcasting (removed!)
    - NO condition broadcasting (removed!)
    """

    def __init__(self, model: nn.Module, embed_fn):
        """
        Args:
            model: The neural network (simplified UNet)
            embed_fn: Function to embed state for model input
        """
        super().__init__(model)
        self.embed_fn = embed_fn

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        """
        Forward pass compatible with RiemannianODESolver

        SIMPLIFIED: No latent, no condition handling!

        Args:
            x: Current state [B, state_dim] (normalized)
            t: Time [B] or scalar

        Returns:
            Velocity [B, state_dim] in tangent space
        """
        # Handle scalar t (expand to batch)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # Embed state for neural network
        x_embedded = self.embed_fn(x)

        # Call the simplified model (no latent, no condition!)
        velocity = self.model(x_embedded, t)

        return velocity


class BaseGaussianNoiseFlowMatcher(pl.LightningModule, ABC):
    """
    Base class for Gaussian Noise Flow Matching (system-agnostic)

    Provides common functionality for:
    - Gaussian perturbation around start states
    - Training and validation loops
    - MAE per-dimension tracking
    - Facebook Flow Matching integration (GeodesicProbPath)

    Subclasses must implement system-specific methods for:
    - Manifold creation (based on system.manifold_components)
    - Dimension naming (using system.manifold_components[idx].name)
    """

    def __init__(self,
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 model_config: Optional[dict] = None,
                 noise_std: float = 0.1,
                 mae_val_frequency: int = 10):
        """
        Initialize Gaussian Noise flow matcher

        Args:
            system: DynamicalSystem instance (any system with manifold_components)
            model: Neural network model (simplified UNet without latent/condition)
            optimizer: Optimizer configuration
            scheduler: LR scheduler configuration
            model_config: Flow matching configuration
            noise_std: Standard deviation of Gaussian perturbation around start state
            mae_val_frequency: Compute endpoint MAE every N epochs
        """
        super().__init__()

        # Store system and model
        self.system = system
        self.model = model
        self.config = model_config or {}
        self.noise_std = noise_std
        self.mae_val_frequency = mae_val_frequency

        # Store optimizer and scheduler configs (will be instantiated in configure_optimizers)
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        # Metrics tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # MAE metrics per dimension for endpoint prediction
        self.val_endpoint_mae_per_dim = nn.ModuleList([
            MeanMetric() for _ in range(self.system.state_dim)
        ])

        # Facebook FM components (subclass creates manifold)
        self.manifold = self._create_manifold()
        self.path = GeodesicProbPath(
            scheduler=CondOTScheduler(),
            manifold=self.manifold
        )

        # Save hyperparameters (exclude model and optimizer/scheduler to avoid pickle issues)
        self.save_hyperparameters(ignore=['model', 'optimizer', 'scheduler', 'system'])

    def sample_perturbed_input(self, start_states: torch.Tensor) -> torch.Tensor:
        """
        Sample perturbed initial states from Gaussian centered at start states

        SYSTEM-AGNOSTIC: Uses system.get_circular_indices() to wrap angles correctly

        Args:
            start_states: Start states [B, state_dim] (normalized)

        Returns:
            Perturbed states [B, state_dim] sampled from Gaussian
        """
        # Sample Gaussian noise
        noise = torch.randn_like(start_states) * self.noise_std

        # Add noise to start state (in normalized space for numerical stability)
        perturbed = start_states + noise

        # Project circular components back onto manifold (wrap angles to [-π, π])
        # SYSTEM-AGNOSTIC: Get circular indices dynamically from system
        circular_indices = self.system.get_circular_indices()
        for idx in circular_indices:
            perturbed[:, idx] = torch.atan2(torch.sin(perturbed[:, idx]), torch.cos(perturbed[:, idx]))

        return perturbed

    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute Gaussian-perturbed flow matching loss

        UNIFIED implementation for all systems using:
        - Gaussian perturbation instead of latent variables
        - GeodesicProbPath for interpolation (automatic geodesics!)
        - path_sample.dx_t for target velocity (automatic via autodiff!)

        Args:
            batch: Dictionary containing 'start_state' and 'end_state'

        Returns:
            Flow matching loss
        """
        # Extract data
        start_states = batch["start_state"]
        end_states = batch["end_state"]

        batch_size = start_states.shape[0]
        device = self.device

        # Normalize states
        start_normalized = self.normalize_state(start_states)
        end_normalized = self.normalize_state(end_states)

        # Sample perturbed inputs from Gaussian centered at start state
        # KEY CHANGE: x₀ ~ N(start_state, σ²I) instead of uniform sampling
        x_0 = self.sample_perturbed_input(start_normalized)

        # Sample random times
        t = torch.rand(batch_size, device=device)

        # Use Facebook FM GeodesicProbPath for geodesic interpolation
        path_sample = self.path.sample(
            x_0=x_0,              # Perturbed start state
            x_1=end_normalized,   # Target endpoint
            t=t                   # Random times
        )

        # Embed interpolated state for neural network input
        x_t_embedded = self.embed_state_for_model(path_sample.x_t)

        # Predict velocity using the SIMPLIFIED model (no latent, no condition!)
        predicted_velocity = self.model(x_t_embedded, t)

        # Use automatic target velocity from path.sample()
        target_velocity = path_sample.dx_t

        # Compute MSE loss
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)

        return loss

    def predict_endpoint(self,
                        start_states: torch.Tensor,
                        num_steps: int = 100,
                        method: str = "euler") -> torch.Tensor:
        """
        Predict endpoints from start states using Gaussian perturbation

        UNIFIED implementation for all systems

        Args:
            start_states: Start states [B, state_dim] in raw coordinates
            num_steps: Number of integration steps for ODE solving
            method: Integration method ("euler", "rk4", "midpoint")

        Returns:
            Predicted endpoints [B, state_dim] in raw coordinates
        """
        batch_size = start_states.shape[0]
        device = start_states.device

        # Ensure model is in eval mode
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                # Normalize start states
                start_normalized = self.normalize_state(start_states)

                # Sample perturbed input from Gaussian
                x_0 = self.sample_perturbed_input(start_normalized)

                # Create SIMPLIFIED model wrapper (no latent, no condition)
                velocity_model = SimpleVelocityWrapper(
                    model=self.model,
                    embed_fn=self.embed_state_for_model
                )

                # Use RiemannianODESolver for integration
                solver = RiemannianODESolver(
                    manifold=self.manifold,
                    velocity_model=velocity_model
                )

                final_states_normalized = solver.sample(
                    x_init=x_0,
                    step_size=1.0/num_steps,
                    method=method,
                    projx=True,   # Use manifold projection (wraps angles)
                    proju=True,   # Use tangent projection
                    time_grid=torch.tensor([0.0, 1.0], device=device)
                )

                # Denormalize back to raw coordinates
                final_states_raw = self.denormalize_state(final_states_normalized)

                return final_states_raw

        finally:
            # Restore original training mode
            if was_training:
                self.train()

    def predict_endpoints_batch(self,
                               start_states: torch.Tensor,
                               num_steps: int = 100,
                               num_samples: int = 1) -> torch.Tensor:
        """
        Predict multiple endpoint samples per start state

        Stochasticity comes from Gaussian perturbation, NOT latent variables

        Args:
            start_states: Start states [B, state_dim] in raw coordinates
            num_steps: Number of integration steps
            num_samples: Number of samples per start state

        Returns:
            Predicted endpoints [B*num_samples, state_dim] in raw coordinates
        """
        if num_samples == 1:
            return self.predict_endpoint(start_states, num_steps)

        all_endpoints = []

        for _ in range(num_samples):
            # Each call samples different Gaussian noise
            endpoints_raw = self.predict_endpoint(start_states, num_steps)
            all_endpoints.append(endpoints_raw)

        # Concatenate all samples: [B*num_samples, state_dim]
        return torch.cat(all_endpoints, dim=0)

    # ===================================================================
    # Abstract methods - System-specific
    # ===================================================================

    @abstractmethod
    def _create_manifold(self):
        """
        Create Facebook FM manifold for this system

        Should build manifold from system.manifold_components

        Returns:
            Product manifold (e.g., S¹×ℝ for pendulum, ℝ²×S¹×ℝ for cartpole)
        """
        pass

    @abstractmethod
    def _get_dimension_name(self, dim_idx: int) -> str:
        """
        Get human-readable name for dimension

        Should use system.manifold_components[idx].name

        Args:
            dim_idx: Dimension index

        Returns:
            Dimension name (e.g., "angle", "cart_position")
        """
        pass

    # ===================================================================
    # System delegate methods
    # ===================================================================

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for normalization"""
        return self.system.normalize_state(state)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for denormalization"""
        return self.system.denormalize_state(normalized_state)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for embedding"""
        return self.system.embed_state_for_model(normalized_state)

    # ===================================================================
    # Training and validation (UNIFIED)
    # ===================================================================

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - common across all variants"""
        loss = self.compute_flow_loss(batch)

        try:
            self.train_loss(loss)
        except Exception:
            self.train_loss.update(loss)

        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with optional endpoint MAE computation"""
        loss = self.compute_flow_loss(batch)

        try:
            self.val_loss(loss)
        except Exception:
            self.val_loss.update(loss)

        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute endpoint MAE every N epochs
        if self.current_epoch % self.mae_val_frequency == 0:
            start_states = batch["start_state"]
            true_endpoints = batch["end_state"]

            with torch.no_grad():
                predicted_endpoints = self.predict_endpoint(
                    start_states=start_states,
                    num_steps=100
                )

            # Compute MAE per dimension using manifold distance
            mae_per_dim = self.manifold.dist(predicted_endpoints, true_endpoints).mean(dim=0)

            # Update metrics
            for dim_idx in range(self.system.state_dim):
                try:
                    self.val_endpoint_mae_per_dim[dim_idx](mae_per_dim[dim_idx])
                except Exception:
                    self.val_endpoint_mae_per_dim[dim_idx].update(mae_per_dim[dim_idx])

                # Log individual dimension MAE
                dim_name = self._get_dimension_name(dim_idx)
                self.log(f'val_endpoint_mae_{dim_name}',
                        self.val_endpoint_mae_per_dim[dim_idx],
                        on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())

        if self.scheduler_config is not None:
            scheduler = hydra.utils.instantiate(self.scheduler_config, optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }

        return optimizer

    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        self.log('train_loss_epoch', self.train_loss.compute())
        self.train_loss.reset()

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        val_loss = self.val_loss.compute()
        self.log('val_loss_epoch', val_loss)
        self.val_loss.reset()

        # Print MAE per dimension every N epochs
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
