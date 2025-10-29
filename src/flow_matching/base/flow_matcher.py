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
import sys
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier/flow_matching')

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

    def __init__(self, model: nn.Module, latent: torch.Tensor,
                 condition: torch.Tensor, embed_fn):
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

    def __init__(self,
                 system,
                 model: nn.Module,
                 optimizer: Any,
                 scheduler: Any,
                 model_config: Optional[FlowMatchingConfig] = None,
                 latent_dim: int = 2,
                 mae_val_frequency: int = 10):
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
        """
        super().__init__()

        # Store system and model
        self.system = system
        self.model = model
        self.config = model_config or FlowMatchingConfig()
        self.latent_dim = latent_dim
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

    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                z: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
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

    def compute_endpoint_mae_per_dim(self,
                                    predicted_endpoints: torch.Tensor,
                                    true_endpoints: torch.Tensor) -> torch.Tensor:
        """
        Compute MAE per dimension using Facebook Product manifold geodesic distance

        The Product manifold correctly computes:
        - Geodesic (circular) distance for S¹ components (e.g., angles)
        - Euclidean distance for ℝ components (e.g., velocities)

        Example for pendulum (S¹×ℝ):
        - θ=3.1 vs θ=-3.1 → distance ≈ 0.08 (wraps around, not 6.2!)
        - θ̇=1.0 vs θ̇=2.0 → distance = 1.0 (Euclidean)

        Args:
            predicted_endpoints: Predicted endpoints [B, state_dim]
            true_endpoints: True endpoints [B, state_dim]

        Returns:
            mae_per_dim: MAE for each dimension [state_dim]
        """
        # Facebook Product manifold returns per-dimension distances
        # Shape: [batch_size, state_dim]
        mae = self.manifold.dist(predicted_endpoints, true_endpoints)
        # Average over batch → [state_dim]
        return mae.mean(dim=0)
    
    @abstractmethod
    def _create_manifold(self):
        """
        Create Facebook FM manifold for this system

        Returns:
            Product manifold (e.g., S¹×ℝ for pendulum, ℝ²×S¹×ℝ for cartpole)
        """
        pass

    @abstractmethod
    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noisy input in system's state space

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Noisy states [batch_size, state_dim]
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

    def _prepare_model_inputs(self,
                              batch_size: int,
                              start_states: torch.Tensor,
                              latent: Optional[torch.Tensor] = None,
                              device: Optional[torch.device] = None):
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

        # Sample and normalize noisy inputs
        x_noise = self.sample_noisy_input(batch_size, device)
        x_noise_normalized = self.normalize_state(x_noise)

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
        x_noise_normalized, z, start_normalized_embedded, _ = self._prepare_model_inputs(
            batch_size=batch_size,
            start_states=start_states,
            latent=None,  # Always sample fresh latents for training
            device=device
        )

        # Sample random times
        t = torch.rand(batch_size, device=device)

        # Normalize data endpoints
        data_normalized = self.normalize_state(data_endpoints)

        # Use Facebook FM GeodesicProbPath for geodesic interpolation
        path_sample = self.path.sample(
            x_0=x_noise_normalized,    # Normalized noise
            x_1=data_normalized,        # Normalized target endpoints
            t=t                         # Random times
        )

        # Embed interpolated state for neural network input
        x_t_embedded = self.embed_state_for_model(path_sample.x_t)

        # Predict velocity using the model
        predicted_velocity = self.forward(x_t_embedded, t, z, condition=start_normalized_embedded)

        # Use automatic target velocity from path.sample()
        target_velocity = path_sample.dx_t

        # Compute MSE loss between predicted and target velocities
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)

        return loss

    def predict_endpoint(self,
                        start_states: torch.Tensor,
                        num_steps: int = 100,
                        latent: Optional[torch.Tensor] = None,
                        method: str = "euler") -> torch.Tensor:
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
            method: Integration method ("euler", "rk4", "midpoint")

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
                x_noise_normalized, z, start_normalized_embedded, start_normalized = self._prepare_model_inputs(
                    batch_size=batch_size,
                    start_states=start_states,
                    latent=latent,  # Use provided or sample
                    device=device
                )

                # Create model wrapper for RiemannianODESolver
                velocity_model = LatentConditionalVelocityWrapper(
                    model=self.model,
                    latent=z,
                    condition=start_normalized_embedded,
                    embed_fn=self.embed_state_for_model
                )

                # Use RiemannianODESolver for integration
                solver = RiemannianODESolver(
                    manifold=self.manifold,
                    velocity_model=velocity_model
                )

                final_states_normalized = solver.sample(
                    x_init=x_noise_normalized,
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
    
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - common across all variants"""
        loss = self.compute_flow_loss(batch)
        
        # Log metrics with compatibility for different TorchMetrics versions
        try:
            # TorchMetrics >= 1.2 style
            self.train_loss(loss)
        except Exception:
            # Fallback for older TorchMetrics versions
            self.train_loss.update(loss)
        
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with optional endpoint MAE computation"""
        loss = self.compute_flow_loss(batch)

        # Log metrics with compatibility for different TorchMetrics versions
        try:
            # TorchMetrics >= 1.2 style
            self.val_loss(loss)
        except Exception:
            # Fallback for older TorchMetrics versions
            self.val_loss.update(loss)

        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute endpoint MAE every N epochs
        if self.current_epoch % self.mae_val_frequency == 0:
            # Extract states from batch
            start_states = self._get_start_states(batch)
            true_endpoints = self._get_end_states(batch)

            # Predict endpoints
            with torch.no_grad():
                predicted_endpoints = self.predict_endpoint(
                    start_states=start_states,
                    num_steps=100,
                    latent=None
                )

            # Compute MAE per dimension/component
            # Note: For Product manifolds, this returns per-component distances
            # (e.g., 65 for Humanoid: 34 Euclidean + 1 Sphere + 30 Euclidean)
            mae_per_dim = self.compute_endpoint_mae_per_dim(predicted_endpoints, true_endpoints)

            # Update metrics - iterate over actual number of distance components
            num_components = len(mae_per_dim)
            for dim_idx in range(num_components):
                try:
                    self.val_endpoint_mae_per_dim[dim_idx](mae_per_dim[dim_idx])
                except Exception:
                    self.val_endpoint_mae_per_dim[dim_idx].update(mae_per_dim[dim_idx])

                # Log individual dimension MAE
                self.log(f'val_endpoint_mae_{self._get_dimension_name(dim_idx)}',
                        self.val_endpoint_mae_per_dim[dim_idx],
                        on_step=False, on_epoch=True, prog_bar=False)

        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Instantiate optimizer from config
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())

        # Handle scheduler if present
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

        # Return just the optimizer if no scheduler
        return optimizer
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Log epoch metrics
        self.log('train_loss_epoch', self.train_loss.compute())
        self.train_loss.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Log epoch metrics
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