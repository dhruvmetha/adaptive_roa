"""
CartPole Latent Conditional Flow Matching implementation using Facebook Flow Matching library

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
from ...utils.fb_manifolds import CartPoleManifold

from flow_matching.utils.manifolds import Product, FlatTorus, Euclidean


class CartPoleLatentConditionalFlowMatcher(BaseFlowMatcher):
    """
    CartPole Latent Conditional Flow Matching using Facebook FM:
    - Uses GeodesicProbPath for geodesic interpolation on ℝ²×S¹×ℝ
    - Uses RiemannianODESolver for manifold-aware ODE integration
    - Neural net takes embedded x_t, time t, latent z, and start state condition
    - Predicts velocity in ℝ²×S¹×ℝ tangent space (x, θ, ẋ, θ̇)

    KEY CHANGES FROM ORIGINAL:
    - ✅ Removed: manual interpolate_r2_s1_r() → uses GeodesicProbPath
    - ✅ Removed: manual compute_target_velocity_r2_s1_r() → automatic via path.sample()
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
        Initialize CartPole latent conditional flow matcher with FB FM integration

        Args:
            system: DynamicalSystem (CartPole with ℝ²×S¹×ℝ structure)
            model: CartPoleLatentConditionalUNet1D model
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

        # Create manifold for ℝ²×S¹×ℝ (CartPole state space)
        self.manifold = Product(input_dim=4, manifolds=[(Euclidean(), 1), (FlatTorus(), 1), (Euclidean(), 2)])

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
        print(f"   - Manifold: CartPoleManifold (ℝ²×S¹×ℝ)")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Latent dim: {latent_dim}")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noisy input in ℝ²×S¹×ℝ space

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Noisy states [batch_size, 4] as (x, θ, ẋ, θ̇)
        """
        # x ~ Uniform[-cart_limit, +cart_limit] (symmetric bounds)
        x = torch.randn(batch_size, 1, device=device) * (self.system.cart_limit) # * 2) - self.system.cart_limit

        # ẋ ~ Uniform[-velocity_limit, +velocity_limit] (symmetric bounds)
        x_dot = torch.randn(batch_size, 1, device=device) * (self.system.velocity_limit) # * 2) - self.system.velocity_limit

        # θ ~ Uniform[-π, π] (wrapped angle)
        theta = torch.randn(batch_size, 1, device=device) * torch.pi # 2  - torch.pi

        # θ̇ ~ Uniform[-angular_velocity_limit, +angular_velocity_limit] (symmetric bounds)
        theta_dot = torch.randn(batch_size, 1, device=device) * (self.system.angular_velocity_limit) # * 2) - self.system.angular_velocity_limit

        return torch.cat([x, theta, x_dot, theta_dot], dim=1)

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

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw state coordinates (x, theta, x_dot, theta_dot) → (x_norm, theta, x_dot_norm, theta_dot_norm)

        Args:
            state: [B, 4] raw cartpole state (theta already wrapped to [-π, π])

        Returns:
            [B, 4] normalized state (theta unchanged)
        """
        # Extract components
        x = state[:, 0]
        theta = state[:, 1]  # Keep as-is (already wrapped)
        x_dot = state[:, 2]
        theta_dot = state[:, 3]

        # Normalize linear quantities to [-1, 1] range using symmetric bounds
        x_norm = x / self.system.cart_limit
        x_dot_norm = x_dot / self.system.velocity_limit
        theta_dot_norm = theta_dot / self.system.angular_velocity_limit

        return torch.stack([x_norm, theta, x_dot_norm, theta_dot_norm], dim=1)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state back to raw coordinates.

        Args:
            normalized_state: [B, 4] normalized state (x_norm, theta, x_dot_norm, theta_dot_norm)

        Returns:
            [B, 4] raw state (x, theta, x_dot, theta_dot)
        """
        x_norm = normalized_state[:, 0]
        theta = normalized_state[:, 1]  # Already in natural coordinates [-π, π]
        x_dot_norm = normalized_state[:, 2]
        theta_dot_norm = normalized_state[:, 3]

        # Denormalize using system bounds
        x = x_norm * self.system.cart_limit
        x_dot = x_dot_norm * self.system.velocity_limit
        theta_dot = theta_dot_norm * self.system.angular_velocity_limit

        return torch.stack([x, theta, x_dot, theta_dot], dim=1)

    def embed_normalized_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Embed normalized state → (x_norm, sin(theta), cos(theta), x_dot_norm, theta_dot_norm)

        Args:
            normalized_state: [B, 4] normalized state

        Returns:
            [B, 5] embedded state
        """
        x_norm = normalized_state[:, 0]
        theta = normalized_state[:, 1]
        x_dot_norm = normalized_state[:, 2]
        theta_dot_norm = normalized_state[:, 3]

        # Embed circular angle as sin/cos
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        return torch.stack([x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm], dim=1)

    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute CartPole latent conditional flow matching loss using Facebook FM

        CHANGES FROM ORIGINAL:
        - Uses GeodesicProbPath for interpolation (automatic geodesics!)
        - Uses path_sample.dx_t for target velocity (automatic via autodiff!)
        - No manual Theseus computation needed

        Args:
            batch: Dictionary containing 'raw_start_state' and 'raw_end_state' in format [x, θ, ẋ, θ̇]

        Returns:
            Flow matching loss
        """
        # Extract data endpoints and start states (raw ℝ²×S¹×ℝ format)
        start_states = batch["raw_start_state"]  # [B, 4] (x, θ, ẋ, θ̇)
        data_endpoints = batch["raw_end_state"]  # [B, 4] (x, θ, ẋ, θ̇)

        batch_size = start_states.shape[0]
        device = self.device

        # Sample noisy inputs in ℝ²×S¹×ℝ
        x_noise = self.sample_noisy_input(batch_size, device)

        # Sample random times
        t = torch.rand(batch_size, device=device)

        # Sample latent vectors
        z = self.sample_latent(batch_size, device)

        # Normalize states for the model
        x_noise_normalized = self.normalize_state(x_noise)
        data_normalized = self.normalize_state(data_endpoints)
        start_normalized = self.normalize_state(start_states)

        # ===================================================================
        # NEW: Use Facebook FM GeodesicProbPath
        # ===================================================================
        # This replaces:
        # - interpolate_r2_s1_r(x_noise, data_endpoints, t)
        # - compute_target_velocity_r2_s1_r(x_noise_normalized, data_normalized, t)

        path_sample = self.path.sample(
            x_0=x_noise_normalized,    # [B, 4] noise in normalized ℝ²×S¹×ℝ
            x_1=data_normalized,       # [B, 4] target endpoints
            t=t                        # [B] random times
        )

        # path_sample contains:
        # - path_sample.x_t: [B, 4] interpolated state (geodesic on ℝ²×S¹×ℝ)
        # - path_sample.dx_t: [B, 4] target velocity (computed via autodiff!)
        # - path_sample.x_0, path_sample.x_1, path_sample.t

        # ===================================================================
        # SAME AS BEFORE: Model prediction
        # ===================================================================

        # Embed interpolated state for neural network input
        x_t_embedded = self.embed_normalized_state(path_sample.x_t)  # [B, 4] → [B, 5]

        # Embed start state for conditioning
        start_embedded = self.embed_normalized_state(start_normalized)  # [B, 4] → [B, 5]

        # Predict velocity using the model (YOUR architecture - unchanged!)
        predicted_velocity = self.forward(x_t_embedded, t, z, condition=start_embedded)

        # ===================================================================
        # NEW: Use automatic target velocity from path.sample()
        # ===================================================================
        # This replaces manual Theseus computation!

        target_velocity = path_sample.dx_t  # [B, 4] automatic geodesic velocity!

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
        
        mae = self.manifold.dist(predicted_endpoints, true_endpoints) # [B, 4]
        return mae.mean(dim=0) # [4]

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
            start_states = batch["raw_start_state"]  # [B, state_dim]
            true_endpoints = batch["raw_end_state"]  # [B, state_dim]

            # Predict endpoints by integrating the flow
            _, predicted_endpoints = self.predict_endpoint(
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
            x_t: Embedded interpolated state [B, 5]
            t: Time parameter [B]
            z: Latent vector [B, latent_dim]
            condition: Embedded start state [B, 5]

        Returns:
            Predicted velocity [B, 4] in tangent space
        """
        return self.model(x_t, t, z, condition)

    # ===================================================================
    # NEW: INFERENCE METHODS using RiemannianODESolver
    # ===================================================================

    def predict_endpoint(self,
                        start_states: torch.Tensor,
                        num_steps: int = 100,
                        latent: Optional[torch.Tensor] = None,
                        method: str = "euler") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict endpoints from start states using Facebook FM's RiemannianODESolver

        NEW METHOD - Uses proper ODE integration with manifold projection

        Args:
            start_states: Start states [B, 4] in raw coordinates (x, θ, ẋ, θ̇)
            num_steps: Number of integration steps for ODE solving
            latent: Optional latent vectors [B, latent_dim]. If None, will sample.
            method: Integration method ("euler", "rk4", "midpoint")

        Returns:
            Tuple of (normalized_endpoints [B, 4], raw_endpoints [B, 4])
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
                x_noise_normalized = self.normalize_state(x_noise)

                # Sample or use provided latent vectors
                if latent is None:
                    z = torch.randn(batch_size, self.latent_dim, device=device)
                else:
                    z = latent

                # Normalize and embed start states for conditioning
                start_normalized = self.normalize_state(start_states)
                start_embedded = self.embed_normalized_state(start_normalized)

                # ===================================================================
                # NEW: Create model wrapper for RiemannianODESolver
                # ===================================================================

                velocity_model = _CartPoleVelocityModelWrapper(
                    model=self.model,
                    latent=z,
                    condition=start_embedded,
                    embed_fn=lambda x: self.embed_normalized_state(x)
                )

                # ===================================================================
                # NEW: Use RiemannianODESolver for integration
                # ===================================================================

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

                return final_states_normalized, final_states_raw

        finally:
            # Restore original training mode
            if was_training:
                self.train()

    def predict_endpoints_batch(self,
                               start_states: torch.Tensor,
                               num_steps: int = 100,
                               num_samples: int = 1) -> torch.Tensor:
        """
        Predict multiple endpoint samples per start state (for stochastic models).

        Args:
            start_states: Start states [B, 4] in raw coordinates
            num_steps: Number of integration steps
            num_samples: Number of samples per start state

        Returns:
            Predicted endpoints [B*num_samples, 4] in raw coordinates
        """
        if num_samples == 1:
            _, raw_endpoints = self.predict_endpoint(start_states, num_steps)
            return raw_endpoints

        batch_size = start_states.shape[0]
        all_endpoints = []

        for _ in range(num_samples):
            # Sample different latent vectors for each sample
            _, endpoints_raw = self.predict_endpoint(start_states, num_steps, latent=None)
            all_endpoints.append(endpoints_raw)

        # Concatenate all samples: [B*num_samples, 4]
        return torch.cat(all_endpoints, dim=0)

    # ===================================================================
    # CHECKPOINT LOADING FOR INFERENCE
    # ===================================================================

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained CartPole LCFM model from checkpoint for inference.

        Args:
            checkpoint_path: Path to Lightning checkpoint file (.ckpt)
            device: Device to load model on ("cuda", "cpu", or None for auto)

        Returns:
            Loaded model ready for inference
        """
        import torch
        import yaml
        from pathlib import Path
        from omegaconf import OmegaConf
        from ...systems.cartpole_lcfm import CartPoleSystemLCFM
        from ...model.cartpole_latent_conditional_unet1d import CartPoleLatentConditionalUNet1D

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = Path(checkpoint_path)
        print(f"🤖 Loading CartPole LCFM checkpoint: {checkpoint_path}")
        print(f"📍 Device: {device}")

        # Verify checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Find the training directory
        if checkpoint_path.parent.name == "checkpoints":
            training_dir = checkpoint_path.parent.parent
        else:
            training_dir = checkpoint_path.parent

        print(f"🗂️  Training directory: {training_dir}")

        # Load Hydra config
        hydra_config = None
        hydra_config_path = training_dir / ".hydra" / "config.yaml"

        if hydra_config_path.exists():
            try:
                print(f"📋 Loading Hydra config: {hydra_config_path}")
                with open(hydra_config_path, 'r') as f:
                    hydra_config = yaml.safe_load(f)
                print("✅ Hydra config loaded successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not load Hydra config: {e}")
                hydra_config = None
        else:
            print(f"⚠️  Hydra config not found at: {hydra_config_path}")

        # Load Lightning checkpoint
        print(f"📦 Loading Lightning checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})
        print("✅ Lightning checkpoint loaded")

        # Extract latent_dim
        latent_dim = hparams.get("latent_dim")
        if latent_dim is None and hydra_config:
            latent_dim = hydra_config.get("flow_matching", {}).get("latent_dim", 2)
        if latent_dim is None:
            latent_dim = 2
            print(f"⚠️  Using default latent_dim: {latent_dim}")

        # Extract model config
        model_config = hparams.get("config", {})
        if not model_config and hydra_config:
            model_config = hydra_config.get("model", {})

        model_config["latent_dim"] = latent_dim

        print(f"📋 Final config - latent_dim: {latent_dim}")
        print(f"📋 Model config keys: {list(model_config.keys())}")

        # Initialize system and model
        system = hparams.get("system")
        if system is None:
            print("🔧 Creating new CartPole system (not found in hparams)")
            system = CartPoleSystemLCFM()
        else:
            print("✅ Restored CartPole system from checkpoint")

        # Create model architecture
        model = CartPoleLatentConditionalUNet1D(
            embedded_dim=model_config.get('embedded_dim', 5),
            latent_dim=model_config.get('latent_dim', latent_dim),
            condition_dim=model_config.get('condition_dim', 5),
            time_emb_dim=model_config.get('time_emb_dim', 64),
            hidden_dims=model_config.get('hidden_dims', [256, 512, 256]),
            output_dim=model_config.get('output_dim', 4),
            use_input_embeddings=model_config.get('use_input_embeddings', False),
            input_emb_dim=model_config.get('input_emb_dim', 64)
        )

        # Create flow matcher instance
        flow_matcher = cls(
            system=system,
            model=model,
            optimizer=None,
            scheduler=None,
            config=model_config,
            latent_dim=latent_dim
        )

        # Load model weights
        print("🔄 Loading model state dict...")
        state_dict = checkpoint["state_dict"]
        model_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}

        if not model_state_dict:
            raise ValueError("No model weights found in checkpoint! Keys: " + str(list(state_dict.keys())[:10]))

        flow_matcher.model.load_state_dict(model_state_dict)

        # Move to device and set eval mode
        flow_matcher = flow_matcher.to(device)
        flow_matcher.eval()

        # Success summary
        print(f"\n✅ Model loaded successfully!")
        print(f"   Checkpoint: {checkpoint_path.name}")
        print(f"   Config sources: {'Hydra + Lightning' if hydra_config else 'Lightning only'}")
        print(f"   System: {type(system).__name__}")
        print(f"   System bounds: cart±{system.cart_limit:.1f}, vel±{system.velocity_limit:.1f}")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Model architecture: {model_config.get('hidden_dims', 'unknown')}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")

        return flow_matcher


# ============================================================================
# MODEL WRAPPER for RiemannianODESolver
# ============================================================================

class _CartPoleVelocityModelWrapper(ModelWrapper):
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
            embed_fn: Function to embed state: (x, θ, ẋ, θ̇) → (x_norm, sin θ, cos θ, ẋ_norm, θ̇_norm)
        """
        super().__init__(model)
        self.latent = latent
        self.condition = condition
        self.embed_fn = embed_fn

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        """
        Forward pass compatible with RiemannianODESolver

        Args:
            x: Current state [B, 4] normalized format (x_norm, θ, ẋ_norm, θ̇_norm)
            t: Time [B] or scalar

        Returns:
            Velocity [B, 4] in tangent space
        """
        # Handle scalar t (expand to batch)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # Embed state for neural network
        x_embedded = self.embed_fn(x)  # [B, 4] → [B, 5]

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
