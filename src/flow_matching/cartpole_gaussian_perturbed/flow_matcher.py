"""
CartPole Gaussian-Perturbed Flow Matching implementation

KEY DIFFERENCES from Latent Conditional variant:
1. NO latent variables (removed z ~ N(0,I))
2. NO conditioning on start state
3. Initial noise sampled from Gaussian centered at start state: x₀ ~ N(start_state, σ²I)
4. Simplified model signature: f(x_t, t) instead of f(x_t, t, z, condition)
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import lightning.pytorch as pl
from torchmetrics import MeanMetric
import sys
import hydra
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier/flow_matching')

from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Product, FlatTorus, Euclidean

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


class CartPoleGaussianPerturbedFlowMatcher(pl.LightningModule):
    """
    CartPole Gaussian-Perturbed Flow Matching using Facebook FM

    Simplified flow matching WITHOUT:
    - Latent variables z
    - Conditioning on start state

    WITH:
    - Gaussian-perturbed initial states: x₀ ~ N(start_state, σ²I)
    - Simplified model: f(x_t, t) → velocity

    Training:
    - Sample x₀ from Gaussian centered at start state
    - Interpolate geodesically from x₀ to target endpoint x₁
    - Train model to predict velocity at interpolated points

    Inference:
    - Sample perturbed initial state x₀ ~ N(start_state, σ²I)
    - Integrate forward using learned velocity field
    - Multiple samples provide uncertainty quantification
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
        Initialize CartPole Gaussian-Perturbed flow matcher

        Args:
            system: DynamicalSystem (CartPole with ℝ²×S¹×ℝ structure)
            model: CartPoleGaussianPerturbedUNet1D model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            model_config: Configuration dict
            noise_std: Standard deviation of Gaussian perturbation around start state
            mae_val_frequency: Compute MAE validation every N epochs
        """
        super().__init__()

        # Store system and model
        self.system = system
        self.model = model
        self.config = model_config or {}
        self.noise_std = noise_std  # NEW: Gaussian noise std
        self.mae_val_frequency = mae_val_frequency

        # Store optimizer and scheduler configs
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        # Metrics tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # MAE metrics per dimension for endpoint prediction
        self.val_endpoint_mae_per_dim = nn.ModuleList([
            MeanMetric() for _ in range(self.system.state_dim)
        ])

        # Facebook FM components
        self.manifold = self._create_manifold()
        self.path = GeodesicProbPath(
            scheduler=CondOTScheduler(),
            manifold=self.manifold
        )

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model', 'optimizer', 'scheduler', 'system'])

        print("✅ Initialized CartPole Gaussian-Perturbed FM:")
        print(f"   - Manifold: ℝ²×S¹×ℝ (Euclidean × FlatTorus × Euclidean)")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Gaussian noise std: {noise_std}")
        print(f"   - NO latent variables")
        print(f"   - NO conditioning on start state")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")

    def _create_manifold(self):
        """Create ℝ²×S¹×ℝ manifold for CartPole"""
        return Product(input_dim=4, manifolds=[(Euclidean(), 1), (FlatTorus(), 1), (Euclidean(), 2)])

    def sample_perturbed_input(self, start_states: torch.Tensor) -> torch.Tensor:
        """
        Sample perturbed initial states from Gaussian centered at start states

        KEY CHANGE: Instead of uniform sampling from state space,
        we sample from N(start_state, σ²I)

        Args:
            start_states: Start states [B, 4] (normalized)

        Returns:
            Perturbed states [B, 4] sampled from Gaussian
        """
        # Sample Gaussian noise
        noise = torch.randn_like(start_states) * self.noise_std

        # Add noise to start state (in normalized space for numerical stability)
        perturbed = start_states + noise

        # Project back onto manifold (wrap angles to [-π, π])
        # For CartPole: index 1 is the angle θ
        perturbed[:, 1] = torch.atan2(torch.sin(perturbed[:, 1]), torch.cos(perturbed[:, 1]))

        return perturbed

    def compute_flow_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute Gaussian-perturbed flow matching loss

        SIMPLIFIED from latent conditional version:
        - NO latent sampling
        - NO condition handling
        - Initial state sampled from Gaussian around start state

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

        SIMPLIFIED: No latent variable, no conditioning

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
            start_states: Start states [B, 4] in raw coordinates
            num_steps: Number of integration steps
            num_samples: Number of samples per start state

        Returns:
            Predicted endpoints [B*num_samples, 4] in raw coordinates
        """
        if num_samples == 1:
            return self.predict_endpoint(start_states, num_steps)

        all_endpoints = []

        for _ in range(num_samples):
            # Each call samples different Gaussian noise
            endpoints_raw = self.predict_endpoint(start_states, num_steps)
            all_endpoints.append(endpoints_raw)

        # Concatenate all samples: [B*num_samples, 4]
        return torch.cat(all_endpoints, dim=0)

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
    # Training and validation
    # ===================================================================

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
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

    def _get_dimension_name(self, dim_idx: int) -> str:
        """Get human-readable dimension name for CartPole"""
        names = ["cart_position", "pole_angle", "cart_velocity", "angular_velocity"]
        return names[dim_idx] if 0 <= dim_idx < len(names) else f"dim_{dim_idx}"

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

    # ===================================================================
    # Checkpoint loading for inference
    # ===================================================================

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained CartPole Gaussian-Perturbed FM model from checkpoint

        Args:
            checkpoint_path: Path to Lightning checkpoint file (.ckpt) OR training folder
            device: Device to load model on ("cuda", "cpu", or None for auto)

        Returns:
            Loaded model ready for inference
        """
        import torch
        import yaml
        from pathlib import Path
        from src.systems.cartpole_lcfm import CartPoleSystemLCFM
        from src.model.cartpole_gaussian_perturbed_unet1d import CartPoleGaussianPerturbedUNet1D

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = Path(checkpoint_path)

        # Check if it's a folder or a .ckpt file
        if checkpoint_path.is_dir():
            print(f"📁 Folder provided: {checkpoint_path}")
            print(f"🔍 Searching for checkpoint in folder...")

            # Look for checkpoints in version_0/checkpoints/
            checkpoint_dir = checkpoint_path / "version_0" / "checkpoints"

            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"No checkpoints directory found at {checkpoint_dir}")

            # Find all .ckpt files (exclude last.ckpt)
            checkpoints = [p for p in checkpoint_dir.glob("*.ckpt") if p.name != "last.ckpt"]

            if not checkpoints:
                raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")

            # Find checkpoint with lowest validation loss
            best_checkpoint = None
            best_val_loss = float('inf')

            for ckpt in checkpoints:
                try:
                    if "val_loss" in ckpt.stem:
                        loss_str = ckpt.stem.split("val_loss")[1]
                        val_loss = float(loss_str)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_checkpoint = ckpt
                except (ValueError, IndexError):
                    continue

            if best_checkpoint is None:
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"   ⚠️  Could not parse val_loss, using most recent checkpoint")
            else:
                checkpoint_path = best_checkpoint
                print(f"   ✓ Found best checkpoint (val_loss={best_val_loss:.4f})")

            print(f"   📄 Using: {checkpoint_path.name}")

        print(f"🤖 Loading CartPole Gaussian-Perturbed FM checkpoint: {checkpoint_path}")
        print(f"📍 Device: {device}")

        # Verify checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Find training directory
        if checkpoint_path.parent.name == "checkpoints":
            potential_version_dir = checkpoint_path.parent.parent
            if potential_version_dir.name.startswith("version_"):
                training_dir = potential_version_dir.parent
            else:
                training_dir = potential_version_dir
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

        # Extract noise_std
        noise_std = hparams.get("noise_std")
        if noise_std is None and hydra_config:
            noise_std = hydra_config.get("flow_matching", {}).get("noise_std", 0.1)
        if noise_std is None:
            noise_std = 0.1
            print(f"⚠️  Using default noise_std: {noise_std}")

        # Extract model config
        config_source = None
        if "model_config" in hparams:
            model_config = hparams["model_config"]
            config_source = "checkpoint (model_config)"
        elif "config" in hparams:
            model_config = hparams["config"]
            config_source = "checkpoint (config)"
        elif hydra_config:
            model_config = hydra_config.get("model", {})
            config_source = "Hydra config"
        else:
            model_config = {}
            config_source = "defaults (empty)"

        # Remove _target_ key if present
        if isinstance(model_config, dict) and "_target_" in model_config:
            model_config = {k: v for k, v in model_config.items() if k != "_target_"}

        print(f"📋 Config source: {config_source}")
        print(f"📋 Final config - noise_std: {noise_std}")
        print(f"📋 Model config keys: {list(model_config.keys())}")

        # Initialize system and model
        system = hparams.get("system")
        if system is None:
            print("🔧 Creating new CartPole system (not found in hparams)")
            system = CartPoleSystemLCFM()
        else:
            print("✅ Restored CartPole system from checkpoint")

        # Create model architecture (simplified - no latent_dim, no condition_dim)
        model = CartPoleGaussianPerturbedUNet1D(
            embedded_dim=model_config.get('embedded_dim', 5),
            time_emb_dim=model_config.get('time_emb_dim', 64),
            hidden_dims=model_config.get('hidden_dims', [256, 512, 1024, 512, 256]),
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
            model_config=model_config,
            noise_std=noise_std
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
        print(f"   Gaussian noise std: {noise_std}")
        print(f"   Model architecture: {model_config.get('hidden_dims', 'unknown')}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")
        print(f"   Variant: Gaussian-Perturbed (no latent, no conditioning)")

        return flow_matcher
