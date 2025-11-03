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

from src.flow_matching.base.flow_matcher import BaseFlowMatcher
from src.systems.base import DynamicalSystem
from src.utils.fb_manifolds import PendulumManifold

from flow_matching.utils.manifolds import Product, FlatTorus, Euclidean

class PendulumLatentConditionalFlowMatcher(BaseFlowMatcher):
    """
    Pendulum Latent Conditional Flow Matching using Facebook FM:
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
        super().__init__(system, model, optimizer, scheduler, model_config, latent_dim, mae_val_frequency)

        print("✅ Initialized Pendulum LCFM with Facebook Flow Matching:")
        print(f"   - Manifold: S¹×ℝ (FlatTorus × Euclidean)")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Latent dim: {latent_dim}")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")

    def _create_manifold(self):
        """Create S¹×ℝ manifold for pendulum"""
        return Product(input_dim=2, manifolds=[(FlatTorus(), 1), (Euclidean(), 1)])

    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract start states from batch"""
        return batch["start_state"]

    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract end states from batch"""
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx: int) -> str:
        """Get human-readable dimension name for pendulum"""
        names = ["angle", "angular_velocity"]
        return names[dim_idx] if 0 <= dim_idx < len(names) else f"dim_{dim_idx}"

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for normalization"""
        return self.system.normalize_state(state)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for denormalization"""
        return self.system.denormalize_state(normalized_state)

    def embed_state_for_model(self, state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for embedding"""
        return self.system.embed_state_for_model(state)

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noisy input uniformly in S¹ × ℝ space

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Noisy states [batch_size, 2] as (θ, θ̇) in raw coordinates
        """
        
        raise NotImplementedError("This method is not implemented for Pendulum Latent Conditional Flow Matching")
        
        # θ ~ Uniform[-π, π] for circular angle
        theta_min = self.system.state_bounds["angle"][0]
        theta_max = self.system.state_bounds["angle"][1]
        theta = torch.randn(batch_size, 1, device=device) * (theta_max - theta_min) + theta_min

        # θ̇ ~ Uniform[-2π, 2π] for angular velocity
        vel_min = self.system.state_bounds["angular_velocity"][0]
        vel_max = self.system.state_bounds["angular_velocity"][1]
        theta_dot = torch.rand(batch_size, 1, device=device) * (vel_max - vel_min) + vel_min

        return torch.cat([theta, theta_dot], dim=1)

    # ===================================================================
    # REMOVED METHODS (now in base class or handled by Facebook FM):
    # ===================================================================
    # ✅ sample_latent() → moved to BaseFlowMatcher
    # ✅ forward() → moved to BaseFlowMatcher
    # ✅ compute_flow_loss() → moved to BaseFlowMatcher (unified implementation)
    # ✅ compute_endpoint_mae_per_dim() → moved to BaseFlowMatcher
    # ✅ validation_step() → moved to BaseFlowMatcher
    # ✅ on_validation_epoch_end() → moved to BaseFlowMatcher
    # ✅ predict_endpoint() → moved to BaseFlowMatcher (unified implementation)
    # ❌ interpolate_s1_x_r() → replaced by self.path.sample()
    # ❌ compute_target_velocity_s1_x_r() → automatic in path_sample.dx_t

    # ===================================================================
    # CHECKPOINT LOADING FOR INFERENCE
    # ===================================================================

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained Pendulum LCFM model from checkpoint for inference.

        Args:
            checkpoint_path: Path to Lightning checkpoint file (.ckpt) OR training folder
                           - If .ckpt file: loads that checkpoint directly
                           - If folder: searches for best checkpoint in folder/version_0/checkpoints/
            device: Device to load model on ("cuda", "cpu", or None for auto)

        Returns:
            Loaded model ready for inference
        """
        import torch
        import yaml
        from pathlib import Path
        from omegaconf import OmegaConf
        from src.systems.pendulum_lcfm import PendulumSystemLCFM
        from src.model.latent_conditional_unet1d import LatentConditionalUNet1D

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

            # Parse validation loss from filename: "epoch{epoch:02d}-val_loss{val_loss:.4f}.ckpt"
            # Find checkpoint with lowest validation loss (best model)
            best_checkpoint = None
            best_val_loss = float('inf')

            for ckpt in checkpoints:
                # Extract val_loss from filename
                try:
                    # Example: "epoch42-val_loss0.4519.ckpt"
                    if "val_loss" in ckpt.stem:
                        loss_str = ckpt.stem.split("val_loss")[1]
                        val_loss = float(loss_str)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_checkpoint = ckpt
                except (ValueError, IndexError):
                    continue

            if best_checkpoint is None:
                # Fallback: use most recent checkpoint
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"   ⚠️  Could not parse val_loss, using most recent checkpoint")
            else:
                checkpoint_path = best_checkpoint
                print(f"   ✓ Found best checkpoint (val_loss={best_val_loss:.4f})")

            print(f"   📄 Using: {checkpoint_path.name}")

        print(f"🤖 Loading Pendulum LCFM checkpoint: {checkpoint_path}")
        print(f"📍 Device: {device}")

        # Verify checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Find the training directory (Hydra root)
        if checkpoint_path.parent.name == "checkpoints":
            # Could be version_0/checkpoints/ or just checkpoints/
            potential_version_dir = checkpoint_path.parent.parent
            if potential_version_dir.name.startswith("version_"):
                # New structure: go up one more level to Hydra root
                training_dir = potential_version_dir.parent
            else:
                # Old structure: already at Hydra root
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

        # Extract latent_dim
        latent_dim = hparams.get("latent_dim")
        if latent_dim is None and hydra_config:
            latent_dim = hydra_config.get("flow_matching", {}).get("latent_dim", 2)
        if latent_dim is None:
            latent_dim = 2
            print(f"⚠️  Using default latent_dim: {latent_dim}")

        # Extract model config (try both 'model_config' and 'config' for backward compatibility)
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

        # Remove _target_ key if present (not needed for reconstruction)
        if isinstance(model_config, dict) and "_target_" in model_config:
            model_config = {k: v for k, v in model_config.items() if k != "_target_"}

        model_config["latent_dim"] = latent_dim

        print(f"📋 Config source: {config_source}")
        print(f"📋 Final config - latent_dim: {latent_dim}")
        print(f"📋 Model config keys: {list(model_config.keys())}")

        # Initialize system and model
        system = hparams.get("system")
        if system is None:
            print("🔧 Creating new Pendulum system (not found in hparams)")
            system = PendulumSystemLCFM()
        else:
            print("✅ Restored Pendulum system from checkpoint")

        # Create model architecture
        model = LatentConditionalUNet1D(
            embedded_dim=model_config.get('embedded_dim', 3),
            latent_dim=model_config.get('latent_dim', latent_dim),
            condition_dim=model_config.get('condition_dim', 3),
            time_emb_dim=model_config.get('time_emb_dim', 64),
            hidden_dims=model_config.get('hidden_dims', [256, 512, 256]),
            output_dim=model_config.get('output_dim', 2),
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
        print(f"   Latent dim: {latent_dim}")
        print(f"   Model architecture: {model_config.get('hidden_dims', 'unknown')}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")

        return flow_matcher
# ============================================================================
# NOTE: VelocityModelWrapper moved to BaseFlowMatcher
# (LatentConditionalVelocityWrapper)
# ============================================================================


