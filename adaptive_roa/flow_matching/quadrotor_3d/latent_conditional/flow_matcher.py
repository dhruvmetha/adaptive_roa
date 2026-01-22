"""
Quadrotor 3D Latent Conditional Flow Matching implementation using Facebook Flow Matching library

Manifold: ℝ³ × SO(3) × ℝ⁶ (13-dimensional state with unit quaternion representation)
- Position: ℝ³ (x, y, z)
- Orientation: SO(3) via unit quaternion (qw, qx, qy, qz) - 4D representation, 3D tangent
- Linear velocity: ℝ³ (ẋ, ẏ, ż)
- Angular velocity: ℝ³ (p, q, r)
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import lightning.pytorch as pl
from torchmetrics import MeanMetric

from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Product, Euclidean, SO3

from adaptive_roa.flow_matching.base.flow_matcher import BaseFlowMatcher
from adaptive_roa.systems.base import DynamicalSystem


class Quadrotor3DLatentConditionalFlowMatcher(BaseFlowMatcher):
    """
    Quadrotor 3D Latent Conditional Flow Matching using Facebook FM:
    - Uses GeodesicProbPath for geodesic interpolation on ℝ³ × SO(3) × ℝ⁶
    - Uses RiemannianODESolver for manifold-aware ODE integration
    - Neural net takes state x_t, time t, latent z, and start state condition
    - Predicts velocity in tangent space

    Manifold structure:
    - Euclidean(3): Position (x, y, z) - indices 0-2
    - SO3(4, 3): Quaternion (qw, qx, qy, qz) - indices 3-6, 4D representation, 3D tangent
    - Euclidean(6): Velocities (ẋ, ẏ, ż, p, q, r) - indices 7-12

    The Product manifold is defined as:
    Product(input_dim=13, manifolds=[(Euclidean(), 3), (SO3(), 4, 3), (Euclidean(), 6)])
    """

    def __init__(self,
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 model_config: Optional[dict] = None,
                 latent_dim: int = 4,
                 mae_val_frequency: int = 10):
        """
        Initialize Quadrotor 3D latent conditional flow matcher with FB FM integration

        Args:
            system: DynamicalSystem (Quadrotor3D with ℝ³×SO(3)×ℝ⁶ structure)
            model: Quadrotor3DUNet model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            model_config: Configuration dict
            latent_dim: Dimension of latent space
            mae_val_frequency: Compute MAE validation every N epochs
        """
        super().__init__(system, model, optimizer, scheduler, model_config, latent_dim, mae_val_frequency)

        print("✅ Initialized Quadrotor3D LCFM with Facebook Flow Matching:")
        print(f"   - Manifold: ℝ³ × SO(3) × ℝ⁶ (Euclidean × SO3 × Euclidean)")
        print(f"   - State dimension: 13 (3 + 4 + 6)")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Latent dim: {latent_dim}")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")

    def _create_manifold(self):
        """
        Create ℝ³ × SO(3) × ℝ⁶ manifold for Quadrotor 3D

        Product manifold structure:
        - Euclidean(3): Position (x, y, z) - 3D
        - SO3(4, 3): Quaternion representation - 4D state, 3D tangent
        - Euclidean(6): Linear + Angular velocity - 6D

        Total: 13D state, 12D tangent (3 + 3 + 6)
        """
        return Product(input_dim=13, manifolds=[
            (Euclidean(), 3),      # Position (x, y, z)
            (SO3(), 4, 3),         # Quaternion (qw, qx, qy, qz) - 4D representation, 3D tangent
            (Euclidean(), 6)       # Velocities (ẋ, ẏ, ż, p, q, r)
        ])

    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract start states from batch"""
        return batch["start_state"]

    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract end states from batch"""
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx: int) -> str:
        """Get human-readable dimension name for Quadrotor 3D"""
        names = [
            "x", "y", "z",           # Position
            "qw", "qx", "qy", "qz",  # Quaternion
            "x_dot", "y_dot", "z_dot",  # Linear velocity
            "p", "q", "r"            # Angular velocity
        ]
        return names[dim_idx] if 0 <= dim_idx < len(names) else f"dim_{dim_idx}"

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noisy input uniformly in ℝ³ × SO(3) × ℝ⁶ space

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Noisy states [batch_size, 13]
        """
        # Position: Gaussian noise (will be projected by manifold if needed)
        position = torch.randn(batch_size, 3, device=device)

        # Quaternion: uniform sampling on SO(3) via normalized Gaussian
        # Sample 4D Gaussian and normalize to get uniform distribution on unit quaternion sphere
        quat = torch.randn(batch_size, 4, device=device)
        quat = quat / torch.norm(quat, dim=1, keepdim=True).clamp(min=1e-8)
        # Canonicalize: ensure qw >= 0
        sign = torch.sign(quat[:, 0:1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        quat = quat * sign

        # Velocities: Gaussian noise
        velocities = torch.randn(batch_size, 6, device=device)

        noisy_input = torch.cat([position, quat, velocities], dim=1)

        # Project onto manifold
        noisy_input = self.manifold.projx(noisy_input)

        return noisy_input

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for normalization"""
        return self.system.normalize_state(state)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for denormalization"""
        return self.system.denormalize_state(normalized_state)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Delegate to system for embedding"""
        return self.system.embed_state_for_model(normalized_state)

    def predict_endpoints_batch(self,
                               start_states: torch.Tensor,
                               num_steps: int = 100,
                               num_samples: int = 1) -> torch.Tensor:
        """
        Predict multiple endpoint samples per start state (for stochastic models).

        Args:
            start_states: Start states [B, 13] in raw coordinates
            num_steps: Number of integration steps
            num_samples: Number of samples per start state

        Returns:
            Predicted endpoints [B*num_samples, 13] in raw coordinates
        """
        if num_samples == 1:
            raw_endpoints = self.predict_endpoint(start_states, num_steps)
            return raw_endpoints

        batch_size = start_states.shape[0]
        all_endpoints = []

        for _ in range(num_samples):
            # Sample different latent vectors for each sample
            endpoints_raw = self.predict_endpoint(start_states, num_steps, latent=None)
            all_endpoints.append(endpoints_raw)

        # Concatenate all samples: [B*num_samples, 13]
        return torch.cat(all_endpoints, dim=0)

    # ===================================================================
    # CHECKPOINT LOADING FOR INFERENCE
    # ===================================================================

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained Quadrotor3D LCFM model from checkpoint for inference.

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
        import os
        from pathlib import Path
        from omegaconf import OmegaConf
        from adaptive_roa.systems.quadrotor3d import Quadrotor3DSystem
        from adaptive_roa.model.quadrotor3d_unet import Quadrotor3DUNet
        from adaptive_roa.utils.env_config import get_net_id, get_exp_dir, get_data_dir, get_env_config

        # Register OmegaConf resolvers for Hydra config loading
        if not OmegaConf.has_resolver("net_id"):
            OmegaConf.register_new_resolver("net_id", lambda: get_net_id())
        if not OmegaConf.has_resolver("exp_dir"):
            OmegaConf.register_new_resolver("exp_dir", lambda: get_exp_dir())
        if not OmegaConf.has_resolver("data_dir"):
            OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
        if not OmegaConf.has_resolver("env"):
            OmegaConf.register_new_resolver("env", lambda key, default="": os.environ.get(key, get_env_config().get(key, default)))

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

            # Parse validation loss from filename
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

        print(f"🤖 Loading Quadrotor3D LCFM checkpoint: {checkpoint_path}")
        print(f"📍 Device: {device}")

        # Verify checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Find the training directory (Hydra root)
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
        hydra_config_path = None

        search_dir = training_dir
        for _ in range(3):
            candidate_path = search_dir / ".hydra" / "config.yaml"
            if candidate_path.exists():
                hydra_config_path = candidate_path
                break
            search_dir = search_dir.parent

        if hydra_config_path:
            try:
                print(f"📋 Loading Hydra config: {hydra_config_path}")
                # Use OmegaConf to load and resolve interpolations (e.g., ${data_dir})
                hydra_omega_config = OmegaConf.load(hydra_config_path)
                # Resolve all interpolations and convert to plain dict
                hydra_config = OmegaConf.to_container(hydra_omega_config, resolve=True)
                print("✅ Hydra config loaded successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not load Hydra config: {e}")
                hydra_config = None
        else:
            print(f"⚠️  Hydra config not found in {training_dir} or parent directories")

        # Load Lightning checkpoint
        print(f"📦 Loading Lightning checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})
        print("✅ Lightning checkpoint loaded")

        # Extract latent_dim
        latent_dim = hparams.get("latent_dim")
        if latent_dim is None and hydra_config:
            latent_dim = hydra_config.get("flow_matching", {}).get("latent_dim", 4)
        if latent_dim is None:
            latent_dim = 4
            print(f"⚠️  Using default latent_dim: {latent_dim}")

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

        model_config["latent_dim"] = latent_dim

        print(f"📋 Config source: {config_source}")
        print(f"📋 Final config - latent_dim: {latent_dim}")
        print(f"📋 Model config keys: {list(model_config.keys())}")

        # Initialize system and model
        system = hparams.get("system")
        if system is None:
            print("🔧 Creating new Quadrotor3D system (not found in hparams)")
            if hydra_config and "system" in hydra_config:
                system_config = hydra_config["system"]
                print(f"   Using system config from Hydra config")
                dataset_dir = system_config.get("dataset_dir", None)
                print(f"   dataset_dir: {dataset_dir}")
                system = Quadrotor3DSystem(dataset_dir=dataset_dir)
            else:
                print("   No Hydra system config found, using defaults")
                system = Quadrotor3DSystem()
        else:
            print("✅ Restored Quadrotor3D system from checkpoint")

        # Create model architecture
        model = Quadrotor3DUNet(
            embedded_dim=model_config.get('embedded_dim', 13),
            latent_dim=model_config.get('latent_dim', latent_dim),
            condition_dim=model_config.get('condition_dim', 13),
            time_emb_dim=model_config.get('time_emb_dim', 64),
            hidden_dims=model_config.get('hidden_dims', [512, 1024, 512]),
            output_dim=model_config.get('output_dim', 13),
            use_input_embeddings=model_config.get('use_input_embeddings', False),
            input_emb_dim=model_config.get('input_emb_dim', 128)
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
