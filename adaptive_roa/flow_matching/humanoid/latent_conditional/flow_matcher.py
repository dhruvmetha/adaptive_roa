"""
Humanoid Latent Conditional Flow Matching implementation using Facebook Flow Matching library

Manifold: ℝ³⁴ × S² × ℝ³⁰ (67-dimensional state)
- Uses GeodesicProbPath for geodesic interpolation
- Uses RiemannianODESolver for manifold-aware ODE integration
- Neural net takes embedded x_t, time t, latent z, and start state condition
- Predicts velocity in tangent space
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from adaptive_roa.flow_matching.base.flow_matcher import BaseFlowMatcher
from adaptive_roa.systems.base import DynamicalSystem
from adaptive_roa.utils.fb_manifolds import HumanoidManifold


class HumanoidLatentConditionalFlowMatcher(BaseFlowMatcher):
    """
    Humanoid Latent Conditional Flow Matching using Facebook FM:
    - Manifold: ℝ³⁴ × S² × ℝ³⁰ (67-dimensional state)
    - Uses GeodesicProbPath for geodesic interpolation
    - Uses RiemannianODESolver for manifold-aware ODE integration
    - Neural net takes embedded x_t, time t, latent z, and start state condition
    - Predicts velocity in ℝ³⁴ × S² × ℝ³⁰ tangent space

    Key manifold components:
    - Euclidean(34): First 34 dimensions (indices 0-33)
    - Sphere(3): Next 3 dimensions (indices 34-36) - 3D unit vector
    - Euclidean(30): Last 30 dimensions (indices 37-66)
    """

    def __init__(self,
                 system: DynamicalSystem,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 model_config: Optional[dict] = None,
                 latent_dim: int = 8,
                 mae_val_frequency: int = 10,
                 use_loss_weights: bool = False,
                 clamp_noise: bool = True,
                 zero_latent: bool = False,
                 val_error_log_file: Optional[str] = None):
        """
        Initialize Humanoid latent conditional flow matcher with FB FM integration

        Args:
            system: DynamicalSystem (Humanoid with ℝ³⁴ × S² × ℝ³⁰ structure)
            model: HumanoidLatentConditionalUNet model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            model_config: Configuration dict
            latent_dim: Dimension of latent space (default: 8)
            mae_val_frequency: Compute MAE validation every N epochs
            use_loss_weights: If True, weight loss by normalization limits
            clamp_noise: If True, clamp noise to [-1, 1] to prevent ODE divergence
            zero_latent: If True, use zero latent vectors instead of random sampling
            val_error_log_file: Path to text file for logging validation errors
        """
        super().__init__(system, model, optimizer, scheduler, model_config, latent_dim, mae_val_frequency, use_loss_weights, clamp_noise, zero_latent, val_error_log_file)

        print("✅ Initialized Humanoid LCFM with Facebook Flow Matching:")
        print(f"   - Manifold: ℝ³⁴ × S² × ℝ³⁰ (Euclidean × Sphere × Euclidean)")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Latent dim: {latent_dim}")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")

    def _create_manifold(self):
        """
        Create ℝ³⁴ × S² × ℝ³⁰ manifold for Humanoid

        Manifold structure:
        - Euclidean(34): Dims 0-33
        - Sphere(3): Dims 34-36 (3D unit vector on S²)
        - Euclidean(30): Dims 37-66
        """
        # Use custom HumanoidManifold that properly handles the product structure
        return HumanoidManifold()

    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract start states from batch"""
        return batch["start_state"]

    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract end states from batch"""
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx: int) -> str:
        """
        Get human-readable dimension name for Humanoid

        Args:
            dim_idx: Dimension index (0-66)

        Returns:
            Human-readable name
        """
        if 0 <= dim_idx < 34:
            return f"euclidean1_{dim_idx}"
        elif 34 <= dim_idx < 37:
            comp_names = ["sphere_x", "sphere_y", "sphere_z"]
            return comp_names[dim_idx - 34]
        elif 37 <= dim_idx < 67:
            return f"euclidean2_{dim_idx - 37}"
        else:
            return f"dim_{dim_idx}"

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample Gaussian noise in normalized space for ℝ³⁴ × S² × ℝ³⁰ manifold.

        Returns noise directly in normalized space (no further normalization needed).
        Sphere component (S²) is projected onto manifold.
        If self.clamp_noise is True, clamps to [-1, 1] to prevent ODE divergence.

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Noisy states [batch_size, 67] in normalized space
        """
        noisy_input = torch.randn(batch_size, 67, device=device)
        if self.clamp_noise:
            noisy_input = torch.clamp(noisy_input, -1.0, 1.0)
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
            start_states: Start states [B, 67] in raw coordinates
            num_steps: Number of integration steps
            num_samples: Number of samples per start state

        Returns:
            Predicted endpoints [B*num_samples, 67] in raw coordinates
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

        # Concatenate all samples: [B*num_samples, 67]
        return torch.cat(all_endpoints, dim=0)

    # ===================================================================
    # CHECKPOINT LOADING FOR INFERENCE
    # ===================================================================

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained Humanoid LCFM model from checkpoint for inference.

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
        from adaptive_roa.systems.humanoid import HumanoidSystem
        from adaptive_roa.model.universal_unet import UniversalUNet
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

        print(f"🤖 Loading Humanoid LCFM checkpoint: {checkpoint_path}")
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

        # Load Hydra config - check current dir and parent directories
        # (adaptive loop stores .hydra in parent, not per-epoch directories)
        hydra_config = None
        hydra_config_path = None

        search_dir = training_dir
        for _ in range(3):  # Check up to 3 parent levels
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
            latent_dim = hydra_config.get("flow_matching", {}).get("latent_dim", 8)
        if latent_dim is None:
            latent_dim = 8
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
            print("🔧 Creating new Humanoid system (not found in hparams)")
            # Use system config from Hydra if available
            if hydra_config and "system" in hydra_config:
                system_config = hydra_config["system"]
                print(f"   Using system config from Hydra config")
                bounds_file = system_config.get("bounds_file", None)
                use_dynamic_bounds = system_config.get("use_dynamic_bounds", False)
                print(f"   bounds_file: {bounds_file}")
                print(f"   use_dynamic_bounds: {use_dynamic_bounds}")
                system = HumanoidSystem(bounds_file=bounds_file, use_dynamic_bounds=use_dynamic_bounds)
            else:
                print("   No Hydra system config found, using defaults")
                system = HumanoidSystem()
        else:
            print("✅ Restored Humanoid system from checkpoint")

        # Create model architecture
        model = UniversalUNet(
            input_dim=model_config.get('input_dim', 142),  # 67 + 67 + 8
            output_dim=model_config.get('output_dim', 67),
            time_embed_dim=model_config.get('time_embed_dim', 128),
            hidden_dims=model_config.get('hidden_dims', [256, 512, 512, 256]),
            dropout=model_config.get('dropout', 0.1),
            activation=model_config.get('activation', 'silu')
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
        print(f"   System bounds: Euclidean ±{system.euclidean_limit:.1f}, Sphere (unit norm)")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Model architecture: {model_config.get('hidden_dims', 'unknown')}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")

        return flow_matcher
