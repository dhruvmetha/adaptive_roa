"""Mountain Car Latent Conditional Flow Matcher

Flow matching implementation for Mountain Car using Facebook Flow Matching library.

Manifold: ℝ² (pure Euclidean, simplest case)
"""

import torch
from typing import Dict, Any, Optional
import os

from src.flow_matching.base.flow_matcher import BaseFlowMatcher
from flow_matching.utils.manifolds import Product, Euclidean


class MountainCarLatentConditionalFlowMatcher(BaseFlowMatcher):
    """Latent conditional flow matcher for Mountain Car.

    State: [position, velocity]
    Manifold: ℝ² (pure Euclidean)
    """

    def __init__(self, system, model, optimizer, scheduler, model_config,
                 mae_val_frequency: int = 10,
                 noise_std: float = 0.1):
        """Initialize Mountain Car flow matcher.

        Args:
            system: MountainCarSystem instance
            model: MountainCarUNet instance
            optimizer: Optimizer config
            scheduler: Scheduler config
            model_config: Model configuration dict
            mae_val_frequency: Frequency for validation MAE computation
            noise_std: Standard deviation for Gaussian noise sampling

        Note: NO latent variables (following Humanoid pattern)
        """
        super().__init__(
            system=system,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            mae_val_frequency=mae_val_frequency
        )

        # Configurable standard deviation for initial noise sampling
        self.noise_std = float(noise_std)

        print("✅ Initialized MountainCar CFM with Facebook Flow Matching:")
        print(f"   Manifold: ℝ² (Euclidean)")
        print(f"   State dim: {system.state_dim}")
        print(f"   NO latent variables (conditional only)")
        print(f"   Initial noise std: {self.noise_std}")

    def _create_manifold(self):
        """Create the manifold for Mountain Car: ℝ² (pure Euclidean).

        CRITICAL: This maps the mathematical manifold to Facebook FM types.

        Mountain Car manifold: ℝ² (position, velocity)
        Facebook FM mapping: Product with single Euclidean component

        Returns:
            Product manifold with Euclidean component
        """
        # Pure Euclidean manifold (simplest case!)
        # Even for single Euclidean, wrap in Product
        return Product(input_dim=2, manifolds=[(Euclidean(), 2)])

    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract start states from batch.

        Args:
            batch: Batch dictionary with 'start_state' key

        Returns:
            Start states tensor [B, 2]
        """
        return batch["start_state"]

    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract end states from batch.

        Args:
            batch: Batch dictionary with 'end_state' key

        Returns:
            End states tensor [B, 2]
        """
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx: int) -> str:
        """Get human-readable name for dimension (for logging).

        Args:
            dim_idx: Dimension index (0 or 1)

        Returns:
            Dimension name string
        """
        names = ["position", "velocity"]
        return names[dim_idx] if dim_idx < len(names) else f"dim_{dim_idx}"

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample initial noisy states ON THE MANIFOLD.

        For pure Euclidean manifold: sample from Gaussian N(0, noise_std²).

        Args:
            batch_size: Number of samples
            device: Device to create tensor on

        Returns:
            Sampled states [B, 2] in normalized coordinates
        """
        # Sample Gaussian noise
        noisy_states = torch.randn(batch_size, 2, device=device) * self.noise_std

        # Project onto manifold (identity for Euclidean, but keeps pattern consistent)
        noisy_states = self.manifold.projx(noisy_states)

        return noisy_states

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state to [-1, 1] range.

        Delegates to system.normalize_state().

        Args:
            state: Unnormalized state [B, 2]

        Returns:
            Normalized state [B, 2]
        """
        return self.system.normalize_state(state)

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Denormalize state from [-1, 1] back to physical units.

        Delegates to system.denormalize_state().

        Args:
            normalized_state: Normalized state [B, 2]

        Returns:
            Denormalized state [B, 2]
        """
        return self.system.denormalize_state(normalized_state)

    def embed_state_for_model(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Embed state for model input.

        For pure Euclidean: no embedding needed (identity mapping).

        Args:
            normalized_state: Normalized state [B, 2]

        Returns:
            Embedded state [B, 2] (same as input)
        """
        return self.system.embed_state_for_model(normalized_state)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """Load a trained Mountain Car flow matcher from checkpoint for inference.

        Args:
            checkpoint_path: Path to Lightning checkpoint file (.ckpt) OR training folder
                           - If .ckpt file: loads that checkpoint directly
                           - If folder: searches for best checkpoint in folder/version_0/checkpoints/
            device: Device to load model on ("cuda", "cpu", or None for auto)

        Returns:
            Loaded model ready for inference
        """
        import yaml
        from pathlib import Path
        from omegaconf import OmegaConf
        from src.systems.mountain_car import MountainCarSystem
        from src.model.mountain_car_unet_film import MountainCarUNetFiLM

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

        print(f"🤖 Loading Mountain Car flow matcher checkpoint: {checkpoint_path}")
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

        print(f"📋 Config source: {config_source}")
        print(f"📋 Model config keys: {list(model_config.keys())}")

        # Initialize system
        system = hparams.get("system")
        if system is None:
            print("🔧 Creating new Mountain Car system (not found in hparams)")
            # Use system config from Hydra if available
            if hydra_config and "system" in hydra_config:
                system_config = hydra_config["system"]
                print(f"   Using system config from Hydra config")
                # Extract bounds configuration
                bounds_file = system_config.get("bounds_file", None)
                if bounds_file:
                    print(f"   bounds_file: {bounds_file}")
                    system = MountainCarSystem(bounds_file=bounds_file)
                else:
                    system = MountainCarSystem()
            else:
                print("   No Hydra system config found, using defaults")
                system = MountainCarSystem()
        else:
            print("✅ Restored Mountain Car system from checkpoint")

        # Create model architecture
        model = MountainCarUNetFiLM(
            embedded_dim=model_config.get('embedded_dim', 2),
            condition_dim=model_config.get('condition_dim', 2),
            time_emb_dim=model_config.get('time_emb_dim', 128),
            hidden_dims=model_config.get('hidden_dims', [256, 512, 256]),
            output_dim=model_config.get('output_dim', 2),
            film_cond_dim=model_config.get('film_cond_dim', 256),
            film_hidden_dims=model_config.get('film_hidden_dims', []),
            use_input_embeddings=model_config.get('use_input_embeddings', False),
            input_emb_dim=model_config.get('input_emb_dim', 128),
            dropout_p=model_config.get('dropout_p', 0.0),
            zero_init_blocks=model_config.get('zero_init_blocks', True),
            zero_init_out=model_config.get('zero_init_out', False)
        )

        # Create flow matcher instance (NO latent_dim for Mountain Car)
        flow_matcher = cls(
            system=system,
            model=model,
            optimizer=None,
            scheduler=None,
            model_config=model_config,
            mae_val_frequency=hparams.get('mae_val_frequency', 10)
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
        print(f"   System bounds: position±{system.position_limit:.1f}, velocity±{system.velocity_limit:.2f}")
        print(f"   Model architecture: {model_config.get('hidden_dims', 'unknown')}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")

        return flow_matcher
