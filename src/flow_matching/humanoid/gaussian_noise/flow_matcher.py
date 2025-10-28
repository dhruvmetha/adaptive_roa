"""
Humanoid Gaussian Noise Flow Matching implementation

INHERITS FROM: BaseGaussianNoiseFlowMatcher

KEY DIFFERENCES from Latent Conditional variant:
1. NO latent variables (removed z ~ N(0,I))
2. NO conditioning on start state
3. Initial noise sampled from Gaussian centered at start state: x₀ ~ N(start_state, σ²I)
4. Simplified model signature: f(x_t, t) instead of f(x_t, t, z, condition)

SPHERE HANDLING:
- After adding Gaussian noise, sphere components (dims 34-36) are renormalized to unit norm
- This ensures the perturbed state stays on the manifold ℝ³⁴ × S² × ℝ³⁰
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier/flow_matching')

from flow_matching.utils.manifolds import Product, Sphere, Euclidean

from src.flow_matching.base.gaussian_noise_flow_matcher import BaseGaussianNoiseFlowMatcher
from src.systems.base import DynamicalSystem


class HumanoidGaussianNoiseFlowMatcher(BaseGaussianNoiseFlowMatcher):
    """
    Humanoid Gaussian Noise Flow Matching using Facebook FM

    Manifold: ℝ³⁴ × S² × ℝ³⁰ (67-dimensional state)

    Simplified flow matching WITHOUT:
    - Latent variables z
    - Conditioning on start state

    WITH:
    - Gaussian-perturbed initial states: x₀ ~ N(start_state, σ²I)
    - Sphere components renormalized to unit norm after perturbation
    - Simplified model: f(x_t, t) → velocity

    REFACTORED: Now inherits from BaseGaussianNoiseFlowMatcher
    - ✅ All generic code moved to base class (~500 lines)
    - ✅ Only Humanoid-specific code remains (~100 lines)
    - ✅ Custom sphere handling for manifold constraint
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
        Initialize Humanoid Gaussian-Perturbed flow matcher

        Args:
            system: DynamicalSystem (Humanoid with ℝ³⁴ × S² × ℝ³⁰ structure)
            model: HumanoidGaussianNoiseUNet model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            model_config: Configuration dict
            noise_std: Standard deviation of Gaussian perturbation around start state
            mae_val_frequency: Compute MAE validation every N epochs
        """
        super().__init__(system, model, optimizer, scheduler, model_config, noise_std, mae_val_frequency)

        print("✅ Initialized Humanoid Gaussian-Perturbed FM (REFACTORED):")
        print(f"   - Manifold: ℝ³⁴ × S² × ℝ³⁰ (Euclidean × Sphere × Euclidean)")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - Gaussian noise std: {noise_std}")
        print(f"   - NO latent variables")
        print(f"   - NO conditioning on start state")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")

    def _create_manifold(self):
        """
        Create ℝ³⁴ × S² × ℝ³⁰ manifold for Humanoid

        Manifold structure:
        - Euclidean(34): Dims 0-33
        - Sphere(3): Dims 34-36 (3D unit vector on S²)
        - Euclidean(30): Dims 37-66
        """
        # Verify manifold structure matches expectations
        assert len(self.system.manifold_components) == 36, \
            f"Humanoid should have 36 components, got {len(self.system.manifold_components)}"

        # Check for sphere component at index 34
        assert self.system.manifold_components[34].manifold_type == "Sphere", \
            "Humanoid component [34] should be Sphere (orientation)"

        # Build manifold: ℝ³⁴ × S² × ℝ³⁰
        return Product(input_dim=67, manifolds=[
            (Euclidean(), 34),  # First Euclidean block (dims 0-33)
            (Sphere(), 3),      # Sphere block (dims 34-36) - 3D unit vector
            (Euclidean(), 30)   # Second Euclidean block (dims 37-66)
        ])

    def _get_dimension_name(self, dim_idx: int) -> str:
        """
        Get human-readable dimension name for Humanoid

        Uses system.manifold_components for consistency
        """
        if 0 <= dim_idx < len(self.system.manifold_components):
            return self.system.manifold_components[dim_idx].name
        return f"dim_{dim_idx}"

    def sample_perturbed_input(self, start_states: torch.Tensor) -> torch.Tensor:
        """
        Sample perturbed initial states from Gaussian centered at start states

        CUSTOM IMPLEMENTATION for Humanoid:
        - Adds Gaussian noise to all components
        - Renormalizes sphere components (dims 34-36) to unit norm

        Args:
            start_states: Start states [B, 67] (normalized)

        Returns:
            Perturbed states [B, 67] sampled from Gaussian with sphere constraint
        """
        # Sample Gaussian noise
        noise = torch.randn_like(start_states) * self.noise_std

        # Add noise to start state (in normalized space for numerical stability)
        perturbed = start_states + noise

        # SPHERE HANDLING: Renormalize sphere components (dims 34-36) to unit norm
        # After adding Gaussian noise, the sphere components may no longer have unit norm
        sphere_components = perturbed[:, 34:37]  # Extract [B, 3]
        sphere_norm = torch.norm(sphere_components, dim=1, keepdim=True)  # [B, 1]

        # Avoid division by zero (should be rare with Gaussian noise)
        sphere_norm = torch.clamp(sphere_norm, min=1e-8)

        # Renormalize to unit sphere
        perturbed[:, 34:37] = sphere_components / sphere_norm

        # No angle wrapping needed for humanoid (no SO2 components)
        # The base class get_circular_indices() returns [] for humanoid

        return perturbed

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
    # CHECKPOINT LOADING FOR INFERENCE
    # ===================================================================

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained Humanoid Gaussian Noise FM model from checkpoint for inference.

        Args:
            checkpoint_path: Path to Lightning checkpoint file (.ckpt) OR training folder
            device: Device to load model on ("cuda", "cpu", or None for auto)

        Returns:
            Loaded model ready for inference
        """
        import torch
        import yaml
        from pathlib import Path
        from src.systems.humanoid import HumanoidSystem
        from src.model.universal_unet import UniversalUNet

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = Path(checkpoint_path)

        # Check if it's a folder or a .ckpt file
        if checkpoint_path.is_dir():
            print(f"📁 Folder provided: {checkpoint_path}")
            print(f"🔍 Searching for checkpoint in folder...")

            checkpoint_dir = checkpoint_path / "version_0" / "checkpoints"
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"No checkpoints directory found at {checkpoint_dir}")

            checkpoints = [p for p in checkpoint_dir.glob("*.ckpt") if p.name != "last.ckpt"]
            if not checkpoints:
                raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")

            # Find best checkpoint
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

        print(f"🤖 Loading Humanoid Gaussian Noise FM checkpoint: {checkpoint_path}")
        print(f"📍 Device: {device}")

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

        if isinstance(model_config, dict) and "_target_" in model_config:
            model_config = {k: v for k, v in model_config.items() if k != "_target_"}

        print(f"📋 Config source: {config_source}")
        print(f"📋 Final config - noise_std: {noise_std}")

        # Initialize system
        system = hparams.get("system")
        if system is None:
            print("🔧 Creating new Humanoid system (not found in hparams)")
            if hydra_config and "system" in hydra_config:
                system_config = hydra_config["system"]
                print(f"   Using system config from Hydra config")
                bounds_file = system_config.get("bounds_file", None)
                use_dynamic_bounds = system_config.get("use_dynamic_bounds", False)
                system = HumanoidSystem(bounds_file=bounds_file, use_dynamic_bounds=use_dynamic_bounds)
            else:
                print("   No Hydra system config found, using defaults")
                system = HumanoidSystem()
        else:
            print("✅ Restored Humanoid system from checkpoint")

        # Create model
        model = UniversalUNet(
            input_dim=model_config.get('input_dim', 68),  # 67 + 1 (time)
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
            noise_std=noise_std
        )

        # Load model weights
        print("🔄 Loading model state dict...")
        state_dict = checkpoint["state_dict"]
        model_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}

        if not model_state_dict:
            raise ValueError("No model weights found in checkpoint!")

        flow_matcher.model.load_state_dict(model_state_dict)

        # Move to device and set eval mode
        flow_matcher = flow_matcher.to(device)
        flow_matcher.eval()

        # Success summary
        print(f"\n✅ Model loaded successfully!")
        print(f"   Checkpoint: {checkpoint_path.name}")
        print(f"   System: {type(system).__name__}")
        print(f"   System bounds: Euclidean ±{system.euclidean_limit:.1f}, Sphere (unit norm)")
        print(f"   Noise std: {noise_std}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")

        return flow_matcher
