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
import sys
sys.path.append('/common/home/dm1487/robotics_research/tripods/olympics-classifier/flow_matching')

from flow_matching.utils.manifolds import Product, Euclidean, Sphere

from src.flow_matching.base.flow_matcher import BaseFlowMatcher
from src.systems.base import DynamicalSystem


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
                 mae_val_frequency: int = 10,
                 noise_std: float = 0.001):
        """
        Initialize Humanoid conditional flow matcher with FB FM integration

        Args:
            system: DynamicalSystem (Humanoid with ℝ³⁴ × S² × ℝ³⁰ structure)
            model: HumanoidUNet model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            model_config: Configuration dict
            mae_val_frequency: Compute MAE validation every N epochs
        """
        super().__init__(system, model, optimizer, scheduler, model_config, mae_val_frequency)

        # Configurable standard deviation for initial noise sampling
        self.noise_std = float(noise_std)
    
        print("✅ Initialized Humanoid CFM with Facebook Flow Matching:")
        print(f"   - Manifold: ℝ³⁴ × S² × ℝ³⁰ (Euclidean × Sphere × Euclidean)")
        print(f"   - Path: GeodesicProbPath with CondOTScheduler")
        print(f"   - MAE validation frequency: every {mae_val_frequency} epochs")
        print(f"   - Initial noise std: {self.noise_std}")

    def _create_manifold(self):
        """
        Create ℝ³⁴ × S² × ℝ³⁰ manifold for Humanoid

        Manifold structure:
        - Euclidean(34): Dims 0-33
        - Sphere(3): Dims 34-36 (3D unit vector on S²)
        - Euclidean(30): Dims 37-66
        """
        return Product(
            input_dim=67,
            manifolds=[
                (Euclidean(), 34),  # First Euclidean block
                (Sphere(), 3),      # Sphere manifold (3D unit vector)
                (Euclidean(), 30)   # Second Euclidean block
            ]
        )
        
        # return Product(
        #     input_dim=67,
        #     manifolds=[
        #         (Euclidean(), 67),  # First Euclidean block
        #     ]
        # )

    def _get_start_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract start states from batch"""
        return batch["start_state"]

    def _get_end_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract end states from batch"""
        return batch["end_state"]

    def _get_dimension_name(self, dim_idx: int) -> str:
        """
        Get human-readable name for dimension

        After expansion in compute_endpoint_mae_per_dim(), we have 67 dimensions:
        - 34 for Euclidean block 1 (dims 0-33)
        - 3 for Sphere (dims 34-36, all with same distance value)
        - 30 for Euclidean block 2 (dims 37-66)

        Args:
            dim_idx: Dimension index (0-66)

        Returns:
            Human-readable name
        """
        if 0 <= dim_idx < 34:
            return f"euclidean1_{dim_idx}"
        elif 34 <= dim_idx < 37:
            # Sphere dimensions (all share same geodesic distance)
            sphere_names = ["sphere_x", "sphere_y", "sphere_z"]
            return sphere_names[dim_idx - 34]
        elif 37 <= dim_idx < 67:
            # Euclidean block 2: dims 37-66
            return f"euclidean2_{dim_idx - 37}"
        else:
            return f"dim_{dim_idx}"

    def sample_noisy_input(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample noisy input uniformly in ℝ³⁴ × S² × ℝ³⁰ space (per-dimension)

        For Euclidean components: sample uniformly within per-dimension bounds
        For Sphere component: sample uniformly on unit sphere

        Args:
            batch_size: Number of samples
            device: Device to create tensors on

        Returns:
            Noisy states [batch_size, 67]
        """
        noisy_states = torch.randn(batch_size, 67, device=device) * self.noise_std
        
        
        noisy_states = self.manifold.projx(noisy_states)
        
        
        return noisy_states

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
            # Generate multiple samples
            endpoints_raw = self.predict_endpoint(start_states, num_steps)
            all_endpoints.append(endpoints_raw)

        # Concatenate all samples: [B*num_samples, 67]
        return torch.cat(all_endpoints, dim=0)

    def compute_endpoint_mae_per_dim(self,
                                    predicted_endpoints: torch.Tensor,
                                    true_endpoints: torch.Tensor) -> torch.Tensor:
        """
        Compute MAE per dimension, expanding Sphere component for better logging

        The Product manifold returns 65 distances:
        - 34 for Euclidean block 1 (dims 0-33)
        - 1 for Sphere component (dims 34-36 as a group)
        - 30 for Euclidean block 2 (dims 37-66)

        For logging purposes, we expand this to 67 by replicating the single
        sphere distance across all 3 sphere dimensions.

        Args:
            predicted_endpoints: Predicted endpoints [B, 67]
            true_endpoints: True endpoints [B, 67]

        Returns:
            mae_per_dim: MAE for each dimension [67] (expanded from 65 components)
        """
        # Get distances from manifold (returns 65 components)
        mae_components = self.manifold.dist(predicted_endpoints, true_endpoints)  # [B, 65]
        mae_components = mae_components.mean(dim=0)  # [65]
        if mae_components.shape[0] == 67:
            return mae_components
        else:
            
            # Expand to 67 dimensions by replicating sphere distance
            # Components: [0-33: Euclidean1, 34: Sphere, 35-64: Euclidean2]
            # Expand to: [0-33: Euclidean1, 34-36: Sphere×3, 37-66: Euclidean2]

            euclidean1 = mae_components[:34]           # [34]
            sphere_dist = mae_components[34]           # scalar
            euclidean2 = mae_components[35:]           # [30]

            # Replicate sphere distance 3 times
            sphere_expanded = sphere_dist.repeat(3)    # [3]

            # Concatenate: [34] + [3] + [30] = [67]
            mae_per_dim = torch.cat([euclidean1, sphere_expanded, euclidean2])

            return mae_per_dim
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for Humanoid Latent Conditional Flow Matching
        """
        flow_loss = super().training_step(batch, batch_idx)
        
        x_1 = self.normalize_state(batch["end_state"])
        condition = self.normalize_state(batch["start_state"])

        log_p0 = lambda x: -0.5 * (x.pow(2).sum(-1) + 67 * torch.log(torch.tensor(2.0) * torch.pi))
        
        step_size = 1.0/10
        
        _, log_loss = self.solver.compute_likelihood( x_1=x_1, log_p0=log_p0, step_size=step_size, exact_divergence=False, enable_grad=True, condition=condition)
        
        
        
        
        return flow_loss + 0.1 * (-log_loss.mean()) 
    
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step for Humanoid Latent Conditional Flow Matching
        """
        flow_loss = super().validation_step(batch, batch_idx)
        return flow_loss

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
        from pathlib import Path
        from src.systems.humanoid import HumanoidSystem
        from src.model.humanoid_unet import HumanoidUNet
        from src.model.humanoid_unet_film import HumanoidUNetFiLM

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

        # No latent variables needed anymore

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

        # Determine which model class to use based on config
        model_target = model_config.get('_target_', 'src.model.humanoid_unet.HumanoidUNet')
        is_film_model = 'film' in model_target.lower()

        # Create model architecture
        if is_film_model:
            model = HumanoidUNetFiLM(
                embedded_dim=model_config.get('embedded_dim', 67),
                condition_dim=model_config.get('condition_dim', 67),
                time_emb_dim=model_config.get('time_emb_dim', 128),
                hidden_dims=model_config.get('hidden_dims', [256, 512, 512, 256]),
                output_dim=model_config.get('output_dim', 67),
                use_input_embeddings=model_config.get('use_input_embeddings', False),
                input_emb_dim=model_config.get('input_emb_dim', 128),
                film_cond_dim=model_config.get('film_cond_dim', 256),
                film_hidden_dims=model_config.get('film_hidden_dims', None),
                dropout_p=model_config.get('dropout_p', 0.0),
                residual_scale=model_config.get('residual_scale', None),
                zero_init_blocks=model_config.get('zero_init_blocks', True),
                zero_init_out=model_config.get('zero_init_out', False)
            )
        else:
            model = HumanoidUNet(
                embedded_dim=model_config.get('embedded_dim', 67),
                condition_dim=model_config.get('condition_dim', 67),
                time_emb_dim=model_config.get('time_emb_dim', 128),
                hidden_dims=model_config.get('hidden_dims', [256, 512, 512, 256]),
                output_dim=model_config.get('output_dim', 67),
                use_input_embeddings=model_config.get('use_input_embeddings', False),
                input_emb_dim=model_config.get('input_emb_dim', 128)
            )

        # Create flow matcher instance
        flow_matcher = cls(
            system=system,
            model=model,
            optimizer=None,
            scheduler=None,
            model_config=model_config
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
        print(f"   System bounds: Per-dimension (ℝ³⁴ × S² × ℝ³⁰)")
        print(f"   Model architecture: {model_config.get('hidden_dims', 'unknown')}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")

        return flow_matcher
