"""
Latent Conditional Flow Matching inference module
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.flow_matching.latent_conditional.flow_matcher import LatentConditionalFlowMatcher
from src.model.latent_conditional_unet1d import LatentConditionalUNet1D
from src.systems.pendulum_lcfm import PendulumSystemLCFM
from src.manifold_integration import TheseusIntegrator


def find_best_checkpoint(folder_path: str) -> str:
    """
    Find the best checkpoint in a timestamped folder.
    
    Algorithm:
    1. Look in folder_path/checkpoints/ for .ckpt files
    2. Exclude 'last.ckpt' (always use validation-based checkpoints)
    3. Parse filename format: epoch=027-step=980-val_loss=0.087929.ckpt
    4. Return the checkpoint with the LOWEST validation loss (best performance)
    
    Args:
        folder_path: Path to timestamped folder (e.g., outputs/name/2024-01-15_14-30-45)
        
    Returns:
        Path to the best checkpoint
        
    Raises:
        FileNotFoundError: If no valid checkpoints found
        ValueError: If validation loss cannot be parsed from filenames
    """
    folder_path = Path(folder_path)
    checkpoints_dir = folder_path / "checkpoints"
    
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Look for checkpoint files (excluding last.ckpt)
    checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
    checkpoint_files = [f for f in checkpoint_files if f.name != "last.ckpt"]
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No validation-based checkpoint files found in: {checkpoints_dir}")
    
    # Find the checkpoint with the lowest validation loss
    best_checkpoint = None
    best_val_loss = float('inf')
    parse_errors = []
    
    print(f"🔍 Parsing {len(checkpoint_files)} checkpoint files...")
    
    for ckpt_file in checkpoint_files:
        # Parse filename to extract val_loss
        # Expected format: epoch=027-step=980-val_loss=0.087929.ckpt
        filename = ckpt_file.stem
        
        if "val_loss=" not in filename:
            parse_errors.append(f"No 'val_loss=' in filename: {filename}")
            continue
            
        try:
            # Handle both single and double equals patterns
            # e.g., "val_loss=0.006688" or "val_loss=val_loss=0.006688"  
            parts = filename.split("val_loss=")
            if len(parts) >= 2:
                # Take the last part (handles double equals case)
                val_loss_part = parts[-1].split("-")[0]
                print(f"  📄 {ckpt_file.name}: val_loss_part='{val_loss_part}'")
                
                val_loss = float(val_loss_part)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint = ckpt_file
            else:
                parse_errors.append(f"Unexpected val_loss format in {filename}")
            
        except (ValueError, IndexError) as e:
            parse_errors.append(f"Could not parse val_loss from {filename}: {e}")
            continue
    
    if best_checkpoint is None:
        error_msg = f"Could not parse validation loss from any checkpoint filenames. Errors:\n"
        error_msg += "\n".join(f"  - {err}" for err in parse_errors)
        raise ValueError(error_msg)
    
    print(f"🏆 Best checkpoint selected (val_loss={best_val_loss:.6f}): {best_checkpoint.name}")
    return str(best_checkpoint)


def find_hydra_config(folder_path: str) -> str:
    """
    Find the hydra config.yaml in a timestamped folder.
    
    Algorithm:
    1. Look for folder_path/.hydra/config.yaml (preferred - contains resolved config)
    2. If not found, raise error (no fallbacks)
    
    Args:
        folder_path: Path to timestamped folder (e.g., outputs/name/2024-01-15_14-30-45)
        
    Returns:
        Path to .hydra/config.yaml
        
    Raises:
        FileNotFoundError: If .hydra/config.yaml not found
    """
    folder_path = Path(folder_path)
    
    # Look for .hydra/config.yaml (contains the resolved configuration)
    hydra_config = folder_path / ".hydra" / "config.yaml"
    if hydra_config.exists():
        print(f"📋 Using Hydra config: {hydra_config}")
        return str(hydra_config)
    
    raise FileNotFoundError(f"Hydra config not found at: {hydra_config}")  


def load_config_from_hydra(config_path: str) -> dict:
    """
    Load configuration from Hydra config file
    
    Args:
        config_path: Path to the config.yaml file
        
    Returns:
        Configuration dictionary
    """
    try:
        # Clear any existing Hydra global state
        GlobalHydra.instance().clear()
        
        # Load the config file directly with OmegaConf
        config = OmegaConf.load(config_path)
        
        print(f"Loaded config from: {config_path}")
        # Try to resolve, but if it fails, return unresolved config
        try:
            return OmegaConf.to_container(config, resolve=True)
        except Exception as resolve_error:
            print(f"Warning: Could not resolve interpolations, using unresolved config: {resolve_error}")
            return OmegaConf.to_container(config, resolve=False)
        
    except Exception as e:
        print(f"Warning: Could not load Hydra config from {config_path}: {e}")
        return {}


class LatentConditionalFlowMatchingInference:
    """
    Inference module for Latent Conditional Flow Matching
    
    Integrates from noisy input to predicted endpoint using learned velocity field
    """
    
    def __init__(self, folder_path: str, device: Optional[str] = None):
        """
        Initialize LCFM inference from timestamped folder
        
        Args:
            folder_path: Path to timestamped folder (e.g., outputs/name/2024-01-15_14-30-45)
            device: Device to run inference on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Validate input is a directory
        path_obj = Path(folder_path)
        if not path_obj.is_dir():
            raise ValueError(f"Path must be a timestamped directory: {folder_path}")
        
        print(f"🗂️ Loading from timestamped folder: {folder_path}")
        
        # Find best checkpoint in the folder (raises error if not found)
        checkpoint_path = find_best_checkpoint(folder_path)
        
        # Find config in the folder (raises error if not found)
        config_path = find_hydra_config(folder_path)
        
        # Load checkpoint (disable weights_only for Lightning checkpoints)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load full Hydra configuration if available
        full_config = {}
        if config_path:
            full_config = load_config_from_hydra(config_path)
        
        # Initialize system (pendulum for S¹ × ℝ)
        self.system = PendulumSystemLCFM()
        
        # Get configuration from checkpoint hyperparameters (most reliable)
        hparams = checkpoint.get("hyper_parameters", {})
        self.latent_dim = hparams.get("latent_dim", 2)
        model_cfg = hparams.get("config", {})
        print(f"Using latent_dim from checkpoint hparams: {self.latent_dim}")
        
        # If model config not in hyperparameters, try full config as fallback
        if not model_cfg and full_config and "model" in full_config:
            model_cfg = full_config.get("model", {})
            flow_matching_cfg = full_config.get("flow_matching", {})
            self.latent_dim = flow_matching_cfg.get("latent_dim", self.latent_dim)
            print(f"Fallback to Hydra config for model parameters")
        
        # Override latent_dim in model config
        if isinstance(model_cfg, dict):
            model_cfg = dict(model_cfg)  # Make a copy
        else:
            # Convert DictConfig to regular dict
            model_cfg = OmegaConf.to_container(model_cfg) if hasattr(model_cfg, '_content') else dict(model_cfg)
        
        model_cfg["latent_dim"] = self.latent_dim
        print(f"Loading model with config: {model_cfg}")
        
        # Use Hydra to instantiate model from config
        try:
            self.model = hydra.utils.instantiate(model_cfg).to(self.device)
        except Exception as e:
            print(f"Error instantiating model with Hydra: {e}")
            print(f"Available model config keys: {list(model_cfg.keys())}")
            # Fallback: use config values with sensible defaults
            from src.model.latent_conditional_unet1d import LatentConditionalUNet1D
            self.model = LatentConditionalUNet1D(
                embedded_dim=model_cfg.get('embedded_dim', 3),
                latent_dim=model_cfg.get('latent_dim', self.latent_dim),
                condition_dim=model_cfg.get('condition_dim', 3),
                time_emb_dim=model_cfg.get('time_emb_dim', 16),
                hidden_dims=model_cfg.get('hidden_dims', [128, 128]),
                output_dim=model_cfg.get('output_dim', 2),
                use_input_embeddings=model_cfg.get('use_input_embeddings', False),
                input_emb_dim=model_cfg.get('input_emb_dim', 64)
            ).to(self.device)
        
        # Load model weights
        state_dict = checkpoint["state_dict"]
        model_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        
        # Initialize integrator for ODE solving
        self.integrator = TheseusIntegrator(self.system)
        
        print(f"✅ Successfully loaded LCFM model!")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Config source: {'Hydra config' if config_path else 'Checkpoint hparams'}")
        print(f"   System: {self.system}")
        print(f"   Latent dim: {self.latent_dim}")
    
    
    def sample_noisy_input(self, batch_size: int) -> torch.Tensor:
        """
        Sample noisy input in S¹ × ℝ space
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Noisy states [batch_size, 2] as (θ, θ̇)
        """
        # θ ~ Uniform[-π, π] 
        theta = torch.rand(batch_size, 1, device=self.device) * 2 * torch.pi - torch.pi
        
        # θ̇ ~ Uniform[-1, 1] (already normalized)
        theta_dot = torch.rand(batch_size, 1, device=self.device) * 2 - 1
        
        noisy_input = torch.cat([theta, theta_dot], dim=1)
        
        # Debug: Print first few values for first call
        if not hasattr(self, '_noise_sample_count'):
            self._noise_sample_count = 0
        self._noise_sample_count += 1
        
        if self._noise_sample_count <= 3:
            print(f"🎯 Noise sample {self._noise_sample_count}: First noise = {noisy_input[0].cpu().numpy()}")
        
        return noisy_input
    
    def sample_latent(self, batch_size: int) -> torch.Tensor:
        """
        Sample Gaussian latent vector
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Latent vectors [batch_size, latent_dim]
        """
        # Ensure truly random sampling by resetting generator state occasionally
        if not hasattr(self, '_sample_count'):
            self._sample_count = 0
        self._sample_count += 1
        
        latents = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        # Debug: Print first few values for first few calls
        if self._sample_count <= 3:
            print(f"🎲 Sample {self._sample_count}: First latent = {latents[0].cpu().numpy()}")
        
        return latents
    
    def velocity_function(self, 
                         t: float, 
                         x_t: torch.Tensor,      # [B, 2] in S¹ × ℝ 
                         z: torch.Tensor,        # [B, 2] latent
                         condition: torch.Tensor # [B, 3] embedded condition
                         ) -> torch.Tensor:
        """
        Velocity function for ODE integration
        
        Args:
            t: Current time (scalar)
            x_t: Current state [B, 2] in S¹ × ℝ
            z: Latent vectors [B, 2]
            condition: Embedded conditioning [B, 3]
            
        Returns:
            Velocity [B, 2] in tangent space
        """
        batch_size = x_t.shape[0]
        
        # Embed current state
        x_t_embedded = self.system.embed_state(x_t)
        
        # Create time tensor
        t_tensor = torch.full((batch_size,), t, device=self.device)
        
        # Predict velocity
        with torch.no_grad():
            velocity = self.model(x_t_embedded, t_tensor, z, condition)
        
        return velocity
    
    def integrate_trajectory(self, 
                           start_state: torch.Tensor,  # [B, 2] in S¹ × ℝ
                           num_steps: int = 100,
                           latent: Optional[torch.Tensor] = None
                           ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Integrate from noisy input to predicted endpoint
        
        Args:
            start_state: Conditioning start states [B, 2] 
            num_steps: Number of integration steps
            latent: Optional latent vectors [B, 2]. If None, will sample.
            
        Returns:
            Tuple of (final_states, trajectory_states)
        """
        batch_size = start_state.shape[0]
        
        # Sample noisy inputs (initial conditions for integration)
        x_current = self.sample_noisy_input(batch_size)
        
        # Sample or use provided latent vectors
        if latent is None:
            z = self.sample_latent(batch_size)
        else:
            z = latent
        
        # Embed conditioning (start states)
        condition_embedded = self.system.embed_state(start_state)
        
        # Integration using TheseusIntegrator batch method
        final_states, trajectory = self.integrator.integrate_batch(
            start_states=x_current,
            velocity_func=lambda t, x: self.velocity_function(t, x, z, condition_embedded),
            num_steps=num_steps
        )
        
        return final_states, trajectory
    
    def predict_endpoint(self, 
                        start_state: torch.Tensor,  # [B, 2] in S¹ × ℝ
                        num_steps: int = 100,
                        num_samples: int = 1
                        ) -> torch.Tensor:
        """
        Predict endpoint given start state
        
        Args:
            start_state: Start states [B, 2] in S¹ × ℝ
            num_steps: Number of integration steps
            num_samples: Number of samples per start state (due to stochastic latent)
            
        Returns:
            Predicted endpoints [B*num_samples, 2] in S¹ × ℝ
        """
        batch_size = start_state.shape[0]
        
        if num_samples == 1:
            # Single sample per start state
            final_states, _ = self.integrate_trajectory(start_state, num_steps)
            return final_states
        else:
            # Multiple samples per start state
            all_endpoints = []
            
            for _ in range(num_samples):
                # Repeat start_state for this sample
                start_repeated = start_state.repeat(1, 1)  # [B, 2]
                
                # Integrate with different latent sample
                final_states, _ = self.integrate_trajectory(start_repeated, num_steps)
                all_endpoints.append(final_states)
            
            # Concatenate all samples
            return torch.cat(all_endpoints, dim=0)  # [B*num_samples, 2]
    
    def predict_trajectory(self, 
                          start_state: torch.Tensor,  # [B, 2]
                          num_steps: int = 100
                          ) -> List[torch.Tensor]:
        """
        Predict full trajectory from noisy input to endpoint
        
        Args:
            start_state: Conditioning start states [B, 2]
            num_steps: Number of integration steps
            
        Returns:
            List of trajectory states [num_steps+1, B, 2]
        """
        _, trajectory = self.integrate_trajectory(start_state, num_steps)
        return trajectory
    
    def __repr__(self) -> str:
        return (f"LatentConditionalFlowMatchingInference("
                f"system={type(self.system).__name__}, "
                f"device={self.device})")