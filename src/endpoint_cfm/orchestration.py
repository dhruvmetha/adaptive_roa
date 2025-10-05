"""
EndpointCFM Orchestration Class

Main interface for training and using conditional flow matching models
for endpoint prediction in dynamical systems.
"""

import os
import shutil
from pathlib import Path
from typing import List, Union, Optional, Tuple
import numpy as np
import torch
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConfig

from ..flow_matching.conditional.flow_matcher import ConditionalFlowMatcher
from ..flow_matching.conditional.inference import ConditionalFlowMatchingInference
from ..flow_matching.base.config import FlowMatchingConfig
from ..data.circular_endpoint_data import CircularEndpointDataModule
from ..model.conditional_unet1d import ConditionalUNet1D


class EndpointCFM:
    """
    Main orchestration class for Conditional Flow Matching endpoint prediction.
    
    Provides high-level interface for:
    - Training models from trajectory data
    - Loading trained models
    - Performing parallel inference for endpoint prediction
    """
    
    def __init__(self):
        """Initialize EndpointCFM orchestrator"""
        self.model = None
        self.inferencer = None
        self.checkpoint_path = None
        
    def train(self, 
              trajectory_files: List[str],
              output_dir: str,
              max_epochs: int = 100,
              batch_size: int = 1024,
              learning_rate: float = 1e-3,
              device: str = "auto",
              num_workers: int = 4,
              validation_split: float = 0.1,
              noise_distribution: str = "uniform",
              noise_scale: float = 1.0,
              num_integration_steps: int = 100,
              hidden_dims: List[int] = None,
              time_emb_dim: int = 128) -> str:
        """
        Train conditional flow matching model from trajectory data.
        
        Args:
            trajectory_files: List of paths to trajectory data files
            output_dir: Directory to save model checkpoints and data
            max_epochs: Maximum number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            device: Device to train on ('auto', 'cpu', 'cuda', or specific GPU)
            num_workers: Number of data loading workers
            validation_split: Fraction of data for validation
            noise_distribution: Type of noise distribution ('uniform' or 'gaussian')
            noise_scale: Scale for gaussian noise (ignored for uniform)
            num_integration_steps: Number of ODE integration steps
            hidden_dims: Hidden dimensions for UNet (default: [64, 128, 256])
            time_emb_dim: Time embedding dimension
            
        Returns:
            Path to best checkpoint
        """
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
            
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Build endpoint dataset from trajectory files
        print("🔨 Building endpoint dataset from trajectory files...")
        dataset_path = self._build_endpoint_dataset(trajectory_files, output_path)
        
        # Step 2: Setup data module
        print("📊 Setting up data module...")
        data_module = CircularEndpointDataModule(
            data_file=str(dataset_path),
            batch_size=batch_size,
            num_workers=num_workers,
            validation_split=validation_split
        )
        
        # Step 3: Setup model architecture
        print("🧠 Creating model architecture...")
        unet_model = ConditionalUNet1D(
            input_dim=3,
            condition_dim=3,
            output_dim=3,
            hidden_dims=hidden_dims,
            time_emb_dim=time_emb_dim
        )
        
        # Step 4: Setup flow matching configuration
        config = FlowMatchingConfig(
            noise_distribution=noise_distribution,
            noise_scale=noise_scale,
            num_integration_steps=num_integration_steps,
            hidden_dims=tuple(hidden_dims),
            time_emb_dim=time_emb_dim
        )
        
        # Step 5: Create optimizer and scheduler
        optimizer = torch.optim.AdamW
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        
        # Step 6: Setup flow matching model
        flow_model = ConditionalFlowMatcher(
            model=unet_model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config
        )
        
        # Override optimizer configuration
        flow_model.optimizer_partial = lambda params: torch.optim.AdamW(
            params, lr=learning_rate, weight_decay=1e-4
        )
        flow_model.scheduler_partial = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        print(f"Model parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
        
        # Step 7: Setup trainer
        checkpoint_dir = output_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="epoch={epoch:03d}-step={step}-val_loss={val_loss:.6f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                every_n_epochs=1
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min", 
                patience=20,
                min_delta=1e-6,
                verbose=True
            )
        ]
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            devices=1 if device != "cpu" else None,
            accelerator=device,
            log_every_n_steps=50,
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
            enable_model_summary=True
        )
        
        # Step 8: Train the model
        print("🚀 Starting training...")
        trainer.fit(model=flow_model, datamodule=data_module)
        
        # Step 9: Get best checkpoint path
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        self.checkpoint_path = best_checkpoint
        
        print(f"✅ Training completed! Best checkpoint: {best_checkpoint}")
        
        return best_checkpoint
    
    def load_model(self, checkpoint_path: str, device: Optional[str] = None) -> None:
        """
        Load a trained conditional flow matching model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to load model on (None for auto-detection)
        """
        print(f"📥 Loading model from {checkpoint_path}...")
        
        self.checkpoint_path = checkpoint_path
        self.inferencer = ConditionalFlowMatchingInference(
            checkpoint_path=checkpoint_path,
            device=device
        )
        
        print("✅ Model loaded successfully!")
        
    def get_final_states(self, 
                        start_states: np.ndarray,
                        num_samples: int = 1,
                        num_steps: int = 100,
                        method: str = 'rk4') -> np.ndarray:
        """
        Get final states from start states using trained model.
        
        Args:
            start_states: Array of start states [N, 2] in (θ, θ̇) format
            num_samples: Number of samples to generate per start state
            num_steps: Number of ODE integration steps
            method: Integration method ('euler' or 'rk4')
            
        Returns:
            Final states array [N, num_samples, 2] in (θ, θ̇) format
            
        Raises:
            RuntimeError: If no model is loaded
        """
        if self.inferencer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        start_states = np.array(start_states)
        if start_states.ndim == 1:
            start_states = start_states.reshape(1, -1)
            
        N, state_dim = start_states.shape
        if state_dim != 2:
            raise ValueError(f"Start states must be [N, 2], got shape {start_states.shape}")
        
        print(f"🔮 Generating {num_samples} samples for {N} start states...")
        
        if num_samples == 1:
            # Single sample per start state - efficient batch processing
            final_states = self.inferencer.predict_endpoint(
                start_states, 
                num_steps=num_steps,
                method=method
            )  # [N, 2]
            
            # Add sample dimension: [N, 1, 2]
            return final_states.cpu().numpy().reshape(N, 1, 2)
        
        else:
            # Multiple samples per start state
            all_final_states = []
            
            for i, start_state in enumerate(start_states):
                # Generate multiple samples for this start state
                samples = self.inferencer.predict_multiple_samples(
                    start_state,
                    num_samples=num_samples,
                    num_steps=num_steps,
                    method=method
                )  # [num_samples, 2]
                
                all_final_states.append(samples.cpu().numpy())
                
                if (i + 1) % 10 == 0 or (i + 1) == N:
                    print(f"  Processed {i + 1}/{N} start states...")
            
            # Stack to [N, num_samples, 2]
            return np.stack(all_final_states, axis=0)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.inferencer is None:
            return {"status": "No model loaded"}
        
        return self.inferencer.get_model_info()
    
    def _build_endpoint_dataset(self, trajectory_files: List[str], output_dir: Path) -> Path:
        """
        Build endpoint dataset from trajectory files.
        
        Args:
            trajectory_files: List of trajectory file paths
            output_dir: Output directory for dataset
            
        Returns:
            Path to created endpoint dataset file
        """
        dataset_dir = output_dir / "endpoint_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        endpoint_file = dataset_dir / "endpoint_data.txt"
        
        print(f"  Processing {len(trajectory_files)} trajectory files...")
        
        with open(endpoint_file, 'w') as outf:
            total_pairs = 0
            
            for traj_file in trajectory_files:
                print(f"    Processing {traj_file}...")
                
                # Read trajectory file
                try:
                    trajectory = np.loadtxt(traj_file)
                    if trajectory.ndim == 1:
                        trajectory = trajectory.reshape(1, -1)
                        
                    # Assume trajectory format: [time_steps, features]
                    # Extract start and end states (assume first 2 cols are θ, θ̇)
                    start_state = trajectory[0, :2]   # First timestep
                    end_state = trajectory[-1, :2]    # Last timestep
                    
                    # Write as: start_θ start_θ̇ end_θ end_θ̇
                    outf.write(f"{start_state[0]:.6f} {start_state[1]:.6f} "
                              f"{end_state[0]:.6f} {end_state[1]:.6f}\n")
                    
                    total_pairs += 1
                    
                except Exception as e:
                    print(f"    Warning: Could not process {traj_file}: {e}")
                    continue
        
        print(f"  Created endpoint dataset with {total_pairs} pairs")
        print(f"  Saved to: {endpoint_file}")
        
        return endpoint_file