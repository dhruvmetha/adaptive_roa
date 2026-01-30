#!/usr/bin/env python3
"""
CartPole Classification Training Script

Trains a binary classifier to predict success/failure from initial CartPole state.

Usage:
    python src/classification/train_cartpole.py
    python src/classification/train_cartpole.py trainer.max_epochs=1000
"""
import sys
from pathlib import Path

# Add project root to path for Hydra to find modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl
from functools import partial


@hydra.main(version_base=None, config_path="../../configs", config_name="train_cartpole_classification")
def main(cfg: DictConfig):
    """Train CartPole classifier using Hydra configuration."""
    
    print("=" * 80)
    print("CartPole Classification Training")
    print("=" * 80)
    print(f"Config: {cfg.get('name', 'unnamed')}")
    print(f"Seed: {cfg.seed}")
    print(f"Max epochs: {cfg.trainer.max_epochs}")
    print("=" * 80)
    print()
    
    # Set random seeds
    pl.seed_everything(cfg.seed)
    
    # Instantiate data module
    print("Loading data...")
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.prepare_data()
    print()
    
    # Create optimizer and scheduler partials
    optimizer_partial = partial(
        torch.optim.AdamW,
        lr=cfg.base_lr,
        weight_decay=1e-4,
    )
    
    scheduler_partial = partial(
        torch.optim.lr_scheduler.CosineAnnealingLR,
        T_max=cfg.trainer.max_epochs,
        eta_min=1e-6,
    )
    
    # Create model config object
    # Input: 5D with manifold embedding (x_norm, sin_theta, cos_theta, x_dot_norm, theta_dot_norm)
    class ModelConfig:
        def __init__(self):
            self.input_dim = 5
            self.output_dim = 1
            self.hidden_channels = [128, 256, 128]
    
    model_config = ModelConfig()
    
    # Instantiate model
    print("Creating model...")
    from src.model.simple_mlp import SimpleMLP
    model = SimpleMLP(
        model=model_config,
        optimizer=optimizer_partial,
        scheduler=scheduler_partial,
    )
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Instantiate trainer
    print("Creating trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)
    print()
    
    # Train
    print("Starting training...")
    print("=" * 80)
    trainer.fit(model, data_module)
    
    # Test - use last model instead of loading checkpoint to avoid pickle issues
    print()
    print("=" * 80)
    print("Running test evaluation...")
    trainer.test(model, data_module)
    
    print()
    print("=" * 80)
    print("Training complete!")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

