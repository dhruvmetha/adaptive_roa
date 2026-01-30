#!/usr/bin/env python3
"""
Quadrotor 3D Classification Training Script

Trains a binary classifier to predict success/failure from initial Quadrotor 3D state.

State space (13D):
- Position: x, y, z
- Orientation: qw, qx, qy, qz (unit quaternion)
- Linear velocity: x_dot, y_dot, z_dot
- Angular velocity: p, q, r

Usage:
    python src/classification/train_quadrotor3d.py
    python src/classification/train_quadrotor3d.py trainer.max_epochs=1000
    python src/classification/train_quadrotor3d.py data.max_samples=2000
"""
import sys
from pathlib import Path

# Add project root to path for Hydra to find modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl


@hydra.main(version_base=None, config_path="../../configs", config_name="train_quadrotor3d_classification")
def main(cfg: DictConfig):
    """Train Quadrotor 3D classifier using Hydra configuration."""

    print("=" * 80)
    print("Quadrotor 3D Classification Training")
    print("=" * 80)
    print(f"Config: {cfg.get('name', 'unnamed')}")
    print(f"Seed: {cfg.seed}")
    print(f"Max epochs: {cfg.trainer.max_epochs}")
    print(f"Max samples: {cfg.data.get('max_samples', 'all')}")
    print(f"Balance samples: {cfg.data.get('balance_samples', False)}")
    print("=" * 80)
    print()

    # Set random seeds
    pl.seed_everything(cfg.seed)

    # Instantiate data module
    print("Loading data...")
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.prepare_data()
    print()

    # Instantiate model
    print("Creating model...")
    from src.model.simple_mlp import SimpleMLP
    
    hidden_channels = cfg.get('hidden_channels', [256, 512, 256])
    # Create model with simple parameters instead of partials to avoid serialization issues
    model = SimpleMLP(
        input_dim=13,  # Full 13D state
        output_dim=1,  # Binary classification (logit)
        hidden_channels=hidden_channels,
        lr=cfg.base_lr,
        weight_decay=1e-4,
        max_epochs=cfg.trainer.max_epochs,
    )
    print(f"Model: {model.__class__.__name__}")
    print(f"Input dim: 13")
    print(f"Hidden layers: {hidden_channels}")
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
