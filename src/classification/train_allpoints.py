#!/usr/bin/env python3
"""
Generic All-Points Classification Training Script

Trains a SimpleMLP classifier on all-points datasets for any system.
Automatically determines input dimension from config.

Usage:
    # Quadrotor 3D
    python src/classification/train_allpoints.py --config-name=train_quadrotor3d_allpoints_classification

    # Quadrotor 2D
    python src/classification/train_allpoints.py --config-name=train_quadrotor2d_allpoints_classification

    # CartPole
    python src/classification/train_allpoints.py --config-name=train_cartpole_allpoints_classification
"""
import sys
from pathlib import Path

# Add project root to path for Hydra to find local modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl

from src.model.simple_mlp import SimpleMLP


# Input dimensions for each system (after embedding)
INPUT_DIMS = {
    "quadrotor3d_allpoints_classification": 13,  # 13D quaternion kept as-is
    "quadrotor2d_allpoints_classification": 7,   # 6D -> 7D (sin/cos for theta)
    "cartpole_allpoints_classification": 5,      # 4D -> 5D (sin/cos for theta)
}


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_quadrotor3d_allpoints_classification",  # Default, override with --config-name
)
def train(cfg: DictConfig):
    """Train all-points classifier."""

    print("=" * 70)
    print(f"All-Points Classification Training")
    print("=" * 70)
    print(f"Config: {cfg.name}")
    print(f"Seed: {cfg.seed}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.base_lr}")
    print(f"Max epochs: {cfg.trainer.max_epochs}")
    print("=" * 70)
    print()

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Determine input dimension from config name
    config_name = cfg.name
    if config_name in INPUT_DIMS:
        input_dim = INPUT_DIMS[config_name]
    else:
        # Try to infer from config name
        if "quadrotor3d" in config_name:
            input_dim = 13
        elif "quadrotor2d" in config_name:
            input_dim = 7
        elif "cartpole" in config_name:
            input_dim = 5
        else:
            raise ValueError(f"Unknown config: {config_name}. Cannot determine input_dim.")

    print(f"Input dimension: {input_dim}")

    # Instantiate data module
    print("\nLoading data...")
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Model configuration
    output_dim = 1  # Binary classification
    hidden_channels = list(cfg.hidden_channels)

    # Create model
    print("\nCreating model...")
    weight_decay = cfg.get('weight_decay', 1e-4)
    model = SimpleMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_channels=hidden_channels,
        lr=cfg.base_lr,
        weight_decay=weight_decay,
        max_epochs=cfg.trainer.max_epochs,
    )

    print(f"  Architecture: {input_dim} -> {hidden_channels} -> {output_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Instantiate trainer
    print("\nCreating trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    trainer.fit(model, datamodule=data_module)

    # Test (only if test file was configured)
    if hasattr(data_module, 'test_file') and data_module.test_file:
        print("\n" + "=" * 70)
        print("Running test evaluation...")
        print("=" * 70)
        trainer.test(model, datamodule=data_module)
    else:
        print("\n(No test set configured, skipping test evaluation)")

    print("\n" + "=" * 70)
    print("Training complete!")
    if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    train()
