#!/usr/bin/env python3
"""
Train quadrotor classifier.

Usage:
    # Quadrotor 2D
    python src/classification/train.py --config-name=train_quadrotor2d_classification

    # Quadrotor 3D
    python src/classification/train.py --config-name=train_quadrotor3d_classification
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl

from src.model.simple_mlp import SimpleMLP
from src.data.quadrotor_classification_data import (
    QuadrotorClassificationDataModule,
    EMBEDDERS,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_quadrotor2d_classification",
)
def train(cfg: DictConfig):
    """Train quadrotor classifier."""

    print("=" * 70)
    print(f"Quadrotor Classification Training")
    print("=" * 70)
    print(f"System: {cfg.system}")
    print(f"Seed: {cfg.seed}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.base_lr}")
    print(f"Max epochs: {cfg.trainer.max_epochs}")
    print("=" * 70)
    print()

    # Set seed
    pl.seed_everything(cfg.seed)

    # Get input dimension from embedder
    embedder = EMBEDDERS[cfg.system]
    input_dim = embedder.OUTPUT_DIM

    print(f"Input dimension: {input_dim}")

    # Create data module
    print("\nLoading data...")
    data_module = QuadrotorClassificationDataModule(
        system=cfg.system,
        train_file=cfg.data.train_file,
        val_file=cfg.data.val_file,
        test_file=cfg.data.get('test_file'),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balance_samples=cfg.data.get('balance_samples', True),
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Create model
    print("\nCreating model...")
    hidden_channels = list(cfg.hidden_channels)
    weight_decay = cfg.get('weight_decay', 1e-4)

    model = SimpleMLP(
        input_dim=input_dim,
        output_dim=1,  # Binary classification
        hidden_channels=hidden_channels,
        lr=cfg.base_lr,
        weight_decay=weight_decay,
        max_epochs=cfg.trainer.max_epochs,
    )

    print(f"  Architecture: {input_dim} -> {hidden_channels} -> 1")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    print("\nCreating trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    trainer.fit(model, datamodule=data_module)

    # Test
    if data_module.test_file:
        print("\n" + "=" * 70)
        print("Running test evaluation...")
        print("=" * 70)
        trainer.test(model, datamodule=data_module)
    else:
        print("\n(No test set configured)")

    print("\n" + "=" * 70)
    print("Training complete!")
    if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    train()
