"""
Training script for Quadrotor 2D binary classification.

Trains a SimpleMLP to predict success/failure from initial state.
Uses sin/cos embedding for theta (S^1 manifold).

Usage:
    python src/classification/train_quadrotor2d.py trainer.max_epochs=500
"""

import sys
from pathlib import Path

# Add project root to path for Hydra to find local modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl

from src.model.simple_mlp import SimpleMLP


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_quadrotor2d_classification",
)
def train(cfg: DictConfig):
    """Train Quadrotor 2D classifier."""

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Instantiate data module
    data_module = hydra.utils.instantiate(cfg.data)

    # Prepare data to get dataset statistics
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Model configuration
    # Input: 7D (x_norm, z_norm, sin_theta, cos_theta, x_dot_norm, z_dot_norm, theta_dot_norm)
    # Output: 1D (binary classification logit)
    input_dim = 7  # 6D state with sin/cos embedding for theta
    output_dim = 1
    hidden_channels = list(cfg.hidden_channels)

    # Create model with direct parameters (avoids serialization issues)
    model = SimpleMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_channels=hidden_channels,
        lr=cfg.base_lr,
        weight_decay=1e-5,
        max_epochs=cfg.trainer.max_epochs,
    )

    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim}D (6D state with sin/cos embedding for theta)")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Output dim: {output_dim}D (binary classification)")
    print(f"  Learning rate: {cfg.base_lr}")
    print(f"  Max epochs: {cfg.trainer.max_epochs}")

    # Instantiate trainer
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule=data_module)

    # Test on validation set
    print("\nTesting on validation set...")
    trainer.test(model, datamodule=data_module)

    print(f"\nTraining complete!")
    print(f"Checkpoints saved to: {cfg.trainer.callbacks[0].dirpath}")


if __name__ == "__main__":
    train()


