#!/usr/bin/env python3
"""Training script for Mountain Car Latent Conditional Flow Matching

Usage:
    python src/flow_matching/mountain_car/latent_conditional/train.py

With config overrides:
    python src/flow_matching/mountain_car/latent_conditional/train.py \
        batch_size=512 \
        flow_matching.latent_dim=4 \
        trainer.max_epochs=1000
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from pathlib import Path


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="train_mountain_car"
)
def main(cfg: DictConfig):
    """Main training function.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("Mountain Car Latent Conditional Flow Matching - Training")
    print("=" * 80)
    print()
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print()

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Instantiate system
    print("Initializing system...")
    system = hydra.utils.instantiate(cfg.system)
    print(system)
    print()

    # Instantiate data module
    print("Setting up data module...")
    data_module = hydra.utils.instantiate(cfg.data)
    print(f"✅ Data module created:")
    print(f"   State dim: {data_module.state_dim}")
    print(f"   Embedded dim: {data_module.embedded_dim}")
    print(f"   Batch size: {cfg.batch_size}")
    print()

    # Instantiate model
    print("Creating model...")
    model = hydra.utils.instantiate(cfg.model)
    print(model.get_model_info())
    print()

    # Create flow matcher
    print("Initializing flow matcher...")
    flow_matcher = hydra.utils.instantiate(
        cfg.flow_matcher,
        system=system,
        model=model,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        mae_val_frequency=cfg.flow_matching.mae_val_frequency,
        _recursive_=False
    )
    print()

    # Instantiate trainer
    print("Setting up trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)
    print(f"✅ Trainer configured:")
    print(f"   Max epochs: {cfg.trainer.max_epochs}")
    print(f"   Devices: {cfg.trainer.devices}")
    print(f"   Accelerator: {cfg.trainer.accelerator}")
    print()

    # Start training
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    print()

    trainer.fit(flow_matcher, data_module)

    print()
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
    print()
    print(f"Checkpoints saved to: {trainer.checkpoint_callback.dirpath}")
    print(f"Logs saved to: {trainer.logger.log_dir}")


if __name__ == "__main__":
    main()
