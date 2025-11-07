"""
Training script for Trajectory MAE.

Usage:
    python src/representation/train.py
    python src/representation/train.py model.mask_ratio=0.5
    python src/representation/train.py model.mask_strategy=block
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
from pathlib import Path


@hydra.main(version_base=None, config_path="../../configs", config_name="train_trajectory_mae")
def main(cfg: DictConfig):
    """Main training function."""

    # Print configuration
    print("=" * 80)
    print("Training Trajectory MAE")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set seed
    if 'seed' in cfg:
        pl.seed_everything(cfg.seed, workers=True)

    # Instantiate data module
    print("\nInstantiating data module...")
    data_module = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    print("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)

    # Print model summary
    print("\nModel architecture:")
    print(f"  State dim: {model.hparams.state_dim}")
    print(f"  Embed dim: {model.hparams.embed_dim}")
    print(f"  Encoder depth: {model.hparams.encoder_depth}")
    print(f"  Decoder depth: {model.hparams.decoder_depth}")
    print(f"  Num heads: {model.hparams.num_heads}")
    print(f"  Mask ratio: {model.hparams.mask_ratio}")
    print(f"  Mask strategy: {model.hparams.mask_strategy}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Setup callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.model_checkpoint)
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stopping_callback = hydra.utils.instantiate(cfg.callbacks.early_stopping)
    callbacks.append(early_stopping_callback)

    # Learning rate monitor
    lr_monitor = hydra.utils.instantiate(cfg.callbacks.learning_rate_monitor)
    callbacks.append(lr_monitor)

    # Setup logger
    logger = hydra.utils.instantiate(cfg.logger)

    # Instantiate trainer
    print("\nInstantiating trainer...")
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print training info
    print("\nTraining configuration:")
    print(f"  Max epochs: {cfg.trainer.max_epochs}")
    print(f"  Batch size: {cfg.data.batch_size}")
    print(f"  Learning rate: {cfg.model.learning_rate}")
    print(f"  Weight decay: {cfg.model.weight_decay}")
    print(f"  Warmup epochs: {cfg.model.warmup_epochs}")
    print(f"  Gradient clip: {cfg.trainer.gradient_clip_val}")
    print(f"  Precision: {cfg.trainer.precision}")

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    trainer.fit(model, datamodule=data_module)

    # Print best checkpoint
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.6f}")

    return trainer, model


if __name__ == "__main__":
    main()
