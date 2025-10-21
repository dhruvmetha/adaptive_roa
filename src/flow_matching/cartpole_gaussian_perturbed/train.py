#!/usr/bin/env python3
"""
Training script for CartPole Gaussian-Perturbed Flow Matching

Simplified variant WITHOUT latent variables or conditioning.
Initial states sampled from Gaussian distributions centered at start states.

Usage:
    python src/flow_matching/cartpole_gaussian_perturbed/train.py

    # With config overrides:
    python src/flow_matching/cartpole_gaussian_perturbed/train.py \
        flow_matching.noise_std=0.2 \
        batch_size=512
"""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from src.flow_matching.cartpole_gaussian_perturbed.flow_matcher import CartPoleGaussianPerturbedFlowMatcher


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_cartpole_gaussian_perturbed")
def main(cfg: DictConfig):
    """
    Main training function for Gaussian-Perturbed Flow Matching

    Key differences from latent conditional training:
    - Model has NO latent_dim or condition_dim parameters
    - Flow matcher uses noise_std instead of latent_dim
    - Simpler model architecture with fewer parameters
    """

    print("="*80)
    print("🚀 CartPole Gaussian-Perturbed Flow Matching Training")
    print("="*80)
    print("\n📋 Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("="*80)

    # Set random seed for reproducibility
    if cfg.get('seed'):
        pl.seed_everything(cfg.seed, workers=True)
        print(f"🎲 Random seed: {cfg.seed}")

    # Set CUDA device if specified
    if 'device' in cfg and 'cuda_visible_devices' in cfg.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device.cuda_visible_devices)
        print(f"🖥️  CUDA_VISIBLE_DEVICES: {cfg.device.cuda_visible_devices}")

    # Instantiate system
    print("\n🔧 Initializing System...")
    system = hydra.utils.instantiate(cfg.system)
    print(f"   System: {system}")

    # Instantiate data module
    print("\n📂 Initializing Data Module...")
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.setup("fit")
    print(f"   Train samples: {len(data_module.train_dataset)}")
    print(f"   Val samples: {len(data_module.val_dataset)}")
    print(f"   Batch size (train): {cfg.batch_size}")
    print(f"   Batch size (val): {cfg.get('val_batch_size', cfg.batch_size)}")

    # Instantiate model (SIMPLIFIED - no latent_dim, no condition_dim)
    print("\n🤖 Initializing Model...")
    model = hydra.utils.instantiate(cfg.model)
    model_info = model.get_model_info()
    print(f"   Model: {model_info['model_type']}")
    print(f"   Embedded dim: {model_info['embedded_dim']}")
    print(f"   Time embedding dim: {model_info['time_emb_dim']}")
    print(f"   Hidden dims: {model_info['hidden_dims']}")
    print(f"   Output dim: {model_info['output_dim']}")
    print(f"   Total parameters: {model_info['total_parameters']:,}")
    print(f"   ⚠️  NO latent variables")
    print(f"   ⚠️  NO conditioning on start state")

    # Instantiate flow matcher
    print("\n🌊 Initializing Flow Matcher...")
    flow_matcher = CartPoleGaussianPerturbedFlowMatcher(
        system=system,
        model=model,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        noise_std=cfg.flow_matching.noise_std,
        mae_val_frequency=cfg.flow_matching.get('mae_val_frequency', 10)
    )
    print(f"   Gaussian noise std: {cfg.flow_matching.noise_std}")
    print(f"   MAE validation frequency: every {cfg.flow_matching.get('mae_val_frequency', 10)} epochs")

    # Instantiate trainer
    print("\n🏋️  Initializing Trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)
    print(f"   Max epochs: {cfg.trainer.max_epochs}")
    print(f"   Accelerator: {cfg.trainer.accelerator}")
    print(f"   Devices: {cfg.trainer.devices}")
    print(f"   Precision: {cfg.trainer.precision}")

    # Print training info
    print("\n" + "="*80)
    print("📊 Training Summary")
    print("="*80)
    print(f"Training samples:    {len(data_module.train_dataset)}")
    print(f"Validation samples:  {len(data_module.val_dataset)}")
    print(f"Batch size:          {cfg.batch_size}")
    print(f"Learning rate:       {cfg.get('base_lr', cfg.optimizer.lr)}")
    print(f"Max epochs:          {cfg.trainer.max_epochs}")
    print(f"Model parameters:    {model_info['total_parameters']:,}")
    print(f"Gaussian noise std:  {cfg.flow_matching.noise_std}")
    print("="*80)

    # Start training
    print("\n🎯 Starting Training...")
    print("="*80)

    trainer.fit(flow_matcher, datamodule=data_module)

    # Training complete
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)

    # Print best checkpoint info
    if hasattr(trainer.checkpoint_callback, 'best_model_path'):
        best_path = trainer.checkpoint_callback.best_model_path
        best_score = trainer.checkpoint_callback.best_model_score
        print(f"🏆 Best checkpoint: {best_path}")
        print(f"   Best val_loss: {best_score:.6f}")

    print(f"📁 Output directory: {os.getcwd()}")
    print("="*80)


if __name__ == "__main__":
    main()
