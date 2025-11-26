#!/usr/bin/env python3
"""
Training script for Trajectory Contrastive Representation Learning

Uses Hydra target instantiation for clean, config-driven architecture.

Self-supervised learning from trajectory temporal structure using:
- Temporal triplet sampling (positives: nearby timesteps, negatives: far timesteps)
- InfoNCE or Triplet loss
- L2-normalized embeddings for metric learning

Usage:
    # Train Humanoid
    python src/contrastive/train_contrastive.py --config-name=train_humanoid_repr

    # Train CartPole
    python src/contrastive/train_contrastive.py --config-name=train_cartpole_repr

    # Train Pendulum
    python src/contrastive/train_contrastive.py --config-name=train_pendulum_repr

    # Override parameters
    python src/contrastive/train_contrastive.py --config-name=train_humanoid_repr \
        model.embedding_dim=256 \
        contrastive.temperature=0.1 \
        trainer.max_epochs=100 \
        batch_size=512
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl
from src.contrastive.contrastive_learner import ContrastiveLearner


@hydra.main(version_base=None, config_path="../../configs", config_name="train_humanoid_repr")
def main(cfg: DictConfig):
    """
    Training for Trajectory Contrastive Representation Learning using Hydra instantiation

    All components are instantiated from config using Hydra's _target_ mechanism.
    """

    # Print configuration
    print("="*80)
    print("🚀 Trajectory Contrastive Representation Learning")
    print("="*80)
    print(f"📋 Config: {cfg.get('name', 'contrastive_repr')}")
    print(f"🎲 Seed: {cfg.seed}")
    print(f"📉 Loss: {cfg.contrastive.loss_type}")
    if cfg.contrastive.loss_type == "triplet":
        print(f"   Margin: {cfg.contrastive.margin}")
    else:
        print(f"   Temperature: {cfg.contrastive.temperature}")
    print("="*80)
    print()

    # Set random seeds
    pl.seed_everything(cfg.seed)

    # ========================================================================
    # INSTANTIATE ALL COMPONENTS FROM CONFIG
    # ========================================================================

    print("📥 Instantiating components from config...")
    print()

    # System (HumanoidSystem, CartPoleSystem, etc.)
    print("  🔧 System...")
    system = hydra.utils.instantiate(cfg.system)
    print(f"     ✅ {system.__class__.__name__}")
    print(f"        {system}")
    print()

    # Data module
    print("  📊 Data module...")
    data_module = hydra.utils.instantiate(cfg.data)
    print(f"     ✅ {data_module.__class__.__name__}")
    print(f"        Train dir: {data_module.train_trajectory_dir}")
    print(f"        Batch size: {cfg.batch_size}")
    print(f"        Temporal window: ±{data_module.temporal_window} timesteps")
    print(f"        Num negatives: {data_module.num_negatives}")
    print()

    # Encoder model
    print("  🏗️  Encoder...")
    encoder = hydra.utils.instantiate(cfg.model)
    print(f"     ✅ {encoder.__class__.__name__}")
    print(f"        Input dim: {encoder.input_dim}")
    print(f"        Embedding dim: {encoder.embedding_dim}")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"        Parameters: {total_params:,}")
    print()

    # Optimizer (keep as config, will be instantiated in configure_optimizers)
    print("  ⚙️  Optimizer...")
    print(f"     ✅ {cfg.optimizer._target_.split('.')[-1]}")
    print(f"        Learning rate: {cfg.base_lr}")
    print()

    # Scheduler (keep as config, will be instantiated in configure_optimizers)
    print("  📈 Scheduler...")
    print(f"     ✅ {cfg.scheduler._target_.split('.')[-1]}")
    print()

    # Contrastive learner (Lightning module)
    print("  🎯 Contrastive learner...")
    learner = ContrastiveLearner(
        system=system,
        encoder=encoder,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        loss_type=cfg.contrastive.loss_type,
        margin=cfg.contrastive.margin,
        temperature=cfg.contrastive.temperature
    )
    print(f"     ✅ {learner.__class__.__name__}")
    print(f"        Loss: {cfg.contrastive.loss_type}")
    print()

    # Trainer
    print("  🏋️  Trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)
    print(f"     ✅ PyTorch Lightning Trainer")
    print(f"        Max epochs: {cfg.trainer.max_epochs}")
    print(f"        Devices: {cfg.trainer.devices}")
    print()

    # ========================================================================
    # TRAINING
    # ========================================================================

    print("="*80)
    print("🎓 Starting Training")
    print("="*80)
    print()

    # Train model
    trainer.fit(learner, datamodule=data_module)

    print()
    print("="*80)
    print("✅ Training Complete!")
    print("="*80)

    # Print best model info
    if hasattr(trainer.checkpoint_callback, 'best_model_path'):
        print(f"📁 Best model: {trainer.checkpoint_callback.best_model_path}")
        print(f"📊 Best metric: {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
