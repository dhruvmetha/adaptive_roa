#!/usr/bin/env python3
"""
UNIFIED Training script for Latent Conditional Flow Matching (Facebook FM)

Uses Hydra target instantiation for clean, config-driven architecture.

Supports multiple systems:
- Pendulum (S¹×ℝ)
- CartPole (ℝ²×S¹×ℝ)

Usage:
    # Train Pendulum
    python src/flow_matching/train_latent_conditional.py --config-name=train_pendulum_lcfm

    # Train CartPole
    python src/flow_matching/train_latent_conditional.py --config-name=train_cartpole_lcfm

    # Override parameters
    python src/flow_matching/train_latent_conditional.py --config-name=train_pendulum_lcfm \
        trainer.max_epochs=200
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


@hydra.main(version_base=None, config_path="../../../../configs", config_name="train_cartpole")
def main(cfg: DictConfig):
    """
    Unified training for Latent Conditional Flow Matching using Hydra instantiation

    All components are instantiated from config using Hydra's _target_ mechanism.
    """

    # Print configuration
    print("="*80)
    print("🚀 Conditional Flow Matching Training (Facebook FM)")
    print("="*80)
    print(f"📋 Config: {cfg.get('name', 'unnamed')}")
    print(f"🎲 Seed: {cfg.seed}")
    print("="*80)
    print()

    # Set random seeds
    pl.seed_everything(cfg.seed)

    # ========================================================================
    # INSTANTIATE ALL COMPONENTS FROM CONFIG
    # ========================================================================

    print("📥 Instantiating components from config...")
    print()

    # System (e.g., PendulumSystemLCFM or CartPoleSystemLCFM)
    print("  🔧 System...")
    system = hydra.utils.instantiate(cfg.system)
    print(f"     ✅ {system.__class__.__name__}")
    print(f"        {system}")
    print()

    # Data module
    print("  📊 Data module...")
    data_module = hydra.utils.instantiate(cfg.data)
    print(f"     ✅ {data_module.__class__.__name__}")
    print(f"        State dim: {data_module.state_dim}")
    print(f"        Embedded dim: {data_module.embedded_dim}")
    print(f"        Batch size: {cfg.batch_size}")
    print()

    # Model (e.g., LatentConditionalUNet1D or CartPoleLatentConditionalUNet1D)
    print("  🏗️  Model...")
    model = hydra.utils.instantiate(cfg.model)
    print(f"     ✅ {model.__class__.__name__}")

    # Get model info
    try:
        model_info = model.get_model_info()
        print(f"        Architecture: {model_info['hidden_dims']}")
        print(f"        Time embedding: {model_info['time_emb_dim']}D")
        print(f"        Input: embedded={model_info['embedded_dim']}, condition={model_info['condition_dim']}")
        print(f"        Output: {model_info['output_dim']}D")
        print(f"        Parameters: {model_info['total_parameters']:,}")
    except (AttributeError, KeyError):
        # Fallback if get_model_info() doesn't exist or doesn't have all keys
        total_params = sum(p.numel() for p in model.parameters())
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

    # Flow matcher (e.g., ConditionalFlowMatcher or CartPoleConditionalFlowMatcher)
    print("  🌊 Flow matcher...")
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
    print(f"     ✅ {flow_matcher.__class__.__name__}")
    print()

    # Trainer
    print("  🎯 Trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)
    print(f"     ✅ Max epochs: {cfg.trainer.max_epochs}")
    print(f"        Devices: {cfg.trainer.devices}")
    print()

    # ========================================================================
    # TRAINING
    # ========================================================================

    print("="*80)
    print("🚀 Starting Training")
    print("="*80)
    print()

    trainer.fit(flow_matcher, data_module)

    print()
    print("="*80)
    print("✅ Training Completed!")
    print("="*80)
    print()
    # Resolve checkpoint directory robustly for Lightning 2.x
    ckpt_dir = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ckpt_dir = cb.dirpath
            break
    print(f"Checkpoints saved to: {ckpt_dir if ckpt_dir is not None else 'N/A'}")
    print(f"Logs saved to: {trainer.logger.log_dir if hasattr(trainer.logger, 'log_dir') else 'N/A'}")


if __name__ == "__main__":
    main()
