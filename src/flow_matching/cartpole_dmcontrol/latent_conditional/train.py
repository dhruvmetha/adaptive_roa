#!/usr/bin/env python3
"""
Training script for CartPole DeepMind Control Suite Latent Conditional Flow Matching

Uses Hydra target instantiation for clean, config-driven architecture.

Usage:
    # Train CartPole DM Control
    python src/flow_matching/cartpole_dmcontrol/latent_conditional/train.py

    # Override parameters
    python src/flow_matching/cartpole_dmcontrol/latent_conditional/train.py \
        trainer.max_epochs=200 \
        batch_size=512
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl


@hydra.main(version_base=None, config_path="../../../../configs", config_name="train_cartpole_dmcontrol")
def main(cfg: DictConfig):
    """
    Training for CartPole DM Control Latent Conditional Flow Matching

    All components are instantiated from config using Hydra's _target_ mechanism.
    """

    # Print configuration
    print("="*80)
    print("🚀 CartPole DM Control - Latent Conditional Flow Matching Training")
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

    # System (CartPoleDMControlSystem)
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

    # Model (reuses CartPoleUNet - same manifold!)
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
        # Fallback if get_model_info() doesn't exist
        total_params = sum(p.numel() for p in model.parameters())
        print(f"        Parameters: {total_params:,}")
    print()

    # Optimizer
    print("  ⚙️  Optimizer...")
    print(f"     ✅ {cfg.optimizer._target_.split('.')[-1]}")
    print(f"        Learning rate: {cfg.base_lr}")
    print()

    # Scheduler
    print("  📈 Scheduler...")
    print(f"     ✅ {cfg.scheduler._target_.split('.')[-1]}")
    print()

    # Flow matcher (CartPoleDMControlLatentConditionalFlowMatcher)
    print("  🌊 Flow matcher...")
    flow_matcher = hydra.utils.instantiate(
        cfg.flow_matcher,
        system=system,
        model=model,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        mae_val_frequency=cfg.flow_matching.mae_val_frequency,
        loss_weights=cfg.flow_matching.get('loss_weights', None),
        _recursive_=False
    )
    print(f"     ✅ {flow_matcher.__class__.__name__}")
    if cfg.flow_matching.get('loss_weights', None) is not None:
        print(f"        Loss weights: {dict(cfg.flow_matching.loss_weights)}")
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
    print(f"Checkpoints saved to: {trainer.checkpoint_callback.dirpath}")
    print(f"Logs saved to: {trainer.logger.log_dir}")


if __name__ == "__main__":
    main()
