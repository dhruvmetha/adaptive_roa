#!/usr/bin/env python3
"""
Training script for Pendulum Cartesian Latent Conditional Flow Matching (Facebook FM)

Uses Hydra target instantiation for clean, config-driven architecture.

Usage:
    # Train Pendulum Cartesian
    python src/flow_matching/pendulum_cartesian/latent_conditional/train.py

    # Override parameters
    python src/flow_matching/pendulum_cartesian/latent_conditional/train.py \
        flow_matching.latent_dim=8 \
        trainer.max_epochs=200
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl


@hydra.main(version_base=None, config_path="../../../../configs", config_name="train_pendulum_cartesian")
def main(cfg: DictConfig):
    """
    Training for Pendulum Cartesian Latent Conditional Flow Matching using Hydra instantiation

    All components are instantiated from config using Hydra's _target_ mechanism.
    """

    # Print configuration
    print("="*80)
    print("🚀 Pendulum Cartesian Latent Conditional Flow Matching Training (Facebook FM)")
    print("="*80)
    print(f"📋 Config: {cfg.get('name', 'pendulum_cartesian_latent_conditional_fm')}")
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

    # System (PendulumCartesianSystem)
    print("  🔧 System...")
    system = hydra.utils.instantiate(cfg.system)
    print(f"     ✅ {system.__class__.__name__}")
    print(f"        {system}")
    print()

    # Data module
    print("  📊 Data module...")
    data_module = hydra.utils.instantiate(cfg.data)
    print(f"     ✅ {data_module.__class__.__name__}")
    print(f"        Dataset: {data_module.data_file}")
    print(f"        Batch size: {cfg.batch_size}")
    print()

    # Model (PendulumCartesianUNet)
    print("  🏗️  Model...")
    model = hydra.utils.instantiate(cfg.model)
    print(f"     ✅ {model.__class__.__name__}")

    # Get model info
    model_info = model.get_model_info()
    print(f"        Architecture: {model_info['hidden_dims']}")
    print(f"        Time embedding: {model_info['time_emb_dim']}D")
    print(f"        Latent dim: {model_info['latent_dim']}D")
    print(f"        Input: embedded={model_info['embedded_dim']}, condition={model_info['condition_dim']}")
    print(f"        Output: {model_info['output_dim']}D")
    print(f"        Parameters: {model_info['total_parameters']:,}")
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

    # Flow matcher (PendulumCartesianLatentConditionalFlowMatcher)
    print("  🌊 Flow matcher...")
    flow_matcher = hydra.utils.instantiate(
        cfg.flow_matcher,
        system=system,
        model=model,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        latent_dim=cfg.flow_matching.latent_dim,
        mae_val_frequency=cfg.flow_matching.mae_val_frequency,
        _recursive_=False,  # Don't recursively instantiate optimizer/scheduler (they need model.parameters())
    )
    print(f"     ✅ {flow_matcher.__class__.__name__}")
    print(f"        Latent dim: {cfg.flow_matching.latent_dim}")
    print(f"        MAE validation: every {cfg.flow_matching.mae_val_frequency} epochs")
    print()

    # Callbacks
    print("  📞 Callbacks...")
    callbacks = []
    for callback_name, callback_cfg in cfg.callbacks.items():
        callback = hydra.utils.instantiate(callback_cfg)
        callbacks.append(callback)
        print(f"     ✅ {callback.__class__.__name__}")
    print()

    # Trainer
    print("  🏋️  Trainer...")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=None,  # Will add logger below
    )
    print(f"     ✅ {trainer.__class__.__name__}")
    print(f"        Max epochs: {cfg.trainer.max_epochs}")
    print(f"        Device: {cfg.trainer.devices}")
    print()

    # ========================================================================
    # TRAIN
    # ========================================================================

    print("="*80)
    print("🏋️  STARTING TRAINING")
    print("="*80)
    print()

    trainer.fit(flow_matcher, data_module)

    print()
    print("="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
