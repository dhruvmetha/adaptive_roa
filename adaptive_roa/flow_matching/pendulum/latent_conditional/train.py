#!/usr/bin/env python3
"""
Pendulum Latent Conditional Flow Matching Training Script

Uses Hydra target instantiation for clean, config-driven architecture.
System: Pendulum (S¹×ℝ manifold)

Usage:
    # Train Pendulum
    python adaptive_roa/flow_matching/pendulum/latent_conditional/train.py

    # Override parameters
    python adaptive_roa/flow_matching/pendulum/latent_conditional/train.py \
        flow_matching.latent_dim=4 \
        trainer.max_epochs=200
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl


@hydra.main(version_base=None, config_path="../../../../configs", config_name="train_pendulum")
def main(cfg: DictConfig):
    """
    Unified training for Latent Conditional Flow Matching using Hydra instantiation

    All components are instantiated from config using Hydra's _target_ mechanism.
    """

    # Print configuration
    print("="*80)
    print("🚀 Latent Conditional Flow Matching Training (Facebook FM)")
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
    print(f"        Dataset: {data_module.data_file}")
    print(f"        Batch size: {cfg.batch_size}")
    print()

    # Model (e.g., LatentConditionalUNet1D or CartPoleLatentConditionalUNet1D)
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

    # Flow matcher (e.g., LatentConditionalFlowMatcher or CartPoleLatentConditionalFlowMatcher)
    print("  🌊 Flow matcher...")
    use_loss_weights = cfg.flow_matching.get('use_loss_weights', False)
    flow_matcher = hydra.utils.instantiate(
        cfg.flow_matcher,
        system=system,
        model=model,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        latent_dim=cfg.flow_matching.latent_dim,
        mae_val_frequency=cfg.flow_matching.mae_val_frequency,
        use_loss_weights=use_loss_weights,
        _recursive_=False
    )
    print(f"     ✅ {flow_matcher.__class__.__name__}")
    if use_loss_weights:
        print(f"        Loss weights: ENABLED (proportional to normalization limits)")
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


if __name__ == "__main__":
    main()
