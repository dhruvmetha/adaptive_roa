#!/usr/bin/env python3
"""
Training script for Humanoid Latent Conditional Flow Matching (Facebook FM)

Uses Hydra target instantiation for clean, config-driven architecture.

Manifold: ℝ³⁴ × S² × ℝ³⁰ (67-dimensional state)

Usage:
    # Train Humanoid
    python src/flow_matching/humanoid/latent_conditional/train.py

    # Override parameters
    python src/flow_matching/humanoid/latent_conditional/train.py \
        trainer.max_epochs=200 \
        batch_size=512
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl


@hydra.main(version_base=None, config_path="../../../../configs", config_name="train_humanoid")
def main(cfg: DictConfig):
    """
    Training for Humanoid Latent Conditional Flow Matching using Hydra instantiation

    All components are instantiated from config using Hydra's _target_ mechanism.
    """

    # Print configuration
    print("="*80)
    print("🚀 Humanoid Conditional Flow Matching Training (Facebook FM)")
    print("="*80)
    print(f"📋 Config: {cfg.get('name', 'humanoid_lcfm')}")
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

    # System (HumanoidSystem)
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

    # Model (UniversalUNet)
    print("  🏗️  Model...")
    model = hydra.utils.instantiate(cfg.model)
    print(f"     ✅ {model.__class__.__name__}")

    # Get model info
    try:
        model_info = model.get_model_info()
        print(f"        Architecture: {model_info['hidden_dims']}")
        print(f"        Time embedding: {model_info.get('time_emb_dim', 'N/A')}D")
        print(f"        Input: {model_info.get('input_dim', 'N/A')}D")
        print(f"        Output: {model_info.get('output_dim', 'N/A')}D")
        print(f"        Parameters: {model_info['total_parameters']:,}")
    except AttributeError:
        # Fallback if get_model_info() doesn't exist
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

    # Flow matcher (HumanoidLatentConditionalFlowMatcher)
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
    print(f"        Manifold: ℝ³⁴ × S² × ℝ³⁰")
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
