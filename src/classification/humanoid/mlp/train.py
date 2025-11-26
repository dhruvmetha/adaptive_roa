#!/usr/bin/env python3
"""
Training script for Humanoid Baseline Classifier.
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
import torch
import sys
from pathlib import Path

# Add src to path
# src/classification/humanoid/mlp -> src/
sys.path.append(str(Path(__file__).resolve().parents[4]))

from src.systems.humanoid import HumanoidSystem
from src.data.humanoid_roa_data import HumanoidROADataModule
from src.classification.humanoid.mlp.classifier import HumanoidBaselineClassifier

@hydra.main(version_base=None, config_path="../../../../configs", config_name="train_humanoid_mlp")
def main(cfg: DictConfig):
    print(f"📋 Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set seed
    pl.seed_everything(cfg.seed)
    
    # Initialize System
    print("🔧 Initializing Humanoid System...")
    system_cfg = cfg.get("system", {})
    # Extract args if it's a dict, or handle if it's from a composed config
    if "_target_" in system_cfg:
        # If instantiated via hydra target, simpler manual init for baseline
        # We just need the bounds logic
        system = HumanoidSystem(
            bounds_file=system_cfg.get("bounds_file"),
            use_dynamic_bounds=system_cfg.get("use_dynamic_bounds", False)
        )
    else:
        system = HumanoidSystem()
        
    # Initialize Data Module
    print("💾 Initializing Data Module...")
    data_module = HumanoidROADataModule(
        data_file=cfg.data.file,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.seed
    )
    
    # Initialize Model
    print("🧠 Initializing Baseline Model...")
    model = HumanoidBaselineClassifier(
        system=system,
        model_config=cfg.model,
        learning_rate=cfg.training.learning_rate
    )
    
    # Instantiate Trainer via Hydra
    print("⚡ Initializing Trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Train
    print("🚀 Starting Training...")
    trainer.fit(model, datamodule=data_module)
    
    # Test
    print("🧪 Starting Testing...")
    trainer.test(model, datamodule=data_module, ckpt_path="best")
    
    print("✅ Done!")

if __name__ == "__main__":
    main()
