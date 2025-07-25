"""
Training script for standard flow matching
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from .flow_matcher import StandardFlowMatcher
from ..base.config import FlowMatchingConfig


@hydra.main(config_path="../../../configs", config_name="train_flow_matching.yaml")
def main(cfg: DictConfig):
    """Main training function for standard flow matching"""
    
    # Set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
    
    # Create configuration
    config = FlowMatchingConfig(
        sigma=cfg.get("sigma", 0.0),
        hidden_dims=tuple(cfg.model.get("hidden_dims", [64, 128, 256])),
        time_emb_dim=cfg.model.get("time_emb_dim", 128),
        num_integration_steps=cfg.get("num_integration_steps", 100)
    )
    
    # Initialize data module
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Initialize model architecture
    model_net = hydra.utils.instantiate(cfg.model)
    
    # Initialize flow matching module
    flow_model = StandardFlowMatcher(
        model=model_net,
        optimizer=cfg.optimizer,
        scheduler=cfg.get("scheduler", None),
        config=config
    )
    
    print("Model architecture:")
    print(flow_model)
    print(f"Total parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    
    # Initialize trainer
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Train the model
    trainer.fit(model=flow_model, datamodule=data_module)
    
    print("\\nTraining completed!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()