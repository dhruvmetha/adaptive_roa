"""
Training script for simple direct latent circular flow matching
"""
import os
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from src.flow_matching.latent_circular.simple_flow_matcher import SimpleLatentCircularFlowMatcher


@hydra.main(config_path="../../../configs", config_name="train_simple_latent_circular.yaml")
def main(cfg: DictConfig):
    """Main training function for simple latent circular flow matching"""
    
    # Set GPU device from config
    if cfg.device.get("device_id") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device.device_id)
        print(f"Set CUDA_VISIBLE_DEVICES={cfg.device.device_id}")
    
    # Set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
    
    # Initialize data module
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Initialize model architecture (MLP)
    model_net = hydra.utils.instantiate(cfg.model)
    
    # Initialize optimizer and scheduler
    optimizer = hydra.utils.instantiate(cfg.optimizer) if cfg.get("optimizer") else None
    scheduler = hydra.utils.instantiate(cfg.get("scheduler")) if cfg.get("scheduler") else None
    
    # Initialize simple latent circular flow matching module
    flow_model = SimpleLatentCircularFlowMatcher(
        model=model_net,
        optimizer=optimizer,
        scheduler=scheduler,
        latent_dim=cfg.latent.latent_dim
    )
    
    print("Simple Latent Circular Flow Matching Model architecture:")
    print(flow_model)
    print(f"Total parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    print(f"Key difference: NO ENCODER - just random noise conditioning!")
    
    # Initialize trainer
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Train the model
    trainer.fit(model=flow_model, datamodule=data_module)
    
    print("\\nSimple latent circular flow matching training completed!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()